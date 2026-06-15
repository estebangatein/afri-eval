"""
Compute evaluation metrics from judge outputs.

Metric semantics by scenario:

  S0 (Phase 1 / SSA-MTE):
        Spearman & Pearson correlation between judge 0–100 score and human DA
        score, per LP and overall. MAE included.

  S1 (Reference-aware audit, A/B/both/neither, position-randomised):
        - gold_confirmed_rate     judge picked gold's position
        - model_confirmed_rate    judge picked model's position
        - both_rate               judge said "both" → auto-metric was too strict
        - neither_rate            judge said "neither" → task quality issue
        - position_bias_decided   |P(A) − 0.5| × 2 among A/B verdicts only

  S2 (Multi-dim rating):
        Per-dimension mean / std / distribution / n, globally and per language.

  S3 (Pairwise, A/B/tie):
        Position bias (A_rate, B_rate, tie_rate), human_pref_rate among
        decided cases.

Skipped records (where the prompt exceeded the judge's max_prompt_chars budget)
are counted but excluded from all rate calculations.

Usage:
    python -m src.compute_metrics
    python -m src.compute_metrics --dataset SSA-MTE
    python -m src.compute_metrics --scenario 1
"""
import argparse
import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import JUDGE_OUTPUTS, RESULTS, LANG_NAMES


# Response parsers 

def parse_scenario_0(filtered):
    if isinstance(filtered, (int, float)):
        return float(filtered)
    if isinstance(filtered, str):
        try:
            return float(filtered)
        except ValueError:
            return None
    return None


def parse_scenario_1(response):
    """Normalise to one of: 'A', 'B', 'both', 'neither', 'unparseable'."""
    r = str(response).strip()
    if r in {"A", "B", "both", "neither", "unparseable", "skipped"}:
        return r
    rl = r.lower()
    if "neither" in rl:
        return "neither"
    if "both" in rl:
        return "both"
    if rl in {"a", '"a"'}:
        return "A"
    if rl in {"b", '"b"'}:
        return "B"
    return "unparseable"


_S2_DIMS = ("adequacy", "fluency", "terminology",
            "informativeness", "coherence", "faithfulness", "overall")


def parse_scenario_2(response):
    if isinstance(response, dict):
        return {k.lower(): int(v) for k, v in response.items()
                if isinstance(v, (int, float)) and 1 <= int(v) <= 5}
    if not isinstance(response, str):
        return {}
    out = {}
    for dim in _S2_DIMS:
        m = re.search(rf"\b{dim}\b\s*\*{{0,2}}\s*[:\-–]\s*\*{{0,2}}\s*([1-5])",
                      response, re.IGNORECASE)
        if m:
            out[dim] = int(m.group(1))
    return out


def parse_scenario_3(response):
    r = str(response).strip().upper()
    if "TIE" in r:
        return "tie"
    if r in {"A", '"A"'}:
        return "A"
    if r in {"B", '"B"'}:
        return "B"
    if "A" in r and "B" not in r:
        return "A"
    if "B" in r and "A" not in r:
        return "B"
    return "unparseable"


def _spearman_pearson(xs, ys):
    try:
        from scipy.stats import spearmanr, pearsonr
        if len(xs) < 3:
            return None, None, len(xs)
        sp = spearmanr(xs, ys).correlation
        pr = pearsonr(xs, ys)[0]
        sp = None if sp is None or sp != sp else round(float(sp), 4)
        pr = None if pr is None or pr != pr else round(float(pr), 4)
        return sp, pr, len(xs)
    except ImportError:
        return None, None, len(xs)


# Per-scenario metrics

def metrics_scenario_0(records):
    pairs, by_lp = [], defaultdict(list)
    unparseable = 0
    for rec in records:
        if rec.get("skipped"):
            continue
        llm = parse_scenario_0(rec.get("judge_filtered_output"))
        try:
            human = float(rec.get("meta", {}).get("human_score"))
        except (TypeError, ValueError):
            human = None
        if llm is None or human is None:
            unparseable += 1
            continue
        lp = rec.get("language_pair") or rec.get("meta", {}).get("language_pair", "?")
        pairs.append((llm, human))
        by_lp[lp].append((llm, human))

    overall_sp, overall_pr, n = _spearman_pearson([p[0] for p in pairs],
                                                   [p[1] for p in pairs])
    mae = None
    if pairs:
        mae = round(sum(abs(a - b) for a, b in pairs) / len(pairs), 2)

    by_lp_results = {}
    for lp, pp in sorted(by_lp.items()):
        sp, pr, nn = _spearman_pearson([p[0] for p in pp], [p[1] for p in pp])
        by_lp_results[lp] = {
            "n": nn, "spearman": sp, "pearson": pr,
            "llm_mean": round(sum(p[0] for p in pp) / max(len(pp), 1), 2),
            "human_mean": round(sum(p[1] for p in pp) / max(len(pp), 1), 2),
            "mae": round(sum(abs(a - b) for a, b in pp) / max(len(pp), 1), 2),
        }
    return {
        "total": len(records),
        "valid": n,
        "unparseable": unparseable,
        "skipped": sum(1 for r in records if r.get("skipped")),
        "overall_spearman": overall_sp,
        "overall_pearson":  overall_pr,
        "overall_mae":      mae,
        "by_language_pair": by_lp_results,
    }


def metrics_scenario_1(records):
    """Reference-aware audit. Uses meta.model_position to map A/B to model/gold."""
    verdicts = Counter()
    by_lang = defaultdict(Counter)
    a_count = b_count = 0
    skipped = 0
    for rec in records:
        if rec.get("skipped"):
            skipped += 1
            continue
        v = parse_scenario_1(rec.get("judge_filtered_output", ""))
        verdicts[v] += 1
        lang = rec.get("language", "?")
        by_lang[lang][v] += 1

        model_pos = rec.get("meta", {}).get("model_position")
        # gold's position = the OTHER position
        gold_pos = "B" if model_pos == "A" else ("A" if model_pos == "B" else None)
        if v == "A":
            a_count += 1
        elif v == "B":
            b_count += 1
        # Tag whether judge picked model or gold
        if v in ("A", "B") and model_pos in ("A", "B"):
            verdicts["__model_confirmed__" if v == model_pos else "__gold_confirmed__"] += 1

    total = sum(verdicts[k] for k in verdicts if not k.startswith("__"))
    decided = a_count + b_count
    return {
        "total": total,
        "skipped": skipped,
        "verdicts": {k: v for k, v in verdicts.items() if not k.startswith("__")},
        "gold_confirmed_rate":  round(verdicts.get("__gold_confirmed__", 0)  / max(total, 1), 4),
        "model_confirmed_rate": round(verdicts.get("__model_confirmed__", 0) / max(total, 1), 4),
        "both_rate":            round(verdicts.get("both", 0)               / max(total, 1), 4),
        "neither_rate":         round(verdicts.get("neither", 0)            / max(total, 1), 4),
        "unparseable_rate":     round(verdicts.get("unparseable", 0)        / max(total, 1), 4),
        "position_bias_decided": (
            round(abs(a_count / decided - 0.5) * 2, 4) if decided > 0 else None
        ),
        "p_A_decided": round(a_count / decided, 4) if decided > 0 else None,
        "by_language": {lang: dict(c) for lang, c in sorted(by_lang.items())},
    }


def metrics_scenario_2(records):
    valid = unparseable = skipped = 0
    by_dim = defaultdict(list)
    by_lang_dim = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec.get("skipped"):
            skipped += 1
            continue
        dims = parse_scenario_2(rec.get("judge_filtered_output", ""))
        if not dims:
            unparseable += 1
            continue
        valid += 1
        lang = rec.get("language", "?")
        for d, v in dims.items():
            by_dim[d].append(v)
            by_lang_dim[lang][d].append(v)

    def _summary(values):
        n = len(values)
        if n == 0:
            return {"n": 0, "mean": None, "std": None, "distribution": {}}
        mean = sum(values) / n
        var  = sum((v - mean) ** 2 for v in values) / n
        return {
            "n": n,
            "mean": round(mean, 3),
            "std":  round(var ** 0.5, 3),
            "distribution": {k: v for k, v in sorted(Counter(values).items())},
        }

    return {
        "total": len(records),
        "valid": valid,
        "unparseable": unparseable,
        "skipped": skipped,
        "dimensions": {d: _summary(vs) for d, vs in sorted(by_dim.items())},
        "by_language": {
            lang: {"dimensions": {dn: _summary(vs) for dn, vs in sorted(d.items())}}
            for lang, d in sorted(by_lang_dim.items())
        },
    }


def metrics_scenario_3(records):
    choices = Counter()
    by_lang = defaultdict(Counter)
    human_pref = model_pref = ties = skipped = 0
    for rec in records:
        if rec.get("skipped"):
            skipped += 1
            continue
        v = parse_scenario_3(rec.get("judge_filtered_output", ""))
        choices[v] += 1
        by_lang[rec.get("language", "?")][v] += 1
        model_pos = rec.get("meta", {}).get("model_position", "?")
        if v == "tie":
            ties += 1
        elif v == model_pos:
            model_pref += 1
        elif v in ("A", "B"):
            human_pref += 1

    total = sum(choices.values())
    decided = model_pref + human_pref
    return {
        "total": total,
        "skipped": skipped,
        "choices": dict(choices),
        "position_bias": {
            "A_rate":   round(choices.get("A", 0) / max(total, 1), 4),
            "B_rate":   round(choices.get("B", 0) / max(total, 1), 4),
            "tie_rate": round(ties / max(total, 1), 4),
            "decided_bias": (round(abs(choices.get("A", 0) /
                                       (choices.get("A", 0) + choices.get("B", 0)) - 0.5) * 2, 4)
                              if (choices.get("A", 0) + choices.get("B", 0)) > 0 else None),
        },
        "preference": {
            "human_preferred": human_pref,
            "model_preferred": model_pref,
            "ties": ties,
            "human_pref_rate": round(human_pref / max(decided, 1), 4),
        },
        "unparseable_rate": round(choices.get("unparseable", 0) / max(total, 1), 4),
        "by_language": {lang: dict(c) for lang, c in sorted(by_lang.items())},
    }


METRIC_FUNCS = {0: metrics_scenario_0, 1: metrics_scenario_1,
                2: metrics_scenario_2, 3: metrics_scenario_3}


# Main

def collect_records(base_dir, dataset=None, scenario=None):
    groups = defaultdict(list)
    for jsonl_path in base_dir.rglob("*.jsonl"):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ds = rec.get("dataset", "?")
                sc = rec.get("scenario", 0)
                jm = rec.get("judge_model", "?")
                if dataset and ds != dataset: continue
                if scenario is not None and sc != scenario: continue
                groups[(ds, sc, jm)].append(rec)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=str(JUDGE_OUTPUTS))
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--scenario", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    groups = collect_records(Path(args.input_dir), args.dataset, args.scenario)
    if not groups:
        print("No judge output records found.")
        return

    all_results = {}
    for (ds, sc, jm), records in sorted(groups.items()):
        key = f"{ds}_scenario{sc}_{jm}"
        print(f"\n{'='*60}\n📊 {ds} | S{sc} | {jm} | n={len(records)}\n{'='*60}")
        func = METRIC_FUNCS.get(sc)
        if not func:
            print(f"  ⚠ No metric function for scenario {sc}")
            continue
        result = func(records)
        all_results[key] = result
        for k, v in result.items():
            if k in ("by_language", "by_language_pair", "dimensions"):
                print(f"  {k}:")
                for sub, data in v.items():
                    label = LANG_NAMES.get(sub, sub) if k == "by_language" else sub
                    print(f"    {label}: {data}")
            else:
                print(f"  {k}: {v}")

    out_path = Path(args.output) if args.output else RESULTS / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
