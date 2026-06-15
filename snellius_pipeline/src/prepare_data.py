"""
Prepare judge input JSONL files.

Scenarios produced:
  S0           ssa-mte/scenario_0       — reference-free DA scoring (0-100)
  S3 (ssa-mte) ssa-mte/scenario_3       — pairwise MT output vs human reference (NEW)
  S1           <dataset>/scenario_1     — reference-aware audit (A/B/both/neither)
  S2           <dataset>/scenario_2     — multi-dim rating of human GOLD reference
  S2_model     <dataset>/scenario_2_model — multi-dim rating of GEMINI OUTPUT (NEW)
  S3           <dataset>/scenario_3     — pairwise model vs human gold

S2 and S2_model share record IDs (suffix differs) so ratings can be compared
pairwise (Gemini vs human reference) per record and per dimension.

Usage:
    python -m src.prepare_data --sample 1            # smoke test
    python -m src.prepare_data --sample 200          # production
    python -m src.prepare_data --sample 200 --skip-ssa-mte
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    AFROBENCH_DATA, SSA_MTE_DATA, JUDGE_INPUTS, DATASETS, METRIC_COLS,
    SSA_MTE_CONFIG, LANG_NAMES,
)
from prompts.judge_prompts import format_judge_prompt


random.seed(42)


# File discovery (AfroBench)

def find_lang_csvs(dataset_key: str) -> dict:
    base = AFROBENCH_DATA / "afrobench" / dataset_key
    if not base.exists():
        return {}
    files = {}
    prefix = dataset_key + "_"
    for csv_path in base.rglob("*.csv"):
        stem = csv_path.stem
        if "combined" in stem or not stem.startswith(prefix) or not stem.endswith("_results"):
            continue
        middle = stem[len(prefix):-len("_results")]
        if not middle or middle == "combined":
            continue
        lang = middle.split("-")[-1] if "-" in middle else middle.split("_")[0]
        if lang and lang.isalpha():
            files[lang] = csv_path
    return files


def load_csv(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# Filters / sampling 

def is_wrong(record: dict, task_type: str) -> bool:
    keys = METRIC_COLS.get(task_type, [])
    if not keys:
        return False
    for k in keys:
        try:
            if float(record.get(k, "")) < 1.0:
                return True
        except (ValueError, TypeError):
            continue
    return False


def sample_records(records: list, n: int) -> list:
    return records if n >= len(records) else random.sample(records, n)


# SSA-MTE loading 

def _resolve_aliases(record: dict, aliases: dict) -> dict:
    out = {}
    for canonical, options in aliases.items():
        for opt in options:
            if opt in record and record[opt] not in (None, ""):
                out[canonical] = record[opt]
                break
    return out


def load_ssa_mte() -> list:
    aliases = SSA_MTE_CONFIG["column_aliases"]
    hf_repo = SSA_MTE_CONFIG["hf_repo"]
    hf_split = SSA_MTE_CONFIG["hf_split"]
    records = None
    try:
        from datasets import load_dataset, get_dataset_config_names
        cfgs = get_dataset_config_names(hf_repo)
        print(f"  Loading {len(cfgs)} per-LP configs from {hf_repo} ...")
        acc = []
        for c in cfgs:
            try:
                ds = load_dataset(hf_repo, c, split=hf_split)
                rows = [dict(r) for r in ds]
                for r in rows:
                    r.setdefault("lp", c)
                acc.extend(rows)
                print(f"    {c}: {len(rows)} rows")
            except Exception as ce:
                print(f"    {c}: FAIL ({type(ce).__name__})")
        if acc:
            records = acc
    except Exception as e:
        print(f"  HF load failed: {type(e).__name__}: {e}")

    if not records:
        raise RuntimeError(
            f"Could not load SSA-MTE from {hf_repo}. If gated, accept the license "
            f"and set HF_TOKEN."
        )

    normalized, skipped = [], 0
    for r in records:
        norm = _resolve_aliases(r, aliases)
        if not all(k in norm for k in ("src", "mt", "lp", "score")):
            skipped += 1
            continue
        lp = str(norm["lp"])
        norm["language"] = lp.split("-")[-1] if "-" in lp else lp
        try:
            norm["score"] = float(norm["score"])
        except (ValueError, TypeError):
            skipped += 1
            continue
        normalized.append(norm)
    if skipped:
        print(f"  ! Skipped {skipped} rows missing required columns / score.")

    sys_filter = SSA_MTE_CONFIG.get("system_filter")
    if sys_filter:
        before = len(normalized)
        normalized = [r for r in normalized if r.get("system") == sys_filter]
        print(f"  Filtered to {len(normalized)}/{before} rows where system == {sys_filter!r}")
    return normalized


def _ssa_by_lp(records):
    by_lp = {}
    for r in records:
        by_lp.setdefault(r["lp"], []).append(r)
    allowed = SSA_MTE_CONFIG.get("language_pairs")
    if allowed:
        before = len(by_lp)
        by_lp = {lp: rs for lp, rs in by_lp.items() if lp in allowed}
        print(f"  Filtered to {len(by_lp)}/{before} configured LPs")
    return by_lp


# Scenario builders

def build_scenario_0_ssa_mte(sample_n: int) -> int:
    """S0: SSA-MTE reference-free meta-evaluation."""
    print(f"\n{'─'*60}\n SSA-MTE — Scenario 0 / Phase 1\n{'─'*60}")
    records = load_ssa_mte()
    if not records:
        print("  ⚠ No SSA-MTE records loaded.")
        return 0
    by_lp = _ssa_by_lp(records)
    out_dir = JUDGE_INPUTS / SSA_MTE_CONFIG["name"] / "scenario_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for lp, recs in sorted(by_lp.items()):
        sampled = sample_records(recs, sample_n)
        src_code, tgt_code = (lp.split("-") + [""])[:2]
        out_path = out_dir / f"{lp}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for i, rec in enumerate(sampled):
                prompt = format_judge_prompt(
                    scenario=0, task_type="translation_qe",
                    src_lang=LANG_NAMES.get(src_code, src_code),
                    tgt_lang=LANG_NAMES.get(tgt_code, tgt_code),
                    src=rec["src"], mt=rec["mt"],
                )
                f.write(json.dumps({
                    "id": f"ssa-mte_p1_{lp}_{i}",
                    "dataset": "SSA-MTE", "language": rec["language"],
                    "language_pair": lp, "scenario": 0, "task_type": "translation_qe",
                    "judge_prompt": prompt,
                    "meta": {
                        "src": rec["src"], "mt": rec["mt"],
                        "human_score": rec["score"],
                        "mt_system": rec.get("system", ""),
                        "evaluated_model": rec.get("system", "various"),
                        "phase": "1_meta_evaluation",
                    },
                }, ensure_ascii=False) + "\n")
                total += 1
        print(f"  {lp}: {len(recs)} → sampled {len(sampled)} → {out_path.name}")
    return total


def build_scenario_3_ssa_mte(sample_n: int) -> int:
    """S3 (NEW): SSA-MTE pairwise — MT output vs human reference, randomised.
    Requires the `ref` column (human reference translation). Skips rows without it."""
    print(f"\n{'─'*60}\n SSA-MTE — Scenario 3 (pairwise MT vs human ref)\n{'─'*60}")
    records = load_ssa_mte()
    records = [r for r in records if r.get("ref")]  # need a human reference
    if not records:
        print("  ⚠ No SSA-MTE rows with a 'ref' column — cannot build pairwise.")
        return 0
    by_lp = _ssa_by_lp(records)
    out_dir = JUDGE_INPUTS / SSA_MTE_CONFIG["name"] / "scenario_3"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for lp, recs in sorted(by_lp.items()):
        sampled = sample_records(recs, sample_n)
        out_path = out_dir / f"{lp}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for i, rec in enumerate(sampled):
                mt_output = rec["mt"]          # Gemini translation
                human_ref = rec["ref"]         # human reference
                if random.random() < 0.5:
                    response_a, response_b, model_position = mt_output, human_ref, "A"
                else:
                    response_a, response_b, model_position = human_ref, mt_output, "B"
                prompt = format_judge_prompt(
                    scenario=3, task_type="translation",
                    input_prompt=rec["src"],
                    response_a=response_a, response_b=response_b,
                )
                f.write(json.dumps({
                    "id": f"ssa-mte_s3_{lp}_{i}",
                    "dataset": "SSA-MTE", "language": rec["language"],
                    "language_pair": lp, "scenario": 3, "task_type": "translation",
                    "judge_prompt": prompt,
                    "meta": {
                        "src": rec["src"], "mt": mt_output, "ref": human_ref,
                        "human_score": rec["score"],
                        "model_position": model_position,
                        "mt_system": rec.get("system", ""),
                        "evaluated_model": rec.get("system", "gemini"),
                    },
                }, ensure_ascii=False) + "\n")
                total += 1
        print(f"  {lp}: sampled {len(sampled)} (mt@{model_position}) → {out_path.name}")
    return total


def build_scenario_1(dataset_name, ds_cfg, sample_n):
    task_type = ds_cfg["task_type"]
    lang_files = find_lang_csvs(ds_cfg["afrobench_key"])
    if not lang_files:
        print(f"  ⚠ No files found for {dataset_name}")
        return 0
    out_dir = JUDGE_INPUTS / dataset_name / "scenario_1"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for lang, csv_path in sorted(lang_files.items()):
        records = load_csv(csv_path)
        wrong = [r for r in records if is_wrong(r, task_type)]
        if not wrong:
            print(f"  {lang}: 0 wrong cases, skipping")
            continue
        sampled = sample_records(wrong, sample_n)
        out_path = out_dir / f"{lang}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in sampled:
                model_text = rec.get("raw_output", "") or rec.get("filtered_output", "")
                gold_text = rec.get("target", "")
                if random.random() < 0.5:
                    response_a, response_b, model_position = model_text, gold_text, "A"
                else:
                    response_a, response_b, model_position = gold_text, model_text, "B"
                prompt = format_judge_prompt(
                    scenario=1, task_type=task_type,
                    input_prompt=rec.get("prompt", ""),
                    response_a=response_a, response_b=response_b,
                )
                f.write(json.dumps({
                    "id": f"{dataset_name}_s1_{lang}_{rec.get('index', total)}",
                    "dataset": dataset_name, "language": lang,
                    "scenario": 1, "task_type": task_type, "judge_prompt": prompt,
                    "meta": {
                        "original_index": rec.get("index", ""),
                        "prompt_no": rec.get("prompt_no", ""),
                        "gold_label": gold_text,
                        "model_raw": rec.get("raw_output", ""),
                        "model_filtered": rec.get("filtered_output", ""),
                        "model_position": model_position,
                        "auto_metrics": {k: rec.get(k, "") for k in METRIC_COLS.get(task_type, [])},
                        "evaluated_model": "gemini-2.5-pro",
                    },
                }, ensure_ascii=False) + "\n")
                total += 1
        print(f"  {lang}: {len(wrong)} wrong → sampled {len(sampled)} (model@{model_position}) → {out_path.name}")
    return total


def _build_scenario_2_generic(dataset_name, ds_cfg, sample_n, rate_target):
    """Shared builder for S2 (rate gold) and S2_model (rate Gemini output).
    rate_target in {'gold', 'model'} chooses which text is rated."""
    task_type = ds_cfg["task_type"]
    lang_files = find_lang_csvs(ds_cfg["afrobench_key"])
    if not lang_files:
        print(f"  ! No files found for {dataset_name}")
        return 0
    subdir = "scenario_2" if rate_target == "gold" else "scenario_2_model"
    id_tag = "s2" if rate_target == "gold" else "s2model"
    out_dir = JUDGE_INPUTS / dataset_name / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for lang, csv_path in sorted(lang_files.items()):
        records = load_csv(csv_path)
        sampled = sample_records(records, sample_n)
        out_path = out_dir / f"{lang}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in sampled:
                gold_text = rec.get("target", "")
                model_text = rec.get("raw_output", "") or rec.get("filtered_output", "")
                rated_text = gold_text if rate_target == "gold" else model_text
                # Shared record id base so gold/model ratings can be paired:
                base_id = f"{dataset_name}_{lang}_{rec.get('index', total)}"
                prompt = format_judge_prompt(
                    scenario=2, task_type=task_type,
                    input_prompt=rec.get("prompt", ""),
                    gold_label=rated_text,   # the rubric template rates this text
                )
                f.write(json.dumps({
                    "id": f"{dataset_name}_{id_tag}_{lang}_{rec.get('index', total)}",
                    "pair_id": base_id,        # identical across gold/model for pairing
                    "dataset": dataset_name, "language": lang,
                    "scenario": 2, "task_type": task_type, "judge_prompt": prompt,
                    "meta": {
                        "original_index": rec.get("index", ""),
                        "prompt_no": rec.get("prompt_no", ""),
                        "rated_text_kind": rate_target,   # 'gold' or 'model'
                        "rated_text": rated_text,
                        "gold_label": gold_text,
                        "model_raw": rec.get("raw_output", ""),
                        "model_filtered": rec.get("filtered_output", ""),
                        "evaluated_model": "gemini-2.5-pro",
                    },
                }, ensure_ascii=False) + "\n")
                total += 1
        print(f"  {lang} [{rate_target}]: sampled {len(sampled)} → {subdir}/{out_path.name}")
    return total


def build_scenario_2(dataset_name, ds_cfg, sample_n):
    """Build BOTH the gold-rating (scenario_2) and model-rating (scenario_2_model)."""
    n_gold = _build_scenario_2_generic(dataset_name, ds_cfg, sample_n, "gold")
    n_model = _build_scenario_2_generic(dataset_name, ds_cfg, sample_n, "model")
    return n_gold + n_model


def build_scenario_3(dataset_name, ds_cfg, sample_n):
    task_type = ds_cfg["task_type"]
    lang_files = find_lang_csvs(ds_cfg["afrobench_key"])
    if not lang_files:
        print(f"  ⚠ No files found for {dataset_name}")
        return 0
    out_dir = JUDGE_INPUTS / dataset_name / "scenario_3"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for lang, csv_path in sorted(lang_files.items()):
        records = load_csv(csv_path)
        sampled = sample_records(records, sample_n)
        out_path = out_dir / f"{lang}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in sampled:
                if task_type == "math_reasoning":
                    model_output = rec.get("raw_output", "")
                else:
                    model_output = rec.get("raw_output", "") or rec.get("filtered_output", "")
                human_label = rec.get("target", "")
                if random.random() < 0.5:
                    response_a, response_b, model_position = model_output, human_label, "A"
                else:
                    response_a, response_b, model_position = human_label, model_output, "B"
                prompt = format_judge_prompt(
                    scenario=3, task_type=task_type,
                    input_prompt=rec.get("prompt", ""),
                    response_a=response_a, response_b=response_b,
                )
                f.write(json.dumps({
                    "id": f"{dataset_name}_s3_{lang}_{rec.get('index', total)}",
                    "dataset": dataset_name, "language": lang,
                    "scenario": 3, "task_type": task_type, "judge_prompt": prompt,
                    "meta": {
                        "original_index": rec.get("index", ""),
                        "prompt_no": rec.get("prompt_no", ""),
                        "gold_label": human_label,
                        "model_raw": model_output,
                        "model_filtered": rec.get("filtered_output", ""),
                        "model_position": model_position,
                        "evaluated_model": "gemini-2.5-pro",
                    },
                }, ensure_ascii=False) + "\n")
                total += 1
        print(f"  {lang}: sampled {len(sampled)} (model@{model_position}) → {out_path.name}")
    return total


# Main

BUILDERS = {1: build_scenario_1, 2: build_scenario_2, 3: build_scenario_3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--scenario", type=int, default=None)
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--skip-ssa-mte", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print(f"PREPARING JUDGE INPUTS (sample={args.sample})")
    print(f"Output: {JUDGE_INPUTS}")
    print("=" * 70)
    grand = 0

    do_ssa = (not args.skip_ssa_mte
              and (args.scenario is None or args.scenario in (0, 3))
              and (args.dataset is None
                   or args.dataset.lower().replace("_", "-") in ("ssa-mte", "ssamte")))
    if do_ssa:
        if args.scenario is None or args.scenario == 0:
            grand += build_scenario_0_ssa_mte(args.sample)
        if args.scenario is None or args.scenario == 3:
            grand += build_scenario_3_ssa_mte(args.sample)

    if args.dataset and args.dataset.lower().replace("_", "-") in ("ssa-mte", "ssamte"):
        targets = {}
    elif args.dataset:
        targets = {args.dataset: DATASETS[args.dataset]}
    else:
        targets = DATASETS

    for ds_name, ds_cfg in targets.items():
        scenarios = [args.scenario] if args.scenario is not None else ds_cfg["scenarios"]
        for sc in scenarios:
            if sc not in BUILDERS or sc not in ds_cfg["scenarios"]:
                continue
            print(f"\n{'─'*60}\n📊 {ds_name} — Scenario {sc} ({ds_cfg['task_type']})\n{'─'*60}")
            grand += BUILDERS[sc](ds_name, ds_cfg, args.sample)

    print(f"\n{'='*70}\nTotal: {grand} judge input records\n{'='*70}")


if __name__ == "__main__":
    main()
