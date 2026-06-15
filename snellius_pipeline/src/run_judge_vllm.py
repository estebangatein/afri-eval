"""
Run judge inference using vLLM — LOADS THE MODEL ONCE per job and iterates over
all scenario directories under the input root.

Usage:
    python -m src.run_judge_vllm --model gemma-3-27b-it
    python -m src.run_judge_vllm --model gemma-3-27b-it --input-root /scratch-shared/egatein/judge_inputs

Output mirrors the input tree:
    judge_inputs/<dataset>/<scenario_dir>/<lang>.jsonl
        -> judge_outputs/<dataset>/<scenario_dir>/<model>/<lang>.jsonl

Robustness:
  • llm.chat() applies each model's chat template (fixes LLaMA-3 empty output).
  • _strip_think() removes <think> blocks (reasoning models) before parsing.
  • Verdict parsers handle verbose, conversational outputs (the chat template
    makes models add reasoning before the final A/B/tie/verdict).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import JUDGE_MODELS, JUDGE_INPUTS, JUDGE_OUTPUTS


# Reasoning-block stripping

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    if not text:
        return text
    stripped = _THINK_RE.sub("", text)
    if "</think>" in stripped.lower():
        idx = stripped.lower().rfind("</think>")
        stripped = stripped[idx + len("</think>"):]
    stripped = stripped.strip()
    return stripped if stripped else text


# Verdict extraction helpers 
# The chat template makes models verbose ("Translation A is better because...").
# These helpers find the FINAL verdict rather than the first stray letter.

# Phrases that introduce a verdict, captured group is the letter.
_VERDICT_PHRASES = [
    r"final answer\s*[:\-]?\s*\**\s*([ab])\b",
    r"answer\s*[:\-]?\s*\**\s*([ab])\b",
    r"verdict\s*[:\-]?\s*\**\s*([ab])\b",
    r"my (?:choice|answer|verdict) is\s*\**\s*([ab])\b",
    r"(?:i )?(?:choose|prefer|select|pick)\s*\**\s*(?:response|translation|summary|option|answer)?\s*\**\s*([ab])\b",
    r"\b(?:response|translation|summary|option|answer)\s+([ab])\s+is\s+(?:better|best|preferred|more)",
    r"\b([ab])\s+is\s+(?:the\s+)?(?:better|best|preferred|more accurate|more fluent)",
]


def _last_letter_verdict(text: str):
    """Return 'A'/'B'/None by taking the LAST clear verdict signal in the text."""
    low = text.lower()
    best_pos, best_letter = -1, None
    # 1. explicit verdict phrases (strongest)
    for pat in _VERDICT_PHRASES:
        for m in re.finditer(pat, low):
            if m.start() > best_pos:
                best_pos, best_letter = m.start(), m.group(1).upper()
    if best_letter:
        return best_letter
    # 2. last standalone A / B token (e.g. final line "A" or "**B**")
    for m in re.finditer(r"(?:^|[\s\"'`*(\[])([ab])(?:[\s\"'`*).\]]|$)", low):
        if m.start() > best_pos:
            best_pos, best_letter = m.start(), m.group(1).upper()
    return best_letter


def filter_scenario_0(raw):
    nums = re.findall(r"\d+(?:\.\d+)?", str(raw).strip())
    if not nums:
        return "unparseable"
    try:
        v = float(nums[0])
    except ValueError:
        return "unparseable"
    return int(round(max(0.0, min(100.0, v))))


def filter_scenario_1(raw):
    """A / B / both / neither, tolerant of verbose output."""
    r = str(raw).strip()
    if not r:
        return "unparseable"
    low = r.lower()
    # 'neither' and 'both' as explicit verdicts take priority.
    has_neither = re.search(r"\bneither\b", low) is not None
    has_both = re.search(r"\bboth\b", low) is not None
    # Find their last positions to compare against any A/B verdict.
    pos = {}
    for key, present in (("neither", has_neither), ("both", has_both)):
        if present:
            pos[key] = low.rfind(key)
    letter = _last_letter_verdict(low)
    letter_pos = -1
    if letter:
        # rough position of the chosen letter verdict = last occurrence
        letter_pos = max(low.rfind(" " + letter.lower()), low.rfind(letter.lower()))
    # Decide by the LAST-stated verdict among {neither, both, A/B}
    candidates = []
    if "neither" in pos: candidates.append((pos["neither"], "neither"))
    if "both" in pos:    candidates.append((pos["both"], "both"))
    if letter:           candidates.append((letter_pos, letter))
    if not candidates:
        return "unparseable"
    candidates.sort()
    return candidates[-1][1]


_S2_DIMENSIONS = (
    "adequacy", "fluency", "terminology",
    "informativeness", "coherence", "faithfulness",
    "overall",
)


def filter_scenario_2(raw):
    """Extract {dimension: 1-5}. Tolerant of formats:
    'Adequacy: 4', 'Adequacy = 4', '**Adequacy**: 4', 'Adequacy (4/5)',
    'Adequacy - 4', 'Adequacy score: 4', '4/5'."""
    if raw is None:
        return "unparseable"
    text = str(raw)
    out = {}
    for dim in _S2_DIMENSIONS:
        # dim ... <sep> ... <score 1-5> optionally followed by /5
        m = re.search(
            rf"\b{dim}\b[^\d\n]{{0,20}}?([1-5])\s*(?:/\s*5)?",
            text, re.IGNORECASE,
        )
        if m:
            out[dim] = int(m.group(1))
    return out if out else "unparseable"


def filter_scenario_3(raw):
    """A / B / tie, tolerant of verbose output."""
    r = str(raw).strip()
    if not r:
        return "unparseable"
    low = r.lower()
    has_tie = re.search(r"\b(?:tie|equal|equally good|same quality|both equal)\b", low) is not None
    tie_pos = low.rfind("tie") if "tie" in low else (
        max(low.rfind("equal"), low.rfind("same quality")) if has_tie else -1)
    letter = _last_letter_verdict(low)
    letter_pos = -1
    if letter:
        letter_pos = max(low.rfind(" " + letter.lower()), low.rfind(letter.lower()))
    candidates = []
    if has_tie:  candidates.append((tie_pos, "tie"))
    if letter:   candidates.append((letter_pos, letter))
    if not candidates:
        return "unparseable"
    candidates.sort()
    return candidates[-1][1]


FILTERS = {0: filter_scenario_0, 1: filter_scenario_1,
           2: filter_scenario_2, 3: filter_scenario_3}


# vLLM loading

def load_vllm_model(model_cfg):
    from vllm import LLM, SamplingParams
    kwargs = dict(
        model=model_cfg["hf_id"],
        tensor_parallel_size=model_cfg.get("tp_size", 1),
        max_model_len=model_cfg.get("max_model_len", 8192),
        trust_remote_code=True,
        enforce_eager=model_cfg.get("enforce_eager", False),
        dtype=model_cfg.get("dtype", "auto"),
    )
    if "gemma-3" in model_cfg["hf_id"].lower():
        kwargs["dtype"] = "bfloat16"
        # Gemma-3 is a vision-language model; we only do text judging.
        # Skip the vision tower so multimodal init doesn't hang on load.
        kwargs["limit_mm_per_prompt"] = {"image": 0}
    llm = LLM(**kwargs)
    max_tokens = 4096 if model_cfg.get("reasoning") else 128
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, top_p=1.0)
    return llm, sampling


def run_inference(llm, sampling, input_dir, output_dir, model_name,
                  max_prompt_chars, is_reasoning=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        return 0, 0
    total, skipped = 0, 0
    for jf in jsonl_files:
        records = []
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            continue
        kept_records, kept_prompts, skipped_records = [], [], []
        for rec in records:
            p = rec["judge_prompt"]
            if len(p) > max_prompt_chars:
                skipped_records.append(rec)
            else:
                kept_records.append(rec)
                kept_prompts.append(p)

        print(f"    {jf.name}: {len(kept_prompts)} prompts "
              f"({len(skipped_records)} skipped) -> ", end="", flush=True)
        outputs, elapsed = [], 0.0
        if kept_prompts:
            conversations = [[{"role": "user", "content": p}] for p in kept_prompts]
            t0 = time.time()
            try:
                outputs = llm.chat(conversations, sampling)
            except Exception as e:
                if "context length" in str(e).lower() or "Validation" in type(e).__name__:
                    print(f"\n    ⚠ batch overflow, retrying individually...", end="")
                    outputs = []
                    for conv in conversations:
                        try:
                            outputs.extend(llm.chat([conv], sampling))
                        except Exception:
                            outputs.append(None)
                else:
                    raise
            elapsed = time.time() - t0
        print(f"{elapsed:.1f}s")

        out_path = output_dir / jf.name
        with open(out_path, "w", encoding="utf-8") as f:
            for rec, output in zip(kept_records, outputs):
                if output is None:
                    skipped_records.append(rec); skipped += 1
                    continue
                raw_output = output.outputs[0].text.strip()
                parse_input = _strip_think(raw_output) if is_reasoning else raw_output
                scenario = rec.get("scenario", 0)
                filtered = FILTERS.get(scenario, lambda x: x)(parse_input)
                f.write(json.dumps({
                    "id": rec["id"],
                    "pair_id": rec.get("pair_id"),
                    "dataset": rec["dataset"], "language": rec["language"],
                    "language_pair": rec.get("language_pair"),
                    "scenario": rec["scenario"], "task_type": rec["task_type"],
                    "judge_model": model_name,
                    "judge_raw_output": raw_output,
                    "judge_filtered_output": filtered,
                    "judge_prompt_tokens": len(output.prompt_token_ids),
                    "judge_completion_tokens": len(output.outputs[0].token_ids),
                    "judge_time_s": round(elapsed / max(len(kept_records), 1), 4),
                    "judge_prompt": rec["judge_prompt"],
                    "skipped": False,
                    "meta": rec.get("meta", {}),
                }, ensure_ascii=False) + "\n")
                total += 1
            for rec in skipped_records:
                f.write(json.dumps({
                    "id": rec["id"],
                    "pair_id": rec.get("pair_id"),
                    "dataset": rec["dataset"], "language": rec["language"],
                    "language_pair": rec.get("language_pair"),
                    "scenario": rec["scenario"], "task_type": rec["task_type"],
                    "judge_model": model_name,
                    "judge_raw_output": "", "judge_filtered_output": "skipped",
                    "judge_prompt_tokens": None, "judge_completion_tokens": None,
                    "judge_time_s": None,
                    "judge_prompt": rec["judge_prompt"][:500] + "... [truncated]",
                    "skipped": True,
                    "skip_reason": f"prompt_chars={len(rec['judge_prompt'])} > budget={max_prompt_chars}",
                    "meta": rec.get("meta", {}),
                }, ensure_ascii=False) + "\n")
                skipped += 1
    return total, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input-root", type=str, default=str(JUDGE_INPUTS))
    parser.add_argument("--output-root", type=str, default=str(JUDGE_OUTPUTS))
    args = parser.parse_args()
    if args.model not in JUDGE_MODELS:
        print(f"Unknown model: {args.model}. Available: {list(JUDGE_MODELS.keys())}")
        sys.exit(1)

    model_cfg = JUDGE_MODELS[args.model]
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    max_chars = model_cfg.get("max_prompt_chars", 28000)
    is_reasoning = bool(model_cfg.get("reasoning", False))

    # All scenario dirs: <dataset>/<scenario_*>
    scenario_dirs = sorted(p for p in input_root.glob("*/scenario_*") if p.is_dir())

    print("=" * 70)
    print(f"JUDGE INFERENCE (load-once): {args.model}")
    print(f"  Model: {model_cfg['hf_id']}")
    print(f"  Input root: {input_root}  ({len(scenario_dirs)} scenario dirs)")
    print(f"  Max prompt chars: {max_chars} | Reasoning: {is_reasoning}")
    print("=" * 70)

    t_load = time.time()
    llm, sampling = load_vllm_model(model_cfg)
    print(f"Model loaded in {time.time()-t_load:.1f}s — processing {len(scenario_dirs)} scenario dirs\n")

    grand_total, grand_skip = 0, 0
    for sdir in scenario_dirs:
        rel = sdir.relative_to(input_root)          # e.g. MAFAND-MT/scenario_2_model
        out_dir = output_root / rel / args.model
        print(f">>> {rel}")
        n, ns = run_inference(llm, sampling, sdir, out_dir, args.model,
                              max_chars, is_reasoning)
        grand_total += n
        grand_skip += ns

    print(f"\nDone: {grand_total} records processed, {grand_skip} skipped for length")


if __name__ == "__main__":
    main()
