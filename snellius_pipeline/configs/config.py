"""
Configuration for the AfroBench LLM-as-a-Judge pipeline.

Scenarios:
  • S0  = SSA-MTE reference-free DA scoring (0-100), correlated with human DA
  • S3 (ssa-mte) = SSA-MTE pairwise: MT output vs human reference (NEW)
  • S1  = reference-aware audit on wrong cases (A/B/both/neither)
  • S2  = multi-dim rating (1-5) of the human GOLD reference
  • S2_model = multi-dim rating (1-5) of the GEMINI OUTPUT (NEW; same rubric,
              same records as S2, so the two can be compared pairwise)
  • S3  = pairwise model vs human gold (A/B/tie, position-randomised)

Cross-model comparisons should intersect on record_id (models with smaller
context windows skip more long-document records; the skip flag makes the common
subset recoverable in analysis).
"""
from pathlib import Path


# Paths
SCRATCH = Path("/scratch-shared/egatein")
AFROBENCH_DATA  = SCRATCH / "afrobench_data"
SSA_MTE_DATA    = SCRATCH / "ssa_mte_data"
JUDGE_INPUTS    = SCRATCH / "judge_inputs"
JUDGE_OUTPUTS   = SCRATCH / "judge_outputs"
RESULTS         = SCRATCH / "results"


# Judge models (24)
JUDGE_MODELS = {
    # LLaMA (Meta)
    "llama-3-70b": {
        "hf_id": "meta-llama/Meta-Llama-3-70B-Instruct",
        "tp_size": 4, "max_model_len": 8192, "max_prompt_chars": 4000,
    },
    "llama-3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "tp_size": 1, "max_model_len": 16384, "max_prompt_chars": 28000,
    },
    "llama-3.1-70b": {
        "hf_id": "meta-llama/Llama-3.1-70B-Instruct",
        "tp_size": 4, "max_model_len": 16384, "max_prompt_chars": 28000,
    },
    "llama-3.3-70b": {
        "hf_id": "meta-llama/Llama-3.3-70B-Instruct",
        "tp_size": 4, "max_model_len": 16384, "max_prompt_chars": 28000,
    },
    "llama-4-scout": {
        "hf_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "tp_size": 4, "max_model_len": 16384, "max_prompt_chars": 28000,
    },
    # Qwen (Alibaba)
    "qwen-2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "tp_size": 1, "max_model_len": 16384, "max_prompt_chars": 28000,
    },
    "qwen-2.5-14b": {
        "hf_id": "Qwen/Qwen2.5-14B-Instruct",
        "tp_size": 1, "max_model_len": 16384, "max_prompt_chars": 28000,
    },
    "qwen-2.5-32b": {
        "hf_id": "Qwen/Qwen2.5-32B-Instruct",
        "tp_size": 2, "max_model_len": 16384, "max_prompt_chars": 28000,
    },
    "qwen-2.5-72b": {
        "hf_id": "Qwen/Qwen2.5-72B-Instruct",
        "tp_size": 4, "max_model_len": 32768, "max_prompt_chars": 55000,
    },
    "qwq-32b": {
        "hf_id": "Qwen/QwQ-32B",
        "tp_size": 2, "max_model_len": 16384, "max_prompt_chars": 28000,
        "reasoning": True,
    },
    # Gemma (Google)
    "gemma-3-4b-it": {
        "hf_id": "google/gemma-3-4b-it",
        "tp_size": 1, "max_model_len": 16384, "max_prompt_chars": 28000,
        "dtype": "bfloat16", "enforce_eager": True,
    },
    "gemma-3-12b-it": {
        "hf_id": "google/gemma-3-12b-it",
        "tp_size": 1, "max_model_len": 16384, "max_prompt_chars": 28000,
        "dtype": "bfloat16", "enforce_eager": True,
    },
    "gemma-3-27b-it": {
        "hf_id": "google/gemma-3-27b-it",
        "tp_size": 4, "max_model_len": 16384, "max_prompt_chars": 28000,
        "dtype": "bfloat16", "enforce_eager": True,
    },
    # Mistral / Mixtral
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "tp_size": 1, "max_model_len": 16384, "max_prompt_chars": 28000,
    },
    "mistral-nemo": {
        "hf_id": "mistralai/Mistral-Nemo-Instruct-2407",
        "tp_size": 1, "max_model_len": 16384, "max_prompt_chars": 28000,
    },
    "mixtral-8x7b": {
        "hf_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "tp_size": 2, "max_model_len": 16384, "max_prompt_chars": 28000,
    },
    "mixtral-8x22b": {
        "hf_id": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "tp_size": 4, "max_model_len": 16384, "max_prompt_chars": 28000,
    },
    # DeepSeek-R1 (reasoning)
    "deepseek-r1-7b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "tp_size": 1, "max_model_len": 16384, "max_prompt_chars": 28000,
        "reasoning": True,
    },
    "deepseek-r1-14b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "tp_size": 1, "max_model_len": 16384, "max_prompt_chars": 28000,
        "reasoning": True,
    },
    "deepseek-r1-32b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "tp_size": 2, "max_model_len": 16384, "max_prompt_chars": 28000,
        "reasoning": True,
    },
    "deepseek-r1-70b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "tp_size": 4, "max_model_len": 16384, "max_prompt_chars": 28000,
        "reasoning": True,
    },
    # Multilingual specialists
    "afroqwen-14b": {
        "hf_id": "McGill-NLP/AfriqueQwen-14B",
        "tp_size": 2, "max_model_len": 16384, "max_prompt_chars": 28000,
    },
    "aya-expanse-8b": {
        "hf_id": "CohereLabs/aya-expanse-8b",
        "tp_size": 1, "max_model_len": 8192, "max_prompt_chars": 4000,
    },
    "aya-expanse-32b": {
        "hf_id": "CohereLabs/aya-expanse-32b",
        "tp_size": 2, "max_model_len": 8192, "max_prompt_chars": 4000,
    },
}


# AfroBench datasets
# Datasets listing scenario 2 automatically also get scenario_2_model built
# (same rubric applied to the Gemini output instead of the gold reference).
DATASETS = {
    "MasakhaNER2": {"afrobench_key": "masakhaner",   "task_type": "token_classification",  "scenarios": [1, 3]},
    "AfriSenti":   {"afrobench_key": "afrisenti",     "task_type": "text_classification",   "scenarios": [1, 3]},
    "MasakhaNEWS": {"afrobench_key": "masakhanews",   "task_type": "text_classification",   "scenarios": [1, 3]},
    "AfriHate":    {"afrobench_key": "afrihate",      "task_type": "text_classification",   "scenarios": [1, 3]},
    "AfriXNLI":    {"afrobench_key": "afrixnli",      "task_type": "nli",                   "scenarios": [1, 3]},
    "AfriQA":      {"afrobench_key": "afriqa",        "task_type": "question_answering",    "scenarios": [1, 3]},
    "Belebele":    {"afrobench_key": "belebele",      "task_type": "reading_comprehension", "scenarios": [1, 3]},
    "AfriMGSM":    {"afrobench_key": "afrimgsm",      "task_type": "math_reasoning",        "scenarios": [1, 3]},
    "MAFAND-MT":   {"afrobench_key": "mafand",        "task_type": "translation",           "scenarios": [2, 3]},
    "XLSum":       {"afrobench_key": "xlsum",         "task_type": "summarization",         "scenarios": [2, 3]},
}


# Auto-metric columns per task type (drives the S1 wrong-case filter)
METRIC_COLS = {
    "token_classification":  ["span_f1", "f1"],
    "text_classification":   ["acc", "f1"],
    "nli":                   ["acc", "f1"],
    "question_answering":    ["f1", "exact_match"],
    "reading_comprehension": ["acc", "f1"],
    "math_reasoning":        ["exact_match"],
    "translation":           ["chrf"],
    "summarization":         ["rougeL"],
}


# SSA-MTE (Phase 1) configuration 
SSA_MTE_CONFIG = {
    "name": "ssa-mte",
    "hf_repo": "McGill-NLP/SSA-MTE",
    "hf_split": "test",
    "local_file": "ssa_mte_test.csv",
    "column_aliases": {
        "src":    ["src_sent", "src", "source", "source_text"],
        "mt":     ["tgt_sent", "mt", "translation", "hypothesis", "candidate"],
        "lp":     ["lp", "language_pair", "lang_pair"],
        "score":  ["score", "z_score", "processed_score", "raw_score",
                   "mean_score", "da_score"],
        "system": ["system", "mt_system", "model"],
        "ref":    ["ref", "reference", "human_reference", "reference_translation"],
    },
    "language_pairs": [
        "eng-amh", "eng-hau", "eng-kik", "eng-kin", "eng-luo",
        "eng-twi", "eng-yor", "fra-ewe", "fra-wol", "por-vmw", "por-nya",
    ],
    "system_filter": "Gemeni-Google",
}


# Language metadata
LANG_NAMES = {
    "swa": "Swahili", "zul": "isiZulu", "xho": "isiXhosa", "sna": "chiShona",
    "nya": "chiChewa", "kin": "Kinyarwanda", "run": "Kirundi", "lug": "Luganda",
    "lin": "Lingala", "tsn": "Setswana", "sot": "Sesotho", "nso": "Sepedi",
    "tso": "Xitsonga", "ven": "Tshivenda", "bem": "Bemba", "kik": "Kikuyu",
    "vmw": "Emakhuwa", "ssw": "Swati", "luo": "Dholuo",
    "yor": "Yoruba", "ibo": "Igbo", "wol": "Wolof", "twi": "Twi", "aka": "Twi",
    "fon": "Fon", "ewe": "Ewe", "bam": "Bambara", "mos": "Mossi",
    "bbj": "Ghomala", "fuv": "Fulfulde", "ful": "Fulfulde", "vai": "Vai",
    "pcm": "Naija",
    "hau": "Hausa", "amh": "Amharic", "tir": "Tigrinya", "orm": "Oromo",
    "som": "Somali", "arq": "Algerian Arabic", "ary": "Moroccan Arabic",
    "plt": "Malagasy", "mlg": "Malagasy",
    "eng": "English", "en": "English",
    "fra": "French",  "fr": "French",
    "por": "Portuguese", "pt": "Portuguese",
}

LANG_FAMILIES = {
    "Niger-Congo (Bantu)": [
        "Swahili", "isiZulu", "isiXhosa", "chiShona", "chiChewa",
        "Kinyarwanda", "Kirundi", "Luganda", "Lingala", "Setswana",
        "Sesotho", "Sepedi", "Xitsonga", "Tshivenda", "Bemba",
        "Kikuyu", "Emakhuwa", "Swati", "Dholuo",
    ],
    "Niger-Congo (non-Bantu)": [
        "Yoruba", "Igbo", "Wolof", "Twi", "Fon", "Ewe", "Bambara",
        "Mossi", "Ghomala", "Fulfulde", "Vai", "Naija",
    ],
    "Afroasiatic": [
        "Hausa", "Amharic", "Tigrinya", "Oromo", "Somali",
        "Algerian Arabic", "Moroccan Arabic",
    ],
    "Austronesian": ["Malagasy"],
    "Indo-European (control)": ["English", "French", "Portuguese"],
}
