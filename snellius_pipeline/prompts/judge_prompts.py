"""
Judge prompt templates for all evaluation phases.

Phase 1 - Scenario 0 (SSA-MTE meta-evaluation):
  Reference-free DA-style scoring on machine translation. Judge sees only
  (source, machine translation) and produces a 0–100 score. Used to validate
  baseline linguistic competency by correlating with human DA scores.

Phase 2 - Scenario 1 (Reference-aware audit on borderline cases):
  Applied to records where the auto-metric flagged the model's output as wrong.
  Judge sees the input plus two responses (model output and gold label) in
  randomised A/B order. Verdict ∈ {A, B, both, neither}.
  - A/B: judge prefers one response (gold's position → gold confirmed;
         model's position → annotation error or model was right)
  - both: both are acceptable → auto-metric was too strict
  - neither: neither is correct → task quality issue

Phase 2 - Scenario 2 (Multi-dimensional rating of human references):
  Applied to generation tasks (MAFAND-MT, XLSum). Judge rates the human
  reference along quality dimensions (1–5 each).

Phase 3 - Scenario 3 (Pairwise model vs human, position-bias measurement):
  Standard pairwise A/B/tie with randomised position. Applied to all 10
  datasets including MAFAND-MT and XLSum (long-source records may be skipped
  for judges with shorter context windows - see truncation handling in
  run_judge_vllm.py).
"""


# PHASE 1 / SCENARIO 0 - SSA-MTE meta-evaluation (reference-free, 0–100 DA)
# Adapted from AfriMTE / SSA-COMET (Li et al., 2025, EMNLP).

SCENARIO_0 = {
    "translation_qe": (
"""Assess the translation adequacy on a continuous scale [0 ~ 100] using the quality levels described below:
[0] Nonsense/No meaning preserved: Nearly all information is lost between the translation and source.
[34] Some meaning preserved: The translation preserves some of the meaning of the source but misses significant parts.
[67] Most meaning preserved: The translation retains most of the meaning of the source.
[100] Perfect meaning: The meaning of the translation is completely consistent with the source.
Source ({src_lang}): {src}
Translation ({tgt_lang}): {mt}
Respond with a single integer between 0 and 100:"""
    ),
}


# SCENARIO 1 - Reference-aware audit (A/B/both/neither, randomised position)
# Applied to borderline (auto-metric-wrong) cases on 8 non-generation datasets.

_S1_TEMPLATE_HEAD = {
    "text_classification":
        "Two responses were given for the following classification task. "
        "Determine which response is more accurate.",
    "token_classification":
        "Two responses were given for the following named-entity recognition task. "
        "Determine which response better identifies the entities.",
    "nli":
        "Two responses were given for the following natural language inference task. "
        "Determine which response better identifies the logical relationship.",
    "question_answering":
        "Two responses were given for the following question. "
        "Determine which response is more accurate and complete.",
    "reading_comprehension":
        "Two responses were given for the following reading comprehension question. "
        "Determine which response is more accurate.",
    "math_reasoning":
        "Two solutions were given for the following math problem. "
        "Evaluate both the quality of the reasoning and the correctness of the final answer.",
}

_S1_BODY = (
    "\n\nTask:\n{input_prompt}\n\n"
    "Response A:\n{response_a}\n\n"
    "Response B:\n{response_b}\n\n"
    "Which response is better? Respond with exactly one of: "
    "\"A\", \"B\", \"both\" (if both are acceptable), or \"neither\" (if neither is correct)."
)

SCENARIO_1 = {tt: head + _S1_BODY for tt, head in _S1_TEMPLATE_HEAD.items()}


# SCENARIO 2 - Multi-dimensional rating of human references (1–5 per dimension)
# Applied to generation tasks: MAFAND-MT (translation), XLSum (summarization).

SCENARIO_2 = {
    "translation": (
        "Rate the following translation on each dimension separately. "
        "For each dimension, provide a score from 1 (worst) to 5 (best).\n\n"
        "Dimensions:\n"
        "- Adequacy (1-5): Does the translation preserve the meaning of the source text?\n"
        "- Fluency (1-5): Is the translation natural and grammatically correct in the target language?\n"
        "- Terminology (1-5): Are domain-specific terms and proper nouns translated correctly?\n"
        "- Overall (1-5): Considering the previous scores, how good is the full translation?\n\n"
        "Source text:\n{input_prompt}\n\n"
        "Translation:\n{gold_label}\n\n"
        "Respond ONLY in the following format:\n"
        "Adequacy: [1-5]\n"
        "Fluency: [1-5]\n"
        "Terminology: [1-5]\n"
        "Overall: [1-5]"
    ),

    "summarization": (
        "Rate the following summary on each dimension separately. "
        "For each dimension, provide a score from 1 (worst) to 5 (best).\n\n"
        "Dimensions:\n"
        "- Informativeness (1-5): Does the summary capture the key information from the source?\n"
        "- Coherence (1-5): Is the summary logically organized and easy to follow?\n"
        "- Faithfulness (1-5): Does the summary avoid introducing information not present in the source?\n"
        "- Fluency (1-5): Is the summary natural and grammatically correct?\n"
        "- Overall (1-5): Considering the previous scores, how good is the full summary?\n\n"
        "Source text:\n{input_prompt}\n\n"
        "Reference summary:\n{gold_label}\n\n"
        "Respond ONLY in the following format:\n"
        "Informativeness: [1-5]\n"
        "Coherence: [1-5]\n"
        "Faithfulness: [1-5]\n"
        "Fluency: [1-5]\n"
        "Overall: [1-5]"
    ),
}


# SCENARIO 3 - Pairwise model vs human (A/B/tie, randomised position)
# Applied to ALL 10 datasets including MAFAND-MT and XLSum.
# For math_reasoning the response is the FULL chain-of-thought (raw_output),
# so the judge evaluates both reasoning quality and final-answer correctness.

SCENARIO_3 = {
    "text_classification": (
        "Two responses were given for the following classification task. "
        "Determine which response is better.\n\n"
        "Task:\n{input_prompt}\n\n"
        "Response A: {response_a}\n"
        "Response B: {response_b}\n\n"
        "Which response is better? Respond with exactly one of: \"A\", \"B\", or \"tie\"."
    ),
    "token_classification": (
        "Two responses were given for the following named-entity recognition task. "
        "Determine which response better identifies the entities.\n\n"
        "Task:\n{input_prompt}\n\n"
        "Response A: {response_a}\n"
        "Response B: {response_b}\n\n"
        "Which response is better? Respond with exactly one of: \"A\", \"B\", or \"tie\"."
    ),
    "nli": (
        "Two responses were given for the following natural language inference task. "
        "Determine which response better identifies the logical relationship.\n\n"
        "Task:\n{input_prompt}\n\n"
        "Response A: {response_a}\n"
        "Response B: {response_b}\n\n"
        "Which response is better? Respond with exactly one of: \"A\", \"B\", or \"tie\"."
    ),
    "question_answering": (
        "Two answers were given for the following question. "
        "Determine which answer is more accurate and complete.\n\n"
        "Task:\n{input_prompt}\n\n"
        "Response A: {response_a}\n"
        "Response B: {response_b}\n\n"
        "Which response is better? Respond with exactly one of: \"A\", \"B\", or \"tie\"."
    ),
    "reading_comprehension": (
        "Two answers were given for the following reading comprehension question. "
        "Determine which answer is more accurate.\n\n"
        "Task:\n{input_prompt}\n\n"
        "Response A: {response_a}\n"
        "Response B: {response_b}\n\n"
        "Which response is better? Respond with exactly one of: \"A\", \"B\", or \"tie\"."
    ),
    "math_reasoning": (
        "Two solutions were given for the following math problem. "
        "Evaluate both the quality of the reasoning process (step-by-step logic, "
        "clarity, correctness of intermediate steps) and the correctness of the "
        "final numerical answer.\n\n"
        "Task:\n{input_prompt}\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}\n\n"
        "Which response is better? Respond with exactly one of: \"A\", \"B\", or \"tie\"."
    ),
    "translation": (
        "Two translations were given for the following source text. "
        "Determine which translation better preserves the meaning and is more "
        "fluent in the target language.\n\n"
        "Source text:\n{input_prompt}\n\n"
        "Translation A:\n{response_a}\n\n"
        "Translation B:\n{response_b}\n\n"
        "Which translation is better? Respond with exactly one of: \"A\", \"B\", or \"tie\"."
    ),
    "summarization": (
        "Two summaries were given for the following source text. "
        "Determine which summary better captures the key information and is more "
        "coherent.\n\n"
        "Source text:\n{input_prompt}\n\n"
        "Summary A:\n{response_a}\n\n"
        "Summary B:\n{response_b}\n\n"
        "Which summary is better? Respond with exactly one of: \"A\", \"B\", or \"tie\"."
    ),
}


# Dispatch helpers

_REGISTRY = {0: SCENARIO_0, 1: SCENARIO_1, 2: SCENARIO_2, 3: SCENARIO_3}


def get_template(scenario: int, task_type: str) -> str:
    templates = _REGISTRY.get(scenario)
    if not templates or task_type not in templates:
        raise ValueError(f"No template for scenario={scenario}, task_type={task_type}")
    return templates[task_type]


def format_judge_prompt(scenario: int, task_type: str, **kwargs) -> str:
    return get_template(scenario, task_type).format(**kwargs)

