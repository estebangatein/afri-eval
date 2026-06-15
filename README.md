# A Unified Benchmark for Assessing the Reliability of LLM-as-a-Judge in African Languages

MSc thesis project (University of Amsterdam / CWI, 2026).  
**Author:** Esteban Gatein · **Supervisor:** Clemencia Siro (CWI)

> *How do open-source LLM judges behave when evaluating African language NLP
> tasks, and what factors explain that behaviour?*

We apply four complementary evaluation scenarios to [AfroBench](https://github.com/McGill-NLP/afrobench) outputs and [SSA-MTE](https://huggingface.co/datasets/McGill-NLP/SSA-MTE) human scores, running 24 open-source judges across 10 datasets and 71 African language varieties.

---

## Repository structure

```
afri-eval/
├── snellius_pipeline/          # Inference pipeline (runs on Snellius HPC)
│   ├── configs/
│   │   └── config.py           # Judge pool, dataset paths, scenario config
│   ├── src/
│   │   ├── prepare_data.py     # Build per-scenario JSONL input files
│   │   ├── run_judge_vllm.py   # vLLM inference runner (one model, all scenarios)
│   │   └── compute_metrics.py  # Per-model metric aggregation -> JSON
│   ├── scripts/
│   │   ├── run_judge_full.sbatch  # SLURM job template (one judge model)
│   │   └── launch_all.sh          # Submit all 24 judges with correct resources
│   └── logs/                   # SLURM stdout/stderr (git-ignored)
│
├── judge_reliability_analysis.ipynb   # Main analysis notebook (all results)
├── requirements.txt                 # Requirement file to create environment
└── EDA.ipynb                        # Exploratory data analysis notebook
```

---

## Evaluation scenarios

| Scenario | Name | Applied to | What it measures |
|---|---|---|---|
| S0 | SSA-MTE meta-evaluation | SSA-MTE DA scores | Judge–human alignment (Spearman ρ) |
| S1 | Borderline case audit | 8 classification/QA datasets | Automatic-metric strictness (both-acceptable rate) |
| S2 | Multi-dimensional rating | MAFAND-MT, XLSum | Reference quality per dimension (1–5 rubric) |
| S3 | Pairwise comparison | All 10 datasets | Position bias, human-preference rate |

---

## Judge pool

24 open-source instruction-tuned models spanning six families, from 4B to 141B
total parameters (MoE models counted by total parameters):

| Family | Models |
|---|---|
| LLaMA | 3-8B, 3-70B, 3.1-8B, 3.1-70B, 3.3-70B, 4-Scout (109B MoE) |
| Qwen | 2.5-7B, 2.5-14B, 2.5-32B, 2.5-72B, QwQ-32B |
| Gemma | 3-4B, 3-12B, 3-27B |
| Mistral | 7B, Nemo-12B, 8x7B (47B MoE), 8x22B (141B MoE) |
| DeepSeek | R1-7B, R1-14B, R1-32B, R1-70B (reasoning models) |
| Aya | Expanse-8B, Expanse-32B |
| AfriQwen | AfriqueQwen-14B (base model, African-adapted) |

---

## Datasets

| Dataset | Task | Scenarios | Languages |
|---|---|---|---|
| SSA-MTE | MT quality (human DA) | S0 | 11 LPs |
| MasakhaNER2 | Token classification | S1, S3 | 20 |
| AfriSenti | Sentiment analysis | S1, S3 | 14 |
| MasakhaNEWS | Topic classification | S1, S3 | 16 |
| AfriHate | Hate speech detection | S1, S3 | 15 |
| AfriXNLI | Natural language inference | S1, S3 | 18 |
| AfriQA | Question answering | S1, S3 | 9 |
| Belebele | Reading comprehension | S1, S3 | 31 |
| AfriMGSM | Math reasoning | S1, S3 | 14 |
| MAFAND-MT | Translation | S2, S3 | 21 |
| XLSum | Summarization | S2, S3 | 11 |

---

## Running the pipeline

### Prerequisites

- Access to [Snellius](https://www.surf.nl/en/services/snellius-the-national-supercomputer) (SURF)
- HuggingFace token with access to gated models (LLaMA, Gemma, Mixtral, Aya)
- AfroBench source data at `/scratch-shared/<user>/afrobench_data/afrobench/`
- Python 3.12 virtualenv at `~/venvs/judge_pipeline` with `vllm`, `transformers`, `datasets`, `pandas`, `scipy`

```bash
# 1. Activate environment
source ~/venvs/judge_pipeline/bin/activate
cd ~/snellius_pipeline

# 2. Build input JSONL files for all scenarios
python -m src.prepare_data --sample 200

# 3. Launch all 24 judge jobs (with correct GPU/walltime per model)
bash scripts/launch_all.sh 200

# 4. Monitor
watch -n 30 squeue -u $USER
```

Jobs are submitted with `sleep 2` between each to avoid SLURM socket timeouts.
Standard models use 12h walltime; reasoning models (DeepSeek-R1, QwQ) use 16h.

### Output structure

```
/scratch-shared/<user>/
├── judge_inputs/<dataset>/<scenario>/<lang>.jsonl   # inputs (per scenario)
├── judge_outputs/<dataset>/<scenario>/<model>/<lang>.jsonl  # verdicts
└── results/
    ├── metrics_<model>.json    # per-model metrics
    └── master_summary.csv      # aggregated across all judges
```

> **Note:** `/scratch-shared` on Snellius has a 14-day auto-cleanup policy.
> Keep AfroBench source data and final outputs in permanent storage
> (home directory or project space) — only intermediate files belong in scratch.

---

## Analysis notebook

`judge_reliability_analysis.ipynb` loads all outputs from `master_summary.csv`
and the per-scenario JSONL files and produces all tables and figures in the
thesis. It is structured in phases matching the evaluation scenarios:

- **Phase 0** — loader, coverage checks, format-compliance analysis
- **Overview** — aggregate stats, English vs African comparison, model size breakdown, task breakdown
- **Phase 1** — SSA-MTE human alignment (Spearman ρ per judge and LP)
- **Phase 2A** — S1 borderline audit (both-acceptable rate, inter-judge κ)
- **Phase 2B** — S2 multi-dimensional ratings (ICC, paired Gemini vs human)
- **Phase 3** — S3 pairwise (position bias, human-preference rate)
- **Phase 4** — Cross-cutting (script, resource level, language family, model size)

The notebook is self-contained given `master_summary.csv` and the saved CSVs
in `results/`. Run it locally after downloading those files from Snellius, or
run it interactively on Snellius via a Jupyter tunnel:

```bash
# On Snellius login node
jupyter notebook --no-browser --port=8888
# On your laptop
ssh -L 8888:localhost:8888 <user>@snellius.surf.nl
```

---

## Key infrastructure notes

- **vLLM** is used for all inference with tensor parallelism (1–4 H100 GPUs depending on model size)
- **Gemma-3** models require `limit_mm_per_prompt={"image": 0}` to suppress the vision tower initialisation
- **Reasoning models** (DeepSeek-R1, QwQ) have `<think>` blocks stripped before verdict parsing
- **Context overflow** for Aya/LLaMA-3-70B (8192-token context) is handled by a try/except fallback that skips individual records
- All inference uses greedy decoding (temperature = 0) for reproducibility

---

## Citation

If you use this framework or data, please cite the underlying benchmarks:

```
@inproceedings{ojo2025afrobench,
  title={{AfroBench}: How Good are Large Language Models on African Languages?},
  author={Ojo, Jessica and others},
  booktitle={ACL},
  year={2025}
}

@article{li2025ssamte,
  title={{SSA-MTE}: A Machine Translation Evaluation Dataset and Metric
         for Sub-Saharan African Languages},
  author={Li, Senyu and others},
  year={2025}
}
```
