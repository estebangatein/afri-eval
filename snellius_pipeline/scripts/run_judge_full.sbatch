#!/bin/bash
#SBATCH --job-name=judge_full
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --account=gisr122612
#SBATCH --output=logs/judge_%x_%j.out
#SBATCH --error=logs/judge_%x_%j.err
#
# Runs ALL scenarios for ONE model in a single process (model loaded once).
# MODEL is passed via --export=ALL,MODEL=<name>. Resource flags (--gpus,
# --cpus-per-task, --time) are overridden per-model by launch_all.sh at submit.

set -euo pipefail

echo "=== Run started at $(date) — MODEL=${MODEL} ==="

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source "$HOME/venvs/judge_pipeline/bin/activate"
cd "$HOME/snellius_pipeline"

export HF_HOME=/scratch-shared/egatein/hf_cache
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")

# Single call — loads the model once, iterates over every scenario dir.
python -m src.run_judge_vllm --model "${MODEL}"

# Per-model metrics rollup (scans all outputs present so far)
python -m src.compute_metrics \
    --output "/scratch-shared/egatein/results/metrics_${MODEL}.json" || true

echo "=== Run finished at $(date) ==="
