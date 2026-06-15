#!/bin/bash
# launch_all.sh — submit prep (optional) + all model jobs + rollup.
#
# Walltimes are REALISTIC (Option A) and now that the model loads ONCE per job
# (load-once refactor) runtimes are much lower, so reservations stay well within
# budget. Tiers below are conservative upper bounds.
#
# Usage:
#   bash scripts/launch_all.sh                 # sample=200, ALL models, runs prep
#   bash scripts/launch_all.sh 200 --no-prep   # skip prep (inputs already built)
#   bash scripts/launch_all.sh 200 "gemma-3-4b-it gemma-3-12b-it"   # subset, runs prep
#   bash scripts/launch_all.sh 200 --no-prep "gemma-3-4b-it gemma-3-12b-it"
#
set -euo pipefail

SAMPLE="${1:-200}"
RUN_PREP=1
MODELS=""

# Parse optional flags / model list
shift || true
for arg in "$@"; do
    if [ "$arg" == "--no-prep" ]; then
        RUN_PREP=0
    else
        MODELS="$arg"
    fi
done

cd "$HOME/snellius_pipeline"
mkdir -p logs

if [ -z "$MODELS" ]; then
    MODELS=$(python -c "from configs.config import JUDGE_MODELS; print(' '.join(JUDGE_MODELS.keys()))")
fi

echo "=== Launching pipeline ==="
echo "Sample: $SAMPLE | Run prep: $RUN_PREP"
echo "Models: $MODELS"
echo ""

# 1. Optional data preparation
PREP_DEP=""
if [ "$RUN_PREP" -eq 1 ]; then
    PREP_JOB=$(sbatch --parsable \
        --job-name=judge_prep --partition=rome \
        --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --time=02:00:00 --account=gisr122612 \
        --output=logs/prep_%j.out --error=logs/prep_%j.err \
        --wrap "
            module purge; module load 2024
            module load Python/3.12.3-GCCcore-13.3.0
            source \$HOME/venvs/judge_pipeline/bin/activate
            cd \$HOME/snellius_pipeline
            export HF_HOME=/scratch-shared/egatein/hf_cache
            export HF_TOKEN=\$(cat ~/.cache/huggingface/token 2>/dev/null || echo '')
            python -m src.prepare_data --sample $SAMPLE
        ")
    echo "Submitted prepare_data: $PREP_JOB"
    PREP_DEP="--dependency=afterok:$PREP_JOB"
fi

# 2. Per-model judge jobs
# Realistic walltime tiers (load-once → lower runtimes). Reasoning models keep
# generous headroom (4096-token outputs).
walltime_for () {  # $1=gpus  $2=reasoning(True/False)
    local g="$1" r="$2"
    if [ "$r" == "True" ]; then
        case "$g" in 1) echo "10:00:00";; 2) echo "12:00:00";; *) echo "12:00:00";; esac
    else
        case "$g" in 1) echo "05:00:00";; 2) echo "06:00:00";; *) echo "06:00:00";; esac
    fi
}
cpus_for () { case "$1" in 1) echo 16;; 2) echo 32;; *) echo 64;; esac; }

JUDGE_JOB_IDS=()
for MODEL in $MODELS; do
    GPUS=$(python -c "from configs.config import JUDGE_MODELS; print(JUDGE_MODELS['$MODEL']['tp_size'])") \
        || { echo "ERROR: unknown model '$MODEL' — skipping."; continue; }
    REASON=$(python -c "from configs.config import JUDGE_MODELS; print(JUDGE_MODELS['$MODEL'].get('reasoning', False))")
    WALL=$(walltime_for "$GPUS" "$REASON")
    CPUS=$(cpus_for "$GPUS")
    JID=$(sbatch --parsable $PREP_DEP \
        --job-name="j_${MODEL}" --partition=gpu_h100 \
        --nodes=1 --ntasks=1 --gpus=$GPUS --cpus-per-task=$CPUS --time=$WALL \
        --export=ALL,MODEL=$MODEL \
        scripts/run_judge_full.sbatch)
    JUDGE_JOB_IDS+=("$JID")
    echo "Submitted $MODEL: $JID  (gpus=$GPUS, reasoning=$REASON, time=$WALL)"
    sleep 2   # avoid SLURM socket timeout on rapid submissions
done

if [ ${#JUDGE_JOB_IDS[@]} -eq 0 ]; then
    echo "No judge jobs submitted — aborting before rollup."
    exit 1
fi

# 3. Final cross-model metrics rollup ──────────────────────────────────────
DEP_LIST=$(IFS=:; echo "${JUDGE_JOB_IDS[*]}")
ROLLUP_JOB=$(sbatch --parsable \
    --dependency=afterany:$DEP_LIST \
    --job-name=judge_rollup --partition=rome \
    --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --time=00:30:00 --account=gisr122612 \
    --output=logs/rollup_%j.out --error=logs/rollup_%j.err \
    --wrap "
        module purge; module load 2024
        module load Python/3.12.3-GCCcore-13.3.0
        source \$HOME/venvs/judge_pipeline/bin/activate
        cd \$HOME/snellius_pipeline
        python -m src.compute_metrics \
            --output /scratch-shared/egatein/results/metrics_all_models.json
    ")
echo "Submitted rollup: $ROLLUP_JOB"
echo ""
echo "All jobs submitted (${#JUDGE_JOB_IDS[@]} model jobs). Track: squeue -u \$USER"
