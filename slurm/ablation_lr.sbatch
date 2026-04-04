#!/bin/bash
#SBATCH --job-name=ablation-lr
#SBATCH --partition=gpu4v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/ablation_lr_%j.out
#SBATCH --error=logs/ablation_lr_%j.err

# ===========================================================================
# Ablation Study: Varying Stage 2 Learning Rate on UTSA Arc HPC
# ---------------------------------------------------------------------------
# Tests 3 learning rates: 2e-5, 1e-5, 5e-6
# Each run: loads Stage 1 checkpoint → trains Stage 2 → runs inference
# Measures the tradeoff between JSON accuracy and Alpaca retention
#
# Partition: gpu1a100 (1x A100 40GB)
# Expected runtime: ~2-3 hours total (3 runs × ~40 min each + inference)
# Prerequisite: Stage 1 checkpoint must exist at checkpoints/stage1/
# ===========================================================================

echo "=============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "=============================================="

# ---------------------------------------------------------------------------
# 1. Environment setup
# ---------------------------------------------------------------------------
module purge
module load cuda/12.1
module load anaconda3
conda activate llm

# ---------------------------------------------------------------------------
# 2. Verify GPU and Stage 1 checkpoint
# ---------------------------------------------------------------------------
nvidia-smi
cd /home/xso947/Sequential-Instruction-Fine-Tuning-of-a-Small-LLM || { echo "Project directory not found!"; exit 1; }

if [ ! -d "checkpoints/stage1" ]; then
    echo "ERROR: Stage 1 checkpoint not found. Run stage1 first."
    exit 1
fi

mkdir -p logs

# ---------------------------------------------------------------------------
# 3. Define ablation learning rates
# ---------------------------------------------------------------------------
LEARNING_RATES=("2e-5" "1e-5" "5e-6")

for LR in "${LEARNING_RATES[@]}"; do
    echo ""
    echo "=============================================="
    echo "ABLATION RUN: Learning Rate = $LR"
    echo "Started at: $(date)"
    echo "=============================================="

    ABLATION_CKPT="checkpoints/ablation_lr_${LR}"
    ABLATION_OUTPUT="outputs/ablation_lr_${LR}"

    mkdir -p "$ABLATION_CKPT" "$ABLATION_OUTPUT"

    # --- Train Stage 2 with this learning rate ---
    echo ">>> Training Stage 2 with lr=$LR ..."
    python modular-code/fine-tuning-pipeline.py \
        --stage 2 \
        --resume-from checkpoints/stage1 \
        --output-dir "$ABLATION_CKPT" \
        --learning-rate "$LR"

    # --- Run inference on both eval sets ---
    echo ">>> Running inference for lr=$LR ..."
    python modular-code/inference.py \
        --checkpoints 2 \
        --checkpoint-path "$ABLATION_CKPT" \
        --alpaca-eval data/alpaca_eval.json \
        --json-eval data/json_eval.json \
        --output-dir "$ABLATION_OUTPUT"

    echo ">>> Ablation lr=$LR complete."
    echo ""
done

# ---------------------------------------------------------------------------
# 4. Compare ablation results
# ---------------------------------------------------------------------------
echo "=============================================="
echo ">>> All ablation runs complete. Comparing results..."
echo "=============================================="

python modular-code/result-aggregation.py \
    --ablation \
    --ablation-dirs outputs/ablation_lr_2e-5 outputs/ablation_lr_1e-5 outputs/ablation_lr_5e-6 \
    --labels "lr=2e-5" "lr=1e-5" "lr=5e-6" \
    --output-file eval_results/ablation_lr_comparison.json

echo ""
echo "=============================================="
echo "Ablation study complete at $(date)"
echo "Results: eval_results/ablation_lr_comparison.json"
echo "=============================================="
