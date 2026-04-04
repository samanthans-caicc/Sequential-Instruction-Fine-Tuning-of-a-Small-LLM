#!/bin/bash
#SBATCH --job-name=stage2-json
#SBATCH --partition=gpu4v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/stage2_json_%j.out
#SBATCH --error=logs/stage2_json_%j.err

# ===========================================================================
# Stage 2: Teacher-Generated JSON Fine-Tuning (QLoRA) on UTSA Arc HPC
# ---------------------------------------------------------------------------
# Partition: gpu1a100 (1x A100 40GB)
# Expected runtime: ~30-60 min for ~5K examples, 5 epochs
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
echo ""
echo "--- GPU Info ---"
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
echo ""

cd /home/xso947/Sequential-Instruction-Fine-Tuning-of-a-Small-LLM || { echo "Project directory not found!"; exit 1; }

# Check that Stage 1 checkpoint exists
if [ ! -d "checkpoints/stage1" ]; then
    echo "ERROR: Stage 1 checkpoint not found at checkpoints/stage1/"
    echo "Run stage1_alpaca_train.sbatch first."
    exit 1
fi
echo ">>> Stage 1 checkpoint found. Continuing to Stage 2..."

mkdir -p logs checkpoints/stage2

# ---------------------------------------------------------------------------
# 3. Run Stage 2 QLoRA fine-tuning (starts from Stage 1 checkpoint)
# ---------------------------------------------------------------------------
echo ">>> Starting Stage 2 training on teacher-generated JSON data..."
python modular-code/fine-tuning-pipeline.py \
    --stage 2 \
    --resume-from checkpoints/stage1 \
    --output-dir checkpoints/stage2

echo ""
echo ">>> Stage 2 training complete."
echo ">>> Checkpoint saved to: checkpoints/stage2/"

# ---------------------------------------------------------------------------
# 4. Run Checkpoint 2 inference on both eval sets
# ---------------------------------------------------------------------------
echo ">>> Running Checkpoint 2 (JSON-tuned) inference..."
python modular-code/inference.py \
    --checkpoints 2 \
    --alpaca-eval data/alpaca_eval.json \
    --json-eval data/json_eval.json \
    --output-dir /home/xso947/Sequential-Instruction-Fine-Tuning-of-a-Small-LLM/outputs

echo ""
echo "=============================================="
echo "Stage 2 job complete at $(date)"
echo "Outputs saved to: outputs/"
echo "=============================================="
