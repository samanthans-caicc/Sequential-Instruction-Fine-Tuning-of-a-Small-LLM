#!/bin/bash
#SBATCH --job-name=stage1-alpaca
#SBATCH --partition=gpu4v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage1_alpaca_%j.out
#SBATCH --error=logs/stage1_alpaca_%j.err

# ===========================================================================
# Stage 1: Alpaca Fine-Tuning (QLoRA) on UTSA Arc HPC
# ---------------------------------------------------------------------------
# Partition: gpu1a100 (1x A100 40GB)
# Expected runtime: ~2-4 hours for ~50K examples, 3 epochs
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
module load cuda/12.1          # adjust to whatever version Arc provides
module load anaconda3           # or miniconda — check with 'module avail'

# Activate your conda environment (create once, reuse across jobs)
# To create:  conda create -n llm python=3.10 -y
#             conda activate llm
#             pip install torch transformers peft bitsandbytes datasets \
#                         accelerate trl scikit-learn tqdm
conda activate llm

# ---------------------------------------------------------------------------
# 2. Verify GPU is available
# ---------------------------------------------------------------------------
echo ""
echo "--- GPU Info ---"
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
echo ""

# ---------------------------------------------------------------------------
# 3. Navigate to project directory
# ---------------------------------------------------------------------------
# IMPORTANT: Update this path to your actual project location on Arc
cd /home/xso947/Sequential-Instruction-Fine-Tuning-of-a-Small-LLM || { echo "Project directory not found!"; exit 1; }

# Create logs directory if it doesn't exist
mkdir -p logs checkpoints/stage1 data

# ---------------------------------------------------------------------------
# 4. Step A — Prepare Alpaca data (CPU-only, fast)
# ---------------------------------------------------------------------------
echo ">>> Preparing Alpaca data..."
python modular-code/alpaca-data-prep.py
echo ">>> Alpaca data ready."
echo ""

# ---------------------------------------------------------------------------
# 5. Step B — Run Stage 1 QLoRA fine-tuning
# ---------------------------------------------------------------------------
echo ">>> Starting Stage 1 training..."
python modular-code/fine-tuning-pipeline.py \
    --stage 1 \
    --output-dir checkpoints/stage1

echo ""
echo ">>> Stage 1 training complete."
echo ">>> Checkpoint saved to: checkpoints/stage1/"

# ---------------------------------------------------------------------------
# 6. Step C — Run Checkpoint 1 inference on both eval sets
# ---------------------------------------------------------------------------
# Checkpoint 0 inference already completed in a separate job.

echo ">>> Running Checkpoint 1 (Alpaca-tuned) inference..."
python modular-code/inference.py \
    --checkpoints 1 \
    --alpaca-eval data/alpaca_eval.json \
    --json-eval data/json_eval.json \
    --output-dir /home/xso947/Sequential-Instruction-Fine-Tuning-of-a-Small-LLM/outputs

echo ""
echo "=============================================="
echo "Stage 1 job complete at $(date)"
echo "Outputs saved to: outputs/"
echo "=============================================="
