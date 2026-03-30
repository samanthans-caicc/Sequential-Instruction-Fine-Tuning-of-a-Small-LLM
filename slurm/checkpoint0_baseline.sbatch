#!/bin/bash
#SBATCH --job-name=checkpoint0-baseline
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/checkpoint0_baseline_%j.out
#SBATCH --error=logs/checkpoint0_baseline_%j.err

# ===========================================================================
# Checkpoint 0: Baseline Evaluation (Untuned Phi-3.5 Mini Instruct)
# ---------------------------------------------------------------------------
# Partition: gpu1a100 (1x A100 40GB)
# Expected runtime: ~1 hour
#   - Alpaca data prep:    ~5 min  (CPU only)
#   - Checkpoint 0 infer:  ~45 min (200 prompts on A100)
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
# 2. Verify GPU
# ---------------------------------------------------------------------------
echo ""
echo "--- GPU Info ---"
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
echo ""

# ---------------------------------------------------------------------------
# 3. Navigate to project directory
# ---------------------------------------------------------------------------
cd /home/xso947/fluffy-umbrella || { echo "Project directory not found!"; exit 1; }

mkdir -p logs data outputs

# ---------------------------------------------------------------------------
# 4. Alpaca data prep (CPU only — generates alpaca_train.json + alpaca_eval.json)
# ---------------------------------------------------------------------------
echo ">>> Preparing Alpaca data..."
python modular-code/alpaca-data-prep.py
echo ">>> Alpaca data ready."
echo ""

# ---------------------------------------------------------------------------
# 5. Checkpoint 0 inference (untuned base model, both eval sets)
# ---------------------------------------------------------------------------
echo ">>> Running Checkpoint 0 (baseline) inference..."
python modular-code/inference.py \
    --checkpoints 0 \
    --alpaca-eval data/alpaca_eval.json \
    --json-eval   data/json_eval.json \
    --output-dir  outputs

echo ""
echo "=============================================="
echo "Checkpoint 0 complete at $(date)"
echo "Outputs saved to: outputs/checkpoint_0_base/"
echo "=============================================="
