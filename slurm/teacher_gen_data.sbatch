#!/bin/bash
#SBATCH --job-name=teacher-gen
#SBATCH --partition=gpu4v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/teacher_gen_%j.out
#SBATCH --error=logs/teacher_gen_%j.err

# ===========================================================================
# Teacher Model: Generate JSON Instruct Dataset via Imitation Learning
# ---------------------------------------------------------------------------
# This script runs Llama 3.1 70B Instruct to generate structured JSON
# responses that become the Stage 2 training data for the student model.
#
# GPU REQUIREMENTS:
#   - Llama 3.1 70B at 4-bit quantization needs ~35-40GB VRAM
#   - A single A100 40GB works but is tight. If you hit OOM errors:
#     Option A: Use a DGX node (change partition)
#     Option B: Use an API service (Together AI, Fireworks, etc.)
#              and set --use-api flag in the script
#
# If using an API instead, you can run this on a CPU-only node:
#   #SBATCH --partition=compute1
#   #SBATCH --gres=          (remove gpu line)
#   #SBATCH --mem=32G
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
echo ""

cd /home/xso947/fluffy-umbrella || { echo "Project directory not found!"; exit 1; }

# HuggingFace token for gated models (Llama 3.1 70B)
export HF_TOKEN=$HF_TOKEN

mkdir -p logs data

# ---------------------------------------------------------------------------
# 3. Generate teacher responses for JSON Instruct dataset
# ---------------------------------------------------------------------------
echo ">>> Generating teacher JSON data via imitation learning..."
echo "    Teacher model: meta-llama/Llama-3.1-70B-Instruct"
echo "    This may take several hours depending on dataset size."
echo ""

python modular-code/teacher-gen-data.py \
    --output-path data/teacher_generated.json \
    --num-samples 5000 \
    --validate-json

# ---------------------------------------------------------------------------
# 4. Validate and report stats
# ---------------------------------------------------------------------------
echo ""
echo ">>> Validating generated dataset..."
python -c "
import json
with open('data/teacher_generated.json') as f:
    data = json.load(f)
valid = sum(1 for d in data if d.get('output'))
print(f'Total examples: {len(data)}')
print(f'Valid examples: {valid}')
print(f'Task types: {set(d.get(\"task_type\", \"unknown\") for d in data)}')
"

echo ""
echo "=============================================="
echo "Teacher data generation complete at $(date)"
echo "Output: data/teacher_generated.json"
echo "=============================================="
