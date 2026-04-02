#!/bin/bash
#SBATCH --job-name=judge-eval
#SBATCH --partition=gpu4v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/judge_eval_%j.out
#SBATCH --error=logs/judge_eval_%j.err

# ===========================================================================
# Judge Evaluation — Llama 3.1 70B Instruct pairwise scoring
# ---------------------------------------------------------------------------
# GPU: 1 × V100-PCIE-32GB — sufficient for 8B at 4-bit (~6 GB)
# Partition: gpu4v100 (4-GPU node)
# Expected runtime: ~6–8 hrs for 3 pairs × 2 eval types × ~250 prompts
#
# Prerequisite: inference outputs must exist at:
#   outputs/checkpoint_0_base/
#   outputs/checkpoint_1_alpaca/
#   outputs/checkpoint_2_json/
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
# 2. Verify GPUs and environment
# ---------------------------------------------------------------------------
echo ""
echo "--- GPU Info ---"
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}')"
echo ""

cd /home/xso947/fluffy-umbrella || { echo "Project directory not found!"; exit 1; }

export HF_TOKEN=$HF_TOKEN

# Check inference outputs exist
missing=0
for ckpt in checkpoint_0_base checkpoint_1_alpaca checkpoint_2_json; do
    if [ ! -d "outputs/$ckpt" ]; then
        echo "ERROR: outputs/$ckpt/ not found."
        missing=1
    else
        echo ">>> Found outputs/$ckpt/"
    fi
done
if [ $missing -eq 1 ]; then
    echo "Missing inference outputs. Run inference first."
    exit 1
fi

mkdir -p logs eval_results

# ---------------------------------------------------------------------------
# 3. Run judge evaluation — all pairs, both eval types
#    Resume is automatic: already-scored prompts are skipped
# ---------------------------------------------------------------------------
echo ""
echo ">>> Running judge evaluation (all pairs, both eval types)..."
echo "    Judge: meta-llama/Llama-3.1-8B-Instruct"
echo "    Pairs: 0v1, 1v2, 0v2"
echo "    Eval types: alpaca, json"
echo ""

python modular-code/judge.py \
    --pairs 0v1 1v2 0v2 \
    --eval-type both \
    --outputs-dir outputs \
    --output-dir eval_results \
    --system-prompt prompts/judge_system.txt \
    --user-template prompts/judge_user_template.txt \
    --max-new-tokens 512 \
    --seed 42

echo ""
echo "=============================================="
echo "Judge evaluation complete at $(date)"
echo "Results saved to: eval_results/"
ls eval_results/
echo "=============================================="
