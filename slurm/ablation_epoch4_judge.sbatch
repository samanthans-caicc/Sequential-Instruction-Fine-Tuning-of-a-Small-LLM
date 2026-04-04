#!/bin/bash
#SBATCH --job-name=ablation-epoch4-judge
#SBATCH --partition=gpu4v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/ablation_epoch4_judge_%j.out
#SBATCH --error=logs/ablation_epoch4_judge_%j.err

echo "=============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Start time:    $(date)"
echo "=============================================="

module purge
module load cuda/12.1
module load anaconda3
conda activate llm

nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

cd /home/xso947/Sequential-Instruction-Fine-Tuning-of-a-Small-LLM || { echo "Project directory not found!"; exit 1; }

# Verify inference outputs exist
for ckpt in checkpoint_1_alpaca ablation_epoch4 checkpoint_2_json; do
    for eval_type in alpaca_eval json_eval; do
        path="outputs/${ckpt}/${eval_type}_responses.json"
        if [ ! -f "$path" ]; then
            echo "ERROR: Missing $path — run inference first."
            exit 1
        fi
    done
done

mkdir -p logs eval_results

python - <<'PYEOF'
import sys, json, logging
sys.path.insert(0, "modular-code")

# Import judge utilities
import judge as judge_mod
from judge import (
    load_judge_model, load_prompt_files, run_judge_pair,
    DEFAULT_SYSTEM_FILE, DEFAULT_USER_TEMPLATE_FILE,
)
from config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Extend checkpoint registry to include ablation epoch-4
judge_mod.CHECKPOINT_INFO["e4"] = {
    "name": "ablation_epoch4",
    "label": "Stage 2 Epoch 4 (checkpoint-1112)",
}

OUTPUTS_DIR = "outputs"
EVAL_DIR    = "eval_results"
PAIRS = [
    ("1", "e4"),   # Stage1 vs Epoch4
    ("e4", "2"),   # Epoch4 vs Stage2-final
]
EVAL_TYPES = ["alpaca", "json"]

model, tokenizer = load_judge_model(cfg.judge_model_id)
system_prompt, user_template = load_prompt_files(DEFAULT_SYSTEM_FILE, DEFAULT_USER_TEMPLATE_FILE)

for ckpt_a, ckpt_b in PAIRS:
    name_a = judge_mod.CHECKPOINT_INFO[ckpt_a]["name"]
    name_b = judge_mod.CHECKPOINT_INFO[ckpt_b]["name"]
    for eval_type in EVAL_TYPES:
        log.info(f"\n{'='*60}")
        log.info(f"Pair: {ckpt_a}v{ckpt_b} | Eval: {eval_type}")
        log.info(f"  A: {name_a}")
        log.info(f"  B: {name_b}")
        log.info(f"{'='*60}")

        responses_a = judge_mod.load_responses(OUTPUTS_DIR, name_a, eval_type)
        responses_b = judge_mod.load_responses(OUTPUTS_DIR, name_b, eval_type)

        output_path = f"{EVAL_DIR}/judge_{ckpt_a}v{ckpt_b}_{eval_type}.json"
        run_judge_pair(
            model, tokenizer,
            system_prompt, user_template,
            responses_a, responses_b,
            ckpt_a, ckpt_b,
            eval_type,
            output_path,
            max_new_tokens=512,
        )

log.info("Ablation epoch-4 judge evaluation complete.")
PYEOF

echo "=============================================="
echo "Ablation judge complete at $(date)"
ls eval_results/judge_*e4*.json
echo "=============================================="
