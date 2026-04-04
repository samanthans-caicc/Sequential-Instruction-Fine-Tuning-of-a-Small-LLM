#!/bin/bash
#SBATCH --job-name=ablation-epoch4-inf
#SBATCH --partition=gpu4v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/ablation_epoch4_inf_%j.out
#SBATCH --error=logs/ablation_epoch4_inf_%j.err

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

mkdir -p logs outputs/ablation_epoch4

python - <<'PYEOF'
import sys, os, json, time, logging
import torch
from tqdm import tqdm

sys.path.insert(0, "modular-code")
import importlib
sm = importlib.import_module("student-model")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ADAPTER_PATH = "checkpoints/stage2/checkpoint-1112"
OUTPUT_DIR   = "outputs/ablation_epoch4"
EVAL_FILES   = {
    "alpaca_eval": "data/alpaca_eval.json",
    "json_eval":   "data/json_eval.json",
}

log.info(f"Loading model from {ADAPTER_PATH} ...")
model, tokenizer = sm.load_student_from_checkpoint(ADAPTER_PATH)
model.eval()
log.info("Model loaded.")

for eval_name, eval_path in EVAL_FILES.items():
    if not os.path.exists(eval_path):
        log.warning(f"{eval_path} not found — skipping.")
        continue

    with open(eval_path) as f:
        data = json.load(f)
    log.info(f"Running inference on {len(data)} {eval_name} examples ...")

    results = []
    for example in tqdm(data, desc=eval_name):
        instruction = example.get("instruction", "")
        input_text  = example.get("input", "") or ""
        reference   = example.get("output", "") or example.get("reference", "")
        prompt_id   = example.get("prompt_id", f"{eval_name}_{len(results):04d}")

        prompt = sm.format_phi35_prompt(instruction, input_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.time() - t0

        prompt_len = inputs["input_ids"].shape[1]
        response = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()

        results.append({
            "prompt_id": prompt_id,
            "instruction": instruction,
            "input": input_text,
            "reference": reference,
            "response": response,
            "generation_time_s": round(elapsed, 2),
            "response_token_count": len(tokenizer.encode(response)),
        })

    out_path = os.path.join(OUTPUT_DIR, f"{eval_name}_responses.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved {len(results)} results -> {out_path}")

log.info("Ablation epoch-4 inference complete.")
PYEOF

echo "=============================================="
echo "Done at $(date)"
echo "=============================================="
