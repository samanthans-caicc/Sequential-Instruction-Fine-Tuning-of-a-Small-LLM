# Sequential Instruction Fine-Tuning of a Small LLM

Sequential QLoRA fine-tuning of **Phi-3.5 Mini Instruct** in two stages, with a strong-model (Llama 3.1 8B) pairwise judge evaluation.

---

## Overview

| Stage | Data | Script |
|-------|------|--------|
| 0 — Baseline | — (untuned base model) | `inference.py` |
| 1 — General instruction following | Alpaca (`tatsu-lab/alpaca`, 5 k samples) | `fine-tuning-pipeline.py --stage 1` |
| 2 — JSON instruction following | Teacher-generated data via Llama 3.1 8B | `fine-tuning-pipeline.py --stage 2` |

Outputs at each checkpoint are compared pairwise with a judge LLM across six quality dimensions.

---

## Repository layout

```
modular-code/
  alpaca-data-prep.py        # download & format Alpaca data
  alpaca-instruct-data.py    # alternate Alpaca formatter
  json-instruct-dataset.py   # load/inspect teacher-generated dataset
  teacher-gen-data.py        # generate JSON training data via teacher model
  fine-tuning-pipeline.py    # QLoRA training (Stage 1 & 2)
  inference.py               # run inference at each checkpoint
  judge.py                   # pairwise LLM-as-a-judge evaluation
  result-aggregation.py      # aggregate & summarise judge scores
  student-model.py           # model loading utility

slurm/                       # SLURM batch scripts for Arc HPC
  stage1_alpaca_train.sbatch
  stage2_json_train.sbatch
  teacher_gen_data.sbatch
  judge_eval.sbatch
  checkpoint0_baseline.sbatch
  ablation_*.sbatch

prompts/
  teacher_gen_system.txt     # system prompt for teacher data generation
  teacher_gen_prompts.json   # 60 evaluation task prompts (5 task types x 15)
  judge_system.txt           # system prompt for judge model
  judge_user_template.txt    # per-example judge user message template

data/
  json_eval.json             # JSON-task evaluation set

config.py                    # all hyperparameters and model IDs
requirements.txt             # Python dependencies
outputs/                     # inference outputs (created at runtime)
checkpoints/                 # LoRA checkpoints (created at runtime)
```

---

## Setup

### Requirements

```bash
conda create -n llm python=3.10 -y
conda activate llm
pip install -r requirements.txt
```

### HuggingFace access

Both `microsoft/Phi-3.5-mini-instruct` and `meta-llama/Llama-3.1-8B-Instruct` are gated. Accept the terms on HuggingFace and authenticate:

```bash
huggingface-cli login
```

---

## Running locally (single GPU)

### Step 1 — Prepare Alpaca data

```bash
python modular-code/alpaca-data-prep.py
```

### Step 2 — Stage 1 training (Alpaca)

```bash
python modular-code/fine-tuning-pipeline.py \
    --stage 1 \
    --output-dir checkpoints/stage1
```

### Step 3 — Generate teacher training data

```bash
python modular-code/teacher-gen-data.py
```

Outputs to `data/teacher_generated.json`.

### Step 4 — Stage 2 training (JSON instruction)

```bash
python modular-code/fine-tuning-pipeline.py \
    --stage 2 \
    --output-dir checkpoints/stage2
```

### Step 5 — Run inference at all checkpoints

```bash
python modular-code/inference.py \
    --checkpoints 0 1 2 \
    --alpaca-eval data/alpaca_eval.json \
    --json-eval data/json_eval.json \
    --output-dir outputs/
```

### Step 6 — Judge evaluation

```bash
python modular-code/judge.py
```

### Step 7 — Aggregate results

```bash
python modular-code/result-aggregation.py
```

---

## Running on Arc HPC (SLURM)

Submit jobs in order:

```bash
# Baseline inference (Checkpoint 0)
sbatch slurm/checkpoint0_baseline.sbatch

# Stage 1 training + Checkpoint 1 inference
sbatch slurm/stage1_alpaca_train.sbatch

# Teacher data generation
sbatch slurm/teacher_gen_data.sbatch

# Stage 2 training
sbatch slurm/stage2_json_train.sbatch

# Judge evaluation
sbatch slurm/judge_eval.sbatch
```

> Update the project path in each `.sbatch` file from `/home/xso947/Sequential-Instruction-Fine-Tuning-of-a-Small-LLM` to your own home directory before submitting.

Logs are written to `logs/<jobname>_<jobid>.out` / `.err`.

---

## Configuration

All hyperparameters, model IDs, and paths live in `config.py`. Key values:

| Parameter | Value |
|-----------|-------|
| Student model | `microsoft/Phi-3.5-mini-instruct` |
| Teacher / Judge model | `meta-llama/Llama-3.1-8B-Instruct` |
| Stage 1 LR | `2e-5`, cosine, 3 epochs |
| Stage 2 LR | `2e-5`, cosine, 5 epochs |
| LoRA rank | 16 (alpha 32) |
| Effective batch size | 16 (both stages) |

---

## Outputs

| Path | Contents |
|------|----------|
| `checkpoints/stage1/` | Stage 1 LoRA adapter |
| `checkpoints/stage2/` | Stage 2 LoRA adapter |
| `outputs/` | Per-checkpoint inference results |
| `eval_results/judge_scores.json` | Raw judge scores |
