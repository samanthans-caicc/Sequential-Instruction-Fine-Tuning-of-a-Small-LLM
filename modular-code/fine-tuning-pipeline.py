"""
Fine-Tuning Pipeline
--------------------
QLoRA-based sequential fine-tuning of Phi-3.5 Mini Instruct.
  Stage 1: Alpaca general instruction data  (tatsu-lab/alpaca)
  Stage 2: Teacher-generated JSON instruction data

Usage:
    python modular-code/fine-tuning-pipeline.py --stage 1 --output-dir checkpoints/stage1
    python modular-code/fine-tuning-pipeline.py --stage 2 --output-dir checkpoints/stage2
"""

import argparse
import importlib
import json
import logging
import os
import random
import sys

import torch
from datasets import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Hyphenated filename — must use importlib
student_model = importlib.import_module("student-model")
from config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# trl compatibility: SFTConfig (trl >= 0.8) vs TrainingArguments fallback
# ---------------------------------------------------------------------------
try:
    from trl import SFTConfig as _TrainingCls
    _USE_SFT_CONFIG = True
    log.info("Using trl.SFTConfig")
except ImportError:
    from transformers import TrainingArguments as _TrainingCls
    _USE_SFT_CONFIG = False
    log.info("trl.SFTConfig not available — using TrainingArguments")

from trl import SFTTrainer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stage1_data(data_dir: str):
    """Load pre-split Alpaca train/eval JSON files produced by alpaca-data-prep.py."""
    train_path = os.path.join(data_dir, "alpaca_train.json")
    eval_path  = os.path.join(data_dir, "alpaca_eval.json")

    for path in (train_path, eval_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' not found. Run alpaca-data-prep.py first."
            )

    with open(train_path) as f:
        train_data = json.load(f)
    with open(eval_path) as f:
        eval_data = json.load(f)

    if cfg.stage1.max_samples and len(train_data) > cfg.stage1.max_samples:
        train_data = train_data[: cfg.stage1.max_samples]
        log.info(f"Stage 1 train capped to {cfg.stage1.max_samples:,} samples.")

    log.info(f"Stage 1 — train: {len(train_data):,} | eval: {len(eval_data):,}")
    return Dataset.from_list(train_data), Dataset.from_list(eval_data)


def load_stage2_data():
    """Load teacher-generated JSON data and split into train/eval."""
    path = cfg.stage2.dataset_path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Stage 2 data not found at '{path}'. Run teacher-gen-data.py first."
        )

    with open(path) as f:
        all_data = json.load(f)

    random.seed(42)
    random.shuffle(all_data)

    n_eval     = max(1, int(len(all_data) * cfg.stage2.eval_split))
    eval_data  = all_data[:n_eval]
    train_data = all_data[n_eval:]

    if cfg.stage2.max_samples and len(train_data) > cfg.stage2.max_samples:
        train_data = train_data[: cfg.stage2.max_samples]

    log.info(f"Stage 2 — train: {len(train_data):,} | eval: {len(eval_data):,}")
    return Dataset.from_list(train_data), Dataset.from_list(eval_data)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def formatting_func(examples: dict) -> list[str]:
    """
    Format a batch of examples into full training strings (prompt + completion).
    SFTTrainer calls this with a batch dict (dict of lists), not a list of dicts.
    Loss is computed over the entire sequence — simpler and more robust than
    using DataCollatorForCompletionOnlyLM with a response template.
    """
    n = len(examples["instruction"])
    inputs = examples.get("input", [""] * n)
    return [
        student_model.format_phi35_training_example(
            instruction=examples["instruction"][i],
            output=examples["output"][i],
            input_text=inputs[i] if inputs[i] else "",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------

def resolve_precision(stage_cfg) -> tuple[bool, bool]:
    """Return (bf16, fp16). Falls back to fp16 on GPUs without bf16 (e.g. V100)."""
    if stage_cfg.bf16 and torch.cuda.is_bf16_supported():
        return True, False
    log.info("bf16 not supported on this GPU — using fp16 instead.")
    return False, True


# ---------------------------------------------------------------------------
# Training args
# ---------------------------------------------------------------------------

def build_training_args(stage_cfg, output_dir: str, bf16: bool, fp16: bool):
    common = dict(
        output_dir=output_dir,
        num_train_epochs=stage_cfg.num_epochs,
        per_device_train_batch_size=stage_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=stage_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=stage_cfg.gradient_accumulation_steps,
        gradient_checkpointing=stage_cfg.gradient_checkpointing,
        learning_rate=stage_cfg.learning_rate,
        lr_scheduler_type=stage_cfg.lr_scheduler,
        warmup_ratio=stage_cfg.warmup_ratio,
        bf16=bf16,
        fp16=fp16,
        save_strategy=stage_cfg.save_strategy,
        save_total_limit=stage_cfg.save_total_limit,
        # Evaluation during training
        eval_strategy=stage_cfg.eval_strategy,
        # load_best_model_at_end=True is unreliable with PEFT — disabled
        load_best_model_at_end=False,
        logging_steps=25,
        report_to="none",
        optim="paged_adamw_8bit",
    )

    return _TrainingCls(**common)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(stage: int, output_dir: str, data_dir: str, resume_from: str | None):
    stage_cfg = cfg.stage1 if stage == 1 else cfg.stage2
    bf16, fp16 = resolve_precision(stage_cfg)

    # --- Load model + tokenizer ---
    if stage == 1:
        log.info("Loading base model for Stage 1 training...")
        model, tokenizer = student_model.load_student_for_training(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
        )
    else:
        checkpoint = resume_from or cfg.stage2.resume_from_checkpoint
        log.info(f"Loading Stage 1 adapter from '{checkpoint}' for Stage 2...")
        model, tokenizer = student_model.load_student_from_checkpoint(checkpoint)
        model.train()
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        model.print_trainable_parameters()

    # --- Load data ---
    train_ds, eval_ds = load_stage1_data(data_dir) if stage == 1 else load_stage2_data()
    log.info(f"Train examples: {len(train_ds):,} | Eval examples: {len(eval_ds):,}")

    # Sanity check — log a formatted example so we can verify the template
    sample = train_ds[0]
    sample_text = student_model.format_phi35_training_example(
        instruction=sample["instruction"],
        output=sample["output"],
        input_text=sample.get("input", ""),
    )
    log.info(f"Sample training text (first 300 chars):\n{sample_text[:300]}")

    # --- Build training args ---
    training_args = build_training_args(stage_cfg, output_dir, bf16, fp16)

    # --- Build SFTTrainer ---
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        formatting_func=formatting_func,
        tokenizer=tokenizer,
        max_seq_length=stage_cfg.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(**trainer_kwargs)

    log.info(
        f"Starting Stage {stage} training | "
        f"epochs={stage_cfg.num_epochs} | lr={stage_cfg.learning_rate} | "
        f"effective_batch={stage_cfg.per_device_train_batch_size * stage_cfg.gradient_accumulation_steps} | "
        f"output={output_dir}"
    )
    log.info(f"Total training steps: {len(trainer.get_train_dataloader()) * stage_cfg.num_epochs}")

    trainer.train()

    log.info(f"Saving adapter checkpoint to '{output_dir}'...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("Stage %d complete.", stage)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning pipeline")
    parser.add_argument(
        "--stage", type=int, choices=[1, 2], required=True,
        help="Training stage (1=Alpaca, 2=JSON)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Adapter checkpoint output directory (default: from config)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Directory containing alpaca_train.json / alpaca_eval.json",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Stage 2 only: path to Stage 1 adapter (default: config value)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (
        cfg.stage1.output_dir if args.stage == 1 else cfg.stage2.output_dir
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    train(
        stage=args.stage,
        output_dir=output_dir,
        data_dir=args.data_dir,
        resume_from=args.resume_from,
    )
