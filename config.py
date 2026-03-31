from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------

STUDENT_MODEL_ID   = "microsoft/Phi-3.5-mini-instruct"
TEACHER_MODEL_ID   = "meta-llama/Llama-3.1-70B-Instruct"
JUDGE_MODEL_ID     = "meta-llama/Llama-3.1-70B-Instruct"


# ---------------------------------------------------------------------------
# LoRA parameters (shared across both stages)
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    r: int             = 16      # adapter rank
    lora_alpha: int    = 32      # scaling factor (rule of thumb: 2 * r)
    lora_dropout: float = 0.05
    bias: str          = "none"
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


# ---------------------------------------------------------------------------
# Stage 1 training — Alpaca general instruction-following
# ---------------------------------------------------------------------------

@dataclass
class Stage1Config:
    # Data
    dataset_name: str       = "tatsu-lab/alpaca"
    max_samples: int        = 52000          # use full Alpaca dataset

    # Tokenisation
    max_seq_length: int     = 512            # Alpaca examples are typically <300 tokens; 512 saves ~2x training time

    # Optimiser
    learning_rate: float    = 2e-5
    lr_scheduler: str       = "cosine"
    warmup_ratio: float     = 0.03

    # Training loop
    num_epochs: int         = 3
    per_device_train_batch_size: int  = 4
    per_device_eval_batch_size: int   = 4
    gradient_accumulation_steps: int  = 4    # effective batch = 4 * 4 = 16
    gradient_checkpointing: bool      = True

    # Precision
    bf16: bool = True
    fp16: bool = False

    # Checkpointing — step-based so jobs can resume within a 24-hour wall time
    output_dir: str              = "checkpoints/stage1"
    save_strategy: str           = "steps"
    save_steps: int              = 500      # ~1 hour on V100
    save_total_limit: int        = 3

    # Evaluation
    eval_strategy: str           = "steps"
    eval_steps: int              = 500
    eval_split: float            = 0.05      # 5 % of stage-1 data held out
    load_best_model_at_end: bool = True
    metric_for_best_model: str   = "eval_loss"


# ---------------------------------------------------------------------------
# Stage 2 training — teacher-generated JSON instruction data
# ---------------------------------------------------------------------------

@dataclass
class Stage2Config:
    # Data
    dataset_path: str       = "data/teacher_generated.json"
    max_samples: int        = 5000           # adjust to how many you generate

    # Tokenisation
    max_seq_length: int     = 2048           # teacher outputs tend to be longer

    # Optimiser — lower LR to preserve Stage 1 knowledge
    learning_rate: float    = 2e-5
    lr_scheduler: str       = "cosine"
    warmup_ratio: float     = 0.03

    # Training loop
    num_epochs: int         = 5
    per_device_train_batch_size: int  = 2
    per_device_eval_batch_size: int   = 2
    gradient_accumulation_steps: int  = 8    # effective batch = 2 * 8 = 16
    gradient_checkpointing: bool      = True

    # Precision
    bf16: bool = True
    fp16: bool = False

    # Checkpointing — continues from stage1 output
    resume_from_checkpoint: str  = "checkpoints/stage1"
    output_dir: str              = "checkpoints/stage2"
    save_strategy: str           = "epoch"
    save_total_limit: int        = 2

    # Evaluation
    eval_strategy: str           = "epoch"
    eval_split: float            = 0.10      # 10 % of stage-2 data held out
    load_best_model_at_end: bool = True
    metric_for_best_model: str   = "eval_loss"


# ---------------------------------------------------------------------------
# Evaluation / judge settings
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    # Checkpoints to compare
    checkpoints: list = field(default_factory=lambda: [
        "base",                    # untrained Phi-3.5 Mini
        "checkpoints/stage1",
        "checkpoints/stage2",
    ])

    # Generation settings (student, during eval)
    max_new_tokens: int    = 512
    temperature: float     = 0.7
    top_p: float           = 0.9
    do_sample: bool        = True

    # Judge generation settings (deterministic for consistency)
    judge_max_new_tokens: int  = 256
    judge_temperature: float   = 0.0

    # Number of test prompts to evaluate per checkpoint
    num_eval_prompts: int  = 100

    # Output
    results_dir: str       = "eval_results"
    results_file: str      = "eval_results/judge_scores.json"


# ---------------------------------------------------------------------------
# Convenience: single object with all configs
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    student_model_id: str  = STUDENT_MODEL_ID
    teacher_model_id: str  = TEACHER_MODEL_ID
    judge_model_id: str    = JUDGE_MODEL_ID
    lora:    LoRAConfig    = field(default_factory=LoRAConfig)
    stage1:  Stage1Config  = field(default_factory=Stage1Config)
    stage2:  Stage2Config  = field(default_factory=Stage2Config)
    eval:    EvalConfig    = field(default_factory=EvalConfig)


# Default instance — import this in other modules
cfg = PipelineConfig()


if __name__ == "__main__":
    from dataclasses import asdict
    import json
    print(json.dumps(asdict(cfg), indent=2))
