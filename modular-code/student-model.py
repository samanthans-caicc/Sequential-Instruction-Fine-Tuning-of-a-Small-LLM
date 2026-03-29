# Student Model: Phi-3.5 Mini Instruct loaded with QLoRA (4-bit) + LoRA adapters via PEFT.
# Used as the base for both Stage 1 and Stage 2 fine-tuning.

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

# LoRA target modules for Phi-3.5 Mini (attention + MLP projections)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def get_bnb_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantization config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> LoraConfig:
    """
    LoRA adapter config.
      r           – rank; higher = more capacity, more VRAM (16 is a good default)
      lora_alpha  – scaling factor; rule of thumb: 2 * r
      lora_dropout – regularization
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
    )


def load_tokenizer() -> AutoTokenizer:
    """Load the Phi-3.5 Mini tokenizer with padding configured for training."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=False,
    )
    # Phi-3.5 Mini uses <|endoftext|> as pad token by default; set explicitly
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required for causal LM training
    return tokenizer


def load_base_model(device_map: str = "auto") -> AutoModelForCausalLM:
    """Load Phi-3.5 Mini in 4-bit quantization (no LoRA adapters yet)."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=get_bnb_config(),
        device_map=device_map,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",  # remove if flash-attn not installed
    )
    model.config.use_cache = False          # required during training
    model.config.pretraining_tp = 1         # disable tensor parallelism for QLoRA
    return model


def load_student_for_training(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    device_map: str = "auto",
):
    """
    Returns (model, tokenizer) ready for SFT training.
    The base model is frozen; only the LoRA adapters are trainable.
    """
    tokenizer = load_tokenizer()
    base_model = load_base_model(device_map=device_map)
    lora_cfg = get_lora_config(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def load_student_from_checkpoint(
    checkpoint_path: str,
    device_map: str = "auto",
):
    """
    Load a previously saved LoRA checkpoint on top of the frozen base model.
    Use this to resume Stage 1 → Stage 2, or for inference.
    """
    tokenizer = load_tokenizer()
    base_model = load_base_model(device_map=device_map)
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    return model, tokenizer


def format_phi35_prompt(instruction: str, input_text: str = "") -> str:
    """
    Format a single example using Phi-3.5 Mini's chat template.
    Phi-3.5 uses: <|user|>...<|end|>\n<|assistant|>
    """
    user_content = instruction if not input_text else f"{instruction}\n\n{input_text}"
    return (
        f"<|user|>\n{user_content}<|end|>\n"
        f"<|assistant|>\n"
    )


def format_phi35_training_example(
    instruction: str,
    output: str,
    input_text: str = "",
) -> str:
    """
    Full training example (prompt + completion + end token).
    The loss is computed over the entire sequence; mask the prompt portion
    in the data collator (e.g., DataCollatorForCompletionOnlyLM) if desired.
    """
    prompt = format_phi35_prompt(instruction, input_text)
    return f"{prompt}{output}<|end|>"


if __name__ == "__main__":
    # Quick sanity check — loads tokenizer only (no GPU needed)
    tokenizer = load_tokenizer()
    example = format_phi35_training_example(
        instruction="Explain what a transformer model is.",
        output="A transformer is a neural network architecture based on self-attention...",
    )
    tokens = tokenizer(example, return_tensors="pt")
    print(f"Tokenizer loaded. Example token count: {tokens['input_ids'].shape[1]}")
    print(example)
