"""
Alpaca Data Preparation
-----------------------
1. Downloads tatsu-lab/alpaca from Hugging Face.
2. Normalizes every example into {instruction, input, output} schema.
3. Validates and removes malformed / low-quality samples.
4. Splits into train / eval subsets (eval is held out and never used for training).
5. Saves both splits to disk as JSON for use by the fine-tuning pipeline.
"""

import json
import logging
import re
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import cfg
from datasets import load_dataset
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"instruction", "output"}   # "input" is optional (may be empty)
MIN_INSTRUCTION_CHARS = 10
MIN_OUTPUT_CHARS = 5
MAX_INSTRUCTION_CHARS = 2000
MAX_OUTPUT_CHARS = 4000


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_alpaca() -> list[dict]:
    """Download tatsu-lab/alpaca and return as a list of raw dicts."""
    log.info("Downloading tatsu-lab/alpaca ...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    records = [dict(row) for row in ds]
    log.info(f"Downloaded {len(records):,} raw examples.")
    return records


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(record: dict) -> dict:
    """
    Coerce a raw Alpaca row into the canonical {instruction, input, output} schema.
    - Strips surrounding whitespace and normalizes internal whitespace.
    - Ensures 'input' key exists (empty string if absent).
    """
    instruction = re.sub(r"\s+", " ", str(record.get("instruction", "")).strip())
    input_text  = re.sub(r"\s+", " ", str(record.get("input",       "")).strip())
    output      = re.sub(r"\s+", " ", str(record.get("output",      "")).strip())
    return {"instruction": instruction, "input": input_text, "output": output}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def is_valid(record: dict) -> tuple[bool, str]:
    """
    Return (True, "") if the record passes all quality checks,
    otherwise (False, reason).
    """
    # Required keys present
    if not REQUIRED_KEYS.issubset(record.keys()):
        return False, f"missing keys: {REQUIRED_KEYS - record.keys()}"

    instr  = record["instruction"]
    output = record["output"]

    # Non-empty
    if not instr:
        return False, "empty instruction"
    if not output:
        return False, "empty output"

    # Length guards
    if len(instr) < MIN_INSTRUCTION_CHARS:
        return False, f"instruction too short ({len(instr)} chars)"
    if len(output) < MIN_OUTPUT_CHARS:
        return False, f"output too short ({len(output)} chars)"
    if len(instr) > MAX_INSTRUCTION_CHARS:
        return False, f"instruction too long ({len(instr)} chars)"
    if len(output) > MAX_OUTPUT_CHARS:
        return False, f"output too long ({len(output)} chars)"

    # Reject placeholder / boilerplate outputs
    boilerplate = [
        "as an ai language model",
        "i cannot",
        "i'm sorry, but",
        "i apologize",
    ]
    output_lower = output.lower()
    for phrase in boilerplate:
        if output_lower.startswith(phrase):
            return False, f"boilerplate output: '{phrase}'"

    # Reject near-duplicate instruction == output
    if instr.strip().lower() == output.strip().lower():
        return False, "instruction equals output"

    return True, ""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def prepare(
    max_samples: int | None = None,
    eval_split: float | None = None,
    output_dir: str = "data",
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Full preparation pipeline.

    Returns (train_data, eval_data) as lists of normalized dicts.
    Also saves them to {output_dir}/alpaca_train.json and alpaca_eval.json.
    """
    max_samples = max_samples or cfg.stage1.max_samples
    eval_split  = eval_split  or cfg.stage1.eval_split

    # 1. Download
    raw = download_alpaca()

    # 2. Normalize
    normalized = [normalize(r) for r in raw]

    # 3. Validate
    valid, rejected = [], []
    for record in normalized:
        ok, reason = is_valid(record)
        if ok:
            valid.append(record)
        else:
            rejected.append((record, reason))

    log.info(f"Valid:    {len(valid):,}")
    log.info(f"Rejected: {len(rejected):,}")

    if rejected:
        reasons: dict[str, int] = {}
        for _, reason in rejected:
            # Bucket reason by its prefix (before the first colon or parenthesis)
            key = re.split(r"[:(]", reason)[0].strip()
            reasons[key] = reasons.get(key, 0) + 1
        log.info("Rejection breakdown: " + str(reasons))

    # 4. Cap at max_samples
    if max_samples and len(valid) > max_samples:
        valid = valid[:max_samples]
        log.info(f"Capped to {max_samples:,} samples.")

    # 5. Train / eval split (stratification not needed for instruction data)
    train_data, eval_data = train_test_split(
        valid, test_size=eval_split, random_state=seed, shuffle=True
    )
    log.info(f"Train: {len(train_data):,} | Eval (held-out): {len(eval_data):,}")

    # 6. Save
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "alpaca_train.json")
    eval_path  = os.path.join(output_dir, "alpaca_eval.json")

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)

    log.info(f"Saved train -> {train_path}")
    log.info(f"Saved eval  -> {eval_path}")

    return train_data, eval_data


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------

def load_split(path: str) -> list[dict]:
    """Load a saved JSON split from disk."""
    with open(path) as f:
        return json.load(f)


def print_sample(data: list[dict], n: int = 3) -> None:
    """Print n random examples for a quick visual sanity check."""
    import random
    for record in random.sample(data, min(n, len(data))):
        print("-" * 60)
        print(f"INSTRUCTION: {record['instruction']}")
        if record["input"]:
            print(f"INPUT:       {record['input']}")
        print(f"OUTPUT:      {record['output'][:200]}...")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train, eval_ = prepare()
    print_sample(train)
