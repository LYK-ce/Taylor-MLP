#!/bin/bash
# ============================================================
# Taylor-MLP GPT-2 Experiment - Environment Setup
# ============================================================
set -e

# ── HuggingFace cache paths ────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
HF_CACHE="${ROOT_DIR}/.hf_cache"

export HF_HOME="${HF_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hub"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"

echo "HF cache root: ${HF_CACHE}"

echo "=== Downloading OpenWebText dataset ==="
python -c "from datasets import load_dataset; load_dataset('openwebtext')"

echo "=== Downloading GPT-2 model ==="
python -c "from transformers import GPT2LMHeadModel, GPT2Tokenizer; GPT2LMHeadModel.from_pretrained('openai-community/gpt2'); GPT2Tokenizer.from_pretrained('openai-community/gpt2')"

echo "=== Setup complete ==="
