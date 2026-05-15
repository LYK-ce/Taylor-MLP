#!/bin/bash
# ============================================================
# Taylor-MLP GPT-2 Experiment - Environment Setup
# ============================================================
set -e

# ── HuggingFace cache paths ────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

export HF_DATASETS_CACHE="/vepfs-mlp2/c20250205/240804016/Datasets"
export HUGGINGFACE_HUB_CACHE="${ROOT_DIR}/Model"

echo "Datasets cache: ${HF_DATASETS_CACHE}"
echo "Model cache:    ${HUGGINGFACE_HUB_CACHE}"

echo "=== Downloading OpenWebText dataset ==="
python -c "from datasets import load_dataset; load_dataset('openwebtext')"

echo "=== Downloading GPT-2 model ==="
python -c "from transformers import GPT2LMHeadModel, GPT2Tokenizer; GPT2LMHeadModel.from_pretrained('openai-community/gpt2'); GPT2Tokenizer.from_pretrained('openai-community/gpt2')"

echo "=== Setup complete ==="
