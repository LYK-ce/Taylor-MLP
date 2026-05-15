#!/bin/bash
# ============================================================
# Taylor-MLP GPT-2 Experiment - Environment Setup
# ============================================================
set -e

echo "=== Downloading OpenWebText dataset ==="
python -c "from datasets import load_dataset; load_dataset('openwebtext')"

echo "=== Downloading GPT-2 model ==="
python -c "from transformers import GPT2Model, GPT2Tokenizer; GPT2Model.from_pretrained('openai-community/gpt2'); GPT2Tokenizer.from_pretrained('openai-community/gpt2')"

echo "=== Setup complete ==="
