#!/bin/bash
# ============================================================
# Taylor-MLP GPT-2 Experiment - Environment Setup
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Download dataset (HF mirror) ──────────────────────
export HF_ENDPOINT="https://hf-mirror.com"
export HF_DATASETS_CACHE="/vepfs-mlp2/c20250205/240804016/Datasets"

echo "=== Downloading OpenWebText dataset (HF mirror) ==="
echo "  Cache: ${HF_DATASETS_CACHE}"
python -c "from datasets import load_dataset; load_dataset('openwebtext')"

# ── Download model (ModelScope) ───────────────────────
MODEL_CACHE="${ROOT_DIR}/Model"

echo ""
echo "=== Downloading GPT-2 model (ModelScope) ==="
echo "  Cache: ${MODEL_CACHE}"
python -c "
from modelscope import snapshot_download
snapshot_download('AI-ModelScope/gpt2', cache_dir='${MODEL_CACHE}')
"

echo ""
echo "=== Setup complete ==="
