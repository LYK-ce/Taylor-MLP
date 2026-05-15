#!/bin/bash
# ============================================================
# Taylor-MLP GPT-2 Experiment - Environment Setup
#
# Downloads dataset and model from ModelScope.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Dataset (ModelScope snapshot) ──────────────────────
echo "=== [1/2] Downloading OpenWebText dataset (ModelScope) ==="
DATASET_DIR="/vepfs-mlp2/c20250205/240804016/Datasets/openwebtext"
mkdir -p "${DATASET_DIR}"
echo "  Target: ${DATASET_DIR}"
python -c "
from modelscope import snapshot_download
snapshot_download('mapjack/openwebtext_dataset', cache_dir='${DATASET_DIR}')
print('Done.')
"

# ── Model ──────────────────────────────────────────────
echo ""
echo "=== [2/2] Downloading GPT-2 model (ModelScope) ==="
MODEL_CACHE="${ROOT_DIR}/Model"
echo "  Cache: ${MODEL_CACHE}"
python -c "
from modelscope import snapshot_download
snapshot_download('openai-community/gpt2', cache_dir='${MODEL_CACHE}')
print('Done.')
"

echo ""
echo "=== Setup complete ==="
