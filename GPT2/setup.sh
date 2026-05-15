#!/bin/bash
# ============================================================
# Taylor-MLP GPT-2 Experiment - Environment Setup
#
# Downloads dataset and model from ModelScope.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Dataset (ModelScope) ───────────────────────────────
echo "=== [1/2] Downloading WikiText-2 dataset (ModelScope) ==="
export MODELSCOPE_DATASETS_CACHE="/vepfs-mlp2/c20250205/240804016/Datasets"
echo "  Cache: ${MODELSCOPE_DATASETS_CACHE}"
python -c "
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('wikitext', subset_name='wikitext-2-v1', split='train')
print(f'Done. {len(ds)} samples.')
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
