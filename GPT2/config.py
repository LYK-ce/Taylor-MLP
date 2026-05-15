"""
Presented by KeJi
Date: 2026-05-15
GPT-2 Taylor-MLP Experiment — Global Configuration.

All Python scripts import parameters from here. Change values here to
switch models, datasets, k ranges, etc. — no need to touch individual scripts.
"""

import torch
import os

# ── Model ──────────────────────────────────────────────────
MODEL_NAME = "openai-community/gpt2"  # GPT-2 Small (12 layers, d_model=768)
# Alternatives: "openai-community/gpt2-medium" (24 layers, d_model=1024)
#              "openai-community/gpt2-large"  (36 layers, d_model=1280)

# ── Dataset ────────────────────────────────────────────────
DATASET_NAME = "openwebtext"
DATASET_CONFIG = None  # OpenWebText has no sub-configs

# ── Sampling ───────────────────────────────────────────────
MAX_TRAIN_SAMPLES = 50000   # tokens for K-means fitting
MAX_TEST_SAMPLES = 2000     # tokens for PPL evaluation
SEQUENCE_LENGTH = 128       # context window for evaluation
BATCH_SIZE = 8              # batch size for feature extraction / PPL eval

# ── K-means k values ───────────────────────────────────────
# FFN input: 768-dim (d_model). Storage balance point k ≈ 96 (all 12 layers).
K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# ── Device ─────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Paths ──────────────────────────────────────────────────
# Workspace root relative to this file
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT_DIR, "Cache", "GPT2")
RESULT_DIR = os.path.join(ROOT_DIR, "Result", "GPT2")

# ── HuggingFace Cache ──────────────────────────────────────
# Custom cache paths for datasets and models.
# Set to None to use HF defaults (~/.cache/huggingface/).
HF_HOME = os.path.join(ROOT_DIR, ".hf_cache")
HF_HUB_CACHE = os.path.join(HF_HOME, "hub")
HF_DATASETS_CACHE = os.path.join(HF_HOME, "datasets")

# Apply HF cache paths before any HF imports
for _key, _val in [("HF_HOME", HF_HOME),
                    ("HUGGINGFACE_HUB_CACHE", HF_HUB_CACHE),
                    ("HF_DATASETS_CACHE", HF_DATASETS_CACHE)]:
    if _key not in os.environ:
        os.environ[_key] = _val

# ── Misc ───────────────────────────────────────────────────
SEED = 42
DTYPE = torch.float32
