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
# Downloaded from ModelScope by setup.sh.
MODELSCOPE_MODEL_ID = "openai-community/gpt2"
MODEL_DIR = os.path.join(ROOT_DIR, "Model")
MODEL_PATH = os.path.join(MODEL_DIR, "openai-community", "gpt2")

# ── Dataset ────────────────────────────────────────────────
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
MODELSCOPE_DATASET_ID = "AI-ModelScope/wikitext"  # for download

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

# ── HF Cache ───────────────────────────────────────────────
HF_DATASETS_CACHE = "/vepfs-mlp2/c20250205/240804016/Datasets"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_DATASETS_CACHE", HF_DATASETS_CACHE)

# ── Misc ───────────────────────────────────────────────────
SEED = 42
DTYPE = torch.float32
