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
# ModelScope mirror for downloading; local path for loading.
MODELSCOPE_MODEL_ID = "AI-ModelScope/gpt2"           # ModelScope 上的模型标识
MODEL_DIR = os.path.join(ROOT_DIR, "Model")           # 本地模型目录
MODEL_PATH = os.path.join(MODEL_DIR, "AI-ModelScope", "gpt2")  # 下载后的本地路径
# Alternatives when switching:
#   Medium: MODELSCOPE_MODEL_ID = "AI-ModelScope/gpt2-medium"
#   Large:  MODELSCOPE_MODEL_ID = "AI-ModelScope/gpt2-large"
#   (update MODEL_PATH accordingly)

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
# Dataset: download via HF mirror, cache to NVMe storage.
# Model: downloaded by setup.sh via ModelScope SDK to MODEL_PATH.
HF_DATASETS_CACHE = "/vepfs-mlp2/c20250205/240804016/Datasets"

# Use HF mirror for datasets (faster in China)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_DATASETS_CACHE", HF_DATASETS_CACHE)

# ── Misc ───────────────────────────────────────────────────
SEED = 42
DTYPE = torch.float32
