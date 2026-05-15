"""
Presented by KeJi
Date: 2026-05-15
GPT-2 Taylor-MLP — Phase B Step 1: Offline Layer Analysis.

Loads precomputed cache from disk, runs Taylor inference on test data,
measures CosSim/MSE per layer per k, and benchmarks FFN vs Taylor timing.
Does NOT modify the model.
"""

import argparse
import csv
import os
import time

import torch
from datasets import load_dataset

import config
from model import GPT2Wrapper
from utils import Load_Cache, Taylor_Predict_Batch
from evaluate import Compute_Cosine_Similarity, Compute_Mse, Benchmark_Ffn


def Analyze_Layers(model_name=None, dataset_name=None, dataset_config=None,
                   max_samples=None, seq_len=None, batch_size=None,
                   k_values=None, device=None):
    """Phase B Step 1: per-layer Taylor accuracy analysis."""

    model_name = model_name or config.MODEL_NAME
    dataset_name = dataset_name or config.DATASET_NAME
    dataset_config = dataset_config or config.DATASET_CONFIG
    max_samples = max_samples or config.MAX_TEST_SAMPLES
    seq_len = seq_len or config.SEQUENCE_LENGTH
    batch_size = batch_size or config.BATCH_SIZE
    k_values = k_values or config.K_VALUES
    device = device or config.DEVICE

    print("=" * 60)
    print("Phase B Step 1: Offline Layer Analysis")
    print("=" * 60)

    # ── 1. Load model + collect test FFN I/O ────────────
    print("\n[1/3] Collecting test FFN I/O...")
    wrapper = GPT2Wrapper(model_name=model_name, device=device)
    wrapper.Load()

    dataset = load_dataset(dataset_name, dataset_config, split="train",
                           trust_remote_code=True)
    # Use last portion of training set as "test" for Taylor evaluation
    texts = []
    total_tokens = 0
    for example in dataset:
        text = example["text"]
        if not text or not text.strip():
            continue
        texts.append(text)
        total_tokens += len(wrapper.tokenizer.encode(text))
        if total_tokens >= max_samples * 2 + seq_len:
            break

    full_text = wrapper.tokenizer.eos_token.join(texts)
    encodings = wrapper.tokenizer(full_text, return_tensors="pt",
                                  truncation=True, max_length=total_tokens)
    input_ids = encodings["input_ids"][0]

    # Use second half for test
    split = len(input_ids) // 2
    test_ids = input_ids[split:split + max_samples]

    chunks = []
    stride = seq_len
    for i in range(0, len(test_ids) - seq_len, stride):
        chunk = test_ids[i:i + seq_len]
        if len(chunk) < seq_len:
            break
        chunks.append(chunk)
    test_chunks = torch.stack(chunks)

    # Run forward with hooks
    wrapper.Register_Ffn_Hooks()
    wrapper.Clear_Ffn_Io()
    for i in range(0, len(test_chunks), batch_size):
        batch = test_chunks[i:i + batch_size].to(device)
        wrapper.Forward(batch)
    wrapper.Remove_Ffn_Hooks()

    ffn_io = wrapper.Get_Ffn_Io()
    print(f"  Test tokens: {test_chunks.numel()}")

    # ── 2. Benchmark original FFN ───────────────────────
    print("\n[2/3] Benchmarking original FFN...")
    sample_input = ffn_io[0]["input"][:batch_size].to(device)
    mlp_orig = wrapper.model.transformer.h[0].mlp
    orig_time = Benchmark_Ffn(mlp_orig, sample_input, device=device)
    print(f"  Original FFN (layer 0): {orig_time:.4f} ms")

    # ── 3. Analyze each layer ───────────────────────────
    print("\n[3/3] Analyzing layers...")
    os.makedirs(config.RESULT_DIR, exist_ok=True)

    all_rows = []

    for layer_idx in range(wrapper.num_layers):
        test_input = ffn_io[layer_idx]["input"]   # (N, d_model)
        test_output = ffn_io[layer_idx]["output"]
        print(f"\n  Layer {layer_idx}: {test_input.shape[0]} samples")

        for k in k_values:
            cache_dir = os.path.join(config.CACHE_DIR,
                                     f"layer_{layer_idx}_k_{k}")
            if not os.path.isdir(cache_dir):
                print(f"    k={k}: cache not found, skipping")
                continue

            t0 = time.time()
            cache = Load_Cache(cache_dir, device=device)

            # Taylor inference
            approx = Taylor_Predict_Batch(test_input.to(device), cache)

            # Metrics
            cos_sim = Compute_Cosine_Similarity(test_output, approx.cpu())
            mse = Compute_Mse(test_output, approx.cpu())

            # Benchmark Taylor FFN
            # Create a Taylor module for timing
            from model import _TaylorFFN
            taylor_module = _TaylorFFN(cache, device)
            taylor_time = Benchmark_Ffn(taylor_module, sample_input.to(device),
                                        device=device)

            speedup = orig_time / taylor_time if taylor_time > 0 else 0

            elapsed = time.time() - t0
            print(f"    k={k:3d}: CosSim={cos_sim:.6f}  MSE={mse:.6f}  "
                  f"Taylor={taylor_time:.4f}ms  Speedup={speedup:.2f}x  "
                  f"({elapsed:.1f}s)")

            all_rows.append({
                "layer": layer_idx,
                "k": k,
                "cosine_sim": cos_sim,
                "mse": mse,
                "ffn_orig_time_ms": orig_time,
                "ffn_taylor_time_ms": taylor_time,
                "speedup": speedup,
            })

    # ── Save ────────────────────────────────────────────
    cosim_path = os.path.join(config.RESULT_DIR, "step1_layer_cosim.csv")
    mse_path = os.path.join(config.RESULT_DIR, "step1_layer_mse.csv")
    summary_path = os.path.join(config.RESULT_DIR, "step1_summary.csv")

    with open(cosim_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "k", "cosine_sim", "speedup"])
        w.writeheader()
        for r in all_rows:
            w.writerow({"layer": r["layer"], "k": r["k"],
                        "cosine_sim": r["cosine_sim"], "speedup": r["speedup"]})

    with open(mse_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "k", "mse"])
        w.writeheader()
        for r in all_rows:
            w.writerow({"layer": r["layer"], "k": r["k"], "mse": r["mse"]})

    with open(summary_path, "w", newline="") as f:
        fields = ["layer", "k", "cosine_sim", "mse",
                  "ffn_orig_time_ms", "ffn_taylor_time_ms", "speedup"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"\nResults saved to {config.RESULT_DIR}/")
    print("Done.")


def Identify_Representative_Layers(cosim_csv_path=None):
    """Identify most-linear, median, and least-linear layers from Step 1 results."""
    if cosim_csv_path is None:
        cosim_csv_path = os.path.join(config.RESULT_DIR, "step1_layer_cosim.csv")

    import pandas as pd
    df = pd.read_csv(cosim_csv_path)

    # Use k=64 as reference
    ref_k = df[df["k"] == 64]
    if ref_k.empty:
        ref_k = df[df["k"] == df["k"].max()]

    sorted_layers = ref_k.sort_values("cosine_sim")
    return {
        "least_linear": int(sorted_layers.iloc[0]["layer"]),
        "median": int(sorted_layers.iloc[len(sorted_layers) // 2]["layer"]),
        "most_linear": int(sorted_layers.iloc[-1]["layer"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase B Step 1: Offline Layer Analysis")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--k", type=str, default=None,
                        help="Comma-separated k values")
    args = parser.parse_args()

    k_values = None
    if args.k:
        k_values = [int(x.strip()) for x in args.k.split(",")]

    Analyze_Layers(
        model_name=args.model,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        k_values=k_values,
    )
