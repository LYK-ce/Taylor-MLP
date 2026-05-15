"""
Presented by KeJi
Date: 2026-05-15
GPT-2 Taylor-MLP — Phase B Step 2: Single-Layer Replacement.

Loads cache from disk, replaces one FFN layer with Taylor approximation,
measures PPL impact and model speedup.
"""

import argparse
import csv
import os

import torch
from datasets import load_dataset

import config
from model import GPT2Wrapper
from utils import Load_Cache
from evaluate import Compute_Ppl_Over_Dataset, Benchmark_Model_Forward


def Test_Single_Replacements(model_name=None, dataset_name=None,
                              dataset_config=None, max_samples=None,
                              seq_len=None, batch_size=None,
                              k_values=None, layer_indices=None, device=None):
    """Phase B Step 2: replace one layer at a time, measure PPL."""

    model_name = model_name or config.MODEL_NAME
    dataset_name = dataset_name or config.DATASET_NAME
    dataset_config = dataset_config or config.DATASET_CONFIG
    max_samples = max_samples or config.MAX_TEST_SAMPLES
    seq_len = seq_len or config.SEQUENCE_LENGTH
    batch_size = batch_size or config.BATCH_SIZE
    k_values = k_values or config.K_VALUES
    device = device or config.DEVICE

    print("=" * 60)
    print("Phase B Step 2: Single-Layer Replacement")
    print("=" * 60)

    # ── 1. Load model ──────────────────────────────────
    print("\n[1/4] Loading model...")
    wrapper = GPT2Wrapper(model_name=model_name, device=device)
    wrapper.Load()

    # ── 2. Tokenize test data ──────────────────────────
    print("[2/4] Preparing test data...")
    dataset = load_dataset(dataset_name, dataset_config, split="train",
                           trust_remote_code=True)
    texts = []
    total_tokens = 0
    for example in dataset:
        text = example["text"]
        if not text or not text.strip():
            continue
        texts.append(text)
        total_tokens += len(wrapper.tokenizer.encode(text))
        if total_tokens >= max_samples + seq_len:
            break

    full_text = wrapper.tokenizer.eos_token.join(texts)
    encodings = wrapper.tokenizer(full_text, return_tensors="pt",
                                  truncation=True, max_length=total_tokens)
    input_ids = encodings["input_ids"][0][:max_samples]

    chunks = []
    stride = seq_len
    for i in range(0, len(input_ids) - seq_len, stride):
        chunk = input_ids[i:i + seq_len]
        if len(chunk) < seq_len:
            break
        chunks.append(chunk)
    test_chunks = torch.stack(chunks)

    # ── 3. Baseline PPL ────────────────────────────────
    print("[3/4] Computing baseline PPL...")
    baseline = Compute_Ppl_Over_Dataset(wrapper.model, test_chunks,
                                        batch_size=batch_size, device=device)
    print(f"  Baseline PPL: {baseline['ppl']:.4f}  "
          f"({baseline['total_time_s']:.2f}s, "
          f"{baseline['time_per_token_ms']:.4f} ms/tok)")

    # Benchmark original model
    sample_batch = test_chunks[:1]
    orig_model_time = Benchmark_Model_Forward(
        wrapper.model, sample_batch, device=device)
    print(f"  Original model forward: {orig_model_time:.4f} ms")

    # ── 4. Test single-layer replacements ──────────────
    print("[4/4] Testing single-layer replacements...")
    os.makedirs(config.RESULT_DIR, exist_ok=True)

    if layer_indices is None:
        # Default: test layers 0, 5, 11 (first, middle, last)
        layer_indices = [0, wrapper.num_layers // 2, wrapper.num_layers - 1]

    all_rows = []

    for layer_idx in layer_indices:
        print(f"\n  Layer {layer_idx}:")
        for k in k_values:
            cache_dir = os.path.join(config.CACHE_DIR,
                                     f"layer_{layer_idx}_k_{k}")
            if not os.path.isdir(cache_dir):
                print(f"    k={k}: cache not found, skipping")
                continue

            cache = Load_Cache(cache_dir, device=device)

            # Replace
            wrapper.Replace_Ffn(layer_idx, cache)

            # Measure PPL
            result = Compute_Ppl_Over_Dataset(
                wrapper.model, test_chunks, batch_size=batch_size, device=device)

            # Measure model speedup
            taylor_model_time = Benchmark_Model_Forward(
                wrapper.model, sample_batch, device=device)
            speedup = orig_model_time / taylor_model_time if taylor_model_time > 0 else 0

            delta_ppl = result["ppl"] - baseline["ppl"]
            delta_pct = (delta_ppl / baseline["ppl"]) * 100

            print(f"    k={k:3d}: PPL={result['ppl']:.4f}  "
                  f"ΔPPL={delta_ppl:+.4f} ({delta_pct:+.2f}%)  "
                  f"Model={taylor_model_time:.4f}ms  "
                  f"Speedup={speedup:.2f}x")

            all_rows.append({
                "layer": layer_idx,
                "k": k,
                "ppl": result["ppl"],
                "delta_ppl": delta_ppl,
                "delta_ppl_pct": delta_pct,
                "baseline_ppl": baseline["ppl"],
                "model_time_orig_ms": orig_model_time,
                "model_time_taylor_ms": taylor_model_time,
                "model_speedup": speedup,
            })

            # Restore
            wrapper.Restore_Ffn(layer_idx)

    # ── Save ────────────────────────────────────────────
    save_path = os.path.join(config.RESULT_DIR, "step2_single_ppl.csv")
    fields = ["layer", "k", "ppl", "delta_ppl", "delta_ppl_pct",
              "baseline_ppl", "model_time_orig_ms",
              "model_time_taylor_ms", "model_speedup"]
    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"\nResults saved to {save_path}")
    print("Done.")
    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase B Step 2: Single-Layer Replacement")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--k", type=str, default=None,
                        help="Comma-separated k values")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices, e.g. '0,5,11'")
    args = parser.parse_args()

    k_values = None
    if args.k:
        k_values = [int(x.strip()) for x in args.k.split(",")]

    layer_indices = None
    if args.layers:
        layer_indices = [int(x.strip()) for x in args.layers.split(",")]

    Test_Single_Replacements(
        model_name=args.model,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        k_values=k_values,
        layer_indices=layer_indices,
    )
