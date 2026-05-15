"""
Presented by KeJi
Date: 2026-05-15
GPT-2 Taylor-MLP — Phase B Step 3: Cumulative Replacement.

Loads cache from disk, replaces FFN layers backward (last → first),
measures PPL and model speedup as layers accumulate.
"""

import argparse
import csv
import os

import torch
from datasets import load_dataset

import config
from model import GPT2Wrapper
from utils import load_cache
from evaluate import compute_ppl_over_dataset, benchmark_model_forward


def test_cumulative(model_name=None, dataset_name=None, dataset_config=None,
                    max_samples=None, seq_len=None, batch_size=None,
                    k_values=None, device=None):
    """Phase B Step 3: cumulative replacement from last layer backward."""

    model_name = model_name or config.MODEL_NAME
    dataset_name = dataset_name or config.DATASET_NAME
    dataset_config = dataset_config or config.DATASET_CONFIG
    max_samples = max_samples or config.MAX_TEST_SAMPLES
    seq_len = seq_len or config.SEQUENCE_LENGTH
    batch_size = batch_size or config.BATCH_SIZE
    k_values = k_values or config.K_VALUES
    device = device or config.DEVICE

    print("=" * 60)
    print("Phase B Step 3: Cumulative Replacement")
    print("=" * 60)

    # ── 1. Load model ──────────────────────────────────
    print("\n[1/4] Loading model...")
    wrapper = GPT2Wrapper(model_name=model_name, device=device)
    wrapper.load()
    n_layers = wrapper.num_layers

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
    baseline = compute_ppl_over_dataset(wrapper.model, test_chunks,
                                        batch_size=batch_size, device=device)
    print(f"  Baseline PPL: {baseline['ppl']:.4f}")

    sample_batch = test_chunks[:1]
    orig_model_time = benchmark_model_forward(
        wrapper.model, sample_batch, device=device)
    print(f"  Original model forward: {orig_model_time:.4f} ms")

    # Compute single-layer FFN params for storage ratio
    d_model = wrapper.d_model
    d_ff = wrapper.model.config.n_embd * 4  # 3072 for GPT-2 Small
    single_ffn_params = 2 * d_model * d_ff   # Linear 768→3072 + 3072→768
    taylor_per_center = d_model * d_model + 2 * d_model  # J + F(X0) + X0

    # ── 4. Cumulative test ─────────────────────────────
    print("[4/4] Running cumulative replacement...")
    os.makedirs(config.RESULT_DIR, exist_ok=True)

    all_rows = []

    for k in k_values:
        print(f"\n  k={k}:")
        # Restore all before starting new k sweep
        wrapper.restore_all_ffns()

        for num_replaced in range(1, n_layers + 1):
            # Layer to replace: from last backward
            layer_idx = n_layers - num_replaced  # 11, 10, 9, ...

            cache_dir = os.path.join(config.CACHE_DIR,
                                     f"layer_{layer_idx}_k_{k}")
            if not os.path.isdir(cache_dir):
                print(f"    Layer {layer_idx}: cache not found, stopping")
                break

            cache = load_cache(cache_dir, device=device)
            wrapper.replace_ffn(layer_idx, cache)

            # Measure PPL
            result = compute_ppl_over_dataset(
                wrapper.model, test_chunks, batch_size=batch_size, device=device)

            # Measure model speedup
            taylor_model_time = benchmark_model_forward(
                wrapper.model, sample_batch, device=device)
            speedup = orig_model_time / taylor_model_time if taylor_model_time > 0 else 0

            delta_ppl = result["ppl"] - baseline["ppl"]
            delta_pct = (delta_ppl / baseline["ppl"]) * 100

            taylor_params = num_replaced * k * taylor_per_center
            storage_ratio = taylor_params / (single_ffn_params * n_layers)

            print(f"    {num_replaced:2d} layers: PPL={result['ppl']:.4f}  "
                  f"ΔPPL={delta_ppl:+.4f} ({delta_pct:+.2f}%)  "
                  f"Speedup={speedup:.2f}x  Storage={storage_ratio:.3f}")

            all_rows.append({
                "k": k,
                "num_replaced": num_replaced,
                "ppl": result["ppl"],
                "delta_ppl": delta_ppl,
                "delta_ppl_pct": delta_pct,
                "baseline_ppl": baseline["ppl"],
                "storage_ratio": storage_ratio,
                "model_time_orig_ms": orig_model_time,
                "model_time_taylor_ms": taylor_model_time,
                "model_speedup": speedup,
            })

    # ── Save ────────────────────────────────────────────
    save_path = os.path.join(config.RESULT_DIR, "step3_cumulative_ppl.csv")
    fields = ["k", "num_replaced", "ppl", "delta_ppl", "delta_ppl_pct",
              "baseline_ppl", "storage_ratio",
              "model_time_orig_ms", "model_time_taylor_ms", "model_speedup"]
    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # Restore model
    wrapper.restore_all_ffns()

    print(f"\nResults saved to {save_path}")
    print("Done.")
    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase B Step 3: Cumulative Replacement")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--k", type=str, default=None,
                        help="Comma-separated k values")
    args = parser.parse_args()

    k_values = None
    if args.k:
        k_values = [int(x.strip()) for x in args.k.split(",")]

    test_cumulative(
        model_name=args.model,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        k_values=k_values,
    )
