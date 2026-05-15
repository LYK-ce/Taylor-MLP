"""
Presented by KeJi
Date: 2026-05-15
GPT-2 Taylor-MLP — Phase A: Data Collection.

Runs GPT-2 on OpenWebText, extracts FFN I/O for all layers,
performs K-means clustering, precomputes Jacobians, and
serializes everything to Cache/GPT2/.
"""

import argparse
import os
import time

import torch

import config
from model import GPT2Wrapper
from utils import Compute_Centers, Precompute_Centers, Save_Cache
from evaluate import Tokenize_And_Chunk


def Collect_Data(model_name=None, dataset_name=None, dataset_config=None,
                 max_samples=None, seq_len=None, batch_size=None, k_values=None,
                 device=None):
    """Phase A: collect FFN I/O, compute K-means centers and Jacobians, save to disk."""

    model_name = model_name or config.MODEL_NAME
    dataset_name = dataset_name or config.DATASET_NAME
    dataset_config = dataset_config or config.DATASET_CONFIG
    max_samples = max_samples or config.MAX_TRAIN_SAMPLES
    seq_len = seq_len or config.SEQUENCE_LENGTH
    batch_size = batch_size or config.BATCH_SIZE
    k_values = k_values or config.K_VALUES
    device = device or config.DEVICE

    print("=" * 60)
    print("Phase A: Data Collection")
    print("=" * 60)
    print(f"  Model:    {model_name}")
    print(f"  Dataset:  {dataset_name}")
    print(f"  k values: {k_values}")
    print(f"  Samples:  {max_samples} tokens")
    print(f"  Device:   {device}")

    # ── 1. Load model ──────────────────────────────────
    print("\n[1/4] Loading model...")
    wrapper = GPT2Wrapper(model_name=model_name, device=device)
    wrapper.Load()

    # ── 2. Load dataset and tokenize ───────────────────
    print("\n[2/4] Tokenizing...")
    input_ids = Tokenize_And_Chunk(
        wrapper.tokenizer, dataset_name, dataset_config,
        max_samples=max_samples, seq_len=seq_len, stride=seq_len,
    )
    n_tokens = input_ids.numel()
    print(f"  Tokenized: {n_tokens} tokens ({len(input_ids)} seqs × {seq_len})")

    # ── 3. Run forward pass w/ hooks ───────────────────
    print("\n[3/4] Running forward pass to collect FFN I/O...")
    wrapper.Register_Ffn_Hooks()
    wrapper.Clear_Ffn_Io()

    t0 = time.time()
    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i:i + batch_size].to(device)
        wrapper.Forward(batch)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    wrapper.Remove_Ffn_Hooks()

    ffn_io = wrapper.Get_Ffn_Io()
    print(f"  Collected I/O for {len(ffn_io)} layers")

    # ── 4. K-means + Jacobian per layer per k ──────────
    print("\n[4/4] Computing K-means centers and Jacobians...")
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    total_centers = sum(k_values) * wrapper.num_layers
    center_count = 0

    for layer_idx in range(wrapper.num_layers):
        layer_inputs = ffn_io[layer_idx]["input"]    # (N, d_model)
        print(f"\n  Layer {layer_idx}: {layer_inputs.shape[0]} tokens")

        # Make a standalone FFN function for Jacobian
        mlp = wrapper.model.transformer.h[layer_idx].mlp

        def make_ffn_fn(m):
            def fn(x):
                return m(x.unsqueeze(0)).squeeze(0)
            return fn

        ffn_fn = make_ffn_fn(mlp)

        for k in k_values:
            t0 = time.time()
            print(f"    k={k} ...", end=" ", flush=True)

            # K-means
            centers = Compute_Centers(layer_inputs, k)

            # Precompute Jacobians
            cache = Precompute_Centers(ffn_fn, centers, verbose=False)

            # Save
            layer_dir = os.path.join(config.CACHE_DIR, f"layer_{layer_idx}_k_{k}")
            Save_Cache(layer_dir, cache, metadata={
                "layer": layer_idx,
                "k": k,
                "model": model_name,
                "n_inputs": layer_inputs.shape[0],
            })

            center_count += k
            elapsed = time.time() - t0
            print(f"{elapsed:.1f}s")

    print(f"\n{'='*60}")
    print(f"Phase A complete. Cache saved to {config.CACHE_DIR}/")
    print(f"Total centers computed: {total_centers}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase A: Data Collection")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--k", type=str, default=None,
                        help="Comma-separated k values, e.g. '1,4,16,64'")
    args = parser.parse_args()

    k_values = None
    if args.k:
        k_values = [int(x.strip()) for x in args.k.split(",")]

    Collect_Data(
        model_name=args.model,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        k_values=k_values,
    )
