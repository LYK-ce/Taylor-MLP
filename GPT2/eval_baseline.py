"""
Presented by KeJi
Date: 2026-05-15
GPT-2 Taylor-MLP — Baseline Evaluation.

Verifies GPT-2 Perplexity on OpenWebText before any Taylor substitution.
Outputs baseline PPL and inference timing.
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import config
from evaluate import Tokenize_And_Chunk


def Evaluate_Baseline(model_name=None, dataset_name=None, dataset_config=None,
                      max_samples=None, seq_len=None, batch_size=None,
                      device=None):
    """Run baseline evaluation and return metrics dict."""

    model_name = model_name or config.MODEL_NAME
    dataset_name = dataset_name or config.DATASET_NAME
    dataset_config = dataset_config or config.DATASET_CONFIG
    max_samples = max_samples or config.MAX_TEST_SAMPLES
    seq_len = seq_len or config.SEQUENCE_LENGTH
    batch_size = batch_size or config.BATCH_SIZE
    device = device or config.DEVICE

    print(f"[Eval] Model: {model_name}")
    print(f"[Eval] Dataset: {dataset_name}")
    print(f"[Eval] Device: {device}")

    # ── Load model ──
    print("[Eval] Loading model...")
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Eval] Model params: {n_params:,}")

    # ── Tokenize & chunk (sliding window) ──
    print("[Eval] Tokenizing...")
    input_ids = Tokenize_And_Chunk(
        tokenizer, dataset_name, dataset_config,
        max_samples=max_samples, seq_len=seq_len, stride=1,
    )

    num_tokens = input_ids.numel()
    print(f"[Eval] Evaluation tokens: {num_tokens} ({len(input_ids)} sequences of {seq_len})")

    # ── Run evaluation ──
    print("[Eval] Running inference...")
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_time = 0.0

    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            batch = input_ids[i:i + batch_size].to(device)
            labels = batch.clone()

            t0 = time.perf_counter()
            outputs = model(batch)
            logits = outputs.logits  # (B, seq_len, vocab_size)
            t1 = time.perf_counter()
            total_time += (t1 - t0)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            total_loss += loss.item()

    avg_loss = total_loss / num_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    avg_time_per_token_ms = (total_time / num_tokens) * 1000

    # ── Report ──
    print(f"\n{'='*50}")
    print(f"  Baseline PPL:        {ppl:.4f}")
    print(f"  Avg loss/token:      {avg_loss:.6f}")
    print(f"  Total inference:     {total_time:.2f} s")
    print(f"  Time per token:      {avg_time_per_token_ms:.4f} ms")
    print(f"{'='*50}")

    # ── Save ──
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    baseline_path = os.path.join(config.RESULT_DIR, "baseline.csv")
    with open(baseline_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "dataset", "ppl", "avg_loss", "num_tokens",
            "total_time_s", "time_per_token_ms", "n_params"
        ])
        w.writeheader()
        w.writerow({
            "model": model_name,
            "dataset": dataset_name,
            "ppl": ppl,
            "avg_loss": avg_loss,
            "num_tokens": num_tokens,
            "total_time_s": total_time,
            "time_per_token_ms": avg_time_per_token_ms,
            "n_params": n_params,
        })

    print(f"\n[Eval] Results saved to {baseline_path}")
    return {
        "ppl": ppl,
        "avg_loss": avg_loss,
        "num_tokens": num_tokens,
        "total_time_s": total_time,
        "time_per_token_ms": avg_time_per_token_ms,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 Baseline Evaluation")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default from config)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (default from config)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max evaluation tokens")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Sequence length")
    args = parser.parse_args()

    Evaluate_Baseline(
        model_name=args.model,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        seq_len=args.seq_len,
    )
