"""
Presented by KeJi
Date: 2026-05-15
GPT-2 Taylor-MLP — Evaluation Utilities.

PPL computation, Cosine Similarity, MSE, and timing benchmarks.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Cosine Similarity & MSE
# ─────────────────────────────────────────────────────────────

def compute_cosine_similarity(y_true, y_pred):
    """
    Batch cosine similarity.

    Args:
        y_true: (N, d) tensor
        y_pred: (N, d) tensor

    Returns:
        avg_cos: float — average cosine similarity over samples
    """
    n = y_true.shape[0]
    cos_sum = 0.0
    for i in range(n):
        cos_sum += F.cosine_similarity(
            y_true[i].unsqueeze(0), y_pred[i].unsqueeze(0)
        ).item()
    return cos_sum / n


def compute_mse(y_true, y_pred):
    """
    Batch MSE.

    Args:
        y_true: (N, d) tensor
        y_pred: (N, d) tensor

    Returns:
        mse: float — mean squared error
    """
    return torch.mean((y_true - y_pred) ** 2).item()


# ─────────────────────────────────────────────────────────────
# PPL Computation
# ─────────────────────────────────────────────────────────────

def compute_ppl(model, input_ids, attention_mask=None, device="cpu",
                label_key="labels"):
    """
    Compute Perplexity over a batch of sequences. Also returns timing.

    Args:
        model: GPT2LMHeadModel (or wrapped with Taylor FFNs)
        input_ids: (B, seq_len) tensor
        attention_mask: optional mask
        device: torch device
        label_key: key for labels in model output (usually "labels" or "loss" pre-computed)

    Returns:
        ppl: float
        total_time_s: float — wall-clock inference time
        total_loss: float — total cross-entropy loss
        total_tokens: int — number of tokens evaluated
    """
    batch = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        t0 = time.perf_counter()
        outputs = model(input_ids=batch, attention_mask=attention_mask)
        logits = outputs.logits  # (B, seq_len, vocab_size)
        t1 = time.perf_counter()

    labels = batch.clone()
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = criterion(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    num_tokens = shift_labels.numel()
    avg_loss = loss.item() / num_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    total_time = t1 - t0

    return ppl, total_time, loss.item(), num_tokens


def compute_ppl_over_dataset(model, input_ids_chunks, batch_size=8, device="cpu"):
    """
    Compute PPL over pre-chunked input sequences.

    Args:
        model: GPT2LMHeadModel
        input_ids_chunks: (num_chunks, seq_len) tensor
        batch_size: inference batch size
        device: torch device

    Returns:
        dict with keys: ppl, total_time_s, time_per_token_ms, num_tokens
    """
    total_loss = 0.0
    total_time = 0.0
    total_tokens = 0

    for i in range(0, len(input_ids_chunks), batch_size):
        batch = input_ids_chunks[i:i + batch_size]
        ppl, elapsed, loss, n_tok = compute_ppl(model, batch, device=device)
        total_loss += loss
        total_time += elapsed
        total_tokens += n_tok

    avg_loss = total_loss / total_tokens
    overall_ppl = torch.exp(torch.tensor(avg_loss)).item()
    time_per_token_ms = (total_time / total_tokens) * 1000

    return {
        "ppl": overall_ppl,
        "total_time_s": total_time,
        "time_per_token_ms": time_per_token_ms,
        "num_tokens": total_tokens,
    }


# ─────────────────────────────────────────────────────────────
# Timing Benchmarks
# ─────────────────────────────────────────────────────────────

def benchmark_ffn(ffn_fn, x_batch, n_warmup=3, n_trials=10, device="cpu"):
    """
    Benchmark a single FFN forward pass (wall-clock time).

    Args:
        ffn_fn: callable (B, d) → (B, d)
        x_batch: (B, d) sample input
        n_warmup: warmup iterations
        n_trials: measurement iterations

    Returns:
        avg_time_ms: float — average time per forward pass in milliseconds
    """
    x_batch = x_batch.to(device)

    # Warmup
    for _ in range(n_warmup):
        _ = ffn_fn(x_batch)

    # Measure
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_trials):
        _ = ffn_fn(x_batch)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_time_ms = ((t1 - t0) / n_trials) * 1000
    return avg_time_ms


def benchmark_model_forward(model, input_ids, n_warmup=3, n_trials=10, device="cpu"):
    """
    Benchmark full model forward pass.

    Args:
        model: GPT2LMHeadModel
        input_ids: (B, seq_len) tensor
        n_warmup, n_trials: as above

    Returns:
        avg_time_ms: float
    """
    input_ids = input_ids.to(device)

    # Warmup
    for _ in range(n_warmup):
        _ = model(input_ids=input_ids)

    # Measure
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_trials):
        _ = model(input_ids=input_ids)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_time_ms = ((t1 - t0) / n_trials) * 1000
    return avg_time_ms
