"""
Presented by KeJi
Date: 2026-05-15
GPT-2 Taylor-MLP — Shared Utilities.

K-means clustering, Jacobian computation, Taylor inference, cache serialization.
"""

import json
import os
import time

import numpy as np
import torch
from sklearn.cluster import KMeans

import config


# ─────────────────────────────────────────────────────────────
# K-means
# ─────────────────────────────────────────────────────────────

def Compute_Centers(data, k):
    """
    K-means clustering on data.

    Args:
        data: (N, d) tensor or ndarray
        k: number of clusters

    Returns:
        centers: (k, d) tensor
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy().astype(np.float32)

    kmeans = KMeans(n_clusters=k, random_state=config.SEED, n_init=10)
    kmeans.fit(data)
    return torch.from_numpy(kmeans.cluster_centers_.astype(np.float32))


# ─────────────────────────────────────────────────────────────
# Jacobian
# ─────────────────────────────────────────────────────────────

def Compute_Jacobian(ffn_fn, x0):
    """
    Compute F(X0) and Jacobian J(X0) at a single center.

    Args:
        ffn_fn: callable that maps (d,) → (d,), the FFN as a standalone function
        x0: (d,) tensor, the expansion center

    Returns:
        f_val: (d,) tensor — F(X0)
        jac: (d, d) tensor — J(X0) = dF/dX|_{X0}
    """
    x0 = x0.clone().detach().requires_grad_(True)

    def fn(x):
        """Wrap ffn_fn for jacobian API: expects (d,) → (d,)"""
        return ffn_fn(x.unsqueeze(0)).squeeze(0)

    f_val = fn(x0).detach()
    jac = torch.autograd.functional.jacobian(fn, x0).detach()
    return f_val, jac


def Precompute_Centers(ffn_fn, centers, verbose=False):
    """
    Precompute F(X0) and J(X0) for all K-means centers.

    Args:
        ffn_fn: callable (d,) → (d,)
        centers: (k, d) tensor of cluster centers

    Returns:
        dict with keys "centers", "f_values", "jacobians"
    """
    k = centers.shape[0]
    f_values_list = []
    jacobians_list = []

    for i in range(k):
        t0 = time.time()
        f_val, jac = Compute_Jacobian(ffn_fn, centers[i])
        elapsed = time.time() - t0
        if verbose:
            print(f"  Center {i+1}/{k}: {elapsed:.1f}s")
        f_values_list.append(f_val)
        jacobians_list.append(jac)

    return {
        "centers": centers,
        "f_values": torch.stack(f_values_list, dim=0),
        "jacobians": torch.stack(jacobians_list, dim=0),
    }


# ─────────────────────────────────────────────────────────────
# Taylor Inference
# ─────────────────────────────────────────────────────────────

def Taylor_Predict(x, cache):
    """
    Taylor inference for a single sample.

    Args:
        x: (d,) tensor — input vector
        cache: dict with "centers", "f_values", "jacobians"

    Returns:
        approx: (d,) tensor — F̂(x)
    """
    centers = cache["centers"]        # (k, d)
    f_values = cache["f_values"]      # (k, d)
    jacobians = cache["jacobians"]    # (k, d, d)

    # Nearest center
    dists = torch.norm(centers - x.unsqueeze(0), dim=1)
    best_idx = torch.argmin(dists).item()

    x0 = centers[best_idx]
    f0 = f_values[best_idx]
    J0 = jacobians[best_idx]
    dx = x - x0
    return f0 + J0 @ dx


def Taylor_Predict_Batch(x_batch, cache):
    """
    Taylor inference for a batch.

    Args:
        x_batch: (N, d) tensor
        cache: dict with "centers", "f_values", "jacobians"

    Returns:
        approx: (N, d) tensor
    """
    centers = cache["centers"]        # (k, d)
    f_values = cache["f_values"]      # (k, d)
    jacobians = cache["jacobians"]    # (k, d, d)
    k = centers.shape[0]
    n = x_batch.shape[0]
    d = x_batch.shape[1]

    # Pairwise distances: (N, k)
    x_norm = (x_batch ** 2).sum(dim=1, keepdim=True)
    c_norm = (centers ** 2).sum(dim=1)
    dists = x_norm + c_norm.unsqueeze(0) - 2 * x_batch @ centers.T
    nearest = torch.argmin(dists, dim=1)  # (N,)

    result = torch.zeros_like(x_batch)
    for c_idx in range(k):
        mask = (nearest == c_idx)
        if not mask.any():
            continue
        x_sub = x_batch[mask]
        x0 = centers[c_idx]
        f0 = f_values[c_idx]
        J0 = jacobians[c_idx]
        dx = x_sub - x0.unsqueeze(0)
        approx = f0.unsqueeze(0) + dx @ J0.T
        result[mask] = approx

    return result


# ─────────────────────────────────────────────────────────────
# Cache Serialization
# ─────────────────────────────────────────────────────────────

def Save_Cache(layer_dir, cache, metadata=None):
    """
    Save precomputed cache to disk.

    Args:
        layer_dir: path like "Cache/GPT2/layer_0"
        cache: dict with "centers", "f_values", "jacobians"
        metadata: optional dict with extra info
    """
    os.makedirs(layer_dir, exist_ok=True)

    torch.save(cache["centers"], os.path.join(layer_dir, "centers.pt"))
    torch.save(cache["f_values"], os.path.join(layer_dir, "f_values.pt"))
    torch.save(cache["jacobians"], os.path.join(layer_dir, "jacobians.pt"))

    meta = metadata or {}
    meta.setdefault("k", cache["centers"].shape[0])
    meta.setdefault("d_in", cache["centers"].shape[1])
    meta.setdefault("d_out", cache["f_values"].shape[1])
    meta.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))

    with open(os.path.join(layer_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def Load_Cache(layer_dir, device="cpu"):
    """
    Load precomputed cache from disk.

    Args:
        layer_dir: path like "Cache/GPT2/layer_0"
        device: torch device

    Returns:
        dict with "centers", "f_values", "jacobians"
    """
    cache = {
        "centers": torch.load(os.path.join(layer_dir, "centers.pt"),
                              map_location=device, weights_only=True),
        "f_values": torch.load(os.path.join(layer_dir, "f_values.pt"),
                               map_location=device, weights_only=True),
        "jacobians": torch.load(os.path.join(layer_dir, "jacobians.pt"),
                                map_location=device, weights_only=True),
    }
    return cache
