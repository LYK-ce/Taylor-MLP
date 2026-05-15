"""
Presented by KeJi
Date: 2026-05-15
Taylor-MLP Pre-Experiment: K-means + 1st-order Taylor approximation accuracy validation.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
import csv
import os
import time


# ============================================================
# Step 1: Build MLP
# ============================================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Step 2: Generate synthetic data
# ============================================================
def generate_data(model, n_train=10000, n_test=2000, seed=42):
    torch.manual_seed(seed)
    x_train = torch.randn(n_train, 64)
    x_test = torch.randn(n_test, 64)
    with torch.no_grad():
        y_train = model(x_train)
        y_test = model(x_test)
    return x_train, y_train, x_test, y_test


# ============================================================
# Step 3: K-means clustering
# ============================================================
def compute_centers(x_train_np, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(x_train_np)
    return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)


# ============================================================
# Step 4: Precompute F(X0) and Jacobian J(X0) at each center
# ============================================================
def compute_jacobian(model, x0):
    """
    Compute F(X0) and Jacobian J = dF/dX at X0.
    J shape: (d_out, d_in)
    """
    x0 = x0.clone().detach().requires_grad_(True)
    f_x0 = model(x0.unsqueeze(0)).squeeze(0)  # (d_out,)
    J = []
    for i in range(f_x0.shape[0]):
        grad = torch.autograd.grad(
            f_x0[i], x0,
            retain_graph=True,
            create_graph=False
        )[0]
        J.append(grad)
    J = torch.stack(J, dim=0)  # (d_out, d_in)
    return f_x0.detach(), J.detach()


def precompute_expansion_points(model, centers):
    """
    Precompute F and J for all k centers.
    Returns: dict mapping center_idx -> (F_X0, J_X0, X0)
    """
    cache = {}
    for i in range(centers.shape[0]):
        x0 = centers[i]
        f_val, jac = compute_jacobian(model, x0)
        cache[i] = (f_val, jac, x0)
    return cache


# ============================================================
# Step 5: Taylor inference and evaluation
# ============================================================
def taylor_predict(x, cache):
    """
    For a single sample x, find nearest center and apply:
    F_hat(x) = F(X0) + J(X0) @ (x - X0)
    """
    # Find nearest center
    best_idx = None
    best_dist = float('inf')
    for idx, (_, _, x0) in cache.items():
        dist = torch.norm(x - x0).item()
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

    f_val, jac, x0 = cache[best_idx]
    dx = x - x0
    approx = f_val + jac @ dx  # (d_out,)
    return approx


def evaluate(model, x_test, y_test, cache, k):
    """
    Evaluate Taylor approximation on test set.
    Returns: avg_mse, avg_cosine_sim
    """
    mse_sum = 0.0
    cos_sum = 0.0
    n = x_test.shape[0]

    with torch.no_grad():
        for i in range(n):
            x_i = x_test[i]
            y_true = y_test[i]
            y_approx = taylor_predict(x_i, cache)
            mse_sum += torch.mean((y_true - y_approx) ** 2).item()
            cos_sum += torch.nn.functional.cosine_similarity(
                y_true.unsqueeze(0), y_approx.unsqueeze(0)
            ).item()

    avg_mse = mse_sum / n
    avg_cos = cos_sum / n
    return avg_mse, avg_cos


# ============================================================
# Main
# ============================================================
def main():
    result_dir = "Result/Pre_Experiment"
    os.makedirs(result_dir, exist_ok=True)

    print("=" * 60)
    print("Taylor-MLP Pre-Experiment")
    print("=" * 60)

    # Build model
    model = MLP()
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] 64→256→512→256→64 (ReLU)")
    print(f"  Total params: {n_params:,}")

    # Generate data
    print("\n[Data] Generating 10K train + 2K test samples (64-dim, N(0,1))...")
    x_train, y_train, x_test, y_test = generate_data(model)
    x_train_np = x_train.numpy()
    print(f"  Train: {x_train.shape}, Test: {x_test.shape}")

    # Storage comparison
    d_in, d_out = 64, 64
    k_values = [1, 2, 4, 8, 16, 32, 64, 128]
    results = []

    print("\n" + "=" * 60)
    print("Running experiments for k =", k_values)
    print("=" * 60)

    for k in k_values:
        t0 = time.time()

        # K-means
        centers = compute_centers(x_train_np, k)

        # Precompute
        cache = precompute_expansion_points(model, centers)

        # Evaluate
        avg_mse, avg_cos = evaluate(model, x_test, y_test, cache, k)

        # Storage: Taylor stores k * (J(d_out*d_in) + F(d_out) + X0(d_in))
        taylor_params = k * (d_out * d_in + d_out + d_in)
        storage_ratio = taylor_params / n_params

        elapsed = time.time() - t0
        print(f"  k={k:3d}: MSE={avg_mse:.6f}  CosSim={avg_cos:.6f}  "
              f"StorageRatio={storage_ratio:.4f}  Time={elapsed:.1f}s")

        results.append({
            "k": k,
            "mse": avg_mse,
            "cosine_sim": avg_cos,
            "taylor_params": taylor_params,
            "orig_params": n_params,
            "storage_ratio": storage_ratio,
            "eval_time_s": elapsed,
        })

    # Save CSVs
    mse_path = os.path.join(result_dir, "k_vs_mse.csv")
    cos_path = os.path.join(result_dir, "k_vs_cosine.csv")
    summary_path = os.path.join(result_dir, "summary.csv")

    with open(mse_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "mse"])
        w.writeheader()
        for r in results:
            w.writerow({"k": r["k"], "mse": r["mse"]})

    with open(cos_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "cosine_sim"])
        w.writeheader()
        for r in results:
            w.writerow({"k": r["k"], "cosine_sim": r["cosine_sim"]})

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "k", "mse", "cosine_sim", "taylor_params",
            "orig_params", "storage_ratio"
        ])
        w.writeheader()
        for r in results:
            w.writerow({k: v for k, v in r.items()
                        if k in ["k", "mse", "cosine_sim",
                                 "taylor_params", "orig_params",
                                 "storage_ratio"]})

    print(f"\n[Results] Saved to {result_dir}/:")
    print(f"  {mse_path}")
    print(f"  {cos_path}")
    print(f"  {summary_path}")

    # Quick analysis
    print("\n[Analysis]")
    best_cos = max(results, key=lambda r: r["cosine_sim"])
    print(f"  Best cosine similarity: {best_cos['cosine_sim']:.6f} at k={best_cos['k']}")

    under_1 = [r for r in results if r["storage_ratio"] <= 1.0]
    if under_1:
        best_under = max(under_1, key=lambda r: r["cosine_sim"])
        print(f"  Best cos with storage <= orig: {best_under['cosine_sim']:.6f} "
              f"at k={best_under['k']} (ratio={best_under['storage_ratio']:.4f})")
    else:
        print("  No k has storage <= original model")

    print("\nDone.")


if __name__ == "__main__":
    main()
