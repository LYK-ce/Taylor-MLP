"""
Presented by KeJi
Date: 2026-05-15
Taylor-MLP Pre-Experiment 2: Smooth activation functions (GELU/SiLU/Swish) comparison.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
import csv
import os
import time


# ============================================================
# Step 1: Build MLP with configurable activation
# ============================================================
ACTIVATION_MAP = {
    "ReLU": nn.ReLU,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,  # also known as Swish
}


class MLP(nn.Module):
    def __init__(self, activation_name="ReLU"):
        super().__init__()
        act_fn = ACTIVATION_MAP[activation_name]
        self.activation_name = activation_name
        self.net = nn.Sequential(
            nn.Linear(64, 256),
            act_fn(),
            nn.Linear(256, 512),
            act_fn(),
            nn.Linear(512, 256),
            act_fn(),
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
# Step 4: Precompute F(X0) and Jacobian J(X0)
# ============================================================
def compute_jacobian(model, x0):
    x0 = x0.clone().detach().requires_grad_(True)
    f_x0 = model(x0.unsqueeze(0)).squeeze(0)
    J = []
    for i in range(f_x0.shape[0]):
        grad = torch.autograd.grad(
            f_x0[i], x0, retain_graph=True, create_graph=False
        )[0]
        J.append(grad)
    J = torch.stack(J, dim=0)
    return f_x0.detach(), J.detach()


def precompute_expansion_points(model, centers):
    cache = {}
    for i in range(centers.shape[0]):
        x0 = centers[i]
        f_val, jac = compute_jacobian(model, x0)
        cache[i] = (f_val, jac, x0)
    return cache


# ============================================================
# Step 5: Taylor inference and evaluation (per sample)
# ============================================================
def taylor_predict(x, cache):
    best_idx = None
    best_dist = float("inf")
    for idx, (_, _, x0) in cache.items():
        dist = torch.norm(x - x0).item()
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    f_val, jac, x0 = cache[best_idx]
    dx = x - x0
    return f_val + jac @ dx


def evaluate(x_test, y_test, cache):
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
    return mse_sum / n, cos_sum / n


# ============================================================
# Main
# ============================================================
def main():
    result_dir = "Result/Pre_Experiment"
    os.makedirs(result_dir, exist_ok=True)

    activations = ["ReLU", "GELU", "SiLU"]
    k_values = [1, 2, 4, 8, 16, 32, 64, 128]
    d_in, d_out = 64, 64
    n_params = 296000  # same for all activations in this structure

    all_results = []

    for act_name in activations:
        print("=" * 60)
        print(f"Activation: {act_name}")
        print("=" * 60)

        model = MLP(activation_name=act_name)
        model.eval()

        x_train, y_train, x_test, y_test = generate_data(model)
        x_train_np = x_train.numpy()

        for k in k_values:
            t0 = time.time()

            centers = compute_centers(x_train_np, k)
            cache = precompute_expansion_points(model, centers)
            avg_mse, avg_cos = evaluate(x_test, y_test, cache)

            taylor_params = k * (d_out * d_in + d_out + d_in)
            storage_ratio = taylor_params / n_params

            elapsed = time.time() - t0
            print(f"  k={k:3d}: MSE={avg_mse:.6f}  CosSim={avg_cos:.6f}  "
                  f"StorageRatio={storage_ratio:.4f}  Time={elapsed:.1f}s")

            all_results.append({
                "activation": act_name,
                "k": k,
                "mse": avg_mse,
                "cosine_sim": avg_cos,
                "taylor_params": taylor_params,
                "orig_params": n_params,
                "storage_ratio": storage_ratio,
            })

    # ── Save per-activation CSVs ──
    for act_name in activations:
        act_results = [r for r in all_results if r["activation"] == act_name]

        with open(os.path.join(result_dir, f"{act_name}_k_vs_mse.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["k", "mse"])
            w.writeheader()
            for r in act_results:
                w.writerow({"k": r["k"], "mse": r["mse"]})

        with open(os.path.join(result_dir, f"{act_name}_k_vs_cosine.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["k", "cosine_sim"])
            w.writeheader()
            for r in act_results:
                w.writerow({"k": r["k"], "cosine_sim": r["cosine_sim"]})

    # ── Save combined comparison CSV ──
    compare_path = os.path.join(result_dir, "activation_comparison.csv")
    with open(compare_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "activation", "k", "mse", "cosine_sim",
            "taylor_params", "orig_params", "storage_ratio"
        ])
        w.writeheader()
        for r in all_results:
            w.writerow(r)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Summary: CosSim @ k=64 (Storage <= original)")
    print("=" * 60)
    for act_name in activations:
        act_results = [r for r in all_results
                       if r["activation"] == act_name and r["k"] == 64]
        if act_results:
            r = act_results[0]
            print(f"  {act_name:5s}: CosSim={r['cosine_sim']:.6f}  "
                  f"MSE={r['mse']:.6f}  StorageRatio={r['storage_ratio']:.4f}")

    print("\nDone. Results in", result_dir)


if __name__ == "__main__":
    main()
