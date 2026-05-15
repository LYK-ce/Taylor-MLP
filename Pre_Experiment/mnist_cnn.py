"""
Presented by KeJi
Date: 2026-05-15
Taylor-MLP Phase 2: MNIST CNN + Taylor-MLP approximation.
Replace MLP classifier with 1st-order Taylor expansion.
Compare ReLU, GELU, SiLU activations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import csv
import os
import time
import numpy as np


# ============================================================
# CNN Model
# ============================================================
ACTIVATION_MAP = {
    "ReLU": nn.ReLU,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
}


class CNN(nn.Module):
    """CNN with separable backbone + MLP classifier."""
    def __init__(self, activation_name="ReLU"):
        super().__init__()
        act_fn = ACTIVATION_MAP[activation_name]
        self.activation_name = activation_name
        # Backbone
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),  # backbone always ReLU
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # MLP Classifier (this part gets Taylor-approximated)
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256),
            act_fn(),
            nn.Linear(256, 64),
            act_fn(),
            nn.Linear(64, 10),
        )

    def forward_features(self, x):
        """Extract 1568-dim feature vector from backbone."""
        return self.conv(x).flatten(1)

    def forward_classifier(self, features):
        """MLP classifier forward pass (R^n → R^10)."""
        return self.classifier(features)

    def forward(self, x):
        features = self.forward_features(x)
        return self.forward_classifier(features)


# ============================================================
# Training
# ============================================================
def train_cnn(model, train_loader, test_loader, epochs=5, lr=1e-3, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, "
              f"test_acc={acc:.4f}")

    return model


# ============================================================
# Feature extraction
# ============================================================
@torch.no_grad()
def extract_features(model, loader, device="cpu"):
    """Extract features and compute ground-truth MLP outputs."""
    all_features = []
    all_mlp_outputs = []
    all_labels = []
    for x, y in loader:
        x = x.to(device)
        feat = model.forward_features(x).cpu()
        mlp_out = model.forward_classifier(feat).cpu()
        all_features.append(feat)
        all_mlp_outputs.append(mlp_out)
        all_labels.append(y)
    return (
        torch.cat(all_features, dim=0),
        torch.cat(all_mlp_outputs, dim=0),
        torch.cat(all_labels, dim=0),
    )


# ============================================================
# K-means
# ============================================================
def compute_centers(features_np, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features_np)
    return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)


# ============================================================
# Jacobian computation (batched via vmap-style loop)
# ============================================================
def compute_jacobian_at_center(model, x0):
    """
    Compute F(X0) and Jacobian J(X0) using torch.autograd.functional.jacobian.
    x0: (d_in,) single center
    Returns: F_X0 (10,), J (10, d_in)
    """
    x0 = x0.clone().detach().requires_grad_(True)

    def fn(x):
        return model.forward_classifier(x.unsqueeze(0)).squeeze(0)

    f_val = fn(x0).detach()
    J = torch.autograd.functional.jacobian(fn, x0).detach()  # (10, d_in)
    return f_val, J


def precompute_centers(model, centers):
    """Precompute F and J for all centers."""
    cache = {}
    for i in range(centers.shape[0]):
        f_val, jac = compute_jacobian_at_center(model, centers[i])
        cache[i] = (f_val, jac, centers[i])
    return cache


# ============================================================
# Taylor batch inference
# ============================================================
@torch.no_grad()
def taylor_predict_batch(features, cache):
    """
    features: (N, d_in)
    Returns: (N, 10) Taylor-approximated outputs
    """
    N = features.shape[0]
    d_out = 10
    outputs = torch.zeros(N, d_out)

    # Collect centers into a matrix
    idxs = list(cache.keys())
    centers_mat = torch.stack([cache[i][2] for i in idxs], dim=0)  # (k, d_in)

    for n in range(N):
        x = features[n]
        # Find nearest center
        dists = torch.norm(centers_mat - x.unsqueeze(0), dim=1)
        best_idx = idxs[torch.argmin(dists).item()]
        f_val, jac, x0 = cache[best_idx]
        outputs[n] = f_val + jac @ (x - x0)

    return outputs


# ============================================================
# Evaluation
# ============================================================
def evaluate_taylor(features, mlp_outputs, labels, cache):
    """Evaluate Taylor approximation vs ground-truth MLP outputs."""
    N = features.shape[0]
    approx = taylor_predict_batch(features, cache)

    # MSE
    mse = torch.mean((approx - mlp_outputs) ** 2).item()

    # Cosine similarity (average over samples)
    cos_sim = 0.0
    for i in range(N):
        cos_sim += F.cosine_similarity(
            approx[i].unsqueeze(0), mlp_outputs[i].unsqueeze(0)
        ).item()
    cos_sim /= N

    # Classification accuracy
    pred_labels = approx.argmax(dim=1)
    acc = (pred_labels == labels).float().mean().item()

    return mse, cos_sim, acc


# ============================================================
# Main
# ============================================================
def main():
    result_dir = "Result/Pre_Experiment"
    os.makedirs(result_dir, exist_ok=True)
    device = "cpu"

    # ── Load MNIST ──
    print("=" * 60)
    print("Loading MNIST...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

    # Use subset for speed: 20K train, 5K test
    train_subset = Subset(train_ds, range(20000))
    test_subset = Subset(test_ds, range(5000))
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)

    # ── Experiment loop ──
    activations = ["ReLU", "GELU", "SiLU"]
    k_values = [1, 4, 8, 12, 24, 48, 96]
    d_in, d_out = 1568, 10
    mlp_params = 1568 * 256 + 256 * 64 + 64 * 10  # 418,432
    taylor_per_center = d_out * d_in + d_out + d_in  # 17,258

    all_results = []

    for act_name in activations:
        print("\n" + "=" * 60)
        print(f"Activation: {act_name}")
        print("=" * 60)

        # Train CNN
        print("  Training CNN...")
        model = CNN(activation_name=act_name)
        full_train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
        model = train_cnn(model, full_train_loader, test_loader, epochs=5, device=device)
        model.eval()

        # Extract features
        print("  Extracting features...")
        full_train_loader = DataLoader(train_subset, batch_size=256, shuffle=False)
        full_test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)
        train_feat, train_mlp, train_labels = extract_features(model, full_train_loader, device)
        test_feat, test_mlp, test_labels = extract_features(model, full_test_loader, device)
        train_feat_np = train_feat.numpy()

        # Baseline accuracy
        with torch.no_grad():
            baseline_pred = test_mlp.argmax(dim=1)
            baseline_acc = (baseline_pred == test_labels).float().mean().item()
        print(f"  Baseline accuracy: {baseline_acc:.4f}")

        for k in k_values:
            t0 = time.time()

            # K-means
            centers = compute_centers(train_feat_np, k)

            # Precompute
            cache = precompute_centers(model, centers)

            # Evaluate
            mse, cos_sim, acc = evaluate_taylor(test_feat, test_mlp, test_labels, cache)

            taylor_params = k * taylor_per_center
            storage_ratio = taylor_params / mlp_params
            acc_drop = baseline_acc - acc

            elapsed = time.time() - t0
            print(f"  k={k:3d}: Acc={acc:.4f} (drop={acc_drop:.4f})  "
                  f"CosSim={cos_sim:.4f}  MSE={mse:.6f}  "
                  f"Storage={storage_ratio:.3f}  Time={elapsed:.1f}s")

            all_results.append({
                "activation": act_name,
                "k": k,
                "baseline_acc": baseline_acc,
                "taylor_acc": acc,
                "acc_drop": acc_drop,
                "mse": mse,
                "cosine_sim": cos_sim,
                "taylor_params": taylor_params,
                "orig_mlp_params": mlp_params,
                "storage_ratio": storage_ratio,
            })

    # ── Save results ──
    summary_path = os.path.join(result_dir, "mnist_phase2_summary.csv")
    with open(summary_path, "w", newline="") as f:
        fields = ["activation", "k", "baseline_acc", "taylor_acc", "acc_drop",
                  "mse", "cosine_sim", "taylor_params", "orig_mlp_params",
                  "storage_ratio"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow(r)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Phase 2 Summary: Accuracy Drop vs Original MLP")
    print("=" * 60)
    print(f"{'Act':5s} {'k':>4s}  {'Baseline':>9s}  {'Taylor':>9s}  "
          f"{'Drop':>8s}  {'CosSim':>7s}  {'Storage':>7s}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['activation']:5s} {r['k']:4d}  {r['baseline_acc']:9.4f}  "
              f"{r['taylor_acc']:9.4f}  {r['acc_drop']:+8.4f}  "
              f"{r['cosine_sim']:7.4f}  {r['storage_ratio']:7.3f}")

    print(f"\nResults saved to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
