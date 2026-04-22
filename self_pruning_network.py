"""

Run:  python self_pruning_network.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Part 1 — PrunableLinear Layer

class PrunableLinear(nn.Module):
   

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

 
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.gate_scores, mean=0.0, std=0.01)

        # Kaiming init for weights (good default for ReLU networks)
        nn.init.kaiming_uniform_(self.weight, a=0, mode="fan_in",
                                 nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """output = x @ (weight * sigmoid(gate_scores))^T + bias"""
        gates = torch.sigmoid(self.gate_scores)        # ∈ (0, 1)
        pruned_weight = self.weight * gates             # element-wise gating
        return F.linear(x, pruned_weight, self.bias)

    @torch.no_grad()
    def get_gates(self) -> torch.Tensor:
        """Gate values (detached, flat 1-D)."""
        return torch.sigmoid(self.gate_scores).detach().flatten()

    @torch.no_grad()
    def sparsity_level(self, threshold: float = 1e-2) -> float:
        """Percentage of gates below threshold."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item() * 100.0


# Network Definition

class SelfPruningNet(nn.Module):
    

    def __init__(self, input_dim: int = 3072, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            PrunableLinear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            PrunableLinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            PrunableLinear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))

    def _prunable_layers(self):
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def all_gates(self) -> torch.Tensor:
        return torch.cat([l.get_gates() for l in self._prunable_layers()])

    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of all gate values (raw sum). Always ≥ 0 since
        gates = sigmoid(...) ≥ 0, so L1 ≡ sum. Part of autograd graph."""
        return sum(
            torch.sigmoid(l.gate_scores).sum()
            for l in self._prunable_layers()
        )

    @torch.no_grad()
    def global_sparsity(self, threshold: float = 1e-2) -> float:
        gates = self.all_gates()
        return (gates < threshold).float().mean().item() * 100.0

    @torch.no_grad()
    def per_layer_sparsity(self, threshold: float = 1e-2) -> list:
        results = []
        for i, layer in enumerate(self._prunable_layers()):
            name = f"Layer {i+1} ({layer.in_features}→{layer.out_features})"
            results.append((name, layer.sparsity_level(threshold)))
        return results

    @torch.no_grad()
    def hard_prune(self, threshold: float = 1e-2):
        """Zero out gate_scores where sigmoid(gate_score) < threshold.
        This makes pruning permanent for evaluation."""
        for layer in self._prunable_layers():
            gates = torch.sigmoid(layer.gate_scores)
            mask = gates < threshold
            layer.gate_scores.data[mask] = -20.0  # sigmoid(-20) ≈ 0


# Data Loading

def get_cifar10_loaders(batch_size: int = 256, num_workers: int = 2):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


# Training & Evaluation


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total * 100.0


def train_one_run(lam, train_loader, test_loader, device,
                  epochs=25, lr=1e-3, warmup_epochs=5) -> dict:
    """Train a fresh SelfPruningNet with sparsity weight λ."""

    model = SelfPruningNet().to(device)

    
    gate_params  = [p for n, p in model.named_parameters()
                    if 'gate_scores' in n]
    other_params = [p for n, p in model.named_parameters()
                    if 'gate_scores' not in n]
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': lr,     'weight_decay': 1e-4},
        {'params': gate_params,  'lr': lr * 5, 'weight_decay': 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    total_gates = sum(l.gate_scores.numel() for l in model._prunable_layers())
    print(f"\n{'='*65}")
    print(f"  Training  λ = {lam:.0e}  |  {epochs} epochs  |  {total_gates:,} gates")
    print(f"{'='*65}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_cls = running_total = 0.0
        n_batches = 0

        
        lam_eff = lam * min(1.0, epoch / warmup_epochs)

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)

            cls_loss   = F.cross_entropy(logits, y)
            spar_loss  = model.sparsity_loss()
            total_loss = cls_loss + lam_eff * spar_loss

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_cls   += cls_loss.item()
            running_total += total_loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            sp = model.global_sparsity()
            g  = model.all_gates()
            print(f"  Epoch {epoch:3d}/{epochs}"
                  f"  |  cls={running_cls/n_batches:.4f}"
                  f"  total={running_total/n_batches:.4f}"
                  f"  |  λ_eff={lam_eff:.2e}"
                  f"  |  Sparsity={sp:.1f}%"
                  f"  |  Gate[min={g.min():.4f},"
                  f" mean={g.mean():.4f}, max={g.max():.4f}]")

    #  Evaluation 
    soft_acc  = evaluate(model, test_loader, device)
    sparsity  = model.global_sparsity()
    gates_np  = model.all_gates().cpu().numpy()
    per_layer = model.per_layer_sparsity()

    # Hard-prune and re-evaluate
    model.hard_prune(threshold=1e-2)
    hard_acc = evaluate(model, test_loader, device)

    print(f"\n  ✔  λ={lam:.0e}  |  Soft Acc={soft_acc:.2f}%  "
          f"|  Hard Acc={hard_acc:.2f}%  |  Sparsity={sparsity:.2f}%")
    for name, sp in per_layer:
        print(f"     {name}: {sp:.1f}% pruned")

    return {
        "lambda": lam, "soft_acc": soft_acc, "hard_acc": hard_acc,
        "sparsity": sparsity, "gates": gates_np,
        "per_layer_sparsity": per_layer,
    }



# Visualisation


def plot_gate_distribution(gates, lam, save_path="gate_distribution.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(gates, bins=150, color="#2563EB", edgecolor="none", alpha=0.85)
    ax.set_xlabel("Gate Value (sigmoid output)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(
        f"Distribution of Gate Values — Best Model (λ = {lam:.0e})\n"
        f"Spike at 0 → pruned  |  Cluster away from 0 → active",
        fontsize=12)
    ax.axvline(x=0.01, color="#DC2626", lw=1.5, ls="--",
               label="Prune threshold (0.01)")
    ax.legend(fontsize=11)
    ax.set_xlim(-0.02, 1.02)

    n_pruned = (gates < 0.01).sum()
    n_total  = len(gates)
    n_active = (gates > 0.5).sum()
    txt = (f"Total gates: {n_total:,}\n"
           f"Pruned (<0.01): {n_pruned:,} ({n_pruned/n_total*100:.1f}%)\n"
           f"Active (>0.50): {n_active:,} ({n_active/n_total*100:.1f}%)")
    ax.text(0.55, 0.85, txt, transform=ax.transAxes, fontsize=10,
            va="top", bbox=dict(boxstyle="round,pad=0.4",
                                facecolor="wheat", alpha=0.8))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n   Gate distribution plot saved → '{save_path}'")



# Main

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else
                          "cpu")
    print(f"\n🔧 Device: {device}")
    train_loader, test_loader = get_cifar10_loaders(batch_size=256)

    # λ values: low → medium-low → medium → high
    # These span the full sparsity–accuracy trade-off curve.
    lambda_values = [1e-5, 1e-4, 1e-3, 1e-1]
    results = []

    for lam in lambda_values:
        results.append(train_one_run(
            lam=lam, train_loader=train_loader,
            test_loader=test_loader, device=device, epochs=25))

    # ── Results table ─────
    print(f"\n\n{'='*70}")
    print(f"  {'Lambda':<10} {'Soft Acc (%)':<14} {'Hard Acc (%)':<14} {'Sparsity (%)'}")
    print(f"  {'-'*62}")
    for r in results:
        print(f"  {r['lambda']:<10.0e} {r['soft_acc']:<14.2f} "
              f"{r['hard_acc']:<14.2f} {r['sparsity']:.2f}")
    print(f"{'='*70}\n")

    # Pick best balanced model: highest accuracy among those with >20% sparsity
    pruned = [r for r in results if r["sparsity"] > 20]
    best = max(pruned, key=lambda r: r["soft_acc"]) if pruned else results[0]
    print(f"   Best → λ={best['lambda']:.0e}  "
          f"Acc={best['soft_acc']:.2f}%  Sparsity={best['sparsity']:.2f}%")

    plot_gate_distribution(best["gates"], lam=best["lambda"])

    # Save summary
    with open("results_summary.txt", "w") as f:
        f.write("Self-Pruning Neural Network — Results Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"{'Lambda':<10} {'Soft Acc(%)':<13} {'Hard Acc(%)':<13} "
                f"{'Sparsity(%)'}\n")
        f.write("-" * 50 + "\n")
        for r in results:
            f.write(f"{r['lambda']:<10.0e} {r['soft_acc']:<13.2f} "
                    f"{r['hard_acc']:<13.2f} {r['sparsity']:.2f}\n")
        f.write(f"\nBest: λ={best['lambda']:.0e}  "
                f"Acc={best['soft_acc']:.2f}%  Sparsity={best['sparsity']:.2f}%\n")
        for name, sp in best["per_layer_sparsity"]:
            f.write(f"  {name}: {sp:.1f}%\n")
    print("   Results saved → 'results_summary.txt'")


if __name__ == "__main__":
    main()
