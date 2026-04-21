

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

# Part 1 – PrunableLinear Layer

class PrunableLinear(nn.Module):
   

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # ── Standard parameters ──────────────────────────────────────────────
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

       
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Initialise weight with Kaiming uniform (good default for ReLU nets)
        nn.init.kaiming_uniform_(self.weight, a=0, mode="fan_in",
                                 nonlinearity="relu")

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map gate_scores → [0, 1] via sigmoid; gradient flows back normally.
        gates = torch.sigmoid(self.gate_scores)          # (out, in)

        # Element-wise gating: weights that have gate ≈ 0 are effectively dead.
        pruned_weight = self.weight * gates              # (out, in)

        # Standard affine transform using F.linear (handles batched matmul).
        return F.linear(x, pruned_weight, self.bias)

    # ── Helpers ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def get_gates(self) -> torch.Tensor:
        """Return gate values (detached) as a flat 1-D tensor."""
        return torch.sigmoid(self.gate_scores).detach().flatten()

    @torch.no_grad()
    def sparsity_level(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below `threshold` (pruned), as a percentage."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item() * 100.0


# Network definition

class SelfPruningNet(nn.Module):
   

    def __init__(self, input_dim: int = 3072, num_classes: int = 10) -> None:
        super().__init__()

        self.net = nn.Sequential(
            PrunableLinear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            PrunableLinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            PrunableLinear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))   # flatten spatial dims

    # ── Gate utilities───
    def _prunable_layers(self):
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def all_gates(self) -> torch.Tensor:
       
        return torch.cat([layer.get_gates() for layer in self._prunable_layers()])

    def sparsity_loss(self) -> torch.Tensor:
        
        gate_sum = sum(
            torch.sigmoid(layer.gate_scores).sum()
            for layer in self._prunable_layers()
        )
        return gate_sum

    @torch.no_grad()
    def global_sparsity(self, threshold: float = 1e-2) -> float:
       
        gates = self.all_gates()
        return (gates < threshold).float().mean().item() * 100.0


# Part 3 – Training & Evaluation

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
        root="./data", train=True,  download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, test_loader


def evaluate(model: SelfPruningNet, loader: DataLoader,
             device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds  = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total * 100.0


def train_one_run(
    lam: float,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
) -> dict:
    
    model = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\n{'='*60}")
    print(f"  Training  λ = {lam:.0e}   |  {epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_cls = running_spar = running_total = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)

            cls_loss  = F.cross_entropy(logits, y)
            spar_loss = model.sparsity_loss()            
            total_loss = cls_loss + lam * spar_loss

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_cls   += cls_loss.item()
            running_spar  += spar_loss.item()
            running_total += total_loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            sparsity = model.global_sparsity()
            print(
                f"  Epoch {epoch:3d}/{epochs}"
                f"  |  cls={running_cls/n_batches:.4f}"
                f"  spar={running_spar/n_batches:.4f}"
                f"  total={running_total/n_batches:.4f}"
                f"  |  Sparsity={sparsity:.1f}%"
            )

    accuracy  = evaluate(model, test_loader, device)
    sparsity  = model.global_sparsity()
    gates_np  = model.all_gates().cpu().numpy()

    print(f"\n  ✔  λ={lam:.0e}  |  Test Acc={accuracy:.2f}%  "
          f"|  Sparsity={sparsity:.2f}%")

    return {
        "lambda":   lam,
        "accuracy": accuracy,
        "sparsity": sparsity,
        "gates":    gates_np,
    }



def plot_gate_distribution(gates: np.ndarray, lam: float,
                           save_path: str = "gate_distribution.png") -> None:
    
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(gates, bins=120, color="#2563EB", edgecolor="none",
            alpha=0.85, density=False)

    ax.set_xlabel("Gate Value  (sigmoid output)", fontsize=13)
    ax.set_ylabel("Count",                         fontsize=13)
    ax.set_title(
        f"Distribution of Gate Values — Best Model  (λ = {lam:.0e})\n"
        f"Spike at 0 → pruned connections | Cluster near 1 → active connections",
        fontsize=12,
    )
    ax.axvline(x=0.01, color="#DC2626", linewidth=1.5,
               linestyle="--", label="Prune threshold (0.01)")
    ax.legend(fontsize=11)
    ax.set_xlim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n  Gate distribution plot saved → '{save_path}'")




def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    train_loader, test_loader = get_cifar10_loaders(batch_size=256)

    lambda_values = [1e-5, 1e-4, 1e-3]   
    results       = []

    for lam in lambda_values:
        result = train_one_run(
            lam=lam,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=20,
        )
        results.append(result)

    # ── Results table ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*55}")
    print(f"  {'Lambda':<12}  {'Test Accuracy (%)':<22}  {'Sparsity Level (%)'}")
    print(f"  {'-'*51}")
    for r in results:
        print(f"  {r['lambda']:<12.0e}  {r['accuracy']:<22.2f}  {r['sparsity']:.2f}")
    print(f"{'='*55}\n")

    # ── Pick best model = highest (accuracy × sparsity) trade-off ────────────
    best = max(results, key=lambda r: r["accuracy"] * r["sparsity"])
    print(f"  Best balanced model → λ={best['lambda']:.0e}  "
          f"(Acc={best['accuracy']:.2f}%  Sparsity={best['sparsity']:.2f}%)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_gate_distribution(best["gates"], lam=best["lambda"])


if __name__ == "__main__":
    main()
