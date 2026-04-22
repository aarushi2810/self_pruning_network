# Self-Pruning Neural Network — CIFAR-10

A feed-forward neural network that learns to **prune its own weights during training**.
Each weight is paired with a learnable sigmoid gate; an L1 sparsity penalty drives
unnecessary connections toward zero while important ones stay active.

## Quick Start

```bash
# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run training (≈20 min on Apple Silicon / GPU)
python self_pruning_network.py
```

## Architecture

| Layer | Shape | Gates |
|-------|-------|-------|
| PrunableLinear 1 | 3072 → 512 | 1,572,864 |
| PrunableLinear 2 | 512 → 256 | 131,072 |
| PrunableLinear 3 | 256 → 128 | 32,768 |
| PrunableLinear 4 | 128 → 10 | 1,280 |
| **Total** | | **1,737,984** |

Each `PrunableLinear` layer computes: `output = x @ (weight ⊙ σ(gate_scores))ᵀ + bias`

## How It Works

1. **Gate mechanism**: Every weight `w_ij` is multiplied by `σ(gate_score_ij)`,
   where σ is the sigmoid function. Gradients flow through both weight and gate.

2. **Sparsity loss**: `L_total = L_classification + λ × Σ σ(gate_scores)`.
   The L1 penalty pushes gates toward 0; classification loss keeps important ones open.

3. **λ warmup**: Sparsity penalty ramps up over the first 5 epochs,
   letting the network learn useful features before pruning begins.

4. **Separate gate learning rate**: Gate parameters use 5× higher lr than weights,
   ensuring gates can traverse the full sigmoid range within the training budget.

## Results

Four λ values explored (low → high sparsity pressure):

| λ | Soft Acc (%) | Hard Acc (%) | Sparsity (%) |
|------|-------------|-------------|-------------|
| 1e-5 | 58.07 | 58.05 | 24.06 |
| 1e-4 | 57.08 | 54.60 | 83.37 |
| 1e-3 | 57.83 | 10.26 | 99.04 |
| 1e-1 | 56.35 | 10.00 | 100.00 |

**Best balanced model (λ=1e-4)**: 83.4% sparsity with 57.08% accuracy.
Per-layer: L1=89.0%, L2=32.4%, L3=18.2%, L4=0.6% — the network automatically
preserves output-layer connections while heavily pruning the input layer.

## Output Files

- `gate_distribution.png` — Histogram of gate values showing pruned/active split
- `results_summary.txt` — Tabulated results with per-layer sparsity breakdown
- `report.md` — Full technical report with analysis

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- torchvision, matplotlib, numpy
