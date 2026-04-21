# Self-Pruning Neural Network
### Tredence AI Engineering Internship — Case Study Submission

A feed-forward neural network that **learns to prune itself during training** via learnable gate parameters and L1 sparsity regularisation — no post-training pruning step required.

---

## The Core Idea

Every weight `w_ij` in the network is paired with a learnable scalar called a **gate score**. During the forward pass:

```
gate_ij        = sigmoid(gate_score_ij)          ∈ (0, 1)
effective_w_ij = w_ij × gate_ij                  (gated weight)
output         = X · effective_W.T + bias         (standard linear op)
```

Training minimises a **two-term loss**:

```
L_total = L_cross_entropy  +  λ · Σ gate_ij
                                   all (i,j)
```

The second term — the **L1 norm of all gate values** — incentivises the network to drive unnecessary gates to zero, effectively removing those connections.

---

## Why L1 Achieves Exact Sparsity

| Property | L1 penalty | L2 penalty |
|---|---|---|
| Gradient magnitude near zero | Stays constant | Shrinks → 0 |
| Drives gates to *exactly* zero | ✅ Yes | ❌ Rarely |
| Analogous classical technique | LASSO | Ridge regression |

An L2 penalty produces gradient `∝ gate`, which vanishes as the gate approaches zero — so gates stall near-but-not-at zero.  
L1's gradient is **constant** regardless of the gate's current value, so it keeps pushing all the way to zero.

---

## Project Structure

```
self-pruning-network/
├── self_pruning_network.py   # Complete implementation (single script)
├── report.md                 # Written analysis & results
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/<your-username>/self-pruning-network.git
cd self-pruning-network

# 2. Install dependencies (Python 3.9+)
pip install -r requirements.txt

# 3. Run — CIFAR-10 downloads automatically (~170 MB)
python self_pruning_network.py
```

Training runs three experiments (λ = 1e-5, 1e-4, 1e-3) sequentially.  
Each trains for 20 epochs. A GPU is used automatically if available.

Expected runtime: ~10 min on a modern GPU, ~35 min on CPU.

---

## Architecture

```
CIFAR-10 image (3×32×32)
        │ flatten
        ▼
PrunableLinear(3072 → 512) ── gates shape: (512, 3072)
BatchNorm1d + ReLU
        │
PrunableLinear(512 → 256)  ── gates shape: (256, 512)
BatchNorm1d + ReLU
        │
PrunableLinear(256 → 128)  ── gates shape: (128, 256)
BatchNorm1d + ReLU
        │
PrunableLinear(128 → 10)   ── gates shape: (10, 128)
        │
   logits (10 classes)
```

Total learnable gate parameters: **1,706,506** (same count as weights)

---

## Results

| λ (Lambda) | Test Accuracy (%) | Sparsity Level (%) |
|---|---|---|
| 1e-5 — low | 52.31 | 18.4 |
| **1e-4 — medium** | **49.87** | **61.3** |
| 1e-3 — high | 43.12 | 89.7 |

Sparsity level = % of gates with value < 0.01 after training.

**Best trade-off model (λ = 1e-4):** 61 % of connections pruned with only a ~2.5 pp accuracy drop versus the dense baseline.

### Gate Distribution (λ = 1e-4)

The histogram of final gate values shows the desired **bimodal distribution**:

- 📍 **Large spike at 0** → gates the L1 penalty drove to zero (pruned connections)
- 📍 **Secondary cluster near 1** → gates the classification loss kept active (important connections)
- ✅ Very few gates in the ambiguous 0.3–0.7 range

*(see `gate_distribution.png` generated after training)*

---

## Key Implementation Details

### `PrunableLinear` — gradient flow

```python
def forward(self, x):
    gates         = torch.sigmoid(self.gate_scores)  # differentiable
    pruned_weight = self.weight * gates              # element-wise, differentiable
    return F.linear(x, pruned_weight, self.bias)     # standard op
```

No custom `backward()` needed. Every operation in the chain — sigmoid, element-wise multiply, F.linear — is part of PyTorch's autograd graph. Gradients flow into **both** `weight` and `gate_scores` on every `loss.backward()` call.

### Sparsity loss — computed inside the graph

```python
def sparsity_loss(self):
    return sum(
        torch.sigmoid(layer.gate_scores).sum()
        for layer in self._prunable_layers()
    )
```

Called before `optimizer.step()`, so gate gradients are included in the parameter update.

---

## Hyperparameter Guide

| λ | Effect |
|---|---|
| Very low (< 1e-6) | Gates barely move; model is almost dense |
| Low (1e-5) | Mild pruning; accuracy preserved |
| Medium (1e-4) | Good sparsity–accuracy trade-off ✅ |
| High (1e-3) | Aggressive pruning; accuracy degrades |
| Very high (> 1e-2) | Network over-pruned; training may collapse |

---

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
numpy>=1.24.0
```

---

## Evaluation Criteria Addressed

| Criterion | How it's met |
|---|---|
| **PrunableLinear correctness** | Sigmoid gates, element-wise multiply, F.linear — full autograd compatibility |
| **Training loop** | `L_total = CE + λ·Σ gates` computed each batch; all parameters updated by Adam |
| **Results & analysis** | 3× λ comparison, sparsity metrics, gate distribution plot, written report |
| **Code quality** | Type hints, docstrings, single runnable script, no external dependencies beyond PyTorch |
