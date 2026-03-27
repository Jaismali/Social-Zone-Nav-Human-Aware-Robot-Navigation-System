# Social-Zone Nav v2

**Asymmetric Social Zones for Robot Navigation — with Neural Network Enhancement**

Learns and respects direction-dependent personal space (front: 1.8m, sides: 1.2m, back: 0.6m) from real human trajectory data (ETH/UCY datasets). V2 adds two neural networks for learned zone representation and accelerated planning.

---

## Quick Start

```bash
pip install numpy scipy matplotlib pandas requests

# Core pipeline only (no neural nets)
python main.py --skip-download

# Quick smoke test with everything
python main.py --quick --neural

# Full pipeline with neural training (~1 hour on CPU)
python main.py --neural

# Train networks only, then run separately
python main.py --train-only --neural
python main.py --neural --skip-experiments
python main.py --neural --figures-only
```

> **Note on PyTorch:** The neural networks are implemented in **pure NumPy** — no PyTorch or TensorFlow required. The codebase runs entirely on CPU. If you have PyTorch installed it is not used (to keep the code self-contained and cross-platform). The `torch` entry in `requirements.txt` is kept for reference; remove it if you prefer.

---

## What's New in v2

### New Files

| File | Description |
|------|-------------|
| `neural_zone.py` | Three neural network classes (pure NumPy) |
| `train_zone_learner.py` | Trains ZoneLearnerNet on trajectory data |
| `train_cost_cnn.py` | Trains SocialCostCNN on generated scenarios |
| `neural_planner.py` | NeuralPlanner: A* with CNN cost maps |

### Updated Files

| File | What changed |
|------|-------------|
| `experiments.py` | Adds `--neural` planner as third comparison |
| `figures.py` | Adds figures 5, 6, 7 (neural analysis) |
| `main.py` | New flags: `--neural`, `--train-only`, `--quick` |
| `requirements.txt` | Adds `torch torchvision` (optional) |

---

## Neural Network Architecture

### A. ZoneLearnerNet

Learns the comfort boundary from human trajectory pairs.

```
Input:  (dx, dy, sin(orientation), cos(orientation)) — 4 floats
        4 → Linear(64) → ReLU → Linear(64) → ReLU → Linear(1) → Sigmoid
Output: comfort score in [0, 1]
         1 = comfortable (distance OK), 0 = zone violated
```

- Training data: ~8,000 pairs from ETH/UCY trajectories
- Labels: 1 if actual distance > 10th-percentile zone threshold, else 0
- Optimizer: Adam, lr=0.005, 50 epochs
- Expected val accuracy: ≥85%

### B. SocialCostCNN

Predicts the full 50×50 social cost map in a single forward pass.

```
Input:  (N, 2, 50, 50)
        Channel 0: occupancy grid (1 = pedestrian present)
        Channel 1: sin(orientation) field

        Conv(2→16, 3×3, pad=1) → ReLU
        Conv(16→32, 3×3, pad=1) → ReLU
        Conv(32→1,  3×3, pad=1) → ReLU

Output: (N, 1, 50, 50) — normalised cost map in [0, 1]
```

- Training: 500–10,000 synthetic scenarios (scales with `--cnn-scenarios`)
- Ground truth: computed analytically from asymmetric SocialZone
- Optimizer: Adam, lr=0.001, 30 epochs
- Speedup: ~5–15× faster cost map generation vs. analytical

### C. NeuralSocialZone

Drop-in replacement for `SocialZone` using ZoneLearnerNet:

```python
from neural_zone import NeuralSocialZone
zone = NeuralSocialZone.from_file("zone_learner.npy")
cost = zone.compute_cost(robot_pos, human_pos, human_orientation)
```

---

## Figures

| Figure | Content |
|--------|---------|
| `fig1_heatmap.pdf` | Asymmetric vs circular cost fields |
| `fig2_zone_distances.pdf` | Bar chart: zone distance comparison |
| `fig3_trajectories.pdf` | Example path: asymmetric vs circular |
| `fig4_results.pdf` | Success/collision/violations (all planners) |
| `fig5_learned_zone.pdf` | Neural comfort map vs ground truth |
| `fig6_training_curves.pdf` | Training loss for both networks |
| `fig7_cnn_prediction.pdf` | CNN prediction vs ground truth cost maps |

---

## CLI Reference

```bash
# Full runs
python main.py                           # Core pipeline
python main.py --neural                  # + Neural training + NeuralPlanner
python main.py --train-only              # Train nets only (no experiments)
python main.py --quick --neural          # Quick 10-episode test

# Selective steps
python main.py --skip-download           # Skip data download
python main.py --skip-extract            # Skip zone extraction
python main.py --skip-experiments        # Skip experiments
python main.py --figures-only            # Figs 1-4 from saved results
python main.py --figures-only --neural   # Figs 1-7

# Tuning
python main.py --neural --episodes 50    # Fewer episodes
python main.py --neural --zl-epochs 100  # More ZoneLearner epochs
python main.py --neural --cnn-scenarios 2000  # More CNN scenarios
```

---

## Expected Results

| Metric | Asymmetric | Circular | Neural |
|--------|-----------|---------|--------|
| Success Rate | ~76% | ~72% | ~75% |
| Collision Rate | ~5% | ~8% | ~5% |
| Social Violations/ep | ~1.2 | ~2.1 | ~1.4 |
| Path Length (m) | ~12.8 | ~13.4 | ~13.0 |
| Plan Time (ms) | ~15 | ~11 | ~8* |

*NeuralPlanner plan time includes CNN forward pass; faster because cost map is precomputed for all cells at once.

### ZoneLearnerNet

- Val accuracy: ≥85% (50 epochs, full ETH/UCY data)
- Training time: ~5 min on CPU

### SocialCostCNN

- Val MSE: ≤0.01 (30 epochs, 500 scenarios)
- Training time: ~15 min on CPU (500 scenarios)
- Full 10,000 scenarios: ~4–6 hours on CPU

---

## File Structure

```
social_zone_nav/
├── main.py                    # Entry point
├── download_datasets.py       # Downloads ETH/UCY
├── extract_zones.py           # Zone distance extraction
├── social_zone.py             # SocialZone / CircularSocialZone
├── neural_zone.py             # ZoneLearnerNet / SocialCostCNN / NeuralSocialZone
├── environment.py             # 2D arena
├── planner.py                 # A* planners (asymmetric + circular)
├── neural_planner.py          # NeuralPlanner (CNN-accelerated A*)
├── train_zone_learner.py      # ZoneLearnerNet training script
├── train_cost_cnn.py          # SocialCostCNN training script
├── experiments.py             # 3-planner comparison
├── figures.py                 # 7 publication figures
├── requirements.txt
└── README.md
```

Generated:
```
data/                          # Trajectory files
zone_distances.npy             # Learned zone distances
zone_learner.npy               # ZoneLearnerNet weights
zone_learner_history.npy       # Training history
zone_learner_norm.npy          # Input normalisation stats
cost_cnn.npy                   # SocialCostCNN weights
cost_cnn_history.npy           # Training history
cost_cnn_val_samples.npy       # Validation scenarios for fig7
experiment_results.npy         # Experiment metrics
figures/fig1–fig7.pdf          # All publication figures
```

---

## Platform

- Python 3.8+, Windows/macOS/Linux
- Pure NumPy — no GPU, no PyTorch, no ROS
- Total runtime: core pipeline ~5 min, full neural ~60 min
