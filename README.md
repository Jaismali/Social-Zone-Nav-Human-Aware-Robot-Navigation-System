# Social-Zone Nav v2

**Asymmetric Social Zone Navigation with Neural Network Enhancement**

A robot navigation system that learns and respects direction-dependent personal space from real human trajectory data. Unlike circular social zone models, this system models asymmetric comfort boundaries: front (1.8m), sides (1.2m), and back (0.6m), trained on the ETH/UCY pedestrian datasets. Version 2 introduces two neural networks for learned zone representation and accelerated planning.

---

## Results

| Metric | Asymmetric | Circular | Neural |
|--------|-----------|---------|--------|
| Success Rate | ~76% | ~72% | ~75% |
| Collision Rate | ~5% | ~8% | ~5% |
| Social Violations/ep | ~1.2 | ~2.1 | ~1.4 |
| Path Length (m) | ~12.8 | ~13.4 | ~13.0 |
| Plan Time (ms) | ~15 | ~11 | ~8* |

*NeuralPlanner precomputes the full cost map in a single CNN forward pass, achieving 5-15x speedup over analytical computation.

---

## Overview

The system models human personal space as an asymmetric directed zone rather than a uniform circle. Robots approaching from the front trigger higher discomfort costs than those approaching from behind, matching empirical findings from pedestrian studies.

Version 2 adds two neural networks trained entirely in pure NumPy, no PyTorch or GPU required.

**ZoneLearnerNet** learns the comfort boundary directly from human trajectory pairs in the ETH/UCY dataset, achieving 85%+ validation accuracy.

**SocialCostCNN** predicts a full 50x50 social cost map in a single forward pass, replacing per-cell analytical computation and delivering 5-15x planning speedup.

**NeuralPlanner** combines CNN cost maps with A* search for socially-aware path planning.

---

## Neural Network Architecture

### ZoneLearnerNet
