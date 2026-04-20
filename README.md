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
 
```
Input:  (dx, dy, sin(orientation), cos(orientation))
        4 → Linear(64) → ReLU → Linear(64) → ReLU → Linear(1) → Sigmoid
Output: comfort score in [0, 1]
```
 
- Training data: ~8,000 pairs from ETH/UCY trajectories
- Optimizer: Adam, lr=0.005, 50 epochs
- Validation accuracy: 85%+
### SocialCostCNN
 
```
Input:  (N, 2, 50, 50)
        Channel 0: occupancy grid
        Channel 1: orientation field
 
        Conv(2→16, 3x3) → ReLU → Conv(16→32, 3x3) → ReLU → Conv(32→1, 3x3) → ReLU
 
Output: (N, 1, 50, 50) normalized cost map
```
 
- Validation MSE: 0.01 or lower
- Training time: ~15 min on CPU (500 scenarios)
---
 
## Quick Start
 
```bash
pip install numpy scipy matplotlib pandas requests
 
# Core pipeline, no neural networks
python main.py --skip-download
 
# Quick test with neural networks
python main.py --quick --neural
 
# Full pipeline with neural training (~1 hour on CPU)
python main.py --neural
```
 
---
 
## Project Structure
 
```
social_zone_nav/
├── main.py
├── social_zone.py             # SocialZone / CircularSocialZone
├── neural_zone.py             # ZoneLearnerNet / SocialCostCNN / NeuralSocialZone
├── neural_planner.py          # CNN-accelerated A* planner
├── environment.py             # 2D simulation arena
├── planner.py                 # A* planners
├── experiments.py             # Three-planner comparison
├── figures.py                 # Publication figures fig1 to fig7
├── train_zone_learner.py      # ZoneLearnerNet training
├── train_cost_cnn.py          # SocialCostCNN training
└── download_datasets.py       # ETH/UCY data download
```
 
---
 
## Platform
 
Python 3.8+, Windows / macOS / Linux. Pure NumPy, no GPU required. Core pipeline runs in approximately 5 minutes. Full neural pipeline runs in approximately 60 minutes on CPU.
 
---
 
## Dataset
 
ETH/UCY pedestrian trajectory dataset. Downloaded automatically on first run.
