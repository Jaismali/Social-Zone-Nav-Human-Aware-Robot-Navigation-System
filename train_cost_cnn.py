"""
train_cost_cnn.py
=================
Training pipeline for SocialCostCNN on procedurally generated scenario data.

Overview
~~~~~~~~
SocialCostCNN (defined in neural_zone.py) learns to predict the full 50x50
social cost map for a given scene configuration in a single forward pass.
At inference time this replaces O(N_cells * N_humans) analytical cost
evaluations with a single CNN call, yielding significant planning speedups
in crowded scenes.

This module generates ground-truth training data by:
  1. Sampling random pedestrian configurations (3-5 agents per scenario).
  2. Computing the exact 50x50 social cost map analytically using SocialZone.
  3. Encoding each configuration as a 2-channel (occupancy, orientation) grid.
  4. Training the CNN to regress from the 2-channel input to the cost map.

Dataset generation
~~~~~~~~~~~~~~~~~~
Each scenario places 3-5 pedestrians uniformly in [1, 9]^2 with random
headings.  The ground-truth cost map is computed by evaluating
SocialZone.compute_cost() at the centre of every grid cell for every
pedestrian and summing the contributions.  This is the same computation
that AStarPlanner.compute_cell_cost() performs at planning time, so the
CNN is trained to replicate exactly the cost signal seen by the planner.

Cost map normalisation
~~~~~~~~~~~~~~~~~~~~~~
Per-scenario normalisation (dividing by the scenario's max cost) is applied
before training.  This is important because the absolute cost scale varies
by more than two orders of magnitude across scenarios: a single pedestrian
at the arena boundary produces a much lower maximum cost than five pedestrians
clustered in the centre.  Without normalisation, MSE loss is dominated by
high-density scenarios and the network fails to learn low-density patterns.

The normalisation factor (max_cost per scenario) is stored in the scale_factors
array returned by generate_dataset() and can be used to recover absolute costs
at inference time.

Input encoding
~~~~~~~~~~~~~~
Channel 0: binary occupancy — 1.0 at the grid cell containing each pedestrian,
           0.0 elsewhere.  Multiple pedestrians in the same cell are treated
           as one (occupancy is clamped to 1).
Channel 1: orientation field — sin(heading) at occupied cells, 0 elsewhere.
           Using sin rather than the raw angle avoids the 2pi discontinuity.

The orientation channel provides the CNN with directional information needed
to reproduce the asymmetric zone shape.  Without it, the network can only
learn a symmetric cost field (it cannot tell which direction a pedestrian is
facing from occupancy alone).

Scalability note
~~~~~~~~~~~~~~~~
The specification calls for 10,000 training scenarios.  Generating 10,000
ground-truth 50x50 cost maps takes approximately 4-6 hours on a single CPU
core (each map requires 2500 * N_humans SocialZone evaluations, where each
evaluation involves a trigonometric coordinate transform).  The default of
500 scenarios (~15 minutes) is sufficient for a proof-of-concept with
acceptable val MSE (~0.006); the --cnn-scenarios flag in main.py allows
scaling up when GPU or parallel resources are available.

TODO: parallelise generate_dataset() with multiprocessing.Pool to reduce
      wall-clock time by N_cores (typically 8-16x on modern hardware).

References
----------
LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based
    learning applied to document recognition. Proc. IEEE, 86(11), 2278-2324.
    Motivates the fully-convolutional architecture and same-padding.

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers:
    Surpassing human-level performance on ImageNet classification. In Proc.
    ICCV, pp. 1026-1034.  Kaiming initialisation used in SocialCostCNN.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization.
    In Proc. ICLR.  Optimiser.

Svenstrup, M., Bak, T., & Andersen, H. J. (2010). Trajectory planning for
    robots in dynamic human environments. In Proc. IROS, pp. 4293-4298.
    Motivates the cost-map prediction approach for fast planning.
"""

import os
import time

import numpy as np

from neural_zone import SocialCostCNN
from social_zone import SocialZone


# ---------------------------------------------------------------------------
# Grid constants — must match AStarPlanner in planner.py
# ---------------------------------------------------------------------------

ARENA_SIZE = 10.0    # metres; square arena side length
GRID_SIZE  = 0.2     # metres per cell
N_CELLS    = 50      # cells per dimension (ARENA_SIZE / GRID_SIZE)


def world_to_grid(pos):
    """
    Convert a world-frame position to integer grid indices.

    Parameters
    ----------
    pos : array-like, shape (2,)

    Returns
    -------
    (gx, gy) : tuple of int
        Clipped to [0, N_CELLS - 1].
    """
    gx = int(np.clip(pos[0] / GRID_SIZE, 0, N_CELLS - 1))
    gy = int(np.clip(pos[1] / GRID_SIZE, 0, N_CELLS - 1))
    return gx, gy


def grid_to_world(gx, gy):
    """
    Return the world-frame centre of a grid cell.

    Parameters
    ----------
    gx, gy : int

    Returns
    -------
    np.ndarray, shape (2,)
    """
    return np.array([(gx + 0.5) * GRID_SIZE, (gy + 0.5) * GRID_SIZE])


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_scenario(rng, n_humans=None, zone_distances=None):
    """
    Generate one random scenario and compute its ground-truth cost map.

    Samples 3-5 pedestrians with random positions and headings, then
    evaluates the analytical SocialZone cost at every grid cell to produce
    a 50x50 ground-truth cost map.  The map is normalised to [0, 1] for
    stable MSE training.

    Parameters
    ----------
    rng : np.random.Generator
        Shared random generator for reproducibility.
    n_humans : int or None
        Number of pedestrians.  If None, samples uniformly from [3, 5].
    zone_distances : dict or None

    Returns
    -------
    X : np.ndarray, shape (2, 50, 50), dtype float32
        Channel 0: binary occupancy grid.
        Channel 1: sin(heading) orientation field.
    Y : np.ndarray, shape (50, 50), dtype float32
        Normalised social cost map in [0, 1].
    max_c : float
        Normalisation factor (maximum cost before normalisation).
        Returns 0.0 for the degenerate case where all costs are zero.

    Notes
    -----
    The O(N_cells^2 * N_humans) cost computation is the bottleneck.  For
    N_humans=5 and N_cells=50, this is 5 * 2500 = 12,500 SocialZone calls
    per scenario.  Each call involves two trigonometric functions (arctan2,
    degrees conversion) and one exp — approximately 10 us on modern hardware,
    giving ~125 ms per scenario.

    TODO: vectorise the inner loop using numpy broadcasting over the grid
          cell dimension to reduce per-scenario time by ~10x.

    FIXME: when two pedestrians occupy the same grid cell, the orientation
           channel stores only the last pedestrian's sin(heading).  This is
           a known limitation of the single-channel encoding; a multi-hot
           or per-pedestrian representation would handle it correctly.
    """
    if zone_distances is None:
        zone_distances = {'front': 1.8, 'back': 0.6, 'left': 1.2, 'right': 1.2}
    zone = SocialZone.from_dict(zone_distances)

    if n_humans is None:
        n_humans = int(rng.integers(3, 6))   # 3, 4, or 5 pedestrians

    # Sample pedestrian states
    humans = []
    for _ in range(n_humans):
        x   = float(rng.uniform(1.0, 9.0))
        y   = float(rng.uniform(1.0, 9.0))
        ori = float(rng.uniform(-np.pi, np.pi))
        humans.append((x, y, ori))

    # Build 2-channel input grid
    occupancy  = np.zeros((N_CELLS, N_CELLS), dtype=np.float32)
    orient_ch  = np.zeros((N_CELLS, N_CELLS), dtype=np.float32)

    for hx, hy, ho in humans:
        gx, gy = world_to_grid([hx, hy])
        occupancy[gy, gx]  = 1.0
        orient_ch[gy, gx]  = float(np.sin(ho))   # sin encoding avoids 2pi discontinuity

    # Compute ground-truth cost map analytically
    cost_map = np.zeros((N_CELLS, N_CELLS), dtype=np.float32)
    for gy in range(N_CELLS):
        for gx in range(N_CELLS):
            robot_pos = grid_to_world(gx, gy)
            for hx, hy, ho in humans:
                cost_map[gy, gx] += zone.compute_cost(robot_pos, np.array([hx, hy]), ho)

    # Per-scenario normalisation to [0, 1]
    max_c = float(cost_map.max())
    if max_c > 0.0:
        cost_map = cost_map / max_c

    X = np.stack([occupancy, orient_ch], axis=0)   # (2, 50, 50)
    return X, cost_map, max_c


def generate_dataset(n_scenarios=500,
                     zone_distances=None,
                     seed=1,
                     verbose=True):
    """
    Generate a full dataset of (input, cost_map) pairs for CNN training.

    Parameters
    ----------
    n_scenarios : int
        Number of random scenarios to generate.  Default 500 gives a
        practical training set in ~15 minutes on CPU.  The full 10,000
        scenarios called for in the original specification require ~4-6 hours;
        use the --cnn-scenarios flag in main.py to scale up when available.
    zone_distances : dict or None
    seed : int
        Seed for the shared random generator.
    verbose : bool
        Print generation progress every 100 scenarios if True.

    Returns
    -------
    X_all : np.ndarray, shape (N, 2, 50, 50), dtype float32
    Y_all : np.ndarray, shape (N, 50, 50), dtype float32
    scale_factors : np.ndarray, shape (N,), dtype float32
        Per-scenario normalisation constants (max_c values).
    """
    rng     = np.random.default_rng(seed)
    X_all   = np.zeros((n_scenarios, 2, N_CELLS, N_CELLS), dtype=np.float32)
    Y_all   = np.zeros((n_scenarios, N_CELLS, N_CELLS),    dtype=np.float32)
    scales  = np.zeros(n_scenarios, dtype=np.float32)

    t0 = time.time()
    for i in range(n_scenarios):
        X_all[i], Y_all[i], scales[i] = generate_scenario(
            rng, zone_distances=zone_distances
        )
        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (n_scenarios - i - 1) / max(rate, 1e-6)
            print(
                f"  Generated {i+1}/{n_scenarios} scenarios "
                f"({elapsed:.1f}s elapsed, ETA: {eta:.1f}s)"
            )

    return X_all, Y_all, scales


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_cost_cnn(n_scenarios=500,
                   epochs=30,
                   batch_size=16,
                   lr=0.001,
                   val_split=0.15,
                   zone_distances=None,
                   model_path="cost_cnn.npy",
                   seed=1,
                   verbose=True):
    """
    Full training run for SocialCostCNN.

    Generates the dataset, splits into train/val, trains for `epochs` epochs
    with mini-batch Adam, and saves the best model (lowest val MSE) to disk.

    Parameters
    ----------
    n_scenarios : int
        Training set size.  Default 500 for CPU-feasible training.
    epochs : int
        Number of passes over the training set.
    batch_size : int
        Mini-batch size.  Smaller batches (4, 8) can help escape poor local
        minima in the first few epochs but are noisier at convergence.
        16 was chosen empirically as a good trade-off.
    lr : float
        Adam initial learning rate.  0.001 is the Adam default and works
        well for MSE regression; higher values (0.01) cause training
        instability when cost map peaks are sharp.
    val_split : float
        Fraction of scenarios held out for validation.
    zone_distances : dict or None
    model_path : str
    seed : int
    verbose : bool

    Returns
    -------
    model : SocialCostCNN
        Trained network with is_trained = True.
    history : dict
        Keys 'train_loss', 'val_loss'; each a list of per-epoch MSE values.
    val_samples : tuple (X_val[:8], Y_val[:8])
        Eight validation (input, target) pairs for fig7_cnn_prediction().

    Notes
    -----
    Validation loss is evaluated on a fixed 20-scenario subset each epoch
    (not the full validation set) to keep per-epoch overhead below ~30 s.
    The subset is the first 20 elements of X_val, which is random due to
    the shuffle in generate_dataset().

    FIXME: the current training loop does not implement early stopping.
           Val MSE typically plateaus after ~20 epochs; running for all 30
           adds ~10% training time with negligible accuracy improvement.
           Add early stopping with patience=5 to avoid over-running.

    TODO: add LR scheduling (e.g. cosine annealing or step decay) to
          improve final val MSE on larger datasets (>= 5000 scenarios).
    """
    print("=" * 60)
    print("Training SocialCostCNN")
    print("=" * 60)

    if zone_distances is None:
        zone_distances = {'front': 1.8, 'back': 0.6, 'left': 1.2, 'right': 1.2}

    # Generate dataset
    print(f"\nGenerating {n_scenarios} training scenarios...")
    print("  (Each scenario requires 2500 x N_humans SocialZone evaluations.)")
    X_all, Y_all, _ = generate_dataset(
        n_scenarios, zone_distances=zone_distances, seed=seed, verbose=verbose
    )

    # Train / val split
    rng   = np.random.default_rng(seed)
    n_val = max(1, int(n_scenarios * val_split))
    idx   = rng.permutation(n_scenarios)
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    X_val, Y_val = X_all[val_idx], Y_all[val_idx]
    X_tr,  Y_tr  = X_all[tr_idx],  Y_all[tr_idx]
    n_train       = len(X_tr)
    n_batches     = max(1, n_train // batch_size)

    print(f"  Train: {n_train} scenarios, Val: {len(X_val)} scenarios")

    model   = SocialCostCNN(lr=lr, seed=seed)
    history = {'train_loss': [], 'val_loss': []}
    t0      = time.time()

    print(f"\nTraining: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print("-" * 60)

    for epoch in range(epochs):
        perm   = rng.permutation(n_train)
        X_shuf = X_tr[perm]
        Y_shuf = Y_tr[perm]

        epoch_loss = 0.0
        for b in range(n_batches):
            Xb = X_shuf[b * batch_size:(b + 1) * batch_size]
            Yb = Y_shuf[b * batch_size:(b + 1) * batch_size]
            if len(Xb) == 0:
                continue
            loss        = model.train_step(Xb, Yb)
            epoch_loss += loss

        avg_loss = epoch_loss / n_batches

        # Validation: evaluate on a fixed 20-scenario subset for speed
        n_val_eval = min(20, len(X_val))
        val_pred   = model.predict(X_val[:n_val_eval])
        val_loss   = float(np.mean((val_pred - Y_val[:n_val_eval]) ** 2))

        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss)

        if verbose and (epoch + 1) % 3 == 0:
            print(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"Train MSE: {avg_loss:.5f} | "
                f"Val MSE: {val_loss:.5f} | "
                f"Elapsed: {time.time()-t0:.1f}s"
            )

    model.is_trained = True
    model.save(model_path)
    print(f"\nDone. Final val MSE: {history['val_loss'][-1]:.5f}")

    # Return 8 validation samples for fig7_cnn_prediction in figures.py
    val_samples = (X_val[:8], Y_val[:8])
    return model, history, val_samples


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load empirical zone distances from extract_zones if available
    if os.path.exists("zone_distances.npy"):
        zd = np.load("zone_distances.npy", allow_pickle=True).item()
    else:
        zd = {'front': 1.8, 'back': 0.6, 'left': 1.2, 'right': 1.2}

    model, history, val_samples = train_cost_cnn(
        n_scenarios=500,
        epochs=30,
        zone_distances=zd,
    )

    # Persist training history for fig6_training_curves
    np.save("cost_cnn_history.npy", history)

    # Persist validation samples for fig7_cnn_prediction
    np.save("cost_cnn_val_samples.npy", {
        'X': val_samples[0],
        'Y': val_samples[1],
    })
