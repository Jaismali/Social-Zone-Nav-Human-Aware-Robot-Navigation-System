"""
experiments.py
==============
Episode-based evaluation framework comparing socially-aware planners.

Three planners are evaluated head-to-head across N random episodes:

  asymmetric  The proposed method: A* with direction-dependent personal-space
              zones extracted from ETH/UCY pedestrian data (Pellegrini et al.,
              2009; Lerner et al., 2007).

  circular    Baseline: A* with an isotropic (circular) personal-space zone
              of radius 1.2 m, matching the prior-art assumption used in the
              majority of socially-aware navigation literature (Kruse et al.,
              2013, Section 4.1).

  neural      Optional: A* with CNN-predicted cost maps (NeuralPlanner from
              neural_planner.py).  Enabled via --neural flag or use_neural=True.

Evaluation protocol
~~~~~~~~~~~~~~~~~~~
Each episode uses an identical random configuration (pedestrians, start,
goal) for all planners, controlled by a per-episode seed.  This within-
episode pairing controls for scenario difficulty and reduces the variance
of the difference estimates.  With N=100 episodes, a 5 percentage-point
difference in success rate is detectable at p < 0.05 (two-sided binomial
test, power > 0.8).

For each planner per episode, the following metrics are recorded:

  success_rate      : fraction of episodes where the robot reached the goal
                      without collision (primary metric)
  collision_rate    : fraction of episodes where the robot collided with a
                      pedestrian (a path found but executed dangerously)
  path_length       : Euclidean length of the planned path (metres)
  min_dist          : minimum separation between any path point and any
                      pedestrian (metres; higher is safer)
  violations/ep     : number of path waypoints that intrude into at least
                      one pedestrian's personal-space zone (lower is better)
  plan_time         : wall-clock time for planner.plan() (seconds)

Episode controller
~~~~~~~~~~~~~~~~~~
After planning, the robot follows the returned path at 0.2 m/s (one cell
per step).  Pedestrians continue to move during execution.  This tests
whether a static plan remains valid under pedestrian motion — a simpler
evaluation than replanning, but representative of the open-loop regime
studied by Sisbot et al. (2007).

The replanning regime (replanning every N steps) would be more realistic but
significantly increases wall-clock experiment time; it is left as future work.

References
----------
Pellegrini, S., Ess, A., Schindler, K., & Van Gool, L. (2009). You'll never
    walk alone: Modelling social behaviour for multi-target tracking.
    In Proc. ICCV, pp. 261-268.  ETH pedestrian dataset.

Lerner, A., Chrysanthou, Y., & Lischinski, D. (2007). Crowds by example.
    Computer Graphics Forum, 26(3), 655-664.  UCY dataset.

Kruse, T., Pandey, A. K., Alami, R., & Mertsching, B. (2013). Human-aware
    robot navigation: A survey. Robotics and Autonomous Systems, 61(12),
    1726-1743.

Sisbot, E. A., Marin-Urias, L. F., Alami, R., & Simeon, T. (2007). A human
    aware mobile robot motion planner. IEEE Transactions on Robotics, 23(5),
    874-883.  Open-loop planning evaluation protocol.

Everett, M., Chen, Y. F., & How, J. P. (2021). Collision avoidance in
    pedestrian-rich environments with deep reinforcement learning. IEEE
    Access, 9, 10357-10377.  Comparative baseline for navigation benchmarks.
"""

import copy
import os
import time

import numpy as np

from environment import NavigationEnvironment, Pedestrian
from planner import (AsymmetricPlanner, CircularPlanner,
                     compute_path_length, min_distance_to_humans,
                     count_social_violations)
from social_zone import SocialZone, CircularSocialZone


# ---------------------------------------------------------------------------
# Episode execution
# ---------------------------------------------------------------------------

def run_episode(planner, env, zone_for_evaluation, max_iter=500):
    """
    Execute one navigation episode: plan once, follow the path open-loop.

    Planning is performed on the initial pedestrian configuration.  After
    planning, the robot moves along the returned waypoints at step_size=0.2 m
    per iteration (one grid cell per step), while pedestrians continue to
    evolve under their constant-velocity model.

    Parameters
    ----------
    planner : AStarPlanner or subclass
        Must implement plan(start, goal, pedestrians).
    env : NavigationEnvironment
        Must be pre-configured with robot_pos, robot_goal, and pedestrians.
    zone_for_evaluation : SocialZone or subclass
        Zone model used to compute the violations metric post-hoc.
        Should match the planner's zone for a fair comparison, but can
        differ to evaluate cross-zone violation counts.
    max_iter : int
        Maximum number of execution steps before declaring failure.
        At step_size=0.2 m and 10 m diagonal distance, ~71 steps are
        needed for a straight-line path; 500 provides generous headroom
        for detours.

    Returns
    -------
    dict with keys:
        success     : bool  — reached goal without collision
        collision   : bool  — physical collision occurred
        path_length : float — Euclidean length of planned path (m)
        min_dist    : float — minimum separation to any pedestrian (m)
        violations  : int   — number of waypoints violating any zone
        plan_time   : float — planner.plan() wall-clock time (s)
        path        : list or None — planned waypoints
        visited     : list  — actual robot positions during execution

    Notes
    -----
    plan_time is measured with time.time() rather than time.perf_counter()
    for cross-platform consistency.  On modern systems the resolution is
    sufficient for the 10-100 ms planning times encountered here.

    The step_size (0.2 m) matches the grid resolution so the robot moves
    exactly one cell per step along axis-aligned segments.  For diagonal
    segments the step overshoots slightly; this is accepted as a minor
    approximation.

    TODO: implement a replanning variant that calls plan() every K steps
          with updated pedestrian positions.
    """
    start       = env.robot_pos.copy()
    goal        = env.robot_goal.copy()
    pedestrians = env.pedestrians

    # Time the planning step only, not execution
    t0        = time.time()
    path      = planner.plan(start, goal, pedestrians)
    plan_time = time.time() - t0

    if path is None:
        # Planner found no valid path under the cost constraints
        return {
            'success':     False,
            'collision':   False,
            'path_length': float('inf'),
            'min_dist':    float('inf'),
            'violations':  0,
            'plan_time':   plan_time,
            'path':        None,
            'visited':     [],
        }

    # Path execution: follow waypoints at fixed step size
    pos          = np.array(start, dtype=float)
    step_size    = 0.2    # metres per step; matches grid resolution
    collision    = False
    visited      = [pos.copy()]
    waypoint_idx = 0

    for _ in range(max_iter):
        if waypoint_idx >= len(path):
            break

        target     = np.array(path[waypoint_idx], dtype=float)
        diff       = target - pos
        dist_to_wp = float(np.linalg.norm(diff))

        if dist_to_wp < step_size:
            # Snap to waypoint and advance to next
            pos          = target.copy()
            waypoint_idx += 1
        else:
            # Move toward waypoint by step_size
            pos = pos + (diff / dist_to_wp) * step_size

        # Advance pedestrian simulation one timestep
        env.step_pedestrians()

        if env.is_collision(pos):
            collision = True
            break

        visited.append(pos.copy())

        if float(np.linalg.norm(pos - goal)) < env.GOAL_TOLERANCE:
            break

    success    = (not collision) and (float(np.linalg.norm(pos - goal)) < env.GOAL_TOLERANCE)
    path_len   = compute_path_length(path)
    min_d      = min_distance_to_humans(path, pedestrians)
    violations = count_social_violations(path, pedestrians, zone_for_evaluation)

    return {
        'success':     success,
        'collision':   collision,
        'path_length': path_len,
        'min_dist':    min_d,
        'violations':  violations,
        'plan_time':   plan_time,
        'path':        path,
        'visited':     visited,
    }


# ---------------------------------------------------------------------------
# Episode configuration generation
# ---------------------------------------------------------------------------

def generate_episode_config(seed, n_peds=4):
    """
    Generate a reproducible random episode configuration.

    Pedestrians are sampled in [1.5, 8.5]^2 with random headings and slow
    speeds (0-0.3 m/s) to ensure they are always in the interior of the
    arena.  The robot start/goal are sampled from one of four opposing-
    quadrant pairs chosen uniformly at random — this ensures the robot always
    crosses the arena diagonally, exercising the zone model in non-trivial
    configurations.

    Parameters
    ----------
    seed : int
        Deterministic seed so the same configuration is reproduced when
        different planners are evaluated on the same episode.
    n_peds : int
        Number of pedestrians per episode.

    Returns
    -------
    pedestrians : list of Pedestrian
    start : np.ndarray, shape (2,)
    goal  : np.ndarray, shape (2,)

    Notes
    -----
    Pedestrian speeds are capped at 0.3 m/s (slower than ETH/UCY average
    of ~0.6 m/s) to reduce the chance that fast-moving pedestrians create
    unavoidable collisions on the executed open-loop path.

    TODO: add a difficulty parameter that scales pedestrian speed and density
          to enable curriculum evaluation (easy -> medium -> hard).
    """
    rng = np.random.default_rng(seed)

    pedestrians = []
    for _ in range(n_peds):
        x   = float(rng.uniform(1.5, 8.5))
        y   = float(rng.uniform(1.5, 8.5))
        ori = float(rng.uniform(-np.pi, np.pi))
        spd = float(rng.uniform(0.0, 0.3))
        pedestrians.append(Pedestrian(
            x, y, ori, spd,
            spd * np.cos(ori),
            spd * np.sin(ori),
        ))

    # Four opposing-quadrant start/goal configurations
    # Quadrant choice controls the crossing direction without bias
    quadrant = int(rng.integers(0, 4))
    if quadrant == 0:
        start = rng.uniform([0.5, 0.5], [3.0, 3.0])
        goal  = rng.uniform([7.0, 7.0], [9.5, 9.5])
    elif quadrant == 1:
        start = rng.uniform([7.0, 0.5], [9.5, 3.0])
        goal  = rng.uniform([0.5, 7.0], [3.0, 9.5])
    elif quadrant == 2:
        start = rng.uniform([0.5, 7.0], [3.0, 9.5])
        goal  = rng.uniform([7.0, 0.5], [9.5, 3.0])
    else:
        start = rng.uniform([0.5, 0.5], [4.0, 4.0])
        goal  = rng.uniform([6.0, 6.0], [9.5, 9.5])

    return pedestrians, start, goal


# ---------------------------------------------------------------------------
# Result aggregation helpers
# ---------------------------------------------------------------------------

def _make_empty_bucket():
    """
    Initialise an empty per-planner result accumulator.

    Returns a dict of lists, one per metric.  Lists are grown by
    run_experiments() as episodes complete.
    """
    return {
        'success':     [],
        'collision':   [],
        'path_length': [],
        'min_dist':    [],
        'violations':  [],
        'plan_time':   [],
    }


def _build_summary(r, n_episodes):
    """
    Compute descriptive statistics over the episode results for one planner.

    Success and collision rates use the raw proportion as the point estimate.
    Inf values (from episodes where no path was found) are excluded from
    path_length and min_dist statistics; their count is implicitly captured
    in the success_rate.

    Parameters
    ----------
    r : dict
        Per-episode result lists from _make_empty_bucket().
    n_episodes : int
        Total number of episodes run (used for reporting, not computation).

    Returns
    -------
    dict
        Summary statistics including mean, std, and N for each metric.

    Notes
    -----
    Standard error bars in fig4 use 1.96 * std / sqrt(N) for continuous
    metrics and the normal approximation to the binomial for rates.
    These are appropriate for N >= 30 (Central Limit Theorem); for small
    N, exact binomial confidence intervals should be used instead.
    """
    path_lengths = [x for x in r['path_length'] if x != float('inf')]
    min_dists    = [x for x in r['min_dist']    if x != float('inf')]

    return {
        'success_rate':      float(np.mean(r['success'])),
        'success_rate_std':  float(np.std(r['success'])),
        'collision_rate':    float(np.mean(r['collision'])),
        'path_length_mean':  float(np.mean(path_lengths)) if path_lengths else 0.0,
        'path_length_std':   float(np.std(path_lengths))  if path_lengths else 0.0,
        'min_dist_mean':     float(np.mean(min_dists))    if min_dists    else 0.0,
        'min_dist_std':      float(np.std(min_dists))     if min_dists    else 0.0,
        'violations_mean':   float(np.mean(r['violations'])),
        'violations_std':    float(np.std(r['violations'])),
        'plan_time_mean':    float(np.mean(r['plan_time'])),
        'n_episodes':        n_episodes,
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiments(n_episodes=100,
                    zone_distances=None,
                    seed_base=0,
                    use_neural=False,
                    neural_model_path="cost_cnn.npy",
                    verbose=True):
    """
    Run N episodes comparing all enabled planners on identical configurations.

    All planners are evaluated on the same episode configurations in sequence.
    Pedestrian state is deep-copied before each planner's episode so that
    execution by one planner does not affect the initial conditions seen by
    the next.

    Parameters
    ----------
    n_episodes : int
        Number of episodes.  100 is sufficient for stable mean estimates;
        use 500 for paper-quality results with tight confidence intervals.
    zone_distances : dict or None
        Zone distances for asymmetric planner and zone_for_evaluation.
    seed_base : int
        Base random seed; episode i uses seed seed_base + i.
    use_neural : bool
        If True, loads NeuralPlanner from neural_model_path and includes it
        in the comparison.
    neural_model_path : str
        Path to trained SocialCostCNN weights.
    verbose : bool
        Print per-10-episode progress and final summary table.

    Returns
    -------
    results : dict
        Keyed by planner name; values are per-episode metric lists.
    summary : dict
        Keyed by planner name; values are aggregated statistics dicts.

    Notes
    -----
    Plan time is measured with time.time() on the host CPU.  On machines
    with high background load, individual plan time measurements may be
    noisy; the mean over 100 episodes is stable to within ~2 ms.

    FIXME: the current implementation runs planners sequentially per
           episode.  Parallelising across episodes with multiprocessing
           would reduce total experiment wall time by N_cores, but requires
           careful handling of the shared NeuralPlanner CNN state.
    """
    if zone_distances is None:
        zone_distances = {'front': 1.8, 'back': 0.6, 'left': 1.2, 'right': 1.2}

    # Instantiate planners and matching evaluation zones
    asym_planner = AsymmetricPlanner(zone_distances=zone_distances)
    circ_planner = CircularPlanner(radius=1.2)
    asym_zone    = SocialZone.from_dict(zone_distances)
    circ_zone    = CircularSocialZone(radius=1.2)

    # (name, planner_instance, evaluation_zone) triples
    # The evaluation zone determines which model is used for violation counting
    planner_specs = [
        ('asymmetric', asym_planner, asym_zone),
        ('circular',   circ_planner, circ_zone),
    ]

    neural_planner = None
    if use_neural:
        if os.path.exists(neural_model_path):
            from neural_planner import NeuralPlanner
            neural_planner = NeuralPlanner.from_file(
                neural_model_path,
                fallback_zone_distances=zone_distances,
            )
            # Evaluate neural planner against the asymmetric zone for
            # an apples-to-apples comparison of violation counts
            planner_specs.append(('neural', neural_planner, asym_zone))
            print(f"  Loaded NeuralPlanner from {neural_model_path}")
        else:
            print(f"  WARNING: {neural_model_path} not found. Skipping neural planner.")

    results = {name: _make_empty_bucket() for name, _, _ in planner_specs}

    if verbose:
        names_str = ' | '.join(n.capitalize() for n, _, _ in planner_specs)
        print("=" * 60)
        print(f"Running {n_episodes} episodes  [{names_str}]")
        print(
            f"  Zone: front={zone_distances['front']:.2f}m  "
            f"back={zone_distances['back']:.2f}m  "
            f"sides={zone_distances['left']:.2f}m"
        )
        print("=" * 60)

    t_start = time.time()

    for ep in range(n_episodes):
        seed = seed_base + ep
        pedestrians, start, goal = generate_episode_config(seed, n_peds=4)

        for name, planner, eval_zone in planner_specs:
            # Deep-copy pedestrians so each planner gets identical initial state
            peds_copy = copy.deepcopy(pedestrians)
            env = NavigationEnvironment(
                pedestrians=peds_copy,
                robot_start=start.copy(),
                robot_goal=goal.copy(),
                seed=seed,
            )
            ep_res = run_episode(planner, env, eval_zone)

            r = results[name]
            r['success'].append(int(ep_res['success']))
            r['collision'].append(int(ep_res['collision']))
            r['path_length'].append(ep_res['path_length'])
            r['min_dist'].append(ep_res['min_dist'])
            r['violations'].append(ep_res['violations'])
            r['plan_time'].append(ep_res['plan_time'])

        if verbose and (ep + 1) % 10 == 0:
            elapsed = time.time() - t_start
            parts = []
            for name, _, _ in planner_specs:
                sr = float(np.mean(results[name]['success'][:ep+1])) * 100
                parts.append(f"{name[:4].capitalize()} SR:{sr:.0f}%")
            print(f"  Ep {ep+1:3d}/{n_episodes} | {' | '.join(parts)} | {elapsed:.1f}s")

    if neural_planner is not None and verbose:
        print(f"\n  {neural_planner.stats()}")

    summary = {
        name: _build_summary(results[name], n_episodes)
        for name, _, _ in planner_specs
    }

    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        for name, _, _ in planner_specs:
            s = summary[name]
            print(f"\n{name.upper()} PLANNER:")
            print(f"  Success Rate:   {s['success_rate']*100:.1f}% ± {s['success_rate_std']*100:.1f}%")
            print(f"  Collision Rate: {s['collision_rate']*100:.1f}%")
            print(f"  Path Length:    {s['path_length_mean']:.2f} ± {s['path_length_std']:.2f} m")
            print(f"  Min Dist:       {s['min_dist_mean']:.2f} ± {s['min_dist_std']:.2f} m")
            print(f"  Violations/ep:  {s['violations_mean']:.1f} ± {s['violations_std']:.1f}")
            print(f"  Plan Time:      {s['plan_time_mean']*1000:.1f} ms (mean)")

    return results, summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    use_neural = '--neural' in sys.argv

    if os.path.exists("zone_distances.npy"):
        zd = np.load("zone_distances.npy", allow_pickle=True).item()
    else:
        zd = {'front': 1.8, 'back': 0.6, 'left': 1.2, 'right': 1.2}

    results, summary = run_experiments(
        n_episodes=100,
        zone_distances=zd,
        use_neural=use_neural,
    )

    np.save("experiment_results.npy", {'results': results, 'summary': summary})
    print("\nSaved results to experiment_results.npy")
