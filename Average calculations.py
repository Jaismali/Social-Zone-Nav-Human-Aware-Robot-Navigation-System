import numpy as np

seeds = [42, 123, 456, 789, 101]
planners = ['asymmetric', 'circular', 'neural']
metrics = ['success_rate', 'collision_rate', 'path_length_mean',
           'violations_mean', 'plan_time_mean']

# Store results
results = {p: {m: [] for m in metrics} for p in planners}

# Load each seed
for seed in seeds:
    try:
        data = np.load(f'experiment_results_seed{seed}.npy', allow_pickle=True).item()
        summary = data['summary']

        for p in planners:
            for m in metrics:
                results[p][m].append(summary[p][m])
    except FileNotFoundError:
        print(f"Warning: experiment_results_seed{seed}.npy not found")

# Calculate averages and std dev
print("\n" + "=" * 70)
print("FINAL AVERAGED RESULTS (over 5 runs)")
print("=" * 70)

for p in planners:
    print(f"\n{p.upper()} PLANNER:")
    print("-" * 50)
    for m in metrics:
        values = results[p][m]
        if values:
            mean = np.mean(values) * (100 if 'rate' in m else 1)
            std = np.std(values) * (100 if 'rate' in m else 1)
            if 'rate' in m:
                print(f"  {m:20s}: {mean:.1f}% ± {std:.1f}%")
            elif 'time' in m:
                print(f"  {m:20s}: {mean * 1000:.1f} ± {std * 1000:.1f} ms")
            else:
                print(f"  {m:20s}: {mean:.2f} ± {std:.2f}")

# Create summary table
print("\n\n" + "=" * 70)
print("SUMMARY TABLE FOR PAPER")
print("=" * 70)
print(f"{'Planner':<12} {'Success %':>12} {'Collision %':>12} {'Path (m)':>10} {'Violations':>12} {'Plan (ms)':>10}")
print("-" * 70)

for p in planners:
    sr = np.mean(results[p]['success_rate']) * 100
    cr = np.mean(results[p]['collision_rate']) * 100
    pl = np.mean(results[p]['path_length_mean'])
    vi = np.mean(results[p]['violations_mean'])
    pt = np.mean(results[p]['plan_time_mean']) * 1000
    print(f"{p:<12} {sr:>11.1f}% {cr:>11.1f}% {pl:>9.2f} {vi:>11.1f} {pt:>9.1f}")