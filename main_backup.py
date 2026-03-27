"""
main.py
Full pipeline for Social-Zone Nav (v2 with neural networks).

Usage:
  python main.py                        # Full pipeline (no neural training)
  python main.py --neural               # Include NeuralPlanner in experiments
  python main.py --train-only           # Train both networks, then exit
  python main.py --quick                # 10-episode smoke test
  python main.py --episodes 20 --neural # 20 episodes with all three planners
  python main.py --skip-download        # Skip dataset download
  python main.py --figures-only         # Regenerate all figures from saved results
  python main.py --figures-only --neural # Include neural figs (5-7)
"""

import argparse
import os
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Social-Zone Nav v2')
    parser.add_argument('--skip-download',     action='store_true')
    parser.add_argument('--skip-extract',      action='store_true')
    parser.add_argument('--skip-experiments',  action='store_true')
    parser.add_argument('--figures-only',      action='store_true')
    parser.add_argument('--train-only',        action='store_true',
                        help='Train neural networks and exit')
    parser.add_argument('--neural',            action='store_true',
                        help='Train neural nets and include NeuralPlanner in experiments')
    parser.add_argument('--quick',             action='store_true',
                        help='10-episode smoke test')
    parser.add_argument('--episodes',          type=int, default=100)
    parser.add_argument('--data-dir',          default='data')
    parser.add_argument('--output-dir',        default='figures')
    parser.add_argument('--zl-epochs',         type=int, default=50,
                        help='ZoneLearnerNet training epochs')
    parser.add_argument('--cnn-epochs',        type=int, default=30,
                        help='SocialCostCNN training epochs')
    parser.add_argument('--cnn-scenarios',     type=int, default=500,
                        help='Number of training scenarios for CNN (full=10000, quick=100)')
    args = parser.parse_args()

    if args.quick:
        args.episodes      = 10
        args.zl_epochs     = 5
        args.cnn_epochs    = 3
        args.cnn_scenarios = 50

    print("=" * 70)
    print("SOCIAL-ZONE NAV v2: Asymmetric + Neural Social Zones")
    print("=" * 70)
    t_total = time.time()

    # ── Step 1: Download ───────────────────────────────────────────────────
    if not args.figures_only and not args.skip_download:
        print("\n[Step 1/6] Downloading ETH/UCY datasets...")
        from download_datasets import download_datasets
        download_datasets(data_dir=args.data_dir)
    else:
        print("\n[Step 1/6] Skipping download.")

    # ── Step 2: Extract zone distances ────────────────────────────────────
    zone_dist_file = "zone_distances.npy"
    if not args.figures_only and not args.skip_extract:
        print("\n[Step 2/6] Extracting social zone distances...")
        from extract_zones import run_extraction
        zone_distances = run_extraction(data_dir=args.data_dir, output_file=zone_dist_file)
        if zone_distances is None:
            zone_distances = {'front': 1.8, 'back': 0.6, 'left': 1.2, 'right': 1.2}
            np.save(zone_dist_file, zone_distances)
    elif os.path.exists(zone_dist_file):
        zone_distances = np.load(zone_dist_file, allow_pickle=True).item()
        print(f"\n[Step 2/6] Loaded zone distances: front={zone_distances['front']:.3f}m "
              f"back={zone_distances['back']:.3f}m sides={zone_distances['left']:.3f}m")
    else:
        zone_distances = {'front': 1.8, 'back': 0.6, 'left': 1.2, 'right': 1.2}
        np.save(zone_dist_file, zone_distances)
        print("\n[Step 2/6] Using default zone distances.")

    # ── Step 3: Train ZoneLearnerNet ──────────────────────────────────────
    zl_history   = None
    cnn_history  = None
    val_samples  = None

    if (args.neural or args.train_only) and not args.figures_only:
        print(f"\n[Step 3/6] Training ZoneLearnerNet ({args.zl_epochs} epochs)...")
        from train_zone_learner import train_zone_learner
        _, zl_history = train_zone_learner(
            data_dir=args.data_dir,
            zone_distances=zone_distances,
            epochs=args.zl_epochs,
            model_path="zone_learner.npy",
        )
        np.save("zone_learner_history.npy", zl_history)
    else:
        print("\n[Step 3/6] Skipping ZoneLearnerNet training.")
        if os.path.exists("zone_learner_history.npy"):
            zl_history = np.load("zone_learner_history.npy", allow_pickle=True).item()

    # ── Step 4: Train SocialCostCNN ───────────────────────────────────────
    if (args.neural or args.train_only) and not args.figures_only:
        print(f"\n[Step 4/6] Training SocialCostCNN "
              f"({args.cnn_scenarios} scenarios, {args.cnn_epochs} epochs)...")
        from train_cost_cnn import train_cost_cnn
        _, cnn_history, val_samples = train_cost_cnn(
            n_scenarios=args.cnn_scenarios,
            epochs=args.cnn_epochs,
            zone_distances=zone_distances,
            model_path="cost_cnn.npy",
        )
        np.save("cost_cnn_history.npy", cnn_history)
        np.save("cost_cnn_val_samples.npy", {'X': val_samples[0], 'Y': val_samples[1]})
    else:
        print("\n[Step 4/6] Skipping SocialCostCNN training.")
        if os.path.exists("cost_cnn_history.npy"):
            cnn_history = np.load("cost_cnn_history.npy", allow_pickle=True).item()

    if args.train_only:
        print("\n[--train-only] Training complete. Exiting.")
        return

    # ── Step 5: Experiments ───────────────────────────────────────────────
    results_file = "experiment_results.npy"

    if not args.figures_only and not args.skip_experiments:
        n_ep = args.episodes
        print(f"\n[Step 5/6] Running {n_ep} navigation episodes...")
        from experiments import run_experiments
        results, summary = run_experiments(
            n_episodes=n_ep,
            zone_distances=zone_distances,
            use_neural=args.neural,
            verbose=True,
        )
        np.save(results_file, {'results': results, 'summary': summary})
        print(f"\n  Results saved to {results_file}")
    elif os.path.exists(results_file):
        data    = np.load(results_file, allow_pickle=True).item()
        summary = data['summary']
        print(f"\n[Step 5/6] Loaded results from {results_file}")
        for k, s in summary.items():
            print(f"  {k}: {s['success_rate']*100:.1f}% success")
    else:
        summary = None
        print("\n[Step 5/6] No results found. Run without --figures-only.")

    # ── Step 6: Figures ───────────────────────────────────────────────────
    print(f"\n[Step 6/6] Generating figures (neural={args.neural or args.figures_only})...")
    from figures import generate_all_figures
    include_neural = args.neural or args.figures_only
    generate_all_figures(
        zone_distances=zone_distances,
        summary=summary,
        output_dir=args.output_dir,
        include_neural=include_neural,
        zl_history=zl_history,
        cnn_history=cnn_history,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE  |  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)

    if summary:
        print(f"\n{'Metric':<28}", end="")
        for k in summary:
            print(f" {k.capitalize():>12}", end="")
        print()
        print("-" * (28 + 13*len(summary)))
        rows = [
            ('Success Rate',      lambda s: f"{s['success_rate']*100:>11.1f}%"),
            ('Collision Rate',    lambda s: f"{s['collision_rate']*100:>11.1f}%"),
            ('Path Length (m)',   lambda s: f"{s['path_length_mean']:>12.2f}"),
            ('Min Dist (m)',      lambda s: f"{s['min_dist_mean']:>12.2f}"),
            ('Violations/ep',    lambda s: f"{s['violations_mean']:>12.1f}"),
            ('Plan Time (ms)',    lambda s: f"{s['plan_time_mean']*1000:>12.1f}"),
        ]
        for label, fn in rows:
            print(f"  {label:<26}", end="")
            for k in summary:
                print(f" {fn(summary[k])}", end="")
            print()

    print(f"\nFigures: {os.path.abspath(args.output_dir)}/")
    print("  fig1–fig4: core results   |   fig5–fig7: neural analysis")


if __name__ == "__main__":
    main()
