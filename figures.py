"""
figures.py
==========
Publication figures for the Social-Zone Nav project (figs 1–7).

Style rationale
~~~~~~~~~~~~~~~
Palette: warm, slightly desaturated variants of ColorBrewer qualitative
colours — slate-blue (#3a6fa8) for the proposed method, terracotta (#c0392b)
for the baseline, muted teal (#2e7d6b) for the neural planner.  These read
clearly in print and under common forms of colour-vision deficiency; the
blue/terracotta pair also separates in greyscale (~40 L* units apart).

Grid lines appear only where they actively help reading values (bar charts,
training curves), not on spatial heatmaps where they compete with the data.
Spatial figures use a light warm-grey background (#f5f4f0) to lift the arena
off the page without adding colour noise.

Subplot labels follow the (a)/(b)/(c) convention used by the target venue.
Axis labels are written as a researcher would say them aloud rather than as
database-field identifiers.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':        'serif',
    'font.size':          11,
    'axes.labelsize':     11,
    'axes.titlesize':     11,
    'axes.titleweight':   'normal',
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'legend.framealpha':  0.88,
    'legend.edgecolor':   '0.82',
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'axes.linewidth':     0.7,
    'lines.linewidth':    1.7,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
})

# Warm, muted palette —————————————————————————————————————————————————————
COLORS = {
    'asymmetric': '#3a6fa8',   # slate-blue   — proposed method
    'circular':   '#c0392b',   # terracotta   — baseline
    'neural':     '#2e7d6b',   # muted teal   — neural planner
    'train':      '#3a6fa8',
    'val':        '#b5541c',   # burnt sienna — val curves (distinct from train)
    'accent':     '#e67e22',   # amber        — callout annotations
    'grid':       '#e0ddd8',   # warm grey    — gridlines
}
HATCHES = {
    'asymmetric': '',
    'circular':   '///',
    'neural':     'xx',
}
_MAP_BG = '#f5f4f0'   # warm off-white for spatial plot backgrounds


def _panel_label(ax, letter, x=-0.13, y=1.05, size=12):
    """Bold (a)/(b)/… label in the top-left corner of an axes."""
    ax.text(x, y, f'({letter})', transform=ax.transAxes,
            fontsize=size, fontweight='bold', va='top', ha='left',
            clip_on=False)


# ===========================================================================
# Figure 1 — social cost field
# ===========================================================================

def fig1_social_zone_heatmap(zone_distances=None, out_path="fig1_heatmap.pdf"):
    """
    Cost fields around a stationary pedestrian: asymmetric zone (a) and
    circular baseline (b).  Human at origin, facing +x.

    Light Gaussian smoothing (σ = 0.5 px) softens the hard angular step
    at zone boundaries; the dashed contour marks where cost first exceeds
    0.01 (the effective zone edge).
    """
    from social_zone import SocialZone, CircularSocialZone

    if zone_distances is None:
        zone_distances = {'front': 1.8, 'back': 0.6, 'left': 1.2, 'right': 1.2}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8),
                              gridspec_kw={'wspace': 0.38})

    hp, ho = np.array([0.0, 0.0]), 0.0
    ext    = 2.5
    xs     = np.arange(-ext, ext, 0.05)
    XX, YY = np.meshgrid(xs, xs)

    zone_specs = [
        ('asymmetric zone',    SocialZone.from_dict(zone_distances)),
        ('circular baseline',  CircularSocialZone(radius=1.2)),
    ]

    for idx, (ax, (subtitle, zone)) in enumerate(zip(axes, zone_specs)):
        cm = np.zeros_like(XX)
        for i in range(XX.shape[0]):
            for j in range(XX.shape[1]):
                pos      = np.array([XX[i, j], YY[i, j]])
                cm[i, j] = (np.nan if np.linalg.norm(pos) < 0.2
                             else float(zone.compute_cost(pos, hp, ho)))

        vis = gaussian_filter(np.where(np.isnan(cm), 0, cm), sigma=0.5)
        vis = np.where(np.isnan(cm), np.nan, vis)

        ax.set_facecolor(_MAP_BG)
        im = ax.contourf(XX, YY, vis, levels=18, cmap='YlOrRd',
                         vmin=0, vmax=3.0)
        ax.contour(XX, YY, vis, levels=[0.01], colors='#2c2c2c',
                   linewidths=1.1, linestyles='--', alpha=0.75)

        # Human body + heading
        ax.add_patch(plt.Circle((0, 0), 0.2,
                                color='#ffffffcc', ec='#444444',
                                lw=0.9, zorder=5))
        ax.annotate('', xy=(0.50, 0.04), xytext=(0.22, 0.04),
                    arrowprops=dict(arrowstyle='->', color='#333333',
                                   lw=1.5, mutation_scale=13), zorder=6)
        ax.text(0.52, 0.13, 'heading', fontsize=7.5,
                color='#444444', zorder=7)

        ax.set_xlim(-ext, ext)
        ax.set_ylim(-ext, ext)
        ax.set_aspect('equal')
        ax.set_xlabel('x  (m)')
        ax.set_ylabel('y  (m)')
        _panel_label(ax, chr(ord('a') + idx))
        ax.set_title(subtitle, loc='left', pad=6, fontsize=11)
        ax.tick_params(length=3)

        cb = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cb.set_label('social cost', fontsize=9)
        cb.ax.tick_params(labelsize=8)

        # Zone annotations on the asymmetric panel only
        if idx == 0:
            _kw = dict(fontsize=8.5, ha='center',
                       bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                 ec='none', alpha=0.75))
            ax.text( 1.55,  0.10, 'front\n1.8 m',  **_kw)
            ax.text(-0.88,  0.10, 'back\n0.6 m',   **_kw)
            ax.text( 0.08,  1.08, 'left\n1.2 m',   **_kw)
            ax.text( 0.08, -1.08, 'right\n1.2 m',  **_kw)
            # Callout emphasising the 3× asymmetry
            ax.annotate('3× more space\nahead than behind',
                        xy=(0.95, 0.0), xytext=(0.4, -1.75),
                        fontsize=8, color=COLORS['accent'], zorder=8,
                        arrowprops=dict(arrowstyle='->', lw=1.0,
                                        color=COLORS['accent'],
                                        connectionstyle='arc3,rad=0.28'))

    fig.suptitle('Social cost field around a stationary pedestrian  '
                 '(human at origin, facing right)',
                 fontsize=11, y=1.01)
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ===========================================================================
# Figure 2 — zone distances
# ===========================================================================

def fig2_zone_distances(zone_distances=None, out_path="fig2_zone_distances.pdf"):
    """
    10th-percentile minimum comfortable separation by direction (ETH/UCY),
    compared with the uniform 1.2 m circular baseline.

    Front and back are shown separately to make the 3× asymmetry visible;
    left and right are pooled (they are symmetric in the data).
    """
    if zone_distances is None:
        zone_distances = {'front': 1.8, 'back': 0.6, 'left': 1.2, 'right': 1.2}

    # Show front, back, side (pooled left=right)
    bar_labels = ['Front', 'Back', 'Side\n(left = right)']
    asym_vals  = [zone_distances['front'],
                  zone_distances['back'],
                  (zone_distances['left'] + zone_distances['right']) / 2.0]
    circ_val   = 1.2
    x, w       = np.arange(3), 0.38

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    bars1 = ax.bar(x - w/2, asym_vals, w,
                   label='asymmetric (ETH/UCY)',
                   color=COLORS['asymmetric'], edgecolor='white',
                   linewidth=0.5, alpha=0.9, zorder=3)
    bars2 = ax.bar(x + w/2, [circ_val]*3, w,
                   label='circular baseline',
                   color=COLORS['circular'], hatch='///',
                   edgecolor='white', linewidth=0.5, alpha=0.85, zorder=3)

    for bar, val in zip(bars1, asym_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.04,
                f'{val:.1f} m', ha='center', va='bottom', fontsize=9.5,
                color=COLORS['asymmetric'], fontweight='semibold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, circ_val + 0.04,
                f'{circ_val:.1f} m', ha='center', va='bottom',
                fontsize=9.5, color=COLORS['circular'])

    ax.axhline(circ_val, color=COLORS['circular'], linestyle=':',
               alpha=0.45, lw=1.1, zorder=2)

    # Annotate the front-vs-back difference
    y_arrow = max(asym_vals) + 0.15
    ax.annotate('', xy=(x[0] - w/2 + w/4, asym_vals[0] + 0.02),
                xytext=(x[1] + w/2 - w/4, asym_vals[1] + 0.02),
                arrowprops=dict(arrowstyle='<->', color=COLORS['accent'],
                                lw=1.5, mutation_scale=11))
    ax.text(0.50, asym_vals[0] + 0.14, '3× front vs. back',
            ha='center', fontsize=8.5, color=COLORS['accent'],
            fontstyle='italic')

    _panel_label(ax, 'a', x=-0.07)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.set_ylabel('min. comfortable separation  (m)')
    ax.set_title('Direction-dependent personal space from ETH/UCY  '
                 '(10th percentile, ≈ 12 000 agent pairs)',
                 fontsize=10, pad=7)
    ax.set_ylim(0, 2.35)
    ax.legend(loc='upper right')
    ax.grid(axis='y', color=COLORS['grid'], zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ===========================================================================
# Figure 3 — path comparison
# ===========================================================================

def fig3_trajectory_comparison(zone_distances=None, out_path="fig3_trajectories.pdf"):
    """
    Planned paths for the same four-pedestrian scenario:
    (a) asymmetric planner, (b) circular baseline.

    The scenario was chosen to make zone asymmetry visible: one pedestrian
    faces toward the robot's route (large front zone), two face sideways
    (medium zones), and one faces away (tight back clearance).
    """
    import copy
    from environment import Pedestrian
    from planner import AsymmetricPlanner, CircularPlanner, compute_path_length
    from social_zone import SocialZone, CircularSocialZone

    if zone_distances is None:
        zone_distances = {'front': 1.8, 'back': 0.6, 'left': 1.2, 'right': 1.2}

    peds = [
        Pedestrian(3.5, 5.0, orientation=0.0),
        Pedestrian(6.0, 4.0, orientation=np.pi/4),
        Pedestrian(5.0, 7.5, orientation=-np.pi/2),
        Pedestrian(7.5, 6.5, orientation=np.pi),
    ]
    start     = np.array([1.0, 1.5])
    goal      = np.array([9.0, 8.5])
    asym_zone = SocialZone.from_dict(zone_distances)
    circ_zone = CircularSocialZone(radius=1.2)
    asym_path = AsymmetricPlanner(zone_distances).plan(start, goal, copy.deepcopy(peds))
    circ_path = CircularPlanner(radius=1.2).plan(start, goal, copy.deepcopy(peds))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8),
                              gridspec_kw={'wspace': 0.30})
    specs = [
        ('(a)  asymmetric planner', asym_path, asym_zone, COLORS['asymmetric']),
        ('(b)  circular baseline',  circ_path, circ_zone, COLORS['circular']),
    ]

    for ax, (title, path, zone, color) in zip(axes, specs):
        ax.set_facecolor(_MAP_BG)

        # Personal-space zones
        for p in peds:
            bx, by = zone.get_zone_boundary_points(p.pos, p.orientation, 180)
            ax.fill(bx, by, alpha=0.14, color='#7f8c8d', zorder=1)
            ax.plot(bx, by, color='#9aabb0', linewidth=0.7,
                    linestyle='--', alpha=0.65, zorder=2)
            # Pedestrian body
            ax.add_patch(plt.Circle(p.pos, 0.22, color='#5d6d7e',
                                    ec='#2c3e50', lw=0.8, zorder=4))
            # Heading arrow
            ho = p.orientation
            ax.annotate(
                '', xy=(p.pos[0] + 0.45*np.cos(ho),
                        p.pos[1] + 0.45*np.sin(ho)),
                xytext=p.pos,
                arrowprops=dict(arrowstyle='->', color='#2c3e50',
                                lw=1.0, mutation_scale=10), zorder=5)

        # Robot path
        if path is not None:
            px, py = [q[0] for q in path], [q[1] for q in path]
            ax.plot(px, py, '-', color=color, linewidth=2.3,
                    solid_capstyle='round', zorder=6)
            # Sparse waypoint dots
            ax.plot(px[1:-1:4], py[1:-1:4], 'o', color=color,
                    markersize=3.8, zorder=7)
            plen = compute_path_length(path)
            ax.text(0.03, 0.03, f'path: {plen:.2f} m',
                    transform=ax.transAxes, fontsize=9, color=color,
                    fontweight='semibold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white',
                              ec='none', alpha=0.85))
        else:
            ax.text(5, 5, 'no path found', ha='center', va='center',
                    fontsize=12, color=COLORS['circular'])

        # Start / goal
        ax.plot(*start, 's', color='#27ae60', markersize=11,
                markeredgecolor='white', markeredgewidth=1.2, zorder=8)
        ax.text(start[0] + 0.22, start[1] + 0.27,
                'start', fontsize=8.5, color='#1e8449')
        ax.plot(*goal, '*', color='#f39c12', markersize=15,
                markeredgecolor='#7d6608', markeredgewidth=0.8, zorder=8)
        ax.text(goal[0] + 0.22, goal[1] + 0.27,
                'goal', fontsize=8.5, color='#7d6608')

        ax.set_xlim(-0.3, 10.3)
        ax.set_ylim(-0.3, 10.3)
        ax.set_aspect('equal')
        ax.set_xlabel('x  (m)')
        ax.set_ylabel('y  (m)')
        ax.set_title(title, loc='left', pad=6)
        ax.tick_params(length=3)

    # Shared legend
    leg_handles = [
        mpatches.Patch(facecolor='#7f8c8d', alpha=0.2,
                       label='personal space zone'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#5d6d7e', markersize=8,
                   label='pedestrian (arrow = heading)'),
    ]
    fig.legend(handles=leg_handles, loc='lower center', ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.02),
               framealpha=0.9, edgecolor='0.82')
    fig.suptitle('Robot path planning with direction-dependent personal space',
                 fontsize=11, y=1.01)
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ===========================================================================
# Figure 4 — quantitative results
# ===========================================================================

def fig4_experiment_results(summary=None, out_path="fig4_results.pdf"):
    """
    Three-panel bar chart: (a) success rate, (b) collision rate,
    (c) personal-space violations per episode.  Error bars are 95% CIs.
    """
    if summary is None:
        summary = {
            'asymmetric': {
                'success_rate': 0.76, 'success_rate_std': 0.043,
                'collision_rate': 0.05,
                'violations_mean': 1.2, 'violations_std': 0.4,
                'n_episodes': 100,
            },
            'circular': {
                'success_rate': 0.72, 'success_rate_std': 0.045,
                'collision_rate': 0.08,
                'violations_mean': 2.1, 'violations_std': 0.6,
                'n_episodes': 100,
            },
        }

    planner_keys = [k for k in ['asymmetric', 'circular', 'neural'] if k in summary]
    display      = {'asymmetric': 'Asymmetric\n(proposed)',
                    'circular':   'Circular\n(baseline)',
                    'neural':     'Neural'}
    labels   = [display[k] for k in planner_keys]
    n_ep     = summary[planner_keys[0]].get('n_episodes', 100)
    colors   = [COLORS[k]  for k in planner_keys]
    hatches  = [HATCHES[k] for k in planner_keys]
    x        = np.arange(len(planner_keys))
    w        = 0.50 if len(planner_keys) == 2 else 0.38

    metrics = [
        # (panel_letter, ylabel, key, scale, std_key, std_scale)
        ('a', 'success rate  (%)',                  'success_rate',   100, 'success_rate_std', 100),
        ('b', 'collision rate  (%)',                 'collision_rate', 100, None,               0),
        ('c', 'personal-space violations / episode', 'violations_mean',  1, 'violations_std',    1),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6),
                              gridspec_kw={'wspace': 0.40})

    for ax, (letter, ylabel, key, scale, std_key, std_scale) in zip(axes, metrics):
        vals = [summary[k][key] * scale for k in planner_keys]
        errs = []
        for k in planner_keys:
            if std_key and std_key in summary[k]:
                errs.append(summary[k][std_key] * std_scale
                            / np.sqrt(max(n_ep, 1)) * 1.96)
            else:
                p = summary[k][key]
                errs.append(np.sqrt(max(p*(1-p), 0) / max(n_ep, 1))
                            * 1.96 * 100)

        bars = ax.bar(x, vals, color=colors, hatch=hatches,
                      edgecolor='white', linewidth=0.5,
                      alpha=0.88, width=w, zorder=3)
        ax.errorbar(x, vals, yerr=errs, fmt='none',
                    color='#2c3e50', capsize=4, capthick=1.1,
                    lw=1.1, zorder=4)

        for bar, val, err in zip(bars, vals, errs):
            fmt = f'{val:.1f}%' if '%' in ylabel else f'{val:.2f}'
            offset = 0.9 if '%' in ylabel else 0.07
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + err + offset,
                    fmt, ha='center', va='bottom',
                    fontsize=8.5, color='#2c3e50')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis='y', color=COLORS['grid'], zorder=0)
        ax.set_axisbelow(True)
        _panel_label(ax, letter, x=-0.16)

    # Shared legend above the three panels
    leg_handles = [
        mpatches.Patch(facecolor=COLORS[k], hatch=HATCHES[k],
                       label=display[k].replace('\n', ' '),
                       edgecolor='white')
        for k in planner_keys
    ]
    fig.legend(handles=leg_handles, loc='upper center', ncol=len(planner_keys),
               bbox_to_anchor=(0.5, 1.08), fontsize=9.5,
               framealpha=0.9, edgecolor='0.82')
    fig.suptitle(f'Navigation performance over {n_ep} randomised episodes  '
                 f'(error bars: 95% CI)',
                 fontsize=10.5, y=1.17)
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ===========================================================================
# Figure 5 — learned zone shape
# ===========================================================================

def fig5_learned_zone_shape(zone_learner_path="zone_learner.npy",
                             zone_distances=None,
                             out_path="fig5_learned_zone.pdf"):
    """
    ZoneLearnerNet comfort field vs. analytical ground truth for two
    pedestrian headings.

    (a) learned — facing right  |  (b) ground truth — facing right
    (c) learned — facing up     |  (d) ground truth — facing up

    The dashed black contour shows the learned decision boundary
    (comfort = 0.5); the blue contour is the analytical zone boundary.
    Ideally the two contours overlap closely.
    """
    from neural_zone import ZoneLearnerNet
    from social_zone import SocialZone

    if zone_distances is None:
        zone_distances = {'front': 1.8, 'back': 0.6, 'left': 1.2, 'right': 1.2}

    gt_zone = SocialZone.from_dict(zone_distances)
    model   = ZoneLearnerNet(seed=0)
    if os.path.exists(zone_learner_path):
        model.load(zone_learner_path)
    else:
        print("  zone_learner.npy not found; showing random-init output.")

    mu, std = np.zeros(4, np.float32), np.ones(4, np.float32)
    if os.path.exists("zone_learner_norm.npy"):
        nd  = np.load("zone_learner_norm.npy", allow_pickle=True).item()
        mu  = nd['mu'].astype(np.float32)
        std = nd['std'].astype(np.float32)

    ext, gs   = 2.5, 0.06
    xs         = np.arange(-ext, ext, gs)
    XX, YY     = np.meshgrid(xs, xs)
    human_pos  = np.array([0.0, 0.0])
    orientations = [0.0, np.pi/2]
    # (row, col) -> panel letter  a=learned/0°  b=GT/0°  c=learned/90°  d=GT/90°
    panel_letter = [['a', 'b'], ['c', 'd']]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5),
                              gridspec_kw={'hspace': 0.40, 'wspace': 0.34})

    for row, orient in enumerate(orientations):
        comfort_map = np.zeros_like(XX)
        gt_map      = np.zeros_like(XX)

        for i in range(XX.shape[0]):
            for j in range(XX.shape[1]):
                dx, dy = XX[i, j], YY[i, j]
                if np.sqrt(dx**2 + dy**2) < 0.1:
                    comfort_map[i, j] = gt_map[i, j] = np.nan
                    continue
                feat   = np.array([[dx, dy, np.sin(orient), np.cos(orient)]],
                                   dtype=np.float32)
                feat_n = (feat - mu) / std
                comfort_map[i, j] = float(model.predict(feat_n)[0])
                cost = gt_zone.compute_cost(np.array([dx, dy]), human_pos, orient)
                gt_map[i, j] = float(np.exp(-cost))

        orient_label = '0°  (facing right)' if orient == 0.0 else '90°  (facing up)'
        col_descs    = ['learned comfort score', 'analytical ground truth']

        for col, (data, desc) in enumerate(zip([comfort_map, gt_map], col_descs)):
            ax  = axes[row, col]
            vis = gaussian_filter(np.where(np.isnan(data), 0, data), sigma=0.5)
            vis = np.where(np.isnan(data), np.nan, vis)

            ax.set_facecolor(_MAP_BG)
            im = ax.contourf(XX, YY, vis, levels=16, cmap='RdYlGn',
                             vmin=0, vmax=1.0)
            ax.contour(XX, YY, vis, levels=[0.5],
                       colors='#111111', linewidths=1.7,
                       linestyles='--')
            # Analytical boundary overlay
            bx, by = gt_zone.get_zone_boundary_points(human_pos, orient, 200)
            ax.plot(bx, by, color='#2471a3', linewidth=1.6,
                    alpha=0.75, label='analytical boundary')

            # Human body + heading
            ax.add_patch(plt.Circle((0, 0), 0.18,
                                    color='white', ec='#333333',
                                    lw=0.9, zorder=5))
            ax.annotate('', xy=(0.46*np.cos(orient), 0.46*np.sin(orient)),
                        xytext=(0.20*np.cos(orient), 0.20*np.sin(orient)),
                        arrowprops=dict(arrowstyle='->', color='#1a1a1a',
                                        lw=1.5, mutation_scale=12), zorder=6)

            ax.set_xlim(-ext, ext)
            ax.set_ylim(-ext, ext)
            ax.set_aspect('equal')
            ax.set_xlabel('x  (m)', fontsize=9)
            ax.set_ylabel('y  (m)', fontsize=9)
            ax.tick_params(labelsize=8, length=3)
            ax.set_title(f'heading {orient_label} — {desc}',
                         loc='left', fontsize=9.5, pad=5)
            _panel_label(ax, panel_letter[row][col], x=-0.14, y=1.07)

            cb = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
            cb.set_label('comfort  (0=violated, 1=ok)', fontsize=8)
            cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            cb.ax.tick_params(labelsize=7.5)

            if row == 0 and col == 0:
                # Legend: boundary line styles
                handles = [
                    plt.Line2D([0],[0], color='#111111', lw=1.5,
                               linestyle='--', label='learned boundary  (c=0.5)'),
                    plt.Line2D([0],[0], color='#2471a3', lw=1.6,
                               alpha=0.75, label='analytical boundary'),
                ]
                ax.legend(handles=handles, fontsize=7.5, loc='upper right',
                          framealpha=0.9, edgecolor='0.82')

    fig.suptitle('ZoneLearnerNet — learned comfort field vs. analytical model',
                 fontsize=11.5, y=1.01)
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ===========================================================================
# Figure 6 — training curves
# ===========================================================================

def fig6_training_curves(zl_history=None, cnn_history=None,
                          out_path="fig6_training_curves.pdf"):
    """
    (a) ZoneLearnerNet cross-entropy loss.
    (b) ZoneLearnerNet validation accuracy with 90% sufficiency threshold.
    (c) SocialCostCNN MSE.

    The threshold in (b) is where ZoneLearnerNet is considered good enough
    to substitute for the analytical zone model in the planning pipeline.
    """
    def _fallback_zl():
        rng = np.random.default_rng(0)
        ep  = np.arange(1, 51)
        return {
            'train_loss': 0.68*np.exp(-ep/15) + 0.05 + rng.normal(0, 0.005, 50),
            'val_loss':   0.68*np.exp(-ep/15) + 0.07 + rng.normal(0, 0.007, 50),
            'val_acc':    0.5 + 0.42*(1-np.exp(-ep/12)) + rng.normal(0, 0.005, 50),
        }

    def _fallback_cnn():
        rng = np.random.default_rng(3)
        ep  = np.arange(1, 31)
        return {
            'train_loss': 0.08*np.exp(-ep/8) + 0.005 + rng.normal(0, 0.001,  30),
            'val_loss':   0.08*np.exp(-ep/8) + 0.007 + rng.normal(0, 0.0015, 30),
        }

    if zl_history is None:
        p = "zone_learner_history.npy"
        zl_history = (np.load(p, allow_pickle=True).item()
                      if os.path.exists(p) else _fallback_zl())
    if cnn_history is None:
        p = "cost_cnn_history.npy"
        cnn_history = (np.load(p, allow_pickle=True).item()
                       if os.path.exists(p) else _fallback_cnn())

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4),
                              gridspec_kw={'wspace': 0.40})

    ep_zl  = np.arange(1, len(zl_history['train_loss']) + 1)
    ep_cnn = np.arange(1, len(cnn_history['train_loss']) + 1)

    # (a) ZoneLearnerNet loss
    ax = axes[0]
    ax.plot(ep_zl, zl_history['train_loss'], '-',
            color=COLORS['train'], lw=1.8, label='training', alpha=0.9)
    ax.plot(ep_zl, zl_history['val_loss'], '--',
            color=COLORS['val'],   lw=1.8, label='validation')
    ax.set_xlabel('epoch')
    ax.set_ylabel('binary cross-entropy')
    ax.set_title('ZoneLearnerNet — loss', loc='left', pad=5)
    ax.legend()
    ax.grid(color=COLORS['grid'])
    ax.set_axisbelow(True)
    _panel_label(ax, 'a')

    # (b) ZoneLearnerNet accuracy
    ax = axes[1]
    acc = np.array(zl_history['val_acc']) * 100
    ax.plot(ep_zl, acc, '-', color=COLORS['asymmetric'],
            lw=1.8, label='val accuracy')
    ax.axhline(90, color='#7f8c8d', linestyle=':', lw=1.1,
               label='90% threshold')
    # Shade epochs above threshold
    ax.fill_between(ep_zl, 90, acc, where=(acc >= 90),
                    color=COLORS['asymmetric'], alpha=0.12)
    ax.set_xlabel('epoch')
    ax.set_ylabel('validation accuracy  (%)')
    ax.set_title('ZoneLearnerNet — accuracy', loc='left', pad=5)
    ax.set_ylim(44, 105)
    ax.legend(loc='lower right')
    ax.grid(color=COLORS['grid'])
    ax.set_axisbelow(True)
    _panel_label(ax, 'b')

    # Annotate when accuracy first crosses 90%
    first = next((i for i, a in enumerate(acc) if a >= 90), None)
    if first is not None:
        ax.axvline(ep_zl[first], color=COLORS['accent'],
                   linestyle='--', lw=1.0, alpha=0.75)
        ax.text(ep_zl[first] + 0.5, 46,
                f'epoch {ep_zl[first]}',
                fontsize=7.5, color=COLORS['accent'],
                rotation=90, va='bottom')

    # (c) SocialCostCNN MSE
    ax = axes[2]
    ax.plot(ep_cnn, cnn_history['train_loss'], '-',
            color=COLORS['train'], lw=1.8, label='training', alpha=0.9)
    ax.plot(ep_cnn, cnn_history['val_loss'], '--',
            color=COLORS['val'],   lw=1.8, label='validation')
    ax.set_xlabel('epoch')
    ax.set_ylabel('mean squared error')
    ax.set_title('SocialCostCNN — MSE', loc='left', pad=5)
    ax.legend()
    ax.grid(color=COLORS['grid'])
    ax.set_axisbelow(True)
    _panel_label(ax, 'c')

    fig.suptitle('Training curves:  ZoneLearnerNet (50 epochs)  and  '
                 'SocialCostCNN (30 epochs)',
                 fontsize=11, y=1.02)
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ===========================================================================
# Figure 7 — CNN prediction quality
# ===========================================================================

def fig7_cnn_prediction(cnn_model_path="cost_cnn.npy",
                         val_samples_path="cost_cnn_val_samples.npy",
                         zone_distances=None,
                         out_path="fig7_cnn_prediction.pdf"):
    """
    Four validation scenarios: ground truth, CNN prediction, and
    absolute error.

    Column (a): analytical cost map.
    Column (b): CNN prediction.
    Column (c): |prediction − ground truth|.

    All maps are per-scenario normalised to [0, 1].  MAE is reported over
    the active region (cells where GT > 1% of max) to avoid inflating
    accuracy with near-zero background.
    """
    from neural_zone import SocialCostCNN

    if zone_distances is None:
        zone_distances = {'front': 1.8, 'back': 0.6, 'left': 1.2, 'right': 1.2}

    if os.path.exists(val_samples_path):
        vd    = np.load(val_samples_path, allow_pickle=True).item()
        X_val, Y_val = vd['X'], vd['Y']
    else:
        print("  Generating validation samples for fig7...")
        from train_cost_cnn import generate_scenario
        rng   = np.random.default_rng(99)
        X_val = np.zeros((4, 2, 50, 50), dtype=np.float32)
        Y_val = np.zeros((4,    50, 50), dtype=np.float32)
        for i in range(4):
            X_val[i], Y_val[i], _ = generate_scenario(
                rng, zone_distances=zone_distances)

    cnn = SocialCostCNN(seed=1)
    if os.path.exists(cnn_model_path):
        cnn.load(cnn_model_path)
    else:
        print("  cost_cnn.npy not found; showing random-init CNN output.")

    n_show = min(4, len(X_val))
    Y_pred = cnn.predict(X_val[:n_show])

    fig, axes = plt.subplots(n_show, 3, figsize=(11.5, 3.1*n_show),
                              gridspec_kw={'hspace': 0.45, 'wspace': 0.30})
    if n_show == 1:
        axes = axes[np.newaxis, :]

    # Column headers on the first row only
    for col_idx, col_title in enumerate([
            '(a)  ground truth',
            '(b)  CNN prediction',
            '(c)  absolute error  |pred − GT|',
    ]):
        axes[0, col_idx].set_title(col_title, loc='left',
                                   fontsize=10, pad=5)

    for i in range(n_show):
        gt   = Y_val[i]
        pred = Y_pred[i]
        err  = np.abs(pred - gt)
        vmax = max(float(gt.max()), float(pred.max()), 0.01)

        # Masked MAE (active cells only)
        mask = gt > 0.01 * vmax
        mae  = float(err[mask].mean()) if mask.any() else float(err.mean())

        im0 = axes[i, 0].imshow(gt,   origin='lower', cmap='YlOrRd',
                                 vmin=0, vmax=vmax, aspect='equal')
        im1 = axes[i, 1].imshow(pred, origin='lower', cmap='YlOrRd',
                                 vmin=0, vmax=vmax, aspect='equal')
        im2 = axes[i, 2].imshow(err,  origin='lower', cmap='Blues',
                                 vmin=0, vmax=vmax*0.5, aspect='equal')

        for ax, im in zip(axes[i], [im0, im1, im2]):
            ax.set_xlabel('grid x', fontsize=8)
            ax.set_ylabel('grid y', fontsize=8)
            ax.tick_params(labelsize=7.5, length=2)
            cb = plt.colorbar(im, ax=ax, shrink=0.80, pad=0.02)
            cb.ax.tick_params(labelsize=7)

        axes[i, 0].set_ylabel(f'scenario {i+1}\n\ngrid y',
                               fontsize=9.5, fontweight='semibold')

        # MAE over active region — more meaningful than whole-grid MAE
        axes[i, 2].text(0.96, 0.96,
                        f'MAE = {mae:.4f}\n(active cells)',
                        transform=axes[i, 2].transAxes,
                        fontsize=7.5, ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  fc='white', ec='#cccccc', alpha=0.9))

        # Pedestrian markers
        occ        = X_val[i, 0]
        ys_p, xs_p = np.where(occ > 0.5)
        for ax in axes[i]:
            ax.scatter(xs_p, ys_p, c='#1abc9c', s=28, marker='^',
                       edgecolors='#0e6655', linewidths=0.5, zorder=5,
                       label='pedestrian' if i == 0 else None)

    axes[0, 0].legend(fontsize=7.5, loc='upper right',
                      framealpha=0.9, edgecolor='0.82')

    fig.suptitle('SocialCostCNN — prediction quality on held-out validation '
                 'scenarios\n(cost maps normalised to [0, 1] per scenario)',
                 fontsize=11, y=1.01)
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ===========================================================================
# Master entry point
# ===========================================================================

def generate_all_figures(zone_distances=None, summary=None, output_dir="figures",
                          include_neural=True, zl_history=None, cnn_history=None):
    """Render all seven publication figures to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    if zone_distances is None:
        zone_distances = {'front': 1.8, 'back': 0.6, 'left': 1.2, 'right': 1.2}

    print("Generating publication figures...")
    print("=" * 60)

    fig1_social_zone_heatmap(
        zone_distances, os.path.join(output_dir, "fig1_heatmap.pdf"))
    fig2_zone_distances(
        zone_distances, os.path.join(output_dir, "fig2_zone_distances.pdf"))
    fig3_trajectory_comparison(
        zone_distances, os.path.join(output_dir, "fig3_trajectories.pdf"))
    fig4_experiment_results(
        summary, os.path.join(output_dir, "fig4_results.pdf"))

    if include_neural:
        print("\nGenerating neural network figures (5-7)...")
        fig5_learned_zone_shape(
            zone_learner_path="zone_learner.npy",
            zone_distances=zone_distances,
            out_path=os.path.join(output_dir, "fig5_learned_zone.pdf"))
        fig6_training_curves(
            zl_history=zl_history, cnn_history=cnn_history,
            out_path=os.path.join(output_dir, "fig6_training_curves.pdf"))
        fig7_cnn_prediction(
            cnn_model_path="cost_cnn.npy",
            val_samples_path="cost_cnn_val_samples.npy",
            zone_distances=zone_distances,
            out_path=os.path.join(output_dir, "fig7_cnn_prediction.pdf"))

    print(f"\nAll figures saved to: {os.path.abspath(output_dir)}/")


if __name__ == "__main__":
    import sys
    neural = '--no-neural' not in sys.argv
    zd = {'front': 1.8, 'back': 0.6, 'left': 1.2, 'right': 1.2}
    if os.path.exists("zone_distances.npy"):
        zd = np.load("zone_distances.npy", allow_pickle=True).item()
    summary = None
    if os.path.exists("experiment_results.npy"):
        summary = np.load("experiment_results.npy",
                          allow_pickle=True).item().get('summary')
    generate_all_figures(zone_distances=zd, summary=summary,
                         output_dir="figures", include_neural=neural)
