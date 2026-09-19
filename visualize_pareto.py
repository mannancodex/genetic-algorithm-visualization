#!/usr/bin/env python3
"""
NSGA-II Cloud Scheduler — Python Pareto Visualizer
====================================================
Generates high-quality static visualizations of the Pareto front
and algorithm convergence from exported simulation data.

Usage
-----
1. Run the simulator in your browser.
2. After converging, click "Export JSON" in the simulator to download
   the history JSON file.
3. Run:
       python visualize_pareto.py results.json

Dependencies
------------
    pip install matplotlib numpy scipy

Optional (for interactive 3-D):
    pip install plotly

The script auto-detects whether plotly is available and generates
an additional interactive HTML file if it is.
"""

import json
import sys
import os
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colormaps
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── colour palette (mirrors the simulator) ──────────────────────────
C_MAKESPAN = '#4285F4'
C_COST     = '#EA4335'
C_ENERGY   = '#F9AB00'
C_SLA      = '#34A853'
OBJ_COLORS = [C_MAKESPAN, C_COST, C_ENERGY, C_SLA]
OBJ_NAMES  = ['Makespan (s)', 'Cost ($)', 'Energy (Wh)', 'SLA Violation (s)']
OBJ_SHORT  = ['Makespan', 'Cost', 'Energy', 'SLA']

# ── dark background style ────────────────────────────────────────────
DARK_BG   = '#0F0F10'
PANEL_BG  = '#1A1B1C'
BORDER    = '#2E2F30'
TEXT_MAIN = '#E8EAED'
TEXT_DIM  = '#9AA0A6'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   PANEL_BG,
    'axes.edgecolor':   BORDER,
    'axes.labelcolor':  TEXT_DIM,
    'axes.titlecolor':  TEXT_MAIN,
    'xtick.color':      TEXT_DIM,
    'ytick.color':      TEXT_DIM,
    'grid.color':       BORDER,
    'grid.linewidth':   0.6,
    'text.color':       TEXT_MAIN,
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif', 'serif'],
    'legend.facecolor': PANEL_BG,
    'legend.edgecolor': BORDER,
    'legend.fontsize':  8,
})


# ════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════

def load_history(path):
    """Load the JSON exported by the browser simulator."""
    with open(path) as f:
        data = json.load(f)
    # Normalise: accept either a raw list (history array) or a wrapped object
    if isinstance(data, list):
        return data
    if 'history' in data:
        return data['history']
    raise ValueError("Unrecognised JSON format — expected a list or {history:[…]}")


def build_demo_history(n_gen=50, pop_size=40, n_tasks=24, n_vms=6, seed=1337):
    """
    Generate synthetic history data so the script works without a browser export.
    The values are purely illustrative — they follow realistic NSGA-II convergence
    curves but are NOT the actual algorithm output.
    """
    rng = np.random.default_rng(seed)
    history = []

    # Base objective ranges (uninformed population)
    base = np.array([30.0, 0.08, 120.0, 5.0])
    spread = np.array([20.0, 0.05, 60.0, 4.0])

    for g in range(n_gen + 1):
        # Convergence factor — fronts tighten as generations pass
        conv = math.exp(-3.0 * g / n_gen)

        # Generate a synthetic population
        objs = []
        for _ in range(pop_size):
            o = base * (1 - 0.35 * (1 - conv)) + spread * conv * rng.standard_normal(4)
            o = np.abs(o)
            objs.append(o.tolist())

        # Rank 0 = first ~20% sorted by sum of normalised objectives
        arr = np.array(objs)
        nrm = (arr - arr.min(0)) / (arr.max(0) - arr.min(0) + 1e-9)
        idx = np.argsort(nrm.sum(1))
        front0 = idx[:max(4, pop_size // 5)].tolist()

        history.append({
            'gen': g,
            'objs': objs,
            'fronts': [front0],
            'pop': [[rng.integers(0, n_vms) for _ in range(n_tasks)]
                    for _ in range(pop_size)],
        })
    return history


# ════════════════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ════════════════════════════════════════════════════════════════════

def pareto_front(history, gen=-1):
    snap = history[gen]
    front_idx = snap['fronts'][0]
    objs = np.array([snap['objs'][i] for i in front_idx])
    return objs, front_idx


def convergence_curves(history):
    """Best value per objective on the Pareto front at each generation."""
    curves = [[] for _ in range(4)]
    for snap in history:
        front_idx = snap['fronts'][0]
        for k in range(4):
            best = min(snap['objs'][i][k] for i in front_idx)
            curves[k].append(best)
    return curves


def normalise(arr):
    mn = arr.min(0)
    mx = arr.max(0)
    return (arr - mn) / (mx - mn + 1e-12)


# ════════════════════════════════════════════════════════════════════
#  PLOT 1 — 3-D Pareto Front (Matplotlib static)
# ════════════════════════════════════════════════════════════════════

def plot_3d_pareto(history, out_path):
    objs, _ = pareto_front(history, gen=-1)
    all_objs = np.array(history[-1]['objs'])

    fig = plt.figure(figsize=(10, 8), facecolor=DARK_BG)
    ax = fig.add_subplot(111, projection='3d', facecolor=PANEL_BG)

    # All population (background)
    ax.scatter(
        all_objs[:, 0], all_objs[:, 1], all_objs[:, 2],
        c=TEXT_DIM, s=14, alpha=0.22, linewidths=0, label='Dominated'
    )

    # Pareto front — colour-map by SLA violation (4th objective)
    sla = objs[:, 3]
    norm_sla = (sla - sla.min()) / (sla.max() - sla.min() + 1e-9)
    cmap = colormaps['RdYlGn_r']
    colours = [cmap(v) for v in norm_sla]

    sc = ax.scatter(
        objs[:, 0], objs[:, 1], objs[:, 2],
        c=sla, cmap='RdYlGn_r', s=70, alpha=0.92,
        edgecolors='white', linewidths=0.4, label='Pareto Front (rank 0)'
    )

    cb = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.55, aspect=14)
    cb.set_label('SLA Violation (s)', color=TEXT_DIM, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=TEXT_DIM)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_DIM, fontsize=7)

    ax.set_xlabel(OBJ_NAMES[0], labelpad=8, fontsize=9)
    ax.set_ylabel(OBJ_NAMES[1], labelpad=8, fontsize=9)
    ax.set_zlabel(OBJ_NAMES[2], labelpad=8, fontsize=9)
    ax.set_title(
        '3-D Pareto Front — Final Generation\n'
        r'$\it{Makespan\ ×\ Cost\ ×\ Energy}$, colour = SLA Violation',
        pad=14, fontsize=11, fontweight='bold'
    )

    ax.tick_params(axis='both', labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(BORDER)
    ax.yaxis.pane.set_edgecolor(BORDER)
    ax.zaxis.pane.set_edgecolor(BORDER)
    ax.grid(True, alpha=0.3)

    leg = ax.legend(loc='upper right', fontsize=8, framealpha=0.6)
    for t in leg.get_texts():
        t.set_color(TEXT_MAIN)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f'  ✓ 3-D Pareto front saved → {out_path}')


# ════════════════════════════════════════════════════════════════════
#  PLOT 2 — Convergence curves
# ════════════════════════════════════════════════════════════════════

def plot_convergence(history, out_path):
    curves = convergence_curves(history)
    gens = list(range(len(curves[0])))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), facecolor=DARK_BG)
    fig.suptitle(
        'Objective Convergence Over Generations',
        fontsize=13, fontweight='bold', y=0.98
    )

    for k, ax in enumerate(axes.flat):
        vals = np.array(curves[k])
        ax.plot(gens, vals, color=OBJ_COLORS[k], linewidth=2.2, zorder=3)
        ax.fill_between(gens, vals, vals.min(), alpha=0.12, color=OBJ_COLORS[k])

        # Annotate min
        best_g = int(np.argmin(vals))
        best_v = vals[best_g]
        ax.scatter([best_g], [best_v], color=OBJ_COLORS[k], s=60, zorder=5,
                   edgecolors='white', linewidths=1.2)
        ax.annotate(
            f'  best: {best_v:.3g} (gen {best_g})',
            xy=(best_g, best_v), fontsize=8, color=OBJ_COLORS[k],
            va='center'
        )

        ax.set_title(OBJ_NAMES[k], fontsize=10, pad=6)
        ax.set_xlabel('Generation', fontsize=8)
        ax.set_ylabel(OBJ_SHORT[k], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.35)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f'  ✓ Convergence chart saved → {out_path}')


# ════════════════════════════════════════════════════════════════════
#  PLOT 3 — Parallel Coordinates
# ════════════════════════════════════════════════════════════════════

def plot_parallel_coordinates(history, out_path):
    snap = history[-1]
    front_idx = set(snap['fronts'][0])
    all_objs  = np.array(snap['objs'])
    norm_all  = normalise(all_objs)

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    xs = np.arange(4)

    # Dominated solutions (faint)
    for i, row in enumerate(norm_all):
        if i not in front_idx:
            ax.plot(xs, row, color=TEXT_DIM, alpha=0.12, linewidth=0.8, zorder=1)

    # Pareto-front solutions
    front_objs_norm = norm_all[list(front_idx)]
    sla_norm = front_objs_norm[:, 3]
    cmap = colormaps['plasma']
    for j, (fi, row) in enumerate(zip(sorted(front_idx), front_objs_norm)):
        colour = cmap(sla_norm[j])
        ax.plot(xs, row, color=colour, alpha=0.78, linewidth=1.8, zorder=3)

    # Axis lines and labels
    for k in range(4):
        ax.axvline(k, color=BORDER, linewidth=1.2, zorder=2)
        ax.text(k, -0.08, OBJ_SHORT[k], ha='center', va='top',
                fontsize=9, color=TEXT_DIM, transform=ax.get_xaxis_transform())

    ax.set_xlim(-0.15, 3.15)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([])
    ax.set_ylabel('Normalised value', fontsize=9)
    ax.set_title(
        'Parallel Coordinates — Final Generation\n'
        'Bold coloured lines: Pareto-optimal (colour gradient = SLA violation)',
        fontsize=11, fontweight='bold', pad=10
    )
    ax.tick_params(labelsize=7)
    ax.grid(False)

    # Colour bar for SLA
    sm = plt.cm.ScalarMappable(cmap='plasma',
                                norm=plt.Normalize(
                                    all_objs[list(front_idx), 3].min(),
                                    all_objs[list(front_idx), 3].max()))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.8)
    cb.set_label('SLA Violation (s)', color=TEXT_DIM, fontsize=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_DIM, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f'  ✓ Parallel coordinates saved → {out_path}')


# ════════════════════════════════════════════════════════════════════
#  PLOT 4 — Pareto Evolution (scatter grid across generations)
# ════════════════════════════════════════════════════════════════════

def plot_pareto_evolution(history, out_path, n_snapshots=6):
    total_gens = len(history)
    snap_gens  = [int(round(i * (total_gens - 1) / (n_snapshots - 1)))
                  for i in range(n_snapshots)]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor=DARK_BG)
    fig.suptitle(
        'Pareto Front Evolution — Makespan vs Cost',
        fontsize=13, fontweight='bold', y=0.98
    )

    for ax, g in zip(axes.flat, snap_gens):
        snap = history[g]
        all_objs  = np.array(snap['objs'])
        front_idx = snap['fronts'][0]
        front_objs = all_objs[front_idx]

        ax.scatter(all_objs[:, 0], all_objs[:, 1],
                   c=TEXT_DIM, s=12, alpha=0.3, linewidths=0, label='All')
        ax.scatter(front_objs[:, 0], front_objs[:, 1],
                   c=C_MAKESPAN, s=36, alpha=0.85, linewidths=0.5,
                   edgecolors='white', zorder=3, label='Pareto Front')

        # Step-wise front line
        srt = front_objs[front_objs[:, 0].argsort()]
        ax.step(srt[:, 0], srt[:, 1], where='post',
                color=C_MAKESPAN, alpha=0.5, linewidth=1.2, linestyle='--')

        ax.set_title(f'Generation {g}', fontsize=9, pad=4)
        ax.set_xlabel(OBJ_SHORT[0], fontsize=7)
        ax.set_ylabel(OBJ_SHORT[1], fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    handles = [
        plt.scatter([], [], c=TEXT_DIM, s=12, alpha=0.4, label='Dominated'),
        plt.scatter([], [], c=C_MAKESPAN, s=36, edgecolors='white',
                    linewidths=0.5, alpha=0.85, label='Pareto Front'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               fontsize=8, framealpha=0.6, bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f'  ✓ Pareto evolution grid saved → {out_path}')


# ════════════════════════════════════════════════════════════════════
#  PLOT 5 — Interactive 3-D Pareto (Plotly) — optional
# ════════════════════════════════════════════════════════════════════

def plot_interactive_3d(history, out_path):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print('  ⚠  plotly not installed — skipping interactive 3-D chart.')
        print('      Install it with:  pip install plotly')
        return

    snap = history[-1]
    front_idx = snap['fronts'][0]
    all_objs  = np.array(snap['objs'])
    front_objs = all_objs[front_idx]

    dom_idx = [i for i in range(len(all_objs)) if i not in set(front_idx)]
    dom_objs = all_objs[dom_idx]

    # Hover text for Pareto points
    hover_front = [
        f'Makespan: {o[0]:.2f} s<br>'
        f'Cost: ${o[1]:.4f}<br>'
        f'Energy: {o[2]:.1f} Wh<br>'
        f'SLA: {o[3]:.3f} s'
        for o in front_objs
    ]
    hover_dom = [
        f'Makespan: {o[0]:.2f} s<br>'
        f'Cost: ${o[1]:.4f}<br>'
        f'Energy: {o[2]:.1f} Wh<br>'
        f'SLA: {o[3]:.3f} s'
        for o in dom_objs
    ]

    fig = go.Figure()

    # Dominated
    fig.add_trace(go.Scatter3d(
        x=dom_objs[:, 0], y=dom_objs[:, 1], z=dom_objs[:, 2],
        mode='markers',
        marker=dict(size=3, color='#5F6368', opacity=0.35),
        text=hover_dom, hoverinfo='text',
        name='Dominated'
    ))

    # Pareto front — colour by SLA
    fig.add_trace(go.Scatter3d(
        x=front_objs[:, 0], y=front_objs[:, 1], z=front_objs[:, 2],
        mode='markers',
        marker=dict(
            size=7,
            color=front_objs[:, 3],
            colorscale='RdYlGn_r',
            opacity=0.92,
            line=dict(width=0.5, color='white'),
            colorbar=dict(
                title='SLA (s)',
                titlefont=dict(color='#9AA0A6'),
                tickfont=dict(color='#9AA0A6'),
                x=1.02
            )
        ),
        text=hover_front, hoverinfo='text',
        name='Pareto Front'
    ))

    fig.update_layout(
        title=dict(
            text='Interactive 3-D Pareto Front — NSGA-II Cloud Scheduler',
            font=dict(family='Times New Roman', size=15, color='#E8EAED')
        ),
        paper_bgcolor='#0F0F10',
        plot_bgcolor='#0F0F10',
        scene=dict(
            xaxis=dict(title='Makespan (s)',  backgroundcolor='#1A1B1C',
                       gridcolor='#2E2F30', color='#9AA0A6'),
            yaxis=dict(title='Cost ($)',      backgroundcolor='#1A1B1C',
                       gridcolor='#2E2F30', color='#9AA0A6'),
            zaxis=dict(title='Energy (Wh)',   backgroundcolor='#1A1B1C',
                       gridcolor='#2E2F30', color='#9AA0A6'),
            bgcolor='#1A1B1C',
        ),
        legend=dict(font=dict(color='#9AA0A6')),
        margin=dict(l=0, r=0, b=0, t=60),
        width=900, height=700
    )

    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f'  ✓ Interactive 3-D Pareto (Plotly) saved → {out_path}')


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate Pareto front visualisations from NSGA-II history JSON.'
    )
    parser.add_argument(
        'json_file', nargs='?', default=None,
        help='Path to the history JSON exported from the browser simulator. '
             'If omitted, a synthetic demo dataset is used.'
    )
    parser.add_argument(
        '--out-dir', default='.',
        help='Directory to write output files into (default: current directory).'
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.json_file:
        print(f'Loading history from: {args.json_file}')
        history = load_history(args.json_file)
    else:
        print('No JSON file supplied — generating synthetic demo data …')
        history = build_demo_history()

    print(f'Loaded {len(history)} generations, '
          f'pop size = {len(history[0]["objs"])}')
    print()

    print('Generating plots …')
    plot_3d_pareto(        history, os.path.join(out_dir, '01_pareto_3d.png'))
    plot_convergence(      history, os.path.join(out_dir, '02_convergence.png'))
    plot_parallel_coordinates(history, os.path.join(out_dir, '03_parallel_coords.png'))
    plot_pareto_evolution( history, os.path.join(out_dir, '04_pareto_evolution.png'))
    plot_interactive_3d(   history, os.path.join(out_dir, '05_pareto_3d_interactive.html'))

    print()
    print('All done. Output files:')
    for fname in sorted(os.listdir(out_dir)):
        if fname.startswith('0') and (fname.endswith('.png') or fname.endswith('.html')):
            full = os.path.join(out_dir, fname)
            size = os.path.getsize(full)
            print(f'  {fname:50s}  {size:>8,} bytes')


if __name__ == '__main__':
    main()
