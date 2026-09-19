# Adaptive NSGA-II — Cloud Task Scheduling Simulator

An interactive, in-browser simulator for cloud task scheduling using NSGA-II
(Non-dominated Sorting Genetic Algorithm II), optimising four objectives simultaneously:

- **Makespan** (s) — total schedule length
- **Cost** ($) — VM-hour billing
- **Energy** (Wh) — active + idle power draw
- **SLA Violation** (s) — total deadline overrun

---

## Quick Start

No build step. No dependencies to install. Open `index.html` in any modern
browser (Chrome, Firefox, Safari, Edge).

---

## Python Visualiser (`visualize_pareto.py`)

After running the simulator, export the history to JSON and generate
high-quality static and interactive charts:

```bash
pip install matplotlib numpy plotly
python visualize_pareto.py results.json --out-dir ./plots
```

Without a JSON argument the script runs on built-in synthetic demo data:

```bash
python visualize_pareto.py
```

### Output files

| File | Description |
|------|-------------|
| `01_pareto_3d.png` | 3-D Pareto front (Makespan × Cost × Energy, colour = SLA) |
| `02_convergence.png` | Best objective value per generation — 4 sub-plots |
| `03_parallel_coords.png` | Parallel coordinates — all 4 objectives, Pareto lines bold |
| `04_pareto_evolution.png` | Pareto evolution grid — 6 snapshots across generations |
| `05_pareto_3d_interactive.html` | Interactive 3-D Plotly chart (requires `plotly`) |

---

## Simulator Panels

### Pareto Front Explorer
Scatter plot of any two objectives. Non-dominated (rank-0) solutions are
highlighted. Click a point to lock it and inspect its Gantt schedule.

### Parallel Coordinates
All four objectives simultaneously, normalised to [0, 1]. Bold coloured lines
are Pareto-optimal; faint lines are dominated solutions.

### Convergence
Best value per objective on the Pareto front, plotted against generation.
Normalised so all four curves fit the same axis.

### VM Schedule (Gantt)
Task → VM assignment timeline for the selected (or auto best-compromise)
solution. SLA-violating tasks are outlined in red.

### Genetic Operators
A real, per-generation log of every parent pair produced by tournament
selection. Each gene shows its actual VM-index digit (0–N), backed by a
compact colour bar. Number chip border colour indicates provenance:

- **Green border** — inherited from Parent A
- **Blue border** — inherited from Parent B
- **Red border / background** — mutated by random reassignment

Parent objective values (Makespan, Cost, Energy, SLA) are shown beneath
each parent for direct comparison with offspring.

---

## Reproducibility

The seed is shown top-right and in the sidebar. Enter any integer and press
**Apply** to pin it, or press **Regenerate workload** for a fresh random seed.
Same seed + same parameters = identical run every time.

## Custom Workload (CSV)

Upload your own tasks and/or VMs as CSV files.

**Tasks CSV** (`sample-data/tasks_sample.csv`):

| column | required | meaning |
|--------|----------|---------|
| `length` | yes | task length in million instructions (MI) |
| `deadline` | no | deadline in seconds; auto-computed if omitted |

**VMs CSV** (`sample-data/vms_sample.csv`):

| column | required | meaning |
|--------|----------|---------|
| `mips` | yes | processing speed (MI/s) |
| `cost_per_hour` | no | billing rate $/h; randomised if omitted |
| `idle_power` | no | idle power in watts; randomised if omitted |
| `max_power` | no | full-load power in watts; randomised if omitted |

---

## Export → Python Visualiser Workflow

1. Configure and run the simulation.
2. Click **⬇ Export History JSON (for Python)** in the sidebar.
3. Move the downloaded `nsga2_seed<N>.json` next to `visualize_pareto.py`.
4. Run `python visualize_pareto.py nsga2_seed<N>.json --out-dir ./plots`.
5. Open the generated PNG files and the interactive Plotly HTML.

---

## File Structure

```
nsga2-cloud-scheduler/
├── index.html                  # Complete simulator (HTML + CSS + JS)
├── visualize_pareto.py         # Python 3-D visualiser
├── README.md                   # This file
└── sample-data/
    ├── tasks_sample.csv
    └── vms_sample.csv
```
