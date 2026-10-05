# CloudSched – NSGA-II task scheduling for modelled cloud infrastructure
Run: `pip install -r requirements.txt && cd backend && uvicorn main:app --port 8000` → open http://localhost:8000
Tests: `python -m pytest tests` (needs internet in the browser for React/Babel CDN scripts).

**Scope:** optimises a *modelled* VM pool and emits an executable plan; no real cloud API is called. Add a provider adapter that returns `VM` objects to connect AWS/Azure/GCP.

## Model (units: MI, MIPS, $/h, W, GB, Mbps)
Chromosome: per task (VM index, random key). Decoder: ready task with smallest key is placed on its VM at `max(arrival, VM free, dep finish)`; one task per VM at a time, non-preemptive. Infeasible VM genes (cpu/mem/storage/bw) are repaired; cycles and unfittable tasks are rejected (HTTP 422).
Objectives (minimised): makespan; cost = Σ busy/3600·price; energy = Σ(P_idle·M+(P_max−P_idle)·busy)/3600 Wh (linear power model); 1−utilisation; imbalance = σ/μ of VM busy time; priority-weighted tardiness (soft deadline/QoS).
NSGA-II: fast non-dominated sort, crowding distance, binary tournament, uniform crossover, mutation, repair, (μ+λ) elitism, optional stagnation stop; seeded RNG. Baseline: DAG-aware Min-Min.
References used for the design (not a literature survey): Deb et al., IEEE TEVC 6(2):182–197, 2002 (NSGA-II); Braun et al., JPDC 61(6):810–837, 2001 (Min-Min); Topcuoglu et al., IEEE TPDS 13(3):260–274, 2002 (DAG list scheduling).
API: POST /api/optimize · GET /api/optimization/{id}[/stream|/pareto|/schedule?solution=k|/results] · POST /api/compare · GET /api/sample

## Visualization workspace

The React frontend is served directly from `frontend/index.html` and now includes a five-step workflow:
Workload → Cloud Resources → Optimization → Results → Pareto Explorer.

The existing NSGA-II engine remains the optimizer of record. During a run, the backend emits actual stage events over Server-Sent Events at:

`GET /api/optimization/{id}/stream`

The stream exposes the real population snapshot, parent selection, crossover, mutation, repair, offspring evaluation, non-dominated sorting, crowding distance, next generation, Pareto front, and generation-complete metrics. The UI uses these events for its live population view and Pareto convergence visualization.

Final analytics are available through the existing schedule/compare APIs and include the actual generated Gantt schedule, VM utilization, task distribution, Pareto solutions, generation convergence, and Min-Min baseline comparison.
