import json, logging, random, threading, time, uuid, pathlib
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from models import Problem, CompareReq, OBJECTIVES
import engine as E

logging.basicConfig(level=logging.INFO); log = logging.getLogger("cloudsched")
app = FastAPI(title="Cloud Task Scheduling - NSGA-II")
JOBS: dict[str, dict] = {}

VM_CATALOG = [  # modelled (not live) instance classes; replace with a provider adapter for real clouds
    ("small", 2, 4, 50, 500, 1000, .04, 30, 70), ("medium", 4, 8, 100, 1000, 2000, .08, 45, 120),
    ("large", 8, 16, 200, 2000, 4000, .17, 70, 220), ("xlarge", 16, 32, 400, 5000, 8000, .34, 110, 380)]

@app.get("/api/sample")
def sample(n: int = 30, m: int = 5, seed: int = 1):
    """Seeded layered-DAG workload (fork/join workflow, lognormal task lengths) + heterogeneous VM pool."""
    r = random.Random(seed); tasks = []; layers = []; i = 0
    while i < n:
        w = min(n - i, r.randint(2, 5)); layers.append(list(range(i, i + w))); i += w
    for li, L in enumerate(layers):
        for i in L:
            deps = [f"T{d}" for d in r.sample(layers[li - 1], r.randint(1, min(2, len(layers[li - 1]))))] if li else []
            ln = round(r.lognormvariate(9.2, .6)); tasks.append(dict(id=f"T{i}", length=ln, cpu=r.choice([1, 2, 2, 4]), mem=r.choice([1, 2, 4, 8]),
                storage=r.choice([5, 10, 20]), bw=r.choice([50, 100, 200]), arrival=0 if li else round(r.uniform(0, 5), 1),
                deadline=None, priority=r.randint(1, 5), deps=deps))
    vms = []
    for j in range(m):
        c = VM_CATALOG[min(j * len(VM_CATALOG) // max(m, 1) + (j % 2), 3)]
        vms.append(dict(id=f"VM{j}-{c[0]}", cpu=c[1], mem=c[2], storage=c[3], bw=c[4], mips=c[5], cost=c[6], p_idle=c[7], p_max=c[8], available_from=0))
    # deadlines: critical-path estimate on the median VM * slack factor
    med = sorted(v["mips"] for v in vms)[len(vms) // 2]; ef = {}
    for t in tasks: ef[t["id"]] = max([ef[d] for d in t["deps"]], default=t["arrival"]) + t["length"] / med; t["deadline"] = round(ef[t["id"]] * r.uniform(1.6, 2.6) + 5, 1)
    return dict(tasks=tasks, vms=vms)

def _parse(tasks, vms):
    from models import Task, VM
    return [Task(**t) for t in tasks], [VM(**v) for v in vms]

def _run(job, P):
    I = job["I"]; p = P.params
    try:
        job["baseline"] = E.min_min(I)
        sols, hist = E.nsga2(
            I, p,
            on_gen=lambda rec: job["history"].append(rec),
            on_event=lambda event: job["events"].append(event),
        )
        job["sols"] = sols; job["rec"] = E.recommend(sols, p.objectives, p.weights)
        job["status"] = "done"
        log.info("job %s done: %d Pareto solutions", job["id"], len(sols))
    except Exception as e:
        log.exception("job failed"); job["status"] = "failed"; job["error"] = str(e)

@app.post("/api/optimize")
def optimize(P: Problem):
    bad = [o for o in P.params.objectives if o not in OBJECTIVES]
    if bad or len(P.params.objectives) < 2: raise HTTPException(422, "choose >=2 valid objectives: " + ", ".join(OBJECTIVES))
    try: I = E.Instance(P.tasks, P.vms)
    except ValueError as e: raise HTTPException(422, str(e))
    jid = uuid.uuid4().hex[:8]
    JOBS[jid] = dict(id=jid, status="running", history=[], events=[], I=I, problem=P, t0=time.time())
    threading.Thread(target=_run, args=(JOBS[jid], P), daemon=True).start()
    return dict(id=jid)

def _job(jid):
    if jid not in JOBS: raise HTTPException(404, "unknown optimization id")
    return JOBS[jid]

@app.get("/api/optimization/{jid}")
def status(jid: str):
    j = _job(jid); return dict(
        id=jid, status=j["status"], error=j.get("error"), generation=len(j["history"]),
        history=j["history"][-400:], elapsed=time.time() - j["t0"]
    )

@app.get("/api/optimization/{jid}/stream")
def stream(jid: str):
    j = _job(jid)
    def gen():
        sent = 0
        while True:
            while sent < len(j["events"]):
                yield f"data: {json.dumps(j['events'][sent])}\n\n"
                sent += 1
            if j["status"] != "running" and sent >= len(j["events"]):
                yield f"event: end\ndata: {j['status']}\n\n"
                return
            time.sleep(.05)
    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

def _need(j):
    if j["status"] != "done": raise HTTPException(409, f"optimization is {j['status']}")

@app.get("/api/optimization/{jid}/pareto")
def pareto(jid: str):
    j = _job(jid); _need(j); p = j["problem"].params
    return dict(objectives=p.objectives, recommended=j["rec"], solutions=[dict(index=i, metrics=s["m"]) for i, s in enumerate(j["sols"])])

@app.get("/api/optimization/{jid}/schedule")
def schedule(jid: str, solution: int = 0, baseline: bool = False):
    j = _job(jid); _need(j); I = j["I"]
    if baseline: a, k = j["baseline"]
    elif 0 <= solution < len(j["sols"]): a, k = j["sols"][solution]["a"], j["sols"][solution]["k"]
    else: raise HTTPException(404, "no such solution")
    return dict(metrics=E.evaluate(I, a, k), tasks=E.schedule_records(I, a, k), vms=[v.model_dump() for v in I.vms])

@app.get("/api/optimization/{jid}/results")
def results(jid: str):
    j = _job(jid); _need(j)
    return dict(
        pareto=pareto(jid),
        baseline=schedule(jid, baseline=True)["metrics"],
        history=j["history"],
        seed=j["problem"].params.seed,
        parameters=j["problem"].params.model_dump(),
    )

@app.post("/api/compare")
def compare(r: CompareReq):
    j = _job(r.id); _need(j); b = schedule(r.id, baseline=True)["metrics"]; g = schedule(r.id, r.solution)["metrics"]; imp = {}
    for k in ("makespan", "cost", "energy", "imbalance", "tardiness", "deadline_misses", "avg_wait"):
        imp[k] = 100 * (b[k] - g[k]) / b[k] if b[k] else 0.0          # + = GA better (lower)
    imp["utilization"] = 100 * (g["utilization"] - b["utilization"]) / b["utilization"]
    return dict(baseline_name="Min-Min", baseline=b, ga=g, improvement_pct=imp)

app.mount("/", StaticFiles(directory=str(pathlib.Path(__file__).parent.parent / "frontend"), html=True))
