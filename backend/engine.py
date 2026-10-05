"""Scheduling engine: decoding, objectives, repair, Min-Min baseline, NSGA-II.
Basis: Deb et al., "A fast and elitist multiobjective genetic algorithm: NSGA-II", IEEE TEVC 6(2), 2002;
Min-Min per Braun et al., JPDC 61(6), 2001; list scheduling over DAGs as in Topcuoglu et al. (HEFT), IEEE TPDS 13(3), 2002."""
import math, random
from statistics import pstdev, mean

INF = float("inf")

class Instance:
    def __init__(s, tasks, vms):
        s.tasks, s.vms, s.n, s.m = tasks, vms, len(tasks), len(vms)
        idx = {t.id: i for i, t in enumerate(tasks)}
        if len(idx) != s.n: raise ValueError("duplicate task ids")
        try: s.deps = [[idx[d] for d in t.deps] for t in tasks]
        except KeyError as e: raise ValueError(f"unknown dependency {e}")
        s.succ = [[] for _ in tasks]
        for i, ds in enumerate(s.deps):
            for d in ds: s.succ[d].append(i)
        s.feas = [[j for j, v in enumerate(vms) if v.cpu >= t.cpu and v.mem >= t.mem and v.storage >= t.storage and v.bw >= t.bw] for t in tasks]
        for i, f in enumerate(s.feas):
            if not f: raise ValueError(f"task {tasks[i].id} fits no VM (cpu/mem/storage/bw)")
        indeg = [len(d) for d in s.deps]; q = [i for i in range(s.n) if not indeg[i]]; seen = 0
        while q:
            i = q.pop(); seen += 1
            for c in s.succ[i]:
                indeg[c] -= 1
                if not indeg[c]: q.append(c)
        if seen != s.n: raise ValueError("dependency graph contains a cycle")

def decode(I, assign, keys):
    """Schedule builder. Gene pair per task = (VM index, random-key). Ready task with smallest key goes next, so any
    chromosome decodes to a precedence-feasible, non-preemptive schedule (one task at a time per VM)."""
    fin = [0.0] * I.n; st = [0.0] * I.n; free = [v.available_from for v in I.vms]
    indeg = [len(d) for d in I.deps]; ready = [i for i in range(I.n) if not indeg[i]]
    while ready:
        i = min(ready, key=keys.__getitem__); ready.remove(i)
        j = assign[i]; t = I.tasks[i]
        s = max(t.arrival, free[j], *(fin[d] for d in I.deps[i]))
        st[i] = s; fin[i] = s + t.length / I.vms[j].mips; free[j] = fin[i]
        for c in I.succ[i]:
            indeg[c] -= 1
            if not indeg[c]: ready.append(c)
    return st, fin

def evaluate(I, assign, keys):
    st, fin = decode(I, assign, keys)
    busy = [0.0] * I.m
    for i in range(I.n): busy[assign[i]] += fin[i] - st[i]
    M = max(fin)
    cost = sum(busy[j] / 3600 * I.vms[j].cost for j in range(I.m))                       # C = sum_j busy_j/3600 * price_j
    energy = sum(v.p_idle * M + (v.p_max - v.p_idle) * busy[j] for j, v in enumerate(I.vms)) / 3600  # Wh, linear power model
    util = sum(busy) / (I.m * M)                                                           # U = sum busy / (m * makespan)
    mu = mean(busy); imb = pstdev(busy) / mu if mu else 0.0                                # coefficient of variation
    tard = sum(t.priority * max(0.0, fin[i] - (t.deadline if t.deadline is not None else INF)) for i, t in enumerate(I.tasks))
    miss = sum(1 for i, t in enumerate(I.tasks) if t.deadline is not None and fin[i] > t.deadline + 1e-9)
    wait = mean(st[i] - I.tasks[i].arrival for i in range(I.n))
    return dict(makespan=M, cost=cost, energy=energy, utilization=util, imbalance=imb, tardiness=tard,
                deadline_misses=miss, avg_wait=wait)

def schedule_records(I, assign, keys):
    st, fin = decode(I, assign, keys); out = []
    for i, t in enumerate(I.tasks):
        v = I.vms[assign[i]]; ex = fin[i] - st[i]
        ok = t.deadline is None or fin[i] <= t.deadline + 1e-9
        out.append(dict(id=t.id, vm=v.id, start=st[i], end=fin[i], exec=ex, wait=st[i] - t.arrival, cost=ex / 3600 * v.cost,
                        energy=(v.p_max - v.p_idle) * ex / 3600, deadline=t.deadline, status="n/a" if t.deadline is None else ("met" if ok else "missed"),
                        cpu=t.cpu, mem=t.mem, storage=t.storage, bw=t.bw, priority=t.priority, deps=t.deps))
    return out

def objective_vector(m, names):  # all minimised; utilisation is maximised so use 1-U
    return [1 - m["utilization"] if k == "utilization" else m[k] for k in names]

def repair(I, assign, rng):
    for i in range(I.n):
        if assign[i] not in I.feas[i]: assign[i] = rng.choice(I.feas[i])
    return assign

def min_min(I):
    """Baseline: among ready tasks pick the (task,VM) pair with the smallest completion time (Braun et al. 2001), DAG-aware."""
    fin = [0.0] * I.n; free = [v.available_from for v in I.vms]; assign = [0] * I.n; order = [0] * I.n
    indeg = [len(d) for d in I.deps]; ready = [i for i in range(I.n) if not indeg[i]]; k = 0
    while ready:
        best = None
        for i in ready:
            for j in I.feas[i]:
                s = max(I.tasks[i].arrival, free[j], *(fin[d] for d in I.deps[i])); e = s + I.tasks[i].length / I.vms[j].mips
                if best is None or e < best[0]: best = (e, i, j)
        e, i, j = best; ready.remove(i); assign[i] = j; fin[i] = e; free[j] = e; order[i] = k; k += 1
        for c in I.succ[i]:
            indeg[c] -= 1
            if not indeg[c]: ready.append(c)
    return assign, [float(o) for o in order]

def dominates(a, b): return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def nd_sort(F):
    n = len(F); S = [[] for _ in range(n)]; cnt = [0] * n; fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q: continue
            if dominates(F[p], F[q]): S[p].append(q)
            elif dominates(F[q], F[p]): cnt[p] += 1
        if not cnt[p]: fronts[0].append(p)
    while fronts[-1]:
        nxt = []
        for p in fronts[-1]:
            for q in S[p]:
                cnt[q] -= 1
                if not cnt[q]: nxt.append(q)
        fronts.append(nxt)
    return fronts[:-1]

def crowding(F, front):
    d = {i: 0.0 for i in front}
    for k in range(len(F[0])):
        s = sorted(front, key=lambda i: F[i][k]); lo, hi = F[s[0]][k], F[s[-1]][k]
        d[s[0]] = d[s[-1]] = INF
        if hi > lo:
            for a in range(1, len(s) - 1): d[s[a]] += (F[s[a + 1]][k] - F[s[a - 1]][k]) / (hi - lo)
    return d

def nsga2(I, params, on_gen=None, on_event=None):
    """Run the existing NSGA-II implementation while optionally exposing its real internal state.

    on_gen receives one compact generation summary. on_event receives stage-level records
    that describe the actual operators as they execute; neither callback changes selection,
    crossover, mutation, repair, or environmental selection.
    """
    rng = random.Random(params.seed); names = params.objectives; n = I.n
    pm = params.pm if params.pm is not None else 1 / n

    def emit(phase, generation, **payload):
        if on_event:
            on_event(dict(phase=phase, generation=generation, **payload))

    def chromosome(p, rank=None, crowd=None):
        return {
            "assign": list(p["a"]),
            "keys": list(p["k"]),
            "metrics": dict(p["m"]),
            "fitness": list(p["f"]),
            "rank": rank,
            "crowding": None if crowd == INF else crowd,
            "feasible": all(p["a"][i] in I.feas[i] for i in range(n)),
        }

    def mk(a, k):
        m = evaluate(I, a, k)
        return {"a": a, "k": k, "m": m, "f": objective_vector(m, names)}

    # constraint-aware initialisation: heuristic seeds + random feasible individuals
    pop = []
    ma, mk_ = min_min(I); pop.append(mk(ma, mk_))
    edf = sorted(range(n), key=lambda i: I.tasks[i].deadline if I.tasks[i].deadline is not None else INF)
    pop.append(mk(
        [min(I.feas[i], key=lambda j: I.tasks[i].length / I.vms[j].mips) for i in range(n)],
        [float(edf.index(i)) for i in range(n)]
    ))
    while len(pop) < params.pop:
        pop.append(mk([rng.choice(I.feas[i]) for i in range(n)], [rng.random() for _ in range(n)]))

    def rank_pop(P):
        F = [p["f"] for p in P]; fr = nd_sort(F); rk = {}; cd = {}
        for r, f in enumerate(fr):
            c = crowding(F, f)
            for i in f: rk[i] = r; cd[i] = c[i]
        return fr, rk, cd

    def snapshot(P, fr, rk, cd, limit=30):
        # Deterministic representative: retain the complete first front when possible,
        # then fill with the best-ranked/crowding-diverse dominated candidates.
        front_idx = list(fr[0]) if fr else []
        ordered = sorted(range(len(P)), key=lambda i: (rk[i], -cd[i] if cd[i] != INF else -1e100, i))
        chosen = front_idx[:limit]
        for i in ordered:
            if len(chosen) >= limit: break
            if i not in chosen: chosen.append(i)
        candidates = [chromosome(P[i], rk[i], cd[i]) for i in chosen]
        return candidates

    fr, rk, cd = rank_pop(pop)
    hist = []; stale = 0; prev = None
    emit("initial_population", 0, populationSize=len(pop), feasibleSolutions=len(pop),
         population=snapshot(pop, fr, rk, cd),
         paretoFront=[chromosome(pop[i], 0, cd[i]) for i in fr[0]])
    emit("fitness_evaluation", 0, populationSize=len(pop), evaluatedSolutions=len(pop))

    for g in range(1, params.generations + 1):
        emit("selection", g, populationSize=len(pop), feasibleSolutions=sum(
            all(p["a"][i] in I.feas[i] for i in range(n)) for p in pop))

        def tour():
            a, b = rng.randrange(len(pop)), rng.randrange(len(pop))
            return a if (rk[a], -cd[a]) < (rk[b], -cd[b]) else b

        kids = []
        op_demo = None
        while len(kids) < params.pop:
            p1, p2 = pop[tour()], pop[tour()]
            a1, a2, k1, k2 = p1["a"][:], p2["a"][:], p1["k"][:], p2["k"][:]
            parent_a, parent_b = a1[:], a2[:]
            demo_pair = op_demo is None
            if demo_pair:
                emit("parent_selection", g, parentA=parent_a, parentB=parent_b)

            if rng.random() < params.pc:
                for i in range(n):
                    if rng.random() < .5: a1[i], a2[i] = a2[i], a1[i]
                    if rng.random() < .5: k1[i], k2[i] = k2[i], k1[i]
                if demo_pair:
                    emit("crossover", g, parentA=parent_a, parentB=parent_b, child=a1[:],
                         crossoverApplied=True)
            else:
                if demo_pair:
                    emit("crossover", g, parentA=parent_a, parentB=parent_b, child=a1[:],
                         crossoverApplied=False)

            for a, k in ((a1, k1), (a2, k2)):
                before_mutation = a[:]
                mutation_count = 0
                for i in range(n):
                    if rng.random() < pm:
                        a[i] = rng.choice(I.feas[i]); mutation_count += 1
                    if rng.random() < pm:
                        k[i] = rng.random(); mutation_count += 1
                mutated = a[:]
                invalid_before_repair = sum(a[i] not in I.feas[i] for i in range(n))
                repaired = repair(I, a, rng)
                if demo_pair:
                    emit("mutation", g, original=before_mutation, mutated=mutated,
                         mutationCount=mutation_count)
                    emit("repair", g, invalidGenes=invalid_before_repair,
                         repaired=repaired[:])
                child = mk(repaired, k)
                kids.append(child)
                if op_demo is None:
                    op_demo = {
                        "parentA": parent_a, "parentB": parent_b,
                        "childAfterCrossover": child["a"][:],
                        "childAfterMutation": mutated,
                        "childAfterRepair": repaired[:],
                    }
                if len(kids) >= params.pop: break

        emit("offspring_evaluation", g, offspringSize=len(kids))
        comb = pop + kids[:params.pop]
        F = [p["f"] for p in comb]
        fronts = nd_sort(F)
        emit("non_dominated_sort", g, combinedSize=len(comb), frontSizes=[len(f) for f in fronts])
        nxt = []
        for f in fronts:
            if len(nxt) + len(f) <= params.pop:
                nxt += f
            else:
                c = crowding(F, f)
                nxt += sorted(f, key=lambda i: -c[i])[:params.pop - len(nxt)]
                break

        pop = [comb[i] for i in nxt]
        fr, rk, cd = rank_pop(pop)
        emit("population_combination", g, combinedSize=len(comb), nextPopulationSize=len(pop))
        emit("next_generation", g, populationSize=len(pop), population=snapshot(pop, fr, rk, cd))

        front = [pop[i] for i in fr[0]]
        ideal = [min(p["f"][k] for p in front) for k in range(len(names))]
        stale = stale + 1 if prev == ideal else 0; prev = ideal
        feasible_count = sum(all(p["a"][i] in I.feas[i] for i in range(n)) for p in pop)
        avg_fitness = mean(sum(p["f"]) for p in pop)
        rec = dict(
            gen=g, phase="generation_complete", front_size=len(front),
            population_size=len(pop), feasible_solutions=feasible_count,
            avg_fitness=avg_fitness,
            mean_front={k: mean(p["m"][k] for p in pop) for k in ("makespan", "cost", "energy")},
            best={k: (max if k == "utilization" else min)(p["m"][k] for p in pop)
                  for k in ("makespan", "cost", "energy", "utilization", "imbalance", "tardiness")},
            pareto=[dict(index=i, metrics=dict(p["m"])) for i, p in enumerate(front)],
            population=snapshot(pop, fr, rk, cd),
            operations=op_demo,
        )
        hist.append(rec)
        emit("pareto_front", g, front_size=len(front),
             pareto=[dict(index=i, metrics=dict(p["m"])) for i, p in enumerate(front)])
        emit("generation_complete", g, **{k: rec[k] for k in (
            "front_size", "population_size", "feasible_solutions", "avg_fitness", "best")})
        if on_gen: on_gen(rec)
        if params.patience and stale >= params.patience: break

    seen = set(); sols = []
    for p in sorted((pop[i] for i in fr[0]), key=lambda p: p["f"]):
        key = tuple(round(x, 6) for x in p["f"])
        if key not in seen:
            seen.add(key); sols.append(p)
    return sols, hist

def recommend(sols, names, weights):
    """Weighted min-max-normalised distance over the Pareto set - used ONLY to suggest a point, never to optimise."""
    if not sols: return 0
    lo = [min(s["f"][k] for s in sols) for k in range(len(names))]; hi = [max(s["f"][k] for s in sols) for k in range(len(names))]
    def sc(s): return sum(weights.get(nm, 1.0) * ((s["f"][k] - lo[k]) / (hi[k] - lo[k]) if hi[k] > lo[k] else 0) for k, nm in enumerate(names))
    return min(range(len(sols)), key=lambda i: sc(sols[i]))
