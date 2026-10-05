import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "backend"))
import pytest
from fastapi.testclient import TestClient
import engine as E, main
from models import Task, VM, Params

def vm(i, mips=1000, cost=.1): return VM(id=i, cpu=4, mem=8, storage=100, bw=1000, mips=mips, cost=cost, p_idle=50, p_max=150)
def tiny(): return E.Instance([Task(id="a", length=1000), Task(id="b", length=1000, deps=["a"])], [vm("v0"), vm("v1", 2000, .2)])

def test_objectives_hand_computed():
    I = tiny(); m = E.evaluate(I, [0, 0], [0., 1.])
    assert m["makespan"] == pytest.approx(2.0) and m["cost"] == pytest.approx(2 / 3600 * .1)
    assert m["utilization"] == pytest.approx(2 / (2 * 2))          # v1 idle
    assert m["energy"] == pytest.approx((150 * 2 + 50 * 2) / 3600)

def test_dependency_respected():
    I = tiny(); st, fin = E.decode(I, [0, 1], [5., 0.]); assert st[1] >= fin[0]

def test_infeasible_and_cycle_rejected():
    with pytest.raises(ValueError): E.Instance([Task(id="a", length=1, cpu=64)], [vm("v")])
    with pytest.raises(ValueError): E.Instance([Task(id="a", length=1, deps=["b"]), Task(id="b", length=1, deps=["a"])], [vm("v")])

def test_repair_and_dominance_and_sort():
    I = E.Instance([Task(id="a", length=1, cpu=8)], [vm("v0"), VM(id="big", cpu=8, mem=8, storage=100, bw=1000, mips=1, cost=1, p_idle=1, p_max=2)])
    import random; assert E.repair(I, [0], random.Random(1)) == [1]
    assert E.dominates([1, 1], [2, 1]) and not E.dominates([1, 2], [2, 1])
    assert E.nd_sort([[1, 3], [2, 2], [3, 3]]) == [[0, 1], [2]]

def test_nsga2_reproducible_and_beats_or_matches_baseline():
    d = main.sample(20, 4, 3); I = E.Instance(*main._parse(d["tasks"], d["vms"])); p = Params(pop=30, generations=25, seed=7)
    s1, _ = E.nsga2(I, p); s2, _ = E.nsga2(I, p); assert [s["f"] for s in s1] == [s["f"] for s in s2]
    bm = E.evaluate(I, *E.min_min(I)); assert min(s["m"]["makespan"] for s in s1) <= bm["makespan"] + 1e-9
    assert all(not E.dominates(a["f"], b["f"]) for a in s1 for b in s1)

def test_api_end_to_end():
    c = TestClient(main.app); P = c.get("/api/sample?n=15&m=3").json(); P["params"] = dict(pop=16, generations=8, seed=1)
    jid = c.post("/api/optimize", json=P).json()["id"]
    for _ in range(100):
        if c.get(f"/api/optimization/{jid}").json()["status"] != "running": break
        time.sleep(.1)
    assert c.get(f"/api/optimization/{jid}/pareto").json()["solutions"]
    assert len(c.get(f"/api/optimization/{jid}/schedule").json()["tasks"]) == 15
    assert "improvement_pct" in c.post("/api/compare", json=dict(id=jid)).json()
    P["tasks"][0]["cpu"] = 999; assert c.post("/api/optimize", json=P).status_code == 422


def test_algorithm_visualization_stream_contains_real_state():
    c = TestClient(main.app)
    P = c.get("/api/sample?n=10&m=3&seed=9").json()
    P["params"] = dict(pop=12, generations=3, seed=5)
    jid = c.post("/api/optimize", json=P).json()["id"]
    for _ in range(100):
        if c.get(f"/api/optimization/{jid}").json()["status"] != "running":
            break
        time.sleep(.02)
    j = main.JOBS[jid]
    phases = {e["phase"] for e in j["events"]}
    assert {"initial_population", "fitness_evaluation", "selection", "parent_selection",
            "crossover", "mutation", "repair", "offspring_evaluation",
            "non_dominated_sort", "next_generation", "pareto_front", "generation_complete"} <= phases
    assert len(j["history"]) == 3
    assert all(h["population_size"] == 12 and h["feasible_solutions"] <= 12 for h in j["history"])
    assert all("population" in h and "pareto" in h and "avg_fitness" in h for h in j["history"])
