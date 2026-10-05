"""Typed schemas (API layer + data model). Units: length=MI, mips=MI/s, cost=$/hour, power=W, memory/storage=GB, bw=Mbps."""
from pydantic import BaseModel, Field

OBJECTIVES = ["makespan", "cost", "energy", "utilization", "imbalance", "tardiness"]

class Task(BaseModel):
    id: str
    length: float = Field(gt=0)
    cpu: float = 1; mem: float = 1; storage: float = 1; bw: float = 1
    arrival: float = Field(0, ge=0)
    deadline: float | None = None
    priority: int = Field(1, ge=1, le=10)
    deps: list[str] = []

class VM(BaseModel):
    id: str
    cpu: float; mem: float; storage: float; bw: float
    mips: float = Field(gt=0)
    cost: float = Field(ge=0)
    p_idle: float = Field(ge=0); p_max: float = Field(ge=0)
    available_from: float = Field(0, ge=0)

class Params(BaseModel):
    pop: int = Field(60, ge=8, le=500)
    generations: int = Field(80, ge=1, le=1000)
    pc: float = Field(0.9, ge=0, le=1)
    pm: float | None = Field(None, ge=0, le=1)   # default 1/n per gene
    seed: int = 1
    objectives: list[str] = ["makespan", "cost", "energy"]
    weights: dict[str, float] = {}               # only used to pick the recommended Pareto point
    patience: int = Field(0, ge=0)               # stop if ideal point unchanged for k generations (0=off)

class Problem(BaseModel):
    tasks: list[Task] = Field(min_length=1)
    vms: list[VM] = Field(min_length=1)
    params: Params = Params()

class CompareReq(BaseModel):
    id: str
    solution: int = 0
