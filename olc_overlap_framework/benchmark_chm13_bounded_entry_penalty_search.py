"""Focused search for the best weak-order entry penalties after a degree-only warmup."""
from __future__ import annotations
import json,time
from pathlib import Path
from dwave.samplers import TabuSampler
from benchmark_chm13_tabu_two_hamiltonians import SEED,bounded_eval,build_bounded_problem
from experimental_chm13_vertex_carry_qubo import BoundedVertexOrderConfig
OUT=Path("debug/qubo/chm13_bounded_entry_penalty_search_20260925.json")
DEGREES=[8,12,16,20,24,32]
WEAK=[(0.125,0.25),(0.25,0.5)]

def solve(problem,init,seed,timeout=3000):
    t=time.perf_counter()
    ss=TabuSampler().sample(problem["model"].to_dimod_bqm(),
        initial_states=[init],initial_states_generator="none",
        num_reads=1,timeout=timeout,seed=seed,
        coefficient_z_first=1000,coefficient_z_restart=250,
        lower_bound_z=180000,num_restarts=70)
    sec=time.perf_counter()-t
    dat=next(ss.data(fields=["sample","energy"],sorted_by="energy"))
    return [int(dat.sample.get(i,0)) for i in range(problem["model"].num_variables)],sec

def build(d,s,p):
    cfg=BoundedVertexOrderConfig(degree_penalty=d,void_degree_penalty=2*d,
        successor_penalty=s,product_penalty=p,edge_cost_scale=1.0)
    return build_bounded_problem(f"d{d}_s{s}_p{p}",cfg)

def main():
    rows=[]
    for di,d in enumerate(DEGREES):
      # independent degree-only warmup for each weak-order pair
      for wi,(s,p) in enumerate(WEAK):
        p0=build(d,0.0,0.0); init=[0]*p0["model"].num_variables
        warm,sec0=solve(p0,init,SEED+100*di+10*wi)
        ev0=bounded_eval(p0,warm)
        p1=build(d,s,p)
        out,sec1=solve(p1,warm,SEED+100*di+10*wi+1)
        ev1=bounded_eval(p1,out)
        row={"degree":d,"void":2*d,"successor":s,"product":p,
             "warmup":{"seconds":sec0,"selected_edges":ev0["selected_edges"],
                       "degree_sq":ev0["read_degree_residual_sq"]},
             "seconds":sec1,**ev1}
        rows.append(row); print(json.dumps(row),flush=True)
    rows.sort(key=lambda r:(r["read_degree_residual_sq"],abs(r["selected_edges"]-143),
                            r["product_violations"],r["successor_residual_sq"]))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"rows":rows},indent=2)+"\n",encoding="utf-8")
    print("\nTOP"); print(json.dumps(rows[:8],indent=2),flush=True)
if __name__=="__main__":main()
