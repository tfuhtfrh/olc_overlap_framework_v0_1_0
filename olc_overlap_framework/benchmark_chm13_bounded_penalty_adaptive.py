"""Adaptive penalty schedule for bounded CHM13 QUBO.

Goal: use weak order penalties first to escape the zero-position maximal-matching
basin, then raise degree penalties to project toward an exact 1-in/1-out
structure, and finally strengthen order/product consistency.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from dwave.samplers import TabuSampler

from benchmark_chm13_tabu_two_hamiltonians import SEED, bounded_eval, build_bounded_problem
from experimental_chm13_vertex_carry_qubo import BoundedVertexOrderConfig

OUT = Path("debug/qubo/chm13_bounded_penalty_adaptive_20260925.json")

SCHEDULES = {
    "A_start16": [
        (16, 32, 0.25, 0.5),
        (64, 128, 0.5, 1.0),
        (256, 512, 1.0, 2.0),
        (1000, 2000, 2.0, 4.0),
        (1000, 2000, 4.0, 8.0),
        (1000, 2000, 8.0, 16.0),
        (1000, 2000, 16.0, 32.0),
    ],
    "B_start32": [
        (32, 64, 0.25, 0.5),
        (128, 256, 0.5, 1.0),
        (512, 1024, 1.0, 2.0),
        (2000, 4000, 2.0, 4.0),
        (2000, 4000, 4.0, 8.0),
        (2000, 4000, 8.0, 16.0),
        (2000, 4000, 16.0, 32.0),
    ],
    "C_start8": [
        (8, 16, 0.25, 0.5),
        (32, 64, 0.5, 1.0),
        (128, 256, 1.0, 2.0),
        (512, 1024, 2.0, 4.0),
        (2000, 4000, 4.0, 8.0),
        (2000, 4000, 8.0, 16.0),
        (2000, 4000, 16.0, 32.0),
    ],
}


def solve(problem, init, seed):
    sampler = TabuSampler()
    t0=time.perf_counter()
    ss=sampler.sample(
        problem["model"].to_dimod_bqm(),
        initial_states=[init],
        initial_states_generator="none",
        num_reads=1,
        timeout=3500,
        seed=seed,
        coefficient_z_first=1000,
        coefficient_z_restart=250,
        lower_bound_z=200000,
        num_restarts=80,
    )
    sec=time.perf_counter()-t0
    datum=next(ss.data(fields=["sample","energy"], sorted_by="energy"))
    sample=[int(datum.sample.get(i,0)) for i in range(problem["model"].num_variables)]
    return sample,sec


def main():
    all_rows=[]
    for si,(name,schedule) in enumerate(SCHEDULES.items()):
        current=None
        print("SCHEDULE",name,flush=True)
        for stage,(d,v,s,p) in enumerate(schedule):
            cfg=BoundedVertexOrderConfig(
                degree_penalty=float(d),
                void_degree_penalty=float(v),
                successor_penalty=float(s),
                product_penalty=float(p),
                edge_cost_scale=1.0,
            )
            problem=build_bounded_problem(f"{name}_{stage}",cfg)
            if current is None:
                current=[0]*problem["model"].num_variables
            current,sec=solve(problem,current,SEED+si*100+stage)
            ev=bounded_eval(problem,current)
            row={"schedule":name,"stage":stage,"degree":d,"void":v,"successor":s,"product":p,"seconds":sec,**ev}
            all_rows.append(row)
            print(json.dumps(row),flush=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"rows":all_rows},indent=2)+"\n",encoding="utf-8")
    print("\nFINAL")
    print(json.dumps([r for r in all_rows if r["stage"]==6],indent=2),flush=True)

if __name__=="__main__":
    main()
