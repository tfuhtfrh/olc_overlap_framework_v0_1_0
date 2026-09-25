"""QBSolv comparison on CHM13 QUBOs.

Historical/decomposition baseline:
- dwave-qbsolv 0.3.4
- internal classical tabu subproblem solver
- energy-impact decomposition
- several subproblem sizes

This is intentionally isolated from main and from the production MST2 Tabu tests.
"""
from __future__ import annotations
import json,time
from pathlib import Path
from dwave_qbsolv import QBSolv, ENERGY_IMPACT

from benchmark_chm13_tabu_two_hamiltonians import (
    SEED,current_eval,bounded_eval,build_current_problem,build_bounded_problem
)
from experimental_chm13_vertex_carry_qubo import BoundedVertexOrderConfig

OUT=Path("debug/qubo/chm13_qbsolv_comparison_20260925.json")

CASES=[
 ("current", build_current_problem, current_eval),
 ("bounded_weak16", lambda: build_bounded_problem(
      "bounded_weak16",
      BoundedVertexOrderConfig(
        degree_penalty=16.0,void_degree_penalty=32.0,
        successor_penalty=0.25,product_penalty=0.5,edge_cost_scale=1.0)), bounded_eval),
 ("bounded_weak24", lambda: build_bounded_problem(
      "bounded_weak24",
      BoundedVertexOrderConfig(
        degree_penalty=24.0,void_degree_penalty=48.0,
        successor_penalty=0.125,product_penalty=0.25,edge_cost_scale=1.0)), bounded_eval),
]

SUBSIZES=[47,100,200]

def run_case(name,builder,evaluator,subsize,seed):
    problem=builder()
    bqm=problem["model"].to_dimod_bqm()
    t0=time.perf_counter()
    ss=QBSolv().sample(
       bqm,
       num_repeats=20,
       seed=seed,
       algorithm=ENERGY_IMPACT,
       verbosity=-1,
       timeout=20,
       solver_limit=subsize,
       solver="tabu",
    )
    sec=time.perf_counter()-t0
    datum=next(ss.data(fields=["sample","energy"],sorted_by="energy"))
    sample=[int(datum.sample.get(i,0)) for i in range(problem["model"].num_variables)]
    ev=evaluator(problem,sample)
    return {
      "model":name,"subproblem_size":subsize,"seconds":sec,
      "variables":problem["model"].num_variables,
      "quadratic_terms":len(problem["model"].quadratic),
      **ev,
    }

def main():
    rows=[]
    for ci,(name,builder,evaluator) in enumerate(CASES):
      for si,size in enumerate(SUBSIZES):
        print("RUN",name,"S",size,flush=True)
        row=run_case(name,builder,evaluator,size,SEED+100*ci+si)
        rows.append(row); print(json.dumps(row),flush=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"solver":"dwave-qbsolv 0.3.4","rows":rows},indent=2)+"\n",encoding="utf-8")
    print("\nSUMMARY")
    print(json.dumps(rows,indent=2),flush=True)
if __name__=="__main__":main()
