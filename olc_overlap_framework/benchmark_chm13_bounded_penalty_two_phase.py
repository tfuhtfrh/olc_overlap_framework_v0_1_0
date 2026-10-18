"""Two-phase penalty schedule seeded by a degree-only stage.

The degree-only stage is deliberately included because previous continuation
runs showed that it creates a much better weak-order starting basin than
turning order penalties on directly from the all-zero state.
"""

from __future__ import annotations
import json, time
from pathlib import Path
from dwave.samplers import TabuSampler
from benchmark_chm13_tabu_two_hamiltonians import SEED, bounded_eval, build_bounded_problem
from experimental_chm13_vertex_carry_qubo import BoundedVertexOrderConfig

OUT=Path("debug/qubo/chm13_bounded_penalty_two_phase_20260925.json")

SCHEDULES={
 "D_aggressive_degree":[
   (16,32,0.0,0.0),
   (16,32,0.25,0.5),
   (64,128,0.5,1.0),
   (256,512,1.0,2.0),
   (1000,2000,2.0,4.0),
   (2000,4000,4.0,8.0),
   (4000,8000,8.0,16.0),
 ],
 "E_gradual_degree":[
   (16,32,0.0,0.0),
   (16,32,0.25,0.5),
   (32,64,0.5,1.0),
   (64,128,1.0,2.0),
   (128,256,2.0,4.0),
   (256,512,4.0,8.0),
   (512,1024,8.0,16.0),
   (1000,2000,16.0,32.0),
 ],
}

def solve(problem,init,seed):
    t0=time.perf_counter()
    ss=TabuSampler().sample(
      problem["model"].to_dimod_bqm(),
      initial_states=[init], initial_states_generator="none",
      num_reads=1, timeout=4000, seed=seed,
      coefficient_z_first=1200, coefficient_z_restart=300,
      lower_bound_z=250000, num_restarts=100,
    )
    sec=time.perf_counter()-t0
    datum=next(ss.data(fields=["sample","energy"],sorted_by="energy"))
    sample=[int(datum.sample.get(i,0)) for i in range(problem["model"].num_variables)]
    return sample,sec

def main():
    rows=[]
    for si,(name,sched) in enumerate(SCHEDULES.items()):
        current=None
        print("SCHEDULE",name,flush=True)
        for stage,(d,v,s,p) in enumerate(sched):
            cfg=BoundedVertexOrderConfig(
                degree_penalty=float(d),void_degree_penalty=float(v),
                successor_penalty=float(s),product_penalty=float(p),
                edge_cost_scale=1.0,
            )
            problem=build_bounded_problem(f"{name}_{stage}",cfg)
            if current is None: current=[0]*problem["model"].num_variables
            current,sec=solve(problem,current,SEED+si*100+stage)
            ev=bounded_eval(problem,current)
            row={"schedule":name,"stage":stage,"degree":d,"void":v,"successor":s,"product":p,"seconds":sec,**ev}
            rows.append(row); print(json.dumps(row),flush=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"rows":rows},indent=2)+"\n",encoding="utf-8")
    print("\nBEST_BY_DEGREE")
    ranked=sorted(rows,key=lambda r:(r["read_degree_residual_sq"]+r["void_residual_sq"],r["successor_residual_sq"]+r["product_violations"],abs(r["selected_edges"]-143)))
    print(json.dumps(ranked[:10],indent=2),flush=True)

if __name__=="__main__": main()
