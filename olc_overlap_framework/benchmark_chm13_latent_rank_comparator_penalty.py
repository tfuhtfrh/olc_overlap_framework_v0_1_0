"""Focused penalty sweep for latent-rank comparator QUBO with QBSolv."""
from __future__ import annotations
import json,time
from pathlib import Path
from dwave_qbsolv import QBSolv,ENERGY_IMPACT
from experimental_chm13_latent_rank_comparator_qubo import (
    build_model,encode_order,audit,LatentRankComparatorConfig,DATASET_DIR
)

OUT=Path("debug/qubo/chm13_latent_rank_comparator_penalty_20260926.json")
SEED=20260926
CASES=[
    (144,144,576,288),
    (288,288,576,288),
    (288,288,1152,576),
    (576,576,1152,576),
    (576,576,2304,1152),
]

def main():
    ref=json.loads((DATASET_DIR/"reference_path.json").read_text())["normalized_nodes"]
    rows=[]
    for i,(deg,void,cmp_,gate) in enumerate(CASES):
        cfg=LatentRankComparatorConfig(
            degree_penalty=float(deg),void_penalty=float(void),
            comparator_penalty=float(cmp_),order_gate_penalty=float(gate),
            edge_cost_scale=1.0,
        )
        m,reward,cfg=build_model(cfg)
        rs=encode_order(m,ref); refE=m.energy(rs)
        print("RUN",deg,cmp_,gate,flush=True)
        t=time.perf_counter()
        ss=QBSolv().sample(m.to_dimod_bqm(),num_repeats=50,seed=SEED+i,
            algorithm=ENERGY_IMPACT,verbosity=-1,timeout=25,
            solver_limit=200,solver="tabu")
        sec=time.perf_counter()-t
        dat=next(ss.data(fields=["sample","energy"],sorted_by="energy"))
        s=[int(dat.sample.get(j,0)) for j in range(m.num_variables)]
        ev=audit(s,m,reward,cfg)
        row={"degree_penalty":deg,"void_penalty":void,
             "comparator_penalty":cmp_,"order_gate_penalty":gate,
             "seconds":sec,"energy_above_reference":ev["energy"]-refE,**ev}
        rows.append(row);print(json.dumps(row),flush=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"rows":rows},indent=2)+"\n")
    print("\nSUMMARY");print(json.dumps(rows,indent=2),flush=True)
if __name__=="__main__":main()
