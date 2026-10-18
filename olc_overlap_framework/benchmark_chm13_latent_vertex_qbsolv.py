"""QBSolv test for the latent vertex-order bounded QUBO."""
from __future__ import annotations
import json,time
from pathlib import Path
from dwave_qbsolv import QBSolv,ENERGY_IMPACT
from experimental_chm13_latent_vertex_order_qubo import (
    build_model,encode_order,audit,LatentVertexOrderConfig,DATASET_DIR
)

OUT=Path("debug/qubo/chm13_latent_vertex_qbsolv_20260926.json")
SIZES=[47,100,200]
SEED=20260926

def main():
    m,reward,cfg=build_model(LatentVertexOrderConfig())
    ref=json.loads((DATASET_DIR/"reference_path.json").read_text())["normalized_nodes"]
    rs=encode_order(m,ref,0); refE=m.energy(rs)
    rows=[]
    for i,size in enumerate(SIZES):
        print("RUN",size,flush=True)
        t=time.perf_counter()
        ss=QBSolv().sample(m.to_dimod_bqm(),num_repeats=50,seed=SEED+i,
            algorithm=ENERGY_IMPACT,verbosity=-1,timeout=30,
            solver_limit=size,solver="tabu")
        sec=time.perf_counter()-t
        dat=next(ss.data(fields=["sample","energy"],sorted_by="energy"))
        s=[int(dat.sample.get(j,0)) for j in range(m.num_variables)]
        ev=audit(s,m,reward,cfg)
        row={"subproblem_size":size,"seconds":sec,
             "energy_above_reference":ev["energy"]-refE,
             "hamming_from_reference":sum(a!=b for a,b in zip(s,rs)),**ev}
        rows.append(row); print(json.dumps(row),flush=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"variables":m.num_variables,"reference_energy":refE,"rows":rows},indent=2)+"\n")
    print("\nSUMMARY");print(json.dumps(rows,indent=2),flush=True)
if __name__=="__main__":main()
