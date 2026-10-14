"""QBSolv sweep for exact N-1 edge-count latent-rank comparator."""
from __future__ import annotations
import json,time
from pathlib import Path
from dwave_qbsolv import QBSolv,ENERGY_IMPACT

from experimental_chm13_count_constrained_comparator_qubo import (
    build_model,encode_order,audit,CountConstrainedComparatorConfig,DATASET_DIR
)

OUT=Path("debug/qubo/chm13_count_constrained_qbsolv_20260926.json")
SEED=20260926
COUNT_PENALTIES=[8.0,32.0,72.0,144.0]


def main():
    ref=json.loads((DATASET_DIR/"reference_path.json").read_text())["normalized_nodes"]
    rows=[]
    for i,count_penalty in enumerate(COUNT_PENALTIES):
        cfg=CountConstrainedComparatorConfig(count_penalty=count_penalty)
        m,reward,cfg=build_model(cfg)
        rs=encode_order(m,ref);refE=m.energy(rs)
        print("RUN",count_penalty,flush=True)
        t=time.perf_counter()
        ss=QBSolv().sample(
            m.to_dimod_bqm(),
            num_repeats=60,
            seed=SEED+i,
            algorithm=ENERGY_IMPACT,
            verbosity=-1,
            timeout=30,
            solver_limit=200,
            solver="tabu",
        )
        sec=time.perf_counter()-t
        dat=next(ss.data(fields=["sample","energy"],sorted_by="energy"))
        s=[int(dat.sample.get(j,0)) for j in range(m.num_variables)]
        ev=audit(s,m,reward,cfg)
        row={
            "count_penalty":count_penalty,
            "seconds":sec,
            "quadratic_terms":len(m.quadratic),
            "max_abs_linear":max(abs(x) for x in m.linear),
            "max_abs_quadratic":max(abs(x) for x in m.quadratic.values()),
            "reference_energy":refE,
            "energy_above_reference":ev["energy"]-refE,
            "hamming_from_reference":sum(a!=b for a,b in zip(s,rs)),
            **ev,
        }
        rows.append(row)
        print(json.dumps(row),flush=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"solver":"dwave-qbsolv 0.3.4","rows":rows},indent=2)+"\n")
    print("\nSUMMARY")
    print(json.dumps(rows,indent=2),flush=True)


if __name__=="__main__":
    main()
