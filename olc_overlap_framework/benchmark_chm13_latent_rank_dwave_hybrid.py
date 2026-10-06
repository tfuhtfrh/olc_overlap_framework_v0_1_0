"""Modern dwave-hybrid classical decomposition benchmark for CHM13 latent-rank comparator.

Closest maintained analogue to historical QBSolv:
- whole-problem interruptible Tabu branch
- energy-impact decomposition branch
- Tabu on selected subproblem
- compose improved subproblem back into global state
- race branches and iterate until no improvement

No QPU is used in this benchmark.
"""
from __future__ import annotations
import json,time
from pathlib import Path
import hybrid

from experimental_chm13_latent_rank_comparator_qubo import (
    build_model,encode_order,audit,LatentRankComparatorConfig,DATASET_DIR
)

OUT=Path("debug/qubo/chm13_latent_rank_dwave_hybrid_20260926.json")
SIZES=[100,200,400]


def solve(bqm,size):
    # Priority-first traversal keeps the energy-impact block connected to the
    # highest-impact seed more often than plain energy ordering.
    subbranch=(
        hybrid.EnergyImpactDecomposer(
            size=size,
            rolling=True,
            rolling_history=0.5,
            traversal="pfs",
        )
        | hybrid.TabuSubproblemSampler(num_reads=1,timeout=250)
        | hybrid.SplatComposer()
    )
    iteration=(
        hybrid.RacingBranches(
            hybrid.InterruptableTabuSampler(timeout=250,max_time=1000),
            subbranch,
        )
        | hybrid.ArgMin()
    )
    workflow=hybrid.LoopUntilNoImprovement(
        iteration,
        convergence=8,
        max_iter=60,
    )
    init=hybrid.State.from_problem(bqm)
    t=time.perf_counter()
    final=workflow.run(init).result()
    sec=time.perf_counter()-t
    return final.samples.first.sample,sec


def main():
    cfg=LatentRankComparatorConfig(
        degree_penalty=288.0,
        void_penalty=288.0,
        comparator_penalty=576.0,
        order_gate_penalty=288.0,
        edge_cost_scale=1.0,
    )
    m,reward,cfg=build_model(cfg)
    ref=json.loads((DATASET_DIR/"reference_path.json").read_text())["normalized_nodes"]
    rs=encode_order(m,ref);refE=m.energy(rs)
    rows=[]
    bqm=m.to_dimod_bqm()
    for size in SIZES:
        print("RUN",size,flush=True)
        sample_map,sec=solve(bqm,size)
        s=[int(sample_map.get(j,0)) for j in range(m.num_variables)]
        ev=audit(s,m,reward,cfg)
        row={
            "subproblem_size":size,
            "seconds":sec,
            "energy_above_reference":ev["energy"]-refE,
            "hamming_from_reference":sum(a!=b for a,b in zip(s,rs)),
            **ev,
        }
        rows.append(row);print(json.dumps(row),flush=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"solver":"dwave-hybrid classical tabu decomposition","rows":rows},indent=2)+"\n")
    print("\nSUMMARY");print(json.dumps(rows,indent=2),flush=True)


if __name__=="__main__":
    main()
