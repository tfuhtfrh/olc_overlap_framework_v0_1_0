"""Extended projected SQA search for CHM13 exact-count edge QUBO.

Goal: determine whether OpenJij SQA can reach the certified weighted optimum,
not merely the first feasible Hamilton path.

Only real edge bits are sampled.  Classical rank projection is recomputed after
every SQA solve.  The order-penalty schedule is repeatedly reset to zero to
allow the path topology to escape the ordering induced by a previously feasible
path, then raised again to remove cycles.
"""
from __future__ import annotations
import json
from pathlib import Path

from benchmark_chm13_projected_edge_sqa import sample_edges_sqa
from benchmark_chm13_projected_edge_hybrid import (
    load_problem,project_rank,graph_metrics,build_bqm_count,
)

OUT=Path("debug/qubo/chm13_projected_edge_sqa_extended_20260926.json")
BASE_SEED=20260926
CERT_SCORE=1343093
SCHEDULE=[0.0,0.25,0.5,1.0,2.0,4.0]


def main():
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    rows=[]
    best=None

    for run in range(10):
        # Different deterministic initial rank per run.
        import random
        rng=random.Random(BASE_SEED+10000*run)
        ranks={r:i for i,r in enumerate(rng.sample(rids,len(rids)))}
        selected=set()

        for it in range(18):
            op=SCHEDULE[it % len(SCHEDULE)]
            bqm=build_bqm_count(
                ep,cost,incoming,outgoing,ranks,len(rids),
                degree_conflict_penalty=288.0,
                count_penalty=32.0,
                order_penalty=op,
            )
            selected,sec,energy=sample_edges_sqa(
                bqm,
                selected if selected else None,
                BASE_SEED+10000*run+it,
            )
            ranks=project_rank(rids,selected)
            metrics=graph_metrics(rids,selected,ranks,reward)
            row={
                "run":run,"iteration":it,"order_penalty":op,
                "seconds":sec,"inner_qubo_energy":energy,**metrics,
            }
            rows.append(row)
            if metrics["valid_path"]:
                if best is None or metrics["path_score"]>best["path_score"]:
                    best=dict(row)
                print("FEASIBLE",json.dumps(row),flush=True)
                if metrics["path_score"]==CERT_SCORE:
                    OUT.parent.mkdir(parents=True,exist_ok=True)
                    OUT.write_text(json.dumps({"best":best,"rows":rows},indent=2)+"\n")
                    print("OPTIMUM",json.dumps(best),flush=True)
                    return

    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"best":best,"rows":rows},indent=2)+"\n")
    print("BEST",json.dumps(best),flush=True)


if __name__=="__main__":
    main()
