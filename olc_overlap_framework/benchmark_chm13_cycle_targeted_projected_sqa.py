"""Cycle-targeted projected SQA for CHM13.

Alternative to global rank/backward-edge penalties:
- inner QUBO uses only real edge bits with weight + degree-conflict + exact N-1 count;
- after each SQA sample, detect the ACTUAL selected directed cycles;
- increase a classical Lagrangian-like linear multiplier only on edges belonging
  to those cycles;
- rebuild and rerun SQA.

This resembles the lazy-cycle idea in the earlier external PHIX design, but
keeps the inner problem a pure edge-only QUBO and uses soft adaptive cycle
multipliers rather than an exact high-order subtour cut.
"""
from __future__ import annotations
import json,random
from pathlib import Path
import networkx as nx

from benchmark_chm13_projected_edge_sqa import sample_edges_sqa
from benchmark_chm13_projected_edge_hybrid import (
    load_problem,project_rank,graph_metrics,build_bqm_count,
)

OUT=Path("debug/qubo/chm13_cycle_targeted_projected_sqa_20260926.json")
BASE_SEED=20260926
CERT=1343093
RUNS=10
ITERS=18
ETA=0.5


def selected_cycles(rids,selected):
    g=nx.DiGraph();g.add_nodes_from(rids);g.add_edges_from(selected)
    return [list(c) for c in nx.simple_cycles(g)]


def cycle_edges(c):
    return list(zip(c,c[1:]+c[:1]))


def main():
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    rows=[];best=None
    for run in range(RUNS):
        penalties={e:0.0 for e in ep}
        selected=set()
        # rank is irrelevant here, pass any fixed rank with order penalty 0.
        ranks={r:i for i,r in enumerate(rids)}

        for it in range(ITERS):
            bqm=build_bqm_count(
                ep,cost,incoming,outgoing,ranks,len(rids),
                degree_conflict_penalty=288.0,
                count_penalty=32.0,
                order_penalty=0.0,
            )
            for e,p in penalties.items():
                if p:
                    bqm.add_variable(e,p)

            selected,sec,energy=sample_edges_sqa(
                bqm,selected if selected else None,BASE_SEED+10000*run+it
            )
            # Classical topological projection only for diagnostics.
            proj=project_rank(rids,selected)
            metrics=graph_metrics(rids,selected,proj,reward)
            cycles=selected_cycles(rids,selected)

            row={
                "run":run,"iteration":it,"eta":ETA,
                "seconds":sec,"inner_qubo_energy":energy,
                "cycle_lengths":sorted(len(c) for c in cycles),
                "penalized_edges":sum(p>0 for p in penalties.values()),
                "max_cycle_multiplier":max(penalties.values(),default=0.0),
                **metrics,
            }
            rows.append(row)
            if metrics["valid_path"]:
                if best is None or metrics["path_score"]>best["path_score"]:
                    best=dict(row)
                print("FEASIBLE",json.dumps(row),flush=True)
                if metrics["path_score"]==CERT:
                    OUT.parent.mkdir(parents=True,exist_ok=True)
                    OUT.write_text(json.dumps({"best":best,"rows":rows},indent=2)+"\n")
                    print("OPTIMUM",json.dumps(best),flush=True)
                    return

            # Soft lazy update: only edges in cycles actually observed are hit.
            for cyc in cycles:
                for e in cycle_edges(cyc):
                    penalties[e]+=ETA

    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"best":best,"rows":rows},indent=2)+"\n")
    print("BEST",json.dumps(best),flush=True)


if __name__=="__main__":
    main()
