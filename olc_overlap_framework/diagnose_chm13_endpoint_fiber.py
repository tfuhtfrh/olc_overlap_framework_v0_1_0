"""Audit whether the CHM13 stuck degree fiber contains any Hamilton path.

Closed alternating-cycle moves R_k preserve every vertex's in/out degree.
Therefore they preserve the source and sink identities of a degree-correct
143-edge state.  If no exact Hamilton path has the same endpoint pair, no
sequence of R_k moves of any order can ever reach a Hamilton path.
"""
from __future__ import annotations
import csv, json, sys
from collections import Counter
from pathlib import Path
import networkx as nx

from benchmark_chm13_projected_edge_hybrid import load_problem, selected_degrees
from demo_chm13_edge_ordered_path_qubo import DATASET_DIR
from diagnose_chm13_move_basis_proposal import replay_stuck_checkpoint

VERIFY=DATASET_DIR/"verification"
sys.path.insert(0,str(VERIFY))
from phase_solver import enumerate_paths  # type: ignore

OUT=Path("debug/qubo/chm13_stuck_endpoint_fiber_20260928.json")


def main():
    rids,ep,reward,cost,incoming,outgoing,selected,ranks,op=replay_stuck_checkpoint()
    indeg,outdeg=selected_degrees(rids,selected)
    sources=[r for r in rids if indeg[r]==0 and outdeg[r]<=1]
    sinks=[r for r in rids if outdeg[r]==0 and indeg[r]<=1]
    assert len(sources)==1 and len(sinks)==1
    source,sink=sources[0],sinks[0]

    graph=nx.read_graphml(DATASET_DIR/"graph.graphml")
    with (DATASET_DIR/"nodes.tsv").open(newline="",encoding="utf-8") as handle:
        rows=list(csv.DictReader(handle,delimiter="\t"))
    oriented_to_normalized={row["node"]:row["read_id"] for row in rows}
    enum=enumerate_paths(graph,limit=1000,timeout_ms=60000)
    paths=enum["paths"]
    normalized_paths=[
        [oriented_to_normalized.get(v,v.rstrip("+-")) for v in p]
        for p in paths
    ]
    endpoint_counts=Counter((p[0],p[-1]) for p in normalized_paths)
    same=[p for p in normalized_paths if p[0]==source and p[-1]==sink]
    reverse=[p for p in normalized_paths if p[0]==sink and p[-1]==source]

    result={
      "stuck_source":source,
      "stuck_sink":sink,
      "exact_hamilton_paths":len(paths),
      "endpoint_ids_compared_after_orientation_to_normalized_mapping":True,
      "distinct_endpoint_pairs":len(endpoint_counts),
      "same_endpoint_hamilton_paths":len(same),
      "reverse_endpoint_hamilton_paths":len(reverse),
      "endpoint_pair_counts":[
        {"source":a,"sink":b,"count":n}
        for (a,b),n in endpoint_counts.most_common()
      ],
      "conclusion":(
        "closed alternating-cycle moves preserve this endpoint pair; "
        + ("the endpoint fiber contains Hamilton paths"
           if same else
           "the endpoint fiber contains NO Hamilton path, so no sequence of R_k closed-cycle moves can solve this stuck state")
      ),
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("RESULT",json.dumps(result),flush=True)

if __name__=="__main__":
    main()
