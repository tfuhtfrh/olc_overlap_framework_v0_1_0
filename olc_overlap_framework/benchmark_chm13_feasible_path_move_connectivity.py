"""Connectivity of the exact CHM13 Hamilton-path solution set under legal relocation moves.

Enumerate all Hamilton paths in the normalized graph using the packaged exact
SMT model, then build a meta-graph whose vertices are feasible Hamilton paths.

Two paths are adjacent when one can be obtained from the other by:
1) relocating one read, or
2) relocating one contiguous segment without reversal.

Because both endpoints are already known feasible Hamilton paths, no extra
edge-feasibility check is needed for the pairwise adjacency test.

This tests connectivity of the FEASIBLE path space under these move classes;
it does not use scores to construct adjacency.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
import networkx as nx

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/"test_data"/"CHM13"/"CHM13_complex144"
VERIFY=DATA/"verification"
sys.path.insert(0,str(VERIFY))
from phase_solver import enumerate_paths  # type: ignore

OUT=Path("debug/qubo/chm13_feasible_path_move_connectivity_20260926.json")


def single_relocation(a,b):
    n=len(a)
    if a==b:return False
    # b must equal a with one item removed and inserted elsewhere.
    for i in range(n):
        x=a[i]
        rem=a[:i]+a[i+1:]
        try:j=b.index(x)
        except ValueError:return False
        if b[:j]+b[j+1:]==rem:
            return True
    return False


def one_segment_relocation(a,b,max_len=None):
    n=len(a)
    if a==b:return False
    posb={v:i for i,v in enumerate(b)}
    # A moved segment remains contiguous and same order in b.
    for i in range(n):
        lim=n if max_len is None else min(n,i+max_len)
        for j in range(i,lim):
            seg=a[i:j+1]
            k=posb[seg[0]]
            if k+len(seg)>n or b[k:k+len(seg)]!=seg:
                continue
            rem_a=a[:i]+a[j+1:]
            rem_b=b[:k]+b[k+len(seg):]
            if rem_a==rem_b:
                return True
    return False


def components(paths,predicate):
    g=nx.Graph();g.add_nodes_from(range(len(paths)))
    for i in range(len(paths)):
        for j in range(i+1,len(paths)):
            if predicate(paths[i],paths[j]) or predicate(paths[j],paths[i]):
                g.add_edge(i,j)
    comps=sorted((sorted(c) for c in nx.connected_components(g)),key=len,reverse=True)
    summaries=[]
    for comp in comps:
        endpoint_pairs=sorted({(paths[i][0],paths[i][-1]) for i in comp})
        starts=sorted({paths[i][0] for i in comp})
        ends=sorted({paths[i][-1] for i in comp})
        summaries.append({
          "size":len(comp),
          "starts":starts,
          "ends":ends,
          "endpoint_pair_count":len(endpoint_pairs),
          "endpoint_pairs":endpoint_pairs,
        })
    return {
      "edges":g.number_of_edges(),
      "components":len(comps),
      "largest_component":len(comps[0]) if comps else 0,
      "component_sizes":[len(c) for c in comps],
      "isolated_paths":sum(1 for c in comps if len(c)==1),
      "component_summaries":summaries,
      "_component_indices":comps,
    }


def main():
    g=nx.read_graphml(DATA/"graph.graphml")
    enum=enumerate_paths(g,limit=1000,timeout_ms=60000)
    paths=enum["paths"];scores=enum["scores"]
    result={
      "n_paths":len(paths),"enumeration_end":enum["end"],
      "single_read_relocation":components(paths,single_relocation),
      "segment_relocation_len_le_4":components(paths,lambda a,b:one_segment_relocation(a,b,4)),
      "segment_relocation_len_le_16":components(paths,lambda a,b:one_segment_relocation(a,b,16)),
      "arbitrary_single_segment_relocation":components(paths,lambda a,b:one_segment_relocation(a,b,None)),
    }
    best=max(range(len(paths)),key=lambda i:scores[i])
    result["optimum_index"]=best;result["optimum_score"]=scores[best]
    for key in ("single_read_relocation","segment_relocation_len_le_4","segment_relocation_len_le_16","arbitrary_single_segment_relocation"):
        comps=result[key].pop("_component_indices")
        result[key]["optimum_component"]=next(ci for ci,comp in enumerate(comps) if best in comp)
        result[key]["component_score_ranges"]=[
            {"min":min(scores[i] for i in comp),"max":max(scores[i] for i in comp)}
            for comp in comps
        ]
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
