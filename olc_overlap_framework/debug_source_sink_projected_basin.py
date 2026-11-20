"""Diagnose the 141-edge source/sink projected-hybrid basin."""
from __future__ import annotations
import json
import networkx as nx

from benchmark_chm13_projected_edge_hybrid import (
    SEED,load_problem,best_endpoints,project_rank,graph_metrics,
    build_bqm_source_sink,sample_edges,selected_degrees,
)

def path_components(rids,selected):
    g=nx.DiGraph();g.add_nodes_from(rids);g.add_edges_from(selected)
    indeg,outdeg=selected_degrees(rids,selected)
    starts=[r for r in rids if indeg[r]==0 and outdeg[r]<=1]
    nxt={u:v for u,v in selected}
    comps=[]
    seen=set()
    for s in starts:
        cur=s;path=[]
        while cur not in seen:
            seen.add(cur);path.append(cur)
            if cur not in nxt:break
            cur=nxt[cur]
        if path:comps.append(path)
    return comps

def fixed_degree_residual(rids,selected,source,sink):
    indeg,outdeg=selected_degrees(rids,selected)
    return sum((1-int(r==source)-indeg[r])**2+(1-int(r==sink)-outdeg[r])**2 for r in rids)

def main():
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    import random
    rng=random.Random(SEED)
    ranks={r:i for i,r in enumerate(rng.sample(rids,len(rids)))}
    selected=set(); source,sink=best_endpoints(rids,selected)
    schedule=[0.0,0.25,0.5,1.0,2.0,4.0,8.0,16.0,32.0]
    for it in range(20):
        op=schedule[min(it,len(schedule)-1)]
        if it>0:source,sink=best_endpoints(rids,selected)
        bqm=build_bqm_source_sink(ep,cost,incoming,outgoing,ranks,source,sink,144.0,op)
        selected=sample_edges(bqm,selected if selected else None,1200,SEED+it)
        ranks=project_rank(rids,selected)

    comps=path_components(rids,selected)
    edge_set=set(ep)
    endpoints=[(c[0],c[-1],len(c)) for c in comps]
    direct=[]
    import itertools
    for perm in itertools.permutations(range(len(comps))):
        conns=[]
        ok=True
        for a,b in zip(perm,perm[1:]):
            e=(comps[a][-1],comps[b][0])
            conns.append(e)
            if e not in edge_set:ok=False
        if ok:direct.append({"component_order":perm,"connectors":conns})

    # Single additions that reduce fixed endpoint degree residual.
    base_res=fixed_degree_residual(rids,selected,source,sink)
    one_add=[]
    for e in edge_set-selected:
        s2=set(selected);s2.add(e)
        rr=fixed_degree_residual(rids,s2,source,sink)
        if rr<base_res:
            one_add.append((rr,e))

    print(json.dumps({
      "selected_edges":len(selected),
      "fixed_source":source,"fixed_sink":sink,
      "fixed_degree_residual_sq":base_res,
      "component_count":len(comps),
      "components":endpoints,
      "direct_two_connector_merges":direct,
      "one_edge_additions_reducing_fixed_degree_residual":sorted(one_add)[:20],
      "num_one_edge_improvers":len(one_add),
    },indent=2))

if __name__=="__main__":main()
