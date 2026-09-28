"""Exact reconnect-state-graph audit for CHM13 complex144.

State space:
- exactly N-1 selected directed edges;
- exact in/out degree profile of the benchmark Hamilton paths:
    source: in=0,out=1
    sink:   in=1,out=0
    others: in=1,out=1.

Therefore every state is one directed source->sink path plus zero or more
vertex-disjoint directed cycles.

Move:
directed 2-switch reconnect
    (a->b, c->d) <-> (a->d, c->b)
using only candidate OLC edges.

This move preserves every vertex's in/out degree and selected-edge count.

The script exhaustively BFSes the connected component(s) induced by the static
reconnect templates and checks whether reconnect-only motion can connect the
two previously observed 96-Hamilton-path components by passing through
path+cycle intermediate states.
"""
from __future__ import annotations
import json, sys
from collections import deque, Counter
from pathlib import Path
import networkx as nx

from benchmark_chm13_projected_edge_hybrid import load_problem
from benchmark_chm13_fixed_cardinality_kernels import build_reconnect_templates
from demo_chm13_edge_ordered_path_qubo import DATASET_DIR

VERIFY=DATASET_DIR/"verification"
sys.path.insert(0,str(VERIFY))
from phase_solver import enumerate_paths  # type: ignore

OUT=Path("debug/qubo/chm13_reconnect_state_graph_20260928.json")
CAP=2_000_000

def state_from_order(order,idx):
    return frozenset(idx[(u,v)] for u,v in zip(order,order[1:]))

def cycle_count_from_state(state,ep,nodes):
    g=nx.DiGraph();g.add_nodes_from(nodes);g.add_edges_from(ep[i] for i in state)
    return len(list(nx.simple_cycles(g)))

def neighbors(state,templates):
    s=set(state)
    for side1,side2 in templates:
        a=all(i in s for i in side1) and all(i not in s for i in side2)
        b=all(i in s for i in side2) and all(i not in s for i in side1)
        if not (a or b):
            continue
        yield frozenset(s.symmetric_difference(side1+side2))

def bfs(start,templates,ham_states):
    q=deque([start]);dist={start:0}
    ham_seen=set()
    cyc_hist=Counter()
    min_cycle_dist={}
    while q:
        st=q.popleft();d=dist[st]
        if st in ham_states:
            ham_seen.add(st)
        cc=cycle_count_from_state(st,EP,RIDS)
        cyc_hist[cc]+=1
        min_cycle_dist.setdefault(cc,d)
        if len(dist)>=CAP:
            return dist,ham_seen,cyc_hist,min_cycle_dist,True
        for nb in neighbors(st,templates):
            if nb not in dist:
                dist[nb]=d+1;q.append(nb)
    return dist,ham_seen,cyc_hist,min_cycle_dist,False

RIDS=[];EP=[]

def main():
    global RIDS,EP
    # Use exactly the same graph that the exact Hamilton-path enumerator uses.
    # This avoids any representation/filtering mismatch with projected-solver
    # helper loaders.
    graph=nx.read_graphml(DATASET_DIR/"graph.graphml")
    RIDS=list(graph.nodes())
    EP=list(graph.edges())
    idx={e:i for i,e in enumerate(EP)}
    templates=build_reconnect_templates(EP)

    enum=enumerate_paths(graph,limit=1000,timeout_ms=60000)
    paths=enum["paths"]
    ham_states={state_from_order(p,idx):i for i,p in enumerate(paths)}
    assert len(ham_states)==192

    ref=json.loads((DATASET_DIR/"reference_path.json").read_text())["nodes"]
    start=state_from_order(ref,idx)

    dist,seen,hist,mincyc,capped=bfs(start,templates,set(ham_states))
    seen_indices=sorted(ham_states[s] for s in seen)

    # If not all Hamilton paths are reachable, start BFS from the first unseen
    # Hamilton path to determine reconnect components.
    components=[{
      "representative_hamilton_index":ham_states[start],
      "states":len(dist),
      "hamilton_paths":len(seen),
      "hamilton_indices":seen_indices,
      "cycle_count_histogram":dict(sorted(hist.items())),
      "min_distance_by_cycle_count":dict(sorted(mincyc.items())),
      "max_bfs_distance":max(dist.values()) if dist else 0,
      "capped":capped,
    }]
    covered=set(seen)
    if not capped:
        for st,i in ham_states.items():
            if st in covered:
                continue
            d2,s2,h2,m2,c2=bfs(st,templates,set(ham_states))
            components.append({
              "representative_hamilton_index":i,
              "states":len(d2),
              "hamilton_paths":len(s2),
              "hamilton_indices":sorted(ham_states[s] for s in s2),
              "cycle_count_histogram":dict(sorted(h2.items())),
              "min_distance_by_cycle_count":dict(sorted(m2.items())),
              "max_bfs_distance":max(d2.values()) if d2 else 0,
              "capped":c2,
            })
            covered.update(s2)
            if c2: break

    result={
      "nodes":len(RIDS),"candidate_edges":len(EP),
      "reconnect_templates":len(templates),
      "exact_hamilton_paths":len(ham_states),
      "enumeration_end":enum["end"],
      "reconnect_components_touching_hamilton_paths":len(components),
      "all_hamilton_paths_covered":len(covered)==len(ham_states),
      "components":components,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
