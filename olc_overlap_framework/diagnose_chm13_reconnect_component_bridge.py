"""Find the minimum degree-preserving switch size that can bridge the two
R2 reconnect components of CHM13 complex144.

Enumerate the two exact connected components under the existing directed
2-switch templates, encode each state as a 261-bit integer, and find the minimum
symmetric-difference distance between states in different components.

For equal degree vectors, the symmetric difference of two states decomposes into
alternating cycles in the bipartite tail/head representation.  If the minimum
cross-component difference is 2r edges total (r removed + r added), then any
single direct bridge must modify at least r selected edges and r unselected
edges.  We additionally decompose one minimum witness into alternating cycles.
"""
from __future__ import annotations
import json, sys
from collections import deque
from pathlib import Path
import networkx as nx

from benchmark_chm13_projected_edge_hybrid import load_problem
from benchmark_chm13_fixed_cardinality_kernels import build_reconnect_templates
from demo_chm13_edge_ordered_path_qubo import DATASET_DIR

VERIFY=DATASET_DIR/"verification"
sys.path.insert(0,str(VERIFY))
from phase_solver import enumerate_paths  # type: ignore

OUT=Path("debug/qubo/chm13_reconnect_component_bridge_20260928.json")

def state_from_order(order,idx):
    return frozenset(idx[(u,v)] for u,v in zip(order,order[1:]))

def neighbors(st,templates):
    s=set(st)
    for side1,side2 in templates:
        a=all(i in s for i in side1) and all(i not in s for i in side2)
        b=all(i in s for i in side2) and all(i not in s for i in side1)
        if a or b:
            yield frozenset(s.symmetric_difference(side1+side2))

def bfs(start,templates):
    q=deque([start]);seen={start}
    while q:
        s=q.popleft()
        for t in neighbors(s,templates):
            if t not in seen:
                seen.add(t);q.append(t)
    return seen

def mask(st):
    z=0
    for i in st:z|=1<<i
    return z

def alternating_components(a,b,ep):
    removed=[ep[i] for i in sorted(a-b)]
    added=[ep[i] for i in sorted(b-a)]
    # bipartite graph: ("o",tail), ("i",head); color 0 removed, 1 added
    g=nx.Graph()
    for color,edges in ((0,removed),(1,added)):
        for u,v in edges:
            g.add_edge(("o",u),("i",v),color=color,edge=(u,v))
    comps=[]
    for nodes in nx.connected_components(g):
        sub=g.subgraph(nodes)
        rem=[d["edge"] for *_,d in sub.edges(data=True) if d["color"]==0]
        add=[d["edge"] for *_,d in sub.edges(data=True) if d["color"]==1]
        comps.append({"removed":rem,"added":add,"r":len(rem),"vertices":len(nodes)})
    return comps

def main():
    graph=nx.read_graphml(DATASET_DIR/"graph.graphml")
    ep=list(graph.edges());idx={e:i for i,e in enumerate(ep)}
    templates=build_reconnect_templates(ep)
    enum=enumerate_paths(graph,limit=1000,timeout_ms=60000)
    paths=enum["paths"];ham=[state_from_order(p,idx) for p in paths]
    c1=bfs(ham[0],templates)
    seed2=next(h for h in ham if h not in c1)
    c2=bfs(seed2,templates)
    assert c1.isdisjoint(c2)

    m1=[(mask(s),s) for s in c1];m2=[(mask(s),s) for s in c2]
    best=10**9;w=None
    for z1,s1 in m1:
        for z2,s2 in m2:
            d=(z1^z2).bit_count()
            if d<best:
                best=d;w=(s1,s2)
    s1,s2=w
    result={
      "component_sizes":[len(c1),len(c2)],
      "min_cross_component_hamming_bits":best,
      "minimum_removed_edges":best//2,
      "minimum_added_edges":best//2,
      "alternating_components":alternating_components(s1,s2,ep),
      "witness_removed":[ep[i] for i in sorted(s1-s2)],
      "witness_added":[ep[i] for i in sorted(s2-s1)],
      "witness1_is_hamilton":s1 in set(ham),
      "witness2_is_hamilton":s2 in set(ham),
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
