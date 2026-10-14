"""Exact feasible-path local-optimum audit under structured moves.

Enumerate all 192 CHM13 Hamilton paths. On the feasible path manifold, connect
paths if they differ by:
- one R2 2-switch,
- one genuine R3 reconnect,
- one simultaneous pair of disjoint R2 templates.

Measure:
- number of local maxima by weighted path score;
- whether every non-ground path has an improving structured neighbor;
- greedy best-improvement basin sizes and steps to ground.
"""
from __future__ import annotations
import csv,json,sys
from collections import Counter
from pathlib import Path
import networkx as nx

import benchmark_chm13_fixed_cardinality_kernels as fc
from diagnose_chm13_move_basis_proposal import build_r3_templates, applicable
from benchmark_chm13_projected_edge_hybrid import load_problem
from demo_chm13_edge_ordered_path_qubo import DATASET_DIR

VERIFY=DATASET_DIR/"verification"
sys.path.insert(0,str(VERIFY))
from phase_solver import enumerate_paths  # type: ignore

OUT=Path("debug/qubo/chm13_feasible_structured_landscape_20260929.json")
GROUND=1_343_093


def normalized():
    graph=nx.read_graphml(DATASET_DIR/"graph.graphml")
    with (DATASET_DIR/"nodes.tsv").open(newline="",encoding="utf-8") as h:
        rows=list(csv.DictReader(h,delimiter="\t"))
    mp={r["node"]:r["read_id"] for r in rows}
    enum=enumerate_paths(graph,limit=1000,timeout_ms=60000)
    return [[mp.get(v,v.rstrip("+-")) for v in p] for p in enum["paths"]]


def main():
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    idx={e:i for i,e in enumerate(ep)}
    states=[];scores=[]
    for p in normalized():
        es=list(zip(p,p[1:]))
        if not all(e in idx for e in es):continue
        states.append(frozenset(idx[e] for e in es))
        scores.append(int(sum(reward[e] for e in es)))
    assert len(states)==192
    state_to_i={s:i for i,s in enumerate(states)}
    r2=fc.build_reconnect_templates(ep);r3=build_r3_templates(ep)
    adj=[{} for _ in states]

    for i,st in enumerate(states):
        # R2 / R3 direct feasible neighbors
        for fam,temps in (("r2",r2),("r3",r3)):
            for ti,tpl in enumerate(temps):
                nb=applicable(st,tpl)
                if nb is not None and nb in state_to_i and nb!=st:
                    j=state_to_i[nb]
                    prev=adj[i].get(j)
                    if prev is None or fam=="r3":
                        adj[i][j]={"family":fam,"templates":[ti]}
        # R2-pair direct feasible neighbors
        elig=[]
        for ti,tpl in enumerate(r2):
            nb=applicable(st,tpl)
            if nb is not None:
                elig.append((ti,set(tpl[0]+tpl[1])))
        for a in range(len(elig)):
            ia,fa=elig[a]
            for b in range(a+1,len(elig)):
                ib,fb=elig[b]
                if fa&fb:continue
                nb=frozenset(set(st).symmetric_difference(fa|fb))
                if nb in state_to_i and nb!=st:
                    j=state_to_i[nb]
                    if j not in adj[i]:
                        adj[i][j]={"family":"r2_pair","templates":[ia,ib]}

    local=[]
    improve_counts=[]
    for i in range(len(states)):
        imp=[j for j in adj[i] if scores[j]>scores[i]]
        improve_counts.append(len(imp))
        if not imp:local.append(i)

    ground_i=max(range(len(scores)),key=lambda i:scores[i])
    assert scores[ground_i]==GROUND

    sinks=Counter();steps=[];routes={}
    for start in range(len(states)):
        cur=start;seen=set();route=[]
        while True:
            if cur in seen:
                sinks[("cycle",cur)]+=1;break
            seen.add(cur)
            imp=[j for j in adj[cur] if scores[j]>scores[cur]]
            if not imp:
                sinks[cur]+=1;break
            nxt=max(imp,key=lambda j:(scores[j],-j))
            route.append({
              "from":cur,"to":nxt,"delta_score":scores[nxt]-scores[cur],
              **adj[cur][nxt],
            })
            cur=nxt
        steps.append(len(route))
        routes[start]={"sink":cur,"route":route}

    result={
      "paths":len(states),"ground_index":ground_i,"ground_score":scores[ground_i],
      "local_maxima_count":len(local),
      "local_maxima":[{"index":i,"score":scores[i],"degree":len(adj[i])} for i in local],
      "paths_with_improving_neighbor":sum(c>0 for c in improve_counts),
      "min_improving_neighbors":min(improve_counts),
      "max_improving_neighbors":max(improve_counts),
      "greedy_ground_basin":sum(1 for s,v in routes.items() if v["sink"]==ground_i),
      "greedy_max_steps":max(steps),
      "greedy_step_histogram":dict(sorted(Counter(steps).items())),
      "ground_neighbors":len(adj[ground_i]),
      "runner_up_indices":[i for i,s in enumerate(scores) if s==1_338_209],
      "runner_up_routes":{str(i):routes[i] for i,s in enumerate(scores) if s==1_338_209},
      "edge_counts_by_family":dict(Counter(v["family"] for d in adj for v in d.values())),
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("RESULT",json.dumps(result),flush=True)

if __name__=="__main__":main()
