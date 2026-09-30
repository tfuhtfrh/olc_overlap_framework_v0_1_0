"""Find the shortest open alternating trail that transfers the CHM13 stuck sink.

A degree-correct N-1 edge state has one source and one sink.  Closed alternating
cycles preserve the source/sink identities.  Here the stuck source matches the
unique source of all exact Hamilton paths, but the sink does not.

Build the residual bipartite graph:
  unselected u->v : u_out -> v_in   (add)
  selected   u->v : v_in  -> u_out  (remove)

A directed residual path from current_sink_out to target_sink_out alternates
add/remove, preserves cardinality, preserves every indegree, preserves all
outdegrees except the two endpoints, and transfers the sink defect.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
from collections import deque
import networkx as nx

import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import graph_metrics, selected_degrees
from demo_chm13_edge_ordered_path_qubo import DATASET_DIR
from diagnose_chm13_move_basis_proposal import replay_stuck_checkpoint

VERIFY=DATASET_DIR/"verification"
sys.path.insert(0,str(VERIFY))
from phase_solver import enumerate_paths  # type: ignore

OUT=Path("debug/qubo/chm13_open_alternating_sink_transfer_20260928.json")


def exact_normalized_paths():
    graph=nx.read_graphml(DATASET_DIR/"graph.graphml")
    with (DATASET_DIR/"nodes.tsv").open(newline="",encoding="utf-8") as handle:
        rows=list(csv.DictReader(handle,delimiter="\t"))
    mp={row["node"]:row["read_id"] for row in rows}
    paths=enumerate_paths(graph,limit=1000,timeout_ms=60000)["paths"]
    return [[mp.get(v,v.rstrip("+-")) for v in p] for p in paths]


def build_residual(ep,selected):
    # vertices are tuples ("out"/"in", read_id); arcs carry edge index and action.
    adj={}
    def add(a,b,idx,action):
        adj.setdefault(a,[]).append((b,idx,action))
    for i,(u,v) in enumerate(ep):
        if (u,v) in selected:
            add(("in",v),("out",u),i,"remove")
        else:
            add(("out",u),("in",v),i,"add")
    return adj


def bfs(adj,start,goal):
    q=deque([start]);prev={start:None};how={}
    while q:
        a=q.popleft()
        if a==goal:break
        for b,idx,action in adj.get(a,()):
            if b in prev:continue
            prev[b]=a;how[b]=(idx,action);q.append(b)
    if goal not in prev:return None
    nodes=[];steps=[];cur=goal
    while cur is not None:
        nodes.append(cur)
        p=prev[cur]
        if p is not None:
            idx,action=how[cur]
            steps.append((p,cur,idx,action))
        cur=p
    nodes.reverse();steps.reverse()
    return nodes,steps


def component_summary(rids,selected):
    indeg,outdeg=selected_degrees(rids,selected)
    return {
      "sources":[r for r in rids if indeg[r]==0],
      "sinks":[r for r in rids if outdeg[r]==0],
      "degree_conflicts":sum(max(0,indeg[r]-1)+max(0,outdeg[r]-1) for r in rids),
    }


def main():
    rids,ep,reward,cost,incoming,outgoing,selected,ranks,op=replay_stuck_checkpoint()
    base=component_summary(rids,selected)
    assert len(base["sources"])==1 and len(base["sinks"])==1
    current_sink=base["sinks"][0]

    paths=exact_normalized_paths()
    sources={p[0] for p in paths};sinks={p[-1] for p in paths}
    assert len(sources)==1 and len(sinks)==1
    target_source=next(iter(sources));target_sink=next(iter(sinks))

    residual=build_residual(ep,selected)
    found=bfs(residual,("out",current_sink),("out",target_sink))
    if found is None:
        result={
          "current_source":base["sources"][0],"current_sink":current_sink,
          "target_source":target_source,"target_sink":target_sink,
          "residual_path_found":False,
        }
    else:
        nodes,steps=found
        flip_idx=[s[2] for s in steps]
        actions=[s[3] for s in steps]
        new=set(selected)
        for i in flip_idx:
            e=ep[i]
            if e in new:new.remove(e)
            else:new.add(e)

        fixed_bqm=fc.build_bqm_fixed_cardinality(
            ep,cost,incoming,outgoing,ranks,
            degree_conflict_penalty=288.0,order_penalty=op)
        abqm=fc.ArrayBQM.from_bqm(fixed_bqm,ep)
        idx={e:i for i,e in enumerate(ep)}
        x=[1 if e in selected else 0 for e in ep]
        de=abqm.delta_flipset(x,tuple(flip_idx))
        new_summary=component_summary(rids,new)
        metrics=graph_metrics(rids,new,ranks,reward)

        result={
          "current_source":base["sources"][0],"current_sink":current_sink,
          "target_source":target_source,"target_sink":target_sink,
          "residual_path_found":True,
          "residual_arc_length":len(steps),
          "exchange_order_k":len(steps)//2,
          "actions":actions,
          "n_add":sum(a=="add" for a in actions),
          "n_remove":sum(a=="remove" for a in actions),
          "flip_edges":[{"edge":list(ep[i]),"action":a} for (_,_,i,a) in steps],
          "classical_delta_fixed_bqm":de,
          "new_degree_summary":new_summary,
          "new_metrics":metrics,
          "directly_hamilton_path":metrics["valid_path"],
          "path_score":metrics["path_score"],
        }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("RESULT",json.dumps(result),flush=True)

if __name__=="__main__":main()
