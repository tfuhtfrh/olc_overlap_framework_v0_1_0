"""Exact runner-up -> ground move-geometry diagnostic for CHM13 complex144.

Re-enumerate the 192 exact Hamilton paths, identify the certified optimum
(1,343,093) and runner-up (1,338,209), then characterize the exact transition:
- symmetric-difference alternating components;
- whether one R2 / R3 / compound R2 pair reaches ground directly;
- feasible-path adjacency by single-read / contiguous-segment relocation;
- shortest feasible-only path under R2+R3 moves;
- rank-free and stale-rank energy differences;
- single-slice versus whole-worldline path-integral action estimates.
"""
from __future__ import annotations

import csv, json, math, sys
from collections import deque
from pathlib import Path
import networkx as nx

import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import (
    load_problem, project_rank, graph_metrics,
)
from diagnose_chm13_move_basis_proposal import build_r3_templates, applicable
from demo_chm13_edge_ordered_path_qubo import DATASET_DIR

VERIFY=DATASET_DIR/"verification"
sys.path.insert(0,str(VERIFY))
from phase_solver import enumerate_paths  # type: ignore

OUT=Path("debug/qubo/chm13_runnerup_ground_geometry_20260929.json")
GROUND=1_343_093
RUNNER=1_338_209
BETA=4.0
P=8
GAMMA=0.03


def normalized_paths():
    graph=nx.read_graphml(DATASET_DIR/"graph.graphml")
    with (DATASET_DIR/"nodes.tsv").open(newline="",encoding="utf-8") as h:
        rows=list(csv.DictReader(h,delimiter="\t"))
    mp={row["node"]:row["read_id"] for row in rows}
    enum=enumerate_paths(graph,limit=1000,timeout_ms=60000)
    return [[mp.get(v,v.rstrip("+-")) for v in p] for p in enum["paths"]]


def score(order,reward):
    es=list(zip(order,order[1:]))
    if not all(e in reward for e in es): return None
    return int(sum(reward[e] for e in es))


def state(order):
    return set(zip(order,order[1:]))


def alternating_components(a,b):
    rem=a-b;add=b-a
    g=nx.Graph()
    for lab,edges in (("remove",rem),("add",add)):
        for u,v in edges:
            g.add_edge(("out",u),("in",v),label=lab,edge=(u,v))
    out=[]
    for nodes in nx.connected_components(g):
        sub=g.subgraph(nodes)
        rr=[];aa=[]
        for _,_,d in sub.edges(data=True):
            (rr if d["label"]=="remove" else aa).append(d["edge"])
        deg1=[n for n in nodes if sub.degree(n)==1]
        out.append({
          "type":"open_path" if deg1 else "cycle",
          "removed":sorted(rr),"added":sorted(aa),
          "k_remove":len(rr),"k_add":len(aa),
          "bipartite_vertices":len(nodes),
        })
    return sorted(out,key=lambda x:(x["type"],x["k_remove"],x["removed"]))


def single_relocation(a,b):
    n=len(a)
    for i,x in enumerate(a):
        rem=a[:i]+a[i+1:]
        j=b.index(x)
        if b[:j]+b[j+1:]==rem:return {"from_index":i,"to_index":j,"read":x}
    return None


def segment_relocation(a,b):
    n=len(a);pos={v:i for i,v in enumerate(b)}
    best=None
    for i in range(n):
        for j in range(i,n):
            seg=a[i:j+1];k=pos[seg[0]]
            if k+len(seg)>n or b[k:k+len(seg)]!=seg:continue
            if a[:i]+a[j+1:]==b[:k]+b[k+len(seg):]:
                cand={"from":[i,j],"to_index":k,"length":len(seg),"segment":seg}
                if best is None or cand["length"]<best["length"]:best=cand
    return best


def array_bqm(ep,cost,incoming,outgoing,ranks,order_penalty):
    bqm=fc.build_bqm_fixed_cardinality(
        ep,cost,incoming,outgoing,ranks,
        degree_conflict_penalty=288.0,order_penalty=order_penalty)
    return fc.ArrayBQM.from_bqm(bqm,ep)


def energy(abqm,ep,sel):
    return abqm.energy([1 if e in sel else 0 for e in ep])


def move_neighbors(st,ep,r2,r3,allowed):
    idx={e:i for i,e in enumerate(ep)}
    ids=frozenset(idx[e] for e in st)
    for fam,templates in (("r2",r2),("r3",r3)):
        for ti,tpl in enumerate(templates):
            nb=applicable(ids,tpl)
            if nb is None:continue
            ns=frozenset(ep[i] for i in nb)
            if ns in allowed:
                yield ns,fam,ti


def shortest_r2_route(start,target,ep,r2,cap=100000,allowed=None):
    idx={e:i for i,e in enumerate(ep)}
    s0=frozenset(idx[e] for e in start);tgt=frozenset(idx[e] for e in target)
    q=deque([s0]);prev={s0:None};how={}
    while q and len(prev)<cap:
        s=q.popleft()
        if s==tgt:break
        for ti,tpl in enumerate(r2):
            nb=applicable(s,tpl)
            if nb is None or nb in prev:continue
            if allowed is not None:
                nbe=frozenset(ep[i] for i in nb)
                if nbe not in allowed:continue
            prev[nb]=s;how[nb]=ti;q.append(nb)
    if tgt not in prev:return None,len(prev)
    route=[];cur=tgt
    while prev[cur] is not None:
        route.append({"template":how[cur],"state_ids":cur})
        cur=prev[cur]
    route.reverse()
    return route,len(prev)


def shortest_feasible_route(start,target,ep,r2,r3,allowed):
    q=deque([frozenset(start)])
    prev={frozenset(start):None};how={}
    tgt=frozenset(target)
    while q:
        s=q.popleft()
        if s==tgt:break
        for nb,fam,ti in move_neighbors(s,ep,r2,r3,allowed):
            if nb not in prev:
                prev[nb]=s;how[nb]=(fam,ti);q.append(nb)
    if tgt not in prev:return None
    route=[];cur=tgt
    while prev[cur] is not None:
        route.append({"family":how[cur][0],"template":how[cur][1]})
        cur=prev[cur]
    route.reverse()
    return route


def main():
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    epset=set(ep)
    rows=[]
    for p in normalized_paths():
        sc=score(p,reward)
        if sc is None:continue
        st=state(p)
        if st<=epset:rows.append((sc,p,st))
    rows.sort(reverse=True,key=lambda z:z[0])
    scores=[x[0] for x in rows]
    assert scores[0]==GROUND and scores[1]==RUNNER
    gsc,gp,gs=rows[0];rsc,rp,rs=rows[1]

    comps=alternating_components(rs,gs)
    r2=fc.build_reconnect_templates(ep);r3=build_r3_templates(ep)
    idx={e:i for i,e in enumerate(ep)}
    rid=frozenset(idx[e] for e in rs);gid=frozenset(idx[e] for e in gs)

    direct=[]
    for fam,templates in (("r2",r2),("r3",r3)):
        for ti,tpl in enumerate(templates):
            nb=applicable(rid,tpl)
            if nb==gid:direct.append({"family":fam,"template":ti})

    # direct disjoint R2-pair
    eligible=[]
    for ti,tpl in enumerate(r2):
        nb=applicable(rid,tpl)
        if nb is not None:eligible.append((ti,tpl))
    for a in range(len(eligible)):
        ia,ta=eligible[a];fa=set(ta[0]+ta[1])
        for b in range(a+1,len(eligible)):
            ib,tb=eligible[b];fb=set(tb[0]+tb[1])
            if fa&fb:continue
            nb=frozenset(set(rid).symmetric_difference(fa|fb))
            if nb==gid:direct.append({"family":"r2_pair","templates":[ia,ib]})

    allowed={frozenset(st) for _,_,st in rows}
    route=shortest_feasible_route(rs,gs,ep,r2,r3,allowed)
    r2_route,r2_component_size=shortest_r2_route(rs,gs,ep,r2)
    r2_feasible_route,_=shortest_r2_route(rs,gs,ep,r2,allowed=allowed)

    runner_rank=project_rank(rids,rs)
    rankfree=array_bqm(ep,cost,incoming,outgoing,runner_rank,0.0)
    stale=array_bqm(ep,cost,incoming,outgoing,runner_rank,4.0)
    de_rankfree=energy(rankfree,ep,gs)-energy(rankfree,ep,rs)
    de_stale=energy(stale,ep,gs)-energy(stale,ep,rs)

    # single-slice action if all neighboring slices remain runner-up.
    kt=-0.5*math.log(math.tanh(BETA*GAMMA/P))
    flip_count=len(rs^gs)
    single_slice_action=(BETA/P)*de_rankfree + 4.0*kt*flip_count
    whole_worldline_action=BETA*de_rankfree

    result={
      "exact_projected_hamilton_paths":len(rows),
      "ground_score":gsc,"runner_up_score":rsc,"score_gap":gsc-rsc,
      "symmetric_difference_bits":len(rs^gs),
      "removed_edges":sorted(rs-gs),"added_edges":sorted(gs-rs),
      "alternating_components":comps,
      "single_read_relocation":single_relocation(rp,gp),
      "minimum_contiguous_segment_relocation":segment_relocation(rp,gp),
      "direct_structured_moves_to_ground":direct,
      "shortest_feasible_only_r2_r3_route":route,
      "ground_reachable_from_runner_by_r2_only":r2_route is not None,
      "runner_r2_component_size":r2_component_size,
      "shortest_r2_only_steps":len(r2_route) if r2_route is not None else None,
      "shortest_feasible_only_r2_steps":len(r2_feasible_route) if r2_feasible_route is not None else None,
      "rankfree_fixed_cardinality_delta_ground_minus_runner":de_rankfree,
      "stale_runner_rank_A4_delta_ground_minus_runner":de_stale,
      "sqa_snapshot":{
        "beta":BETA,"P":P,"gamma":GAMMA,"K":kt,
        "direct_flip_count":flip_count,
        "single_slice_direct_action_estimate":single_slice_action,
        "whole_worldline_direct_action":whole_worldline_action,
      },
      "runner_metrics":graph_metrics(rids,rs,project_rank(rids,rs),reward),
      "ground_metrics":graph_metrics(rids,gs,project_rank(rids,gs),reward),
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("RESULT",json.dumps(result),flush=True)


if __name__=="__main__":main()
