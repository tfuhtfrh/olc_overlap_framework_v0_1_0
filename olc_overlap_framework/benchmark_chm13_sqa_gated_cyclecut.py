"""Phase-gated current-cycle cuts for CHM13 OpenJij SQA.

A cycle cut is activated only when the previous outer state is already on the
fixed-cardinality, degree-correct manifold but remains cyclic:
    selected_count == N-1
    degree_conflicts == 0
    cycle_count > 0

No cycle penalty is used while degree/cardinality are still being repaired, and
all cycle cuts are removed immediately once the state is a Hamilton path.

This tests whether cycle cuts are useful as a *repair trigger* rather than as a
persistent global guidance term.
"""
from __future__ import annotations

import json, random, time
from pathlib import Path
import networkx as nx
import openjij as oj

from benchmark_chm13_projected_edge_hybrid import (
    load_problem, project_rank, graph_metrics, build_bqm_count, add_square,
)

OUT=Path("debug/qubo/chm13_sqa_gated_cyclecut_20260929.json")
CERT=1_343_093
SEEDS=(202609291,202609292,202609293,202609294)
OUTER=12
CONFIGS=("none","gated_1","gated_2","gated_4","current_4")


def bounded_weights(u):
    if u<=0:return []
    ws=[];rem=int(u);p=1
    while rem>0:
        w=min(p,rem);ws.append(w);rem-=w;p*=2
    return ws


def encode_value(weights,value):
    value=int(value);bits=[0]*len(weights)
    for i in range(len(weights)-1,-1,-1):
        if weights[i]<=value:
            bits[i]=1;value-=weights[i]
    if value!=0:
        from itertools import product
        for cand in product((0,1),repeat=len(weights)):
            if sum(w*b for w,b in zip(weights,cand))==value:return list(cand)
        raise ValueError
    return bits


def cycles(rids,selected):
    if not selected:return []
    g=nx.DiGraph();g.add_nodes_from(rids);g.add_edges_from(selected)
    ans=[]
    for comp in nx.strongly_connected_components(g):
        if len(comp)<=1:continue
        sub=g.subgraph(comp).copy()
        try:cyc=nx.find_cycle(sub,orientation="original")
        except nx.NetworkXNoCycle:continue
        ans.append(tuple(sorted((u,v) for u,v,*_ in cyc)))
    return sorted(set(ans))


def add_cuts(bqm,cuts,penalty):
    meta=[]
    for ci,cyc in enumerate(cuts):
        L=len(cyc);ws=bounded_weights(L-1)
        coeff={e:1.0 for e in cyc};labels=[]
        for bi,w in enumerate(ws):
            lab=("cycle_slack",ci,bi);coeff[lab]=float(w);labels.append(lab)
        add_square(bqm,coeff,-float(L-1),penalty)
        meta.append({"cycle":cyc,"labels":labels,"weights":ws})
    return meta


def warm(bqm,ep,selected,meta):
    sm={v:0 for v in bqm.variables}
    if selected is not None:
        for e in ep:sm[e]=int(e in selected)
    for item in meta:
        chosen=sum(int(e in selected) for e in item["cycle"]) if selected else 0
        val=max(0,len(item["cycle"])-1-chosen)
        bits=encode_value(item["weights"],val)
        for lab,b in zip(item["labels"],bits):sm[lab]=b
    return sm


def sqa(bqm,ep,selected,meta,seed):
    vs=list(bqm.variables);idx={v:i for i,v in enumerate(vs)};q={}
    for v,b in bqm.linear.items():
        if b:q[(idx[v],idx[v])]=q.get((idx[v],idx[v]),0.0)+float(b)
    for (u,v),b in bqm.quadratic.items():
        if not b:continue
        i,j=idx[u],idx[v]
        if i>j:i,j=j,i
        q[(i,j)]=q.get((i,j),0.0)+float(b)
    kw={"num_reads":8,"num_sweeps":1200,"trotter":8,"seed":seed}
    if selected is not None:
        ini=warm(bqm,ep,selected,meta)
        kw["initial_state"]={idx[v]:int(ini[v]) for v in vs}
    t=time.perf_counter();resp=oj.SQASampler().sample_qubo(q,**kw);sec=time.perf_counter()-t
    sm={vs[i]:int(x) for i,x in resp.first.sample.items()}
    return {e for e in ep if sm.get(e,0)},sec,len(vs),len(bqm.quadratic)


def parse(config):
    if config=="none":return "none",0.0
    mode,p=config.split("_");return mode,float(p)


def run(config,seed):
    rids,ep,reward,cost,incoming,outgoing=load_problem();N=len(rids)
    rng=random.Random(seed);ranks={r:i for i,r in enumerate(rng.sample(rids,N))}
    selected=None;prev=None;rows=[]
    mode,pen=parse(config)
    for it in range(OUTER):
        use=False
        if selected is not None:
            if mode=="current":use=True
            elif mode=="gated":
                use=(len(selected)==N-1 and prev is not None
                     and prev["degree_conflicts"]==0 and prev["cycle_count"]>0)
        active=cycles(rids,selected) if use else []

        bqm=build_bqm_count(
            ep,cost,incoming,outgoing,ranks,N,
            degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)
        meta=add_cuts(bqm,active,pen) if active else []
        selected,sec,nv,nq=sqa(bqm,ep,selected,meta,seed+1000*it)
        ranks=project_rank(rids,selected);met=graph_metrics(rids,selected,ranks,reward)
        row={
          "config":config,"seed":seed,"iteration":it,
          "cut_active":bool(active),"active_cuts":len(active),
          "bqm_variables":nv,"bqm_quadratic":nq,"seconds":sec,**met,
        }
        rows.append(row);prev=met;print("ROW",json.dumps(row),flush=True)
        if met["valid_path"] and met["path_score"]==CERT:break
    feas=[r for r in rows if r["valid_path"]]
    best=max((r["path_score"] for r in feas),default=None)
    return rows,{
      "config":config,"seed":seed,"ever_feasible":bool(feas),
      "best_score":best,"ground_hit":best==CERT,
      "cut_iterations":sum(r["cut_active"] for r in rows),
      "max_variables":max(r["bqm_variables"] for r in rows),
    }


def main():
    rows=[];ss=[]
    for config in CONFIGS:
        for seed in SEEDS:
            rr,s=run(config,seed);rows+=rr;ss.append(s);print("SUMMARY",json.dumps(s),flush=True)
    agg={}
    for c in CONFIGS:
        aa=[s for s in ss if s["config"]==c];scores=[s["best_score"] for s in aa if s["best_score"] is not None]
        agg[c]={
          "runs":len(aa),"feasible_runs":sum(s["ever_feasible"] for s in aa),
          "ground_hits":sum(s["ground_hit"] for s in aa),
          "best_score":max(scores,default=None),
          "median_score":sorted(scores)[len(scores)//2] if scores else None,
          "mean_cut_iterations":sum(s["cut_iterations"] for s in aa)/len(aa),
          "max_variables":max(s["max_variables"] for s in aa),
        }
    result={"aggregate":agg,"summaries":ss,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)


if __name__=="__main__":main()
