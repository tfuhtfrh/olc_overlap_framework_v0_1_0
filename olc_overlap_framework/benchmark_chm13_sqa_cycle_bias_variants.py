"""SQA benchmark of smarter acyclicity biases for CHM13 complex144.

Compare:
- none: no acyclicity term
- rank: existing projected backward-edge schedule
- cycle_avg: current-cycle rank-ensemble bias. For a simple L-cycle,
  averaging all L equivalent cyclic rank rotations gives A/L on every edge.
- cycle_cut_slack: exact current-cycle cut via bounded slack
- cycle_cut_break: exact current-cycle cut via a symmetric break-choice variable

The break-choice encoding for a cycle C is
    A (sum b_e - 1)^2 + A sum_e b_e x_e.
If all cycle edges are selected, minimum penalty is A. If any edge is absent,
choose b on an absent edge and the penalty is zero. Thus it exactly forbids only
full-cycle selection while treating cycle edges symmetrically.
"""
from __future__ import annotations

import json, random, time
from pathlib import Path
import networkx as nx
import openjij as oj

from benchmark_chm13_projected_edge_hybrid import (
    load_problem, project_rank, graph_metrics, build_bqm_count, add_square,
)

OUT=Path("debug/qubo/chm13_sqa_cycle_bias_variants_20260929.json")
CERT=1_343_093
SEEDS=(202609291,202609292,202609293,202609294)
OUTER=12
ORDER_SCHEDULE=(0.0,0.25,0.5,1.0,2.0,4.0,8.0,16.0,32.0)
CONFIGS=("none","rank","avg_1","avg_4","slack_4","break_4","rosen_4")


def current_cycles(rids,selected):
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


def bounded_weights(u):
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


def add_avg_bias(bqm,cycles,A):
    for cyc in cycles:
        if not cyc:continue
        w=float(A)/len(cyc)
        for e in cyc:bqm.add_linear(e,w)
    return []


def add_slack_cuts(bqm,cycles,A):
    meta=[]
    for ci,cyc in enumerate(cycles):
        L=len(cyc);ws=bounded_weights(L-1);coeff={e:1.0 for e in cyc};labels=[]
        for bi,w in enumerate(ws):
            lab=("slack",ci,bi);coeff[lab]=float(w);labels.append(lab)
        add_square(bqm,coeff,-float(L-1),float(A))
        meta.append({"kind":"slack","cycle":cyc,"labels":labels,"weights":ws})
    return meta


def add_break_cuts(bqm,cycles,A):
    meta=[]
    for ci,cyc in enumerate(cycles):
        labels=[("break",ci,bi) for bi in range(len(cyc))]
        coeff={lab:1.0 for lab in labels}
        add_square(bqm,coeff,-1.0,float(A))
        for e,lab in zip(cyc,labels):
            bqm.add_quadratic(e,lab,float(A))
        meta.append({"kind":"break","cycle":cyc,"labels":labels})
    return meta


def add_rosenberg_cuts(bqm,cycles,A):
    """Sparse exact quadratization of A*prod_{e in C} x_e.

    Chain products for the first L-1 variables using Rosenberg AND penalties,
    then couple the final partial product quadratically to the last edge.
    M=4A > A makes cheating on an auxiliary more expensive than the forbidden
    full-cycle penalty.
    """
    meta=[]
    M=4.0*float(A)
    for ci,cyc in enumerate(cycles):
        L=len(cyc)
        if L<2:continue
        if L==2:
            bqm.add_quadratic(cyc[0],cyc[1],float(A))
            meta.append({"kind":"rosen","cycle":cyc,"labels":[]})
            continue
        labels=[]
        prev=cyc[0]
        # z_i = prev * x_i for i=1..L-2
        for ai,i in enumerate(range(1,L-1)):
            x=cyc[i];z=("rosen",ci,ai);labels.append(z)
            # M*(prev*x - 2 prev*z - 2 x*z + 3z)
            bqm.add_quadratic(prev,x,M)
            bqm.add_quadratic(prev,z,-2.0*M)
            bqm.add_quadratic(x,z,-2.0*M)
            bqm.add_linear(z,3.0*M)
            prev=z
        bqm.add_quadratic(prev,cyc[-1],float(A))
        meta.append({"kind":"rosen","cycle":cyc,"labels":labels})
    return meta


def warm(bqm,ep,selected,meta):
    sm={v:0 for v in bqm.variables}
    if selected is not None:
        for e in ep:sm[e]=int(e in selected)
    for item in meta:
        if item["kind"]=="slack":
            chosen=sum(int(e in selected) for e in item["cycle"])
            val=max(0,len(item["cycle"])-1-chosen)
            bits=encode_value(item["weights"],val)
            for lab,b in zip(item["labels"],bits):sm[lab]=b
        elif item["kind"]=="break":
            # choose an absent edge if available, otherwise arbitrary first edge
            pick=0
            for i,e in enumerate(item["cycle"]):
                if e not in selected:pick=i;break
            sm[item["labels"][pick]]=1
        elif item["kind"]=="rosen":
            vals=[int(e in selected) for e in item["cycle"]]
            prod=vals[0]
            for lab,v in zip(item["labels"],vals[1:-1]):
                prod*=v
                sm[lab]=prod
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


def parse(c):
    if c in ("none","rank"):return c,0.0
    k,a=c.split("_");return k,float(a)


def run(config,seed):
    rids,ep,reward,cost,incoming,outgoing=load_problem();N=len(rids)
    rng=random.Random(seed);ranks={r:i for i,r in enumerate(rng.sample(rids,N))}
    selected=None;rows=[]
    kind,A=parse(config)
    for it in range(OUTER):
        cyc=current_cycles(rids,selected)
        op=ORDER_SCHEDULE[min(it,len(ORDER_SCHEDULE)-1)] if kind=="rank" else 0.0
        bqm=build_bqm_count(
            ep,cost,incoming,outgoing,ranks,N,
            degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=op)
        meta=[]
        if kind=="avg":meta=add_avg_bias(bqm,cyc,A)
        elif kind=="slack":meta=add_slack_cuts(bqm,cyc,A)
        elif kind=="break":meta=add_break_cuts(bqm,cyc,A)
        elif kind=="rosen":meta=add_rosenberg_cuts(bqm,cyc,A)

        selected,sec,nv,nq=sqa(bqm,ep,selected,meta,seed+1000*it)
        ranks=project_rank(rids,selected);met=graph_metrics(rids,selected,ranks,reward)
        row={
          "config":config,"seed":seed,"iteration":it,
          "current_cycles_used":len(cyc),"order_penalty":op,
          "variables":nv,"quadratic":nq,"seconds":sec,**met,
        }
        rows.append(row);print("ROW",json.dumps(row),flush=True)
        if met["valid_path"] and met["path_score"]==CERT:break
    feas=[r for r in rows if r["valid_path"]]
    best=max((r["path_score"] for r in feas),default=None)
    return rows,{
      "config":config,"seed":seed,"ever_feasible":bool(feas),
      "best_score":best,"ground_hit":best==CERT,
      "max_variables":max(r["variables"] for r in rows),
      "max_quadratic":max(r["quadratic"] for r in rows),
    }


def main():
    rows=[];ss=[]
    for c in CONFIGS:
        for seed in SEEDS:
            rr,s=run(c,seed);rows+=rr;ss.append(s);print("SUMMARY",json.dumps(s),flush=True)
    agg={}
    for c in CONFIGS:
        aa=[s for s in ss if s["config"]==c];scores=[s["best_score"] for s in aa if s["best_score"] is not None]
        agg[c]={
          "runs":len(aa),"feasible_runs":sum(s["ever_feasible"] for s in aa),
          "ground_hits":sum(s["ground_hit"] for s in aa),
          "best_score":max(scores,default=None),
          "median_score":sorted(scores)[len(scores)//2] if scores else None,
          "max_variables":max(s["max_variables"] for s in aa),
          "max_quadratic":max(s["max_quadratic"] for s in aa),
        }
    result={"aggregate":agg,"summaries":ss,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)


if __name__=="__main__":main()
