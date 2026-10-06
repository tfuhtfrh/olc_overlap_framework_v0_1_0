"""Standard-SQA Hamiltonian baseline with weak acyclicity guidance.

All inner solves use ONLY OpenJij SQASampler's built-in SingleSpinFlip updater.
No endpoint/R2/R3/worldline/custom PIMC move is used.

This is an outer hybrid only because the Hamiltonian can be rebuilt between
standard-SQA episodes from the current graph topology.

Methods
-------
none:
    H_weight + H_degree + H_count.

exact_cut:
    For each current directed cycle C, add the exact inequality
        sum_{e in C} x_e <= |C|-1
    with bounded slack at weak penalty A=0.02.

rank_ensemble:
    For each current directed cycle C, add a symmetric cycle-averaged rank
    pressure with per-edge coefficient 0.02:
        0.02 * sum_{e in C} x_e.
    Equivalently total cycle coefficient is 0.02*|C|.

Schedules
---------
old_default:
    beta=5, gamma=1, P=8, OpenJij quartic schedule.
p16:
    beta=6, gamma=2, P=16, OpenJij quartic schedule.

Fixed 12 outer iterations. Ground score is never used to stop or steer.
OpenJij standard output is used: each read already returns its lowest-energy
Trotter slice; resp.first is the lowest-energy read. In addition, diagnostics
record whether any of the 8 standard output reads is an HP/ground.
"""
from __future__ import annotations

import json, random, time
from pathlib import Path
import networkx as nx
import openjij as oj

from benchmark_chm13_projected_edge_hybrid import (
    load_problem, graph_metrics, project_rank, build_bqm_count, add_square,
)

OUT=Path("debug/qubo/chm13_standard_sqa_acyclicity_baseline_20260930.json")
GROUND=1_343_093
OUTER=12
READS=8
SWEEPS=1400
SEEDS=(202609301,202609302,202609303,202609304)
STARTS=("zero","random")
METHODS=("none","exact_cut","rank_ensemble")
CUT_A=0.02
RANK_EDGE_A=0.02
SCHEDULES={
  "old_default":{"beta":5.0,"gamma":1.0,"P":8},
  "p16":{"beta":6.0,"gamma":2.0,"P":16},
}


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


def bounded_weights(u):
    ws=[];rem=int(u);p=1
    while rem>0:
        w=min(p,rem);ws.append(w);rem-=w;p*=2
    return ws


def encode_value(weights,value):
    value=int(value)
    from itertools import product
    for cand in product((0,1),repeat=len(weights)):
        if sum(w*b for w,b in zip(weights,cand))==value:return list(cand)
    raise ValueError((weights,value))


def add_exact(bqm,cycs,A):
    meta=[]
    for ci,cyc in enumerate(cycs):
        L=len(cyc);ws=bounded_weights(L-1)
        coeff={e:1.0 for e in cyc};labels=[]
        for bi,w in enumerate(ws):
            lab=("cycle_slack",ci,bi);labels.append(lab);coeff[lab]=float(w)
        add_square(bqm,coeff,-float(L-1),A)
        meta.append({"cycle":cyc,"labels":labels,"weights":ws})
    return meta


def add_ensemble(bqm,cycs):
    for cyc in cycs:
        for e in cyc:bqm.add_linear(e,RANK_EDGE_A)
    return []


def warm(vs,idx,ep,selected,meta,start,rng,N):
    sm={v:0 for v in vs}
    if selected is not None:
        for e in ep:sm[e]=int(e in selected)
    elif start=="random":
        p=(N-1)/len(ep)
        for e in ep:sm[e]=int(rng.random()<p)
    for item in meta:
        chosen=sum(int(e in selected) for e in item["cycle"]) if selected is not None else 0
        slack=max(0,len(item["cycle"])-1-chosen)
        bits=encode_value(item["weights"],slack)
        for lab,b in zip(item["labels"],bits):sm[lab]=b
    return {idx[v]:int(sm[v]) for v in vs}


def make_qubo(bqm):
    vs=list(bqm.variables);idx={v:i for i,v in enumerate(vs)};q={}
    for v,b in bqm.linear.items():
        if b:q[(idx[v],idx[v])]=q.get((idx[v],idx[v]),0.0)+float(b)
    for (u,v),b in bqm.quadratic.items():
        if not b:continue
        i,j=idx[u],idx[v]
        if i>j:i,j=j,i
        q[(i,j)]=q.get((i,j),0.0)+float(b)
    return vs,idx,q


def decode_arr(arr,labels,vs,ep):
    sm={vs[int(label)]:int(arr[col]) for col,label in enumerate(labels)}
    return {e for e in ep if sm.get(e,0)}


def sample_standard(bqm,ep,selected,meta,start,rng,seed,cfg,rids,reward,N):
    vs,idx,q=make_qubo(bqm)
    ini=warm(vs,idx,ep,selected,meta,start,rng,N)
    t=time.perf_counter()
    resp=oj.SQASampler().sample_qubo(
      q,beta=cfg["beta"],gamma=cfg["gamma"],trotter=cfg["P"],
      num_reads=READS,num_sweeps=SWEEPS,seed=seed,
      initial_state=ini,updater="single spin flip")
    sec=time.perf_counter()-t

    first=resp.first
    sm={vs[i]:int(x) for i,x in first.sample.items()}
    primary={e for e in ep if sm.get(e,0)}

    labels=list(resp.variables)
    read_metrics=[]
    for arr,en in zip(resp.record.sample,resp.record.energy):
        sel=decode_arr(arr,labels,vs,ep)
        met=graph_metrics(rids,sel,project_rank(rids,sel),reward)
        read_metrics.append({
          "energy":float(en+bqm.offset),
          "valid_path":bool(met["valid_path"]),
          "path_score":met["path_score"],
          "selected_edges":met["selected_edges"],
          "degree_conflicts":met["degree_conflicts"],
          "cycle_count":met["cycle_count"],
        })
    return primary,sec,float(first.energy+bqm.offset),read_metrics,len(vs),len(bqm.quadratic)


def run(method,schedule_name,start,seed):
    cfg=SCHEDULES[schedule_name]
    rids,ep,reward,cost,incoming,outgoing=load_problem();N=len(rids)
    dummy={r:0 for r in rids}
    rng=random.Random(seed)
    selected=None;rows=[]
    best_hp=None;ever_ground_any_read=False

    for it in range(OUTER):
        active=cycles(rids,selected)
        bqm=build_bqm_count(
          ep,cost,incoming,outgoing,dummy,N,
          degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)
        if method=="exact_cut":
            meta=add_exact(bqm,active,CUT_A)
        elif method=="rank_ensemble":
            meta=add_ensemble(bqm,active)
        else:
            meta=[]

        selected,sec,en,rm,nv,nq=sample_standard(
          bqm,ep,selected,meta,start,rng,seed+1000*it,cfg,rids,reward,N)
        met=graph_metrics(rids,selected,project_rank(rids,selected),reward)

        read_hp=[x["path_score"] for x in rm if x["valid_path"]]
        if read_hp:
            cur=max(read_hp)
            best_hp=cur if best_hp is None else max(best_hp,cur)
        if any(x["path_score"]==GROUND for x in rm):
            ever_ground_any_read=True

        row={
          "method":method,"schedule":schedule_name,"start":start,"seed":seed,
          "iteration":it,"active_cycles":len(active),
          "variables":nv,"quadratic":nq,"seconds":sec,"primary_energy":en,
          "any_read_hp":bool(read_hp),
          "best_read_hp_score":max(read_hp,default=None),
          "any_read_ground":any(x["path_score"]==GROUND for x in rm),
          "read_metrics":rm,
          **met,
        }
        rows.append(row);print("ROW",json.dumps(row),flush=True)

    primary_hp=[r["path_score"] for r in rows if r["valid_path"]]
    return rows,{
      "method":method,"schedule":schedule_name,"start":start,"seed":seed,
      "primary_ever_hp":bool(primary_hp),
      "primary_best_hp":max(primary_hp,default=None),
      "any_read_best_hp":best_hp,
      "any_read_ground":ever_ground_any_read,
      "final_primary_valid":rows[-1]["valid_path"],
      "final_primary_score":rows[-1]["path_score"],
      "max_variables":max(r["variables"] for r in rows),
    }


def main():
    rows=[];summ=[]
    for sched in SCHEDULES:
      for method in METHODS:
        for start in STARTS:
          for si,base_seed in enumerate(SEEDS):
            seed=base_seed+10000*si
            rr,ss=run(method,sched,start,seed)
            rows+=rr;summ.append(ss);print("SUMMARY",json.dumps(ss),flush=True)

    agg={}
    for sched in SCHEDULES:
      for method in METHODS:
        for start in STARTS:
          aa=[s for s in summ if s["schedule"]==sched and s["method"]==method and s["start"]==start]
          primary=[s["primary_best_hp"] for s in aa if s["primary_best_hp"] is not None]
          anyread=[s["any_read_best_hp"] for s in aa if s["any_read_best_hp"] is not None]
          agg[f"{sched}|{method}|{start}"]={
            "runs":len(aa),
            "primary_hp_runs":sum(s["primary_ever_hp"] for s in aa),
            "primary_best_hp":max(primary,default=None),
            "any_read_hp_runs":sum(s["any_read_best_hp"] is not None for s in aa),
            "any_read_best_hp":max(anyread,default=None),
            "any_read_ground_runs":sum(s["any_read_ground"] for s in aa),
            "final_primary_hp_runs":sum(s["final_primary_valid"] for s in aa),
          }
    result={"aggregate":agg,"summaries":summ,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)


if __name__=="__main__":
    main()
