"""Compare current-cycle-only cuts against stale rank, no cycle guidance, and
historically accumulated cycle cuts for CHM13 complex144 OpenJij SQA.

Key hypothesis
--------------
A cycle cut should represent only the topology defect observed in the current
outer state, analogous to projected rank.  Historical cycles are discarded.

For each currently observed directed cycle C:
    sum_{e in C} x_e <= |C|-1
is encoded exactly with bounded nonnegative slack:
    A_cut (sum x_e + s_C - (|C|-1))^2.

All configurations use the same seeds and the same SQA budget.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import networkx as nx
import openjij as oj

from benchmark_chm13_projected_edge_hybrid import (
    load_problem, project_rank, graph_metrics, build_bqm_count, add_square,
)

OUT=Path("debug/qubo/chm13_sqa_current_cyclecut_20260929.json")
CERT=1_343_093
BASE_SEEDS=(202609291,202609292,202609293,202609294)
OUTER=12
ORDER_SCHEDULE=(0.0,0.25,0.5,1.0,2.0,4.0,8.0,16.0,32.0)
CONFIGS=("none","rank","current_4","current_16","accum_4")


def bounded_weights(u):
    if u<=0:return []
    ws=[];remaining=int(u);p=1
    while remaining>0:
        w=min(p,remaining)
        ws.append(w);remaining-=w;p*=2
    return ws


def encode_value(weights,value):
    value=int(value);bits=[0]*len(weights)
    for i in range(len(weights)-1,-1,-1):
        if weights[i]<=value:
            bits[i]=1;value-=weights[i]
    if value!=0:
        from itertools import product
        for cand in product((0,1),repeat=len(weights)):
            if sum(w*b for w,b in zip(weights,cand))==int(value):
                return list(cand)
        raise ValueError((weights,value))
    return bits


def canonical_cycle(edges):
    return tuple(sorted(tuple(e) for e in edges))


def current_cycles(rids,selected):
    """One explicit directed cycle per nontrivial SCC.

    Once degree constraints are nearly satisfied, every nontrivial SCC is
    normally a simple directed cycle.  Before that, one witness cycle per SCC
    keeps the current-cut overhead bounded.
    """
    if selected is None:
        return []
    g=nx.DiGraph();g.add_nodes_from(rids);g.add_edges_from(selected)
    found=[]
    for comp in nx.strongly_connected_components(g):
        if len(comp)<=1:
            continue
        sub=g.subgraph(comp).copy()
        try:
            cyc=nx.find_cycle(sub,orientation="original")
        except nx.NetworkXNoCycle:
            continue
        found.append(canonical_cycle([(u,v) for u,v,*_ in cyc]))
    return sorted(set(found))


def add_cycle_cuts(bqm,cuts,penalty):
    meta=[]
    for ci,cycle in enumerate(cuts):
        L=len(cycle)
        if L<2:continue
        weights=bounded_weights(L-1)
        coeff={e:1.0 for e in cycle}
        labels=[]
        for bi,w in enumerate(weights):
            lab=("cycle_slack",ci,bi)
            coeff[lab]=float(w);labels.append(lab)
        add_square(bqm,coeff,-float(L-1),penalty)
        meta.append({"cycle":cycle,"labels":labels,"weights":weights})
    return meta


def warm_initial(bqm,ep,selected,cut_meta):
    sample={v:0 for v in bqm.variables}
    if selected is not None:
        for e in ep:
            sample[e]=int(e in selected)
    for item in cut_meta:
        chosen=sum(int(e in selected) for e in item["cycle"]) if selected else 0
        slack=max(0,len(item["cycle"])-1-chosen)
        bits=encode_value(item["weights"],slack)
        for lab,b in zip(item["labels"],bits):
            sample[lab]=b
    return sample


def sample_sqa(bqm,ep,selected,cut_meta,seed):
    vars_=list(bqm.variables);idx={v:i for i,v in enumerate(vars_)}
    qubo={}
    for v,bias in bqm.linear.items():
        if bias:
            i=idx[v];qubo[(i,i)]=qubo.get((i,i),0.0)+float(bias)
    for (u,v),bias in bqm.quadratic.items():
        if bias:
            i,j=idx[u],idx[v]
            if i>j:i,j=j,i
            qubo[(i,j)]=qubo.get((i,j),0.0)+float(bias)
    kwargs={"num_reads":8,"num_sweeps":1200,"trotter":8,"seed":seed}
    if selected is not None:
        init=warm_initial(bqm,ep,selected,cut_meta)
        kwargs["initial_state"]={idx[v]:int(init[v]) for v in vars_}
    t0=time.perf_counter()
    resp=oj.SQASampler().sample_qubo(qubo,**kwargs)
    sec=time.perf_counter()-t0
    first=resp.first
    sample={vars_[i]:int(val) for i,val in first.sample.items()}
    selected2={e for e in ep if sample.get(e,0)}
    return selected2,sec,float(first.energy+bqm.offset),len(vars_),len(bqm.quadratic)


def config_kind(config):
    if config=="none":return "none",0.0
    if config=="rank":return "rank",0.0
    kind,p=config.split("_")
    return kind,float(p)


def run(config,seed):
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    rng=random.Random(seed)
    ranks={r:i for i,r in enumerate(rng.sample(rids,len(rids)))}
    selected=None
    historical=[];histset=set()
    rows=[]

    for it in range(OUTER):
        kind,pen=config_kind(config)
        if kind=="rank":
            op=ORDER_SCHEDULE[min(it,len(ORDER_SCHEDULE)-1)]
            active_cuts=[]
        else:
            op=0.0
            if kind=="current":
                active_cuts=current_cycles(rids,selected)
            elif kind=="accum":
                active_cuts=list(historical)
            else:
                active_cuts=[]

        bqm=build_bqm_count(
            ep,cost,incoming,outgoing,ranks,len(rids),
            degree_conflict_penalty=288.0,
            count_penalty=32.0,
            order_penalty=op,
        )
        cut_meta=add_cycle_cuts(bqm,active_cuts,pen) if active_cuts else []

        selected,sec,en,nvars,nquad=sample_sqa(
            bqm,ep,selected,cut_meta,seed+1000*it)
        ranks=project_rank(rids,selected)
        met=graph_metrics(rids,selected,ranks,reward)

        observed=current_cycles(rids,selected)
        if kind=="accum":
            for cyc in observed:
                if cyc not in histset:
                    histset.add(cyc);historical.append(cyc)

        row={
          "config":config,"seed":seed,"iteration":it,
          "order_penalty":op,
          "active_cycle_cuts":len(active_cuts),
          "observed_cycles_for_next":len(observed),
          "historical_cycle_cuts":len(historical),
          "bqm_variables":nvars,"bqm_quadratic":nquad,
          "seconds":sec,"inner_energy":en,**met,
        }
        rows.append(row);print("ROW",json.dumps(row),flush=True)
        if met["valid_path"] and met["path_score"]==CERT:
            break

    feas=[r for r in rows if r["valid_path"]]
    best=max((r["path_score"] for r in feas),default=None)
    return rows,{
      "config":config,"seed":seed,
      "ever_feasible":bool(feas),
      "first_feasible_iteration":next((r["iteration"] for r in rows if r["valid_path"]),None),
      "best_score":best,"ground_hit":best==CERT,
      "max_variables":max(r["bqm_variables"] for r in rows),
      "max_active_cuts":max(r["active_cycle_cuts"] for r in rows),
      "final_historical_cuts":len(historical),
    }


def main():
    rows=[];summ=[]
    for config in CONFIGS:
        for seed in BASE_SEEDS:
            rr,ss=run(config,seed)
            rows.extend(rr);summ.append(ss)
            print("SUMMARY",json.dumps(ss),flush=True)

    agg={}
    for config in CONFIGS:
        ss=[s for s in summ if s["config"]==config]
        scores=[s["best_score"] for s in ss if s["best_score"] is not None]
        firsts=[s["first_feasible_iteration"] for s in ss if s["first_feasible_iteration"] is not None]
        agg[config]={
          "runs":len(ss),
          "feasible_runs":sum(s["ever_feasible"] for s in ss),
          "ground_hits":sum(s["ground_hit"] for s in ss),
          "best_score":max(scores,default=None),
          "median_score":sorted(scores)[len(scores)//2] if scores else None,
          "median_first_feasible_iteration":sorted(firsts)[len(firsts)//2] if firsts else None,
          "max_variables":max(s["max_variables"] for s in ss),
          "max_active_cuts":max(s["max_active_cuts"] for s in ss),
        }

    result={
      "configs":list(CONFIGS),
      "same_seeds":list(BASE_SEEDS),
      "aggregate":agg,"summaries":summ,"rows":rows,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)


if __name__=="__main__":
    main()
