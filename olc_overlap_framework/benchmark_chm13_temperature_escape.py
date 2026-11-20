"""Temperature/escape diagnostic from the CHM13 path+cycle stuck state.

Use only degree-preserving R2/R3 alternating-cycle moves on the fixed-cardinality
projected BQM.  The move set is static and every template is an involution;
choosing uniformly from the union of static templates (with null proposals when
ineligible) gives a symmetric proposal.

Goal: distinguish "need larger k" from "R2/R3 can escape if uphill sequences
are thermally accessible".
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import graph_metrics
from diagnose_chm13_move_basis_proposal import (
    replay_stuck_checkpoint,
    build_r3_templates,
    applicable,
)

OUT=Path("debug/qubo/chm13_temperature_escape_20260928.json")
BETAS=(0.02,0.05,0.10,0.25,0.50,1.0,4.0)
SEEDS=(202609281,202609282,202609283,202609284)
ATTEMPTS=80000
CERT=1_343_093


def accept(beta,de,rng):
    return de<=0.0 or math.log(max(rng.random(),1e-300)) < -beta*de


def path_info(rids,ep,reward,state):
    selected={ep[i] for i in state}
    indeg={r:0 for r in rids};outdeg={r:0 for r in rids};nxt={}
    for u,v in selected:
        outdeg[u]+=1;indeg[v]+=1;nxt[u]=v
    if any(d>1 for d in indeg.values()) or any(d>1 for d in outdeg.values()):
        return False,None,None
    sources=[r for r in rids if indeg[r]==0]
    sinks=[r for r in rids if outdeg[r]==0]
    if len(sources)!=1 or len(sinks)!=1 or len(selected)!=len(rids)-1:
        return False,None,None
    cur=sources[0];seen=[];used=set()
    while cur not in used:
        used.add(cur);seen.append(cur)
        if cur not in nxt:break
        cur=nxt[cur]
    if len(seen)!=len(rids) or seen[-1]!=sinks[0]:
        return False,None,None
    score=int(sum(reward[(u,v)] for u,v in zip(seen,seen[1:])))
    return True,score,seen


def cycle_count_degree_correct(rids,ep,state):
    selected={ep[i] for i in state}
    nxt={u:v for u,v in selected}
    seen_global=set();cycles=0
    for start in rids:
        if start in seen_global:continue
        cur=start;local={}
        while cur not in seen_global and cur not in local and cur in nxt:
            local[cur]=len(local);cur=nxt[cur]
        if cur in local:cycles+=1
        seen_global.update(local)
        if cur not in nxt:seen_global.add(cur)
    return cycles


def run(ctx,beta,seed,anneal=False):
    rids=ctx["rids"];ep=ctx["ep"];reward=ctx["reward"];selected=ctx["selected"]
    abqm=ctx["abqm"];idx=ctx["idx"];r2=ctx["r2"];r3=ctx["r3"];templates=ctx["templates"]
    state=frozenset(idx[e] for e in selected)
    x=[1 if i in state else 0 for i in range(len(ep))]
    e0=abqm.energy(x);best_e=e0
    rng=random.Random(seed)
    attempted={"r2":0,"r3":0};eligible={"r2":0,"r3":0};accepted={"r2":0,"r3":0}
    uphill=0;first_feasible=None;best_score=None;best_cycles=cycle_count_degree_correct(rids,ep,state)
    max_dist=0

    for t in range(ATTEMPTS):
        if anneal:
            frac=t/max(1,ATTEMPTS-1)
            # warm exploration -> cold refinement
            b=0.02*((4.0/0.02)**frac)
        else:
            b=beta
        fam,tpl=rng.choice(templates);attempted[fam]+=1
        nb=applicable(state,tpl)
        if nb is None:continue
        eligible[fam]+=1
        flips=tuple(sorted(state.symmetric_difference(nb)))
        de=abqm.delta_flipset(x,flips)
        if not accept(b,de,rng):continue
        if de>1e-12:uphill+=1
        for i in flips:x[i]^=1
        state=nb;accepted[fam]+=1
        en=abqm.energy(x);best_e=min(best_e,en)
        max_dist=max(max_dist,len(state.symmetric_difference(
            frozenset(idx[e] for e in selected))))
        cyc=cycle_count_degree_correct(rids,ep,state);best_cycles=min(best_cycles,cyc)
        valid,score,_=path_info(rids,ep,reward,state)
        if valid:
            if first_feasible is None:first_feasible=t
            best_score=score if best_score is None else max(best_score,score)

    return {
      "mode":"anneal_0.02_to_4" if anneal else "fixed_beta",
      "beta":None if anneal else beta,
      "seed":seed,"attempts":ATTEMPTS,
      "template_counts":{"r2":len(r2),"r3":len(r3)},
      "attempted":attempted,"eligible":eligible,"accepted":accepted,
      "accepted_total":sum(accepted.values()),
      "uphill_accepted":uphill,
      "best_energy_delta":best_e-e0,
      "best_cycle_count":best_cycles,
      "max_symmetric_difference_from_start":max_dist,
      "first_feasible_attempt":first_feasible,
      "best_score":best_score,
      "ground_hit":best_score==CERT,
    }


def main():
    rids,ep,reward,cost,incoming,outgoing,selected,ranks,op=replay_stuck_checkpoint()
    fixed_bqm=fc.build_bqm_fixed_cardinality(
        ep,cost,incoming,outgoing,ranks,
        degree_conflict_penalty=288.0,order_penalty=op)
    abqm=fc.ArrayBQM.from_bqm(fixed_bqm,ep)
    idx={e:i for i,e in enumerate(ep)}
    r2=fc.build_reconnect_templates(ep);r3=build_r3_templates(ep)
    ctx={
      "rids":rids,"ep":ep,"reward":reward,"selected":selected,
      "abqm":abqm,"idx":idx,"r2":r2,"r3":r3,
      "templates":[("r2",t) for t in r2]+[("r3",t) for t in r3],
    }
    rows=[]
    for beta in BETAS:
        for seed in SEEDS:
            r=run(ctx,beta,seed,False);rows.append(r);print("ROW",json.dumps(r),flush=True)
    for seed in SEEDS:
        r=run(ctx,0.0,seed,True);rows.append(r);print("ROW",json.dumps(r),flush=True)
    agg={}
    modes=sorted(set((r["mode"],r["beta"]) for r in rows),key=str)
    for mode,beta in modes:
        rr=[r for r in rows if r["mode"]==mode and r["beta"]==beta]
        agg[str((mode,beta))]={
          "runs":len(rr),
          "feasible_runs":sum(r["best_score"] is not None for r in rr),
          "ground_hits":sum(r["ground_hit"] for r in rr),
          "best_score":max((r["best_score"] for r in rr if r["best_score"] is not None),default=None),
          "min_cycle_count":min(r["best_cycle_count"] for r in rr),
          "mean_accepted":sum(r["accepted_total"] for r in rr)/len(rr),
          "mean_uphill_accepted":sum(r["uphill_accepted"] for r in rr)/len(rr),
          "max_distance":max(r["max_symmetric_difference_from_start"] for r in rr),
        }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"aggregate":agg,"rows":rows},indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)


if __name__=="__main__":main()
