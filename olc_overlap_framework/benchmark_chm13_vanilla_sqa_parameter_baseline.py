"""Vanilla OpenJij SQA parameter baseline for CHM13 complex144.

This file intentionally uses ONLY OpenJij SQASampler's built-in
SingleSpinFlip updater. No endpoint/R2/R3/worldline/custom PIMC proposals.

Purpose
-------
Re-establish a clean standard-SQA baseline after the structure-aware PIMC
experiments.

Phase A here keeps the QUBO fixed:
    H = H_weight + H_degree + H_count
with 261 edge variables.

Starts:
- zero
- random Bernoulli with E[count] = N-1

Output:
- OpenJij standard optimization output: for each read, get_solution selects the
  lowest classical-energy Trotter slice; resp.first then selects the lowest
  returned read.
- no ground-score early stop;
- ground score is checked only after the fixed budget.

Schedules tested:
- old_default: reproduces the previous standard-SQA settings as closely as
  possible: beta=5, gamma=1, P=8, quartic default schedule.
- quartic variants: beta/gamma/P only.
- pause-quench variants: standard OpenJij custom (s,beta,steps) schedule using
  the same SingleSpinFlip updater; no custom move kernel.

A pause-quench schedule remains standard SQA: only the annealing schedule is
changed.
"""
from __future__ import annotations

import json, random, time
from pathlib import Path
import openjij as oj

from benchmark_chm13_projected_edge_hybrid import (
    load_problem, graph_metrics, project_rank, build_bqm_count,
)

OUT=Path("debug/qubo/chm13_vanilla_sqa_parameter_baseline_20260930.json")
GROUND=1_343_093
READS=8
SWEEPS=1400
SEEDS=(202609301,202609302,202609303,202609304)
STARTS=("zero","random")

CONFIGS={
  "old_default": {"beta":5.0,"gamma":1.0,"P":8,"kind":"quartic"},
  "b6_g1_P8": {"beta":6.0,"gamma":1.0,"P":8,"kind":"quartic"},
  "b6_g2_P8": {"beta":6.0,"gamma":2.0,"P":8,"kind":"quartic"},
  "b6_g4_P8": {"beta":6.0,"gamma":4.0,"P":8,"kind":"quartic"},
  "b8_g2_P8": {"beta":8.0,"gamma":2.0,"P":8,"kind":"quartic"},
  "b6_g2_P16": {"beta":6.0,"gamma":2.0,"P":16,"kind":"quartic"},
  "pause_b6_g2_s07": {"beta":6.0,"gamma":2.0,"P":8,"kind":"pause","s_pause":0.70},
  "pause_b6_g2_s08": {"beta":6.0,"gamma":2.0,"P":8,"kind":"pause","s_pause":0.80},
  "pause_b6_g4_s08": {"beta":6.0,"gamma":4.0,"P":8,"kind":"pause","s_pause":0.80},
}


def schedule_pause(beta,s_pause,total=SWEEPS):
    # Standard SQA schedule: gradual forward ramp, pause in the finite-field
    # region, then a short final ramp/quench to s=1.
    n1=int(total*0.40); n2=int(total*0.45); n3=total-n1-n2
    out=[]
    for k in range(n1):
        f=k/max(1,n1-1)
        s=s_pause*f
        out.append((s,beta,1))
    for _ in range(n2):
        out.append((s_pause,beta,1))
    for k in range(n3):
        f=(k+1)/max(1,n3)
        s=s_pause+(1.0-s_pause)*f
        out.append((min(1.0,s),beta,1))
    return out


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


def init_state(vs,idx,ep,start,rng,N):
    sm={v:0 for v in vs}
    if start=="random":
        p=(N-1)/len(ep)
        for e in ep:sm[e]=int(rng.random()<p)
    return {idx[v]:int(sm[v]) for v in vs}


def run(config_name,start,seed):
    cfg=CONFIGS[config_name]
    rids,ep,reward,cost,incoming,outgoing=load_problem();N=len(rids)
    dummy={r:0 for r in rids}
    bqm=build_bqm_count(
      ep,cost,incoming,outgoing,dummy,N,
      degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)
    vs,idx,q=make_qubo(bqm)
    rng=random.Random(seed)
    ini=init_state(vs,idx,ep,start,rng,N)

    kw={
      "beta":cfg["beta"],"gamma":cfg["gamma"],"trotter":cfg["P"],
      "num_reads":READS,"seed":seed,"initial_state":ini,
      "updater":"single spin flip",
    }
    if cfg["kind"]=="quartic":
        kw["num_sweeps"]=SWEEPS
    else:
        kw["schedule"]=schedule_pause(cfg["beta"],cfg["s_pause"])

    t=time.perf_counter()
    resp=oj.SQASampler().sample_qubo(q,**kw)
    sec=time.perf_counter()-t
    first=resp.first
    sm={vs[i]:int(x) for i,x in first.sample.items()}
    selected={e for e in ep if sm.get(e,0)}
    met=graph_metrics(rids,selected,project_rank(rids,selected),reward)

    # Diagnostic only: OpenJij stores final Trotter states/energies in info.
    # The returned sample itself already follows OpenJij's standard
    # lowest-energy-slice optimization rule.
    info=resp.info
    trotter_energies=info.get("trotter_energies")
    te=None
    if trotter_energies is not None:
        try:te=[float(x) for x in trotter_energies]
        except Exception:te=None

    return {
      "config":config_name,"start":start,"seed":seed,
      "beta":cfg["beta"],"gamma":cfg["gamma"],"P":cfg["P"],
      "schedule_kind":cfg["kind"],"s_pause":cfg.get("s_pause"),
      "reads":READS,"sweeps_equiv":SWEEPS,
      "seconds":sec,
      "returned_energy":float(first.energy+bqm.offset),
      "trotter_energies":te,
      **met,
    }


def main():
    rows=[]
    for config in CONFIGS:
        for start in STARTS:
            for si,seed in enumerate(SEEDS):
                r=run(config,start,seed+10000*si)
                rows.append(r);print("ROW",json.dumps(r),flush=True)

    agg={}
    for config in CONFIGS:
      for start in STARTS:
        rr=[r for r in rows if r["config"]==config and r["start"]==start]
        hp=[r for r in rr if r["valid_path"]]
        scores=[r["path_score"] for r in hp]
        agg[f"{config}|{start}"]={
          "runs":len(rr),
          "hp_hits":len(hp),
          "ground_hits":sum(r["path_score"]==GROUND for r in rr),
          "best_hp_score":max(scores,default=None),
          "median_hp_score":sorted(scores)[len(scores)//2] if scores else None,
          "mean_selected_edges":sum(r["selected_edges"] for r in rr)/len(rr),
          "mean_degree_conflicts":sum(r["degree_conflicts"] for r in rr)/len(rr),
          "mean_cycle_count":sum(r["cycle_count"] for r in rr)/len(rr),
          "mean_seconds":sum(r["seconds"] for r in rr)/len(rr),
        }
    result={"configs":CONFIGS,"aggregate":agg,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)


if __name__=="__main__":
    main()
