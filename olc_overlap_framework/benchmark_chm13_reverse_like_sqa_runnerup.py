"""Reverse-annealing-like standard-SQA diagnostic from CHM13 runner-up.

Purpose
-------
The hybrid now reliably reaches the exact runner-up HP (1,338,209), while a
fresh forward SQA episode from that incumbent almost always destroys the HP and
never reaches the certified ground (1,343,093).

This diagnostic keeps:
- the same classical H0 = weight + degree + count QUBO;
- OpenJij SQASampler;
- built-in SingleSpinFlip updater;
- standard transverse-field Hamiltonian.

Only the annealing schedule changes.

A warm-started reverse-like schedule starts at high s, lowers s to reintroduce
quantum/thermal mixing locally, pauses, then raises s again:

    s_start -> s_min -> pause -> s_final

OpenJij custom schedules only validate 0 <= s <= 1 and permit non-monotonic
lists. We avoid exactly s=1 to prevent the path-integral coupling from becoming
formally singular.

This is a simulated reverse-annealing-like SQA protocol, not a real-time QPU
reverse anneal.

Ground is used only for offline evaluation.
"""
from __future__ import annotations

import json, time
from pathlib import Path
import openjij as oj

from benchmark_chm13_projected_edge_hybrid import (
    load_problem, graph_metrics, project_rank, build_bqm_count,
)
from diagnose_chm13_runnerup_ground_geometry import normalized_paths, score, state

OUT=Path("debug/qubo/chm13_reverse_like_sqa_runnerup_20260930.json")
RUNNER=1_338_209
GROUND=1_343_093
SEEDS=tuple(202610400+i for i in range(24))
READS=8
BETA=5.0
GAMMA=1.0
P=8

CONFIGS={
  "forward_default":{"kind":"forward"},
  "rev_s08":{"kind":"reverse","smin":0.80},
  "rev_s07":{"kind":"reverse","smin":0.70},
  "rev_s06":{"kind":"reverse","smin":0.60},
  "rev_s05":{"kind":"reverse","smin":0.50},
  "rev_s04":{"kind":"reverse","smin":0.40},
  "rev_s03":{"kind":"reverse","smin":0.30},
}


def find_runner(reward,epset):
    for p in normalized_paths():
        if score(p,reward)==RUNNER:
            st=state(p)
            if st<=epset:return st
    raise RuntimeError("runner-up not found")


def make_bqm(ctx):
    rids,ep,reward,cost,incoming,outgoing=ctx
    dummy={r:0 for r in rids}
    return build_bqm_count(
      ep,cost,incoming,outgoing,dummy,len(rids),
      degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)


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


def reverse_schedule(smin):
    # Total updater steps = 500 down + 1000 pause + 500 up = 2000.
    s0=0.95
    sf=0.995
    n=100
    sch=[]
    for k in range(n):
        f=(k+1)/n
        s=s0+(smin-s0)*f
        sch.append((float(s),BETA,5))
    sch.append((float(smin),BETA,1000))
    for k in range(n):
        f=(k+1)/n
        s=smin+(sf-smin)*f
        sch.append((float(s),BETA,5))
    return sch


def run_one(cfg,seed,ctx,runner,bqm,vs,idx,q):
    rids,ep,reward,*_=ctx
    ini={idx[v]:0 for v in vs}
    for e in ep:ini[idx[e]]=int(e in runner)

    kw=dict(
      beta=BETA,gamma=GAMMA,trotter=P,num_reads=READS,seed=seed,
      initial_state=ini,updater="single spin flip",
    )
    spec=CONFIGS[cfg]
    if spec["kind"]=="forward":
        kw["num_sweeps"]=2000
    else:
        kw["schedule"]=reverse_schedule(spec["smin"])

    t=time.perf_counter()
    resp=oj.SQASampler().sample_qubo(q,**kw)
    sec=time.perf_counter()-t

    labels=list(resp.variables);mets=[]
    for arr in resp.record.sample:
        sm={vs[int(lab)]:int(arr[col]) for col,lab in enumerate(labels)}
        sel={e for e in ep if sm.get(e,0)}
        m=graph_metrics(rids,sel,project_rank(rids,sel),reward)
        mets.append(m)

    hp=[m["path_score"] for m in mets if m["valid_path"]]
    return {
      "config":cfg,"seed":seed,"seconds":sec,
      "any_hp":bool(hp),
      "any_ground":any(x==GROUND for x in hp),
      "runner_returned":any(x==RUNNER for x in hp),
      "best_hp":max(hp,default=None),
      "mean_cycles":sum(m["cycle_count"] for m in mets)/len(mets),
      "mean_degree_conflicts":sum(m["degree_conflicts"] for m in mets)/len(mets),
    }


def main():
    ctx=load_problem();rids,ep,reward,*_=ctx
    runner=find_runner(reward,set(ep))
    bqm=make_bqm(ctx);vs,idx,q=make_qubo(bqm)
    rows=[]
    for cfg in CONFIGS:
        for seed in SEEDS:
            r=run_one(cfg,seed,ctx,runner,bqm,vs,idx,q)
            rows.append(r);print("ROW",json.dumps(r),flush=True)

    agg={}
    for cfg in CONFIGS:
        rr=[r for r in rows if r["config"]==cfg]
        hp=[r["best_hp"] for r in rr if r["best_hp"] is not None]
        agg[cfg]={
          "runs":len(rr),
          "hp_runs":sum(r["any_hp"] for r in rr),
          "ground_runs":sum(r["any_ground"] for r in rr),
          "runner_return_runs":sum(r["runner_returned"] for r in rr),
          "best_hp":max(hp,default=None),
          "median_best_hp":sorted(hp)[len(hp)//2] if hp else None,
          "mean_cycles":sum(r["mean_cycles"] for r in rr)/len(rr),
          "mean_degree_conflicts":sum(r["mean_degree_conflicts"] for r in rr)/len(rr),
        }
    result={"configs":CONFIGS,"aggregate":agg,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":
    main()
