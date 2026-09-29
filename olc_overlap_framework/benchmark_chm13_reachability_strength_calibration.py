"""Reachability-rank strength calibration on known HP states.

Diagnostic only: runner-up and certified ground are used as controlled starting
states to measure the exploration/retention tradeoff. They are not used as
solver oracles.

For each start HP, build a fixed vanilla-SQA Hamiltonian:
    H0 + A * sum_{e in R(start)} x_e
where R(start) contains candidate edges whose addition would close a directed
cycle with respect to the current HP condensation/reachability order.

Sweep A and run one standard OpenJij SQASampler episode from that HP.

Questions:
- ground start: how large must A be to retain an HP / retain the ground?
- runner-up start: when does A become so strong that it suppresses movement
  away from the runner-up, including any chance of finding the ground?

Every run uses OpenJij SingleSpinFlip only.
"""
from __future__ import annotations

import json,time
from pathlib import Path
import openjij as oj

import benchmark_chm13_standard_sqa_acyclicity_baseline as b
from benchmark_chm13_projected_edge_hybrid import build_bqm_count
from benchmark_chm13_reachability_rank import reachability_bad_edges
from diagnose_chm13_runnerup_ground_geometry import normalized_paths,score,state

OUT=Path("debug/qubo/chm13_reachability_strength_calibration_20260930.json")
GROUND=1_343_093
RUNNER=1_338_209
AS=(0.0,0.01,0.02,0.05,0.10,0.20,0.50)
SEEDS=tuple(202609500+i for i in range(12))
READS=8
SWEEPS=2000


def find_state(target,reward,epset):
    for p in normalized_paths():
        if score(p,reward)==target:
            st=state(p)
            if st<=epset:return st
    raise RuntimeError(target)


def make_bqm(rids,ep,cost,incoming,outgoing,selected,A):
    dummy={r:0 for r in rids}
    bqm=build_bqm_count(
      ep,cost,incoming,outgoing,dummy,len(rids),
      degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)
    bad=reachability_bad_edges(rids,ep,selected)
    for e in bad:
        bqm.add_linear(e,A)
    return bqm,bad


def make_qubo(bqm):
    vs=list(bqm.variables);idx={v:i for i,v in enumerate(vs)};q={}
    for v,bias in bqm.linear.items():
        if bias:q[(idx[v],idx[v])]=q.get((idx[v],idx[v]),0.0)+float(bias)
    for (u,v),bias in bqm.quadratic.items():
        if not bias:continue
        i,j=idx[u],idx[v]
        if i>j:i,j=j,i
        q[(i,j)]=q.get((i,j),0.0)+float(bias)
    return vs,idx,q


def run_one(name,selected,A,seed,ctx):
    rids,ep,reward,cost,incoming,outgoing=ctx
    bqm,bad=make_bqm(rids,ep,cost,incoming,outgoing,selected,A)
    vs,idx,q=make_qubo(bqm)
    ini={idx[v]:0 for v in vs}
    for e in ep:ini[idx[e]]=int(e in selected)

    t=time.perf_counter()
    resp=oj.SQASampler().sample_qubo(
      q,beta=5.0,gamma=1.0,trotter=8,
      num_reads=READS,num_sweeps=SWEEPS,seed=seed,
      initial_state=ini,updater="single spin flip")
    sec=time.perf_counter()-t

    labels=list(resp.variables);mets=[]
    for arr in resp.record.sample:
        sm={vs[int(lab)]:int(arr[col]) for col,lab in enumerate(labels)}
        sel={e for e in ep if sm.get(e,0)}
        met=b.graph_metrics(rids,sel,b.project_rank(rids,sel),reward)
        mets.append(met)

    first=resp.first
    sm={vs[i]:int(x) for i,x in first.sample.items()}
    psel={e for e in ep if sm.get(e,0)}
    pm=b.graph_metrics(rids,psel,b.project_rank(rids,psel),reward)

    return {
      "start":name,"A":A,"seed":seed,
      "reachability_edges":len(bad),"seconds":sec,
      "primary_valid":bool(pm["valid_path"]),"primary_score":pm["path_score"],
      "primary_ground":pm["path_score"]==GROUND,
      "any_read_hp":any(m["valid_path"] for m in mets),
      "any_read_ground":any(m["path_score"]==GROUND for m in mets),
      "best_read_hp":max((m["path_score"] for m in mets if m["valid_path"]),default=None),
      "mean_read_cycles":sum(m["cycle_count"] for m in mets)/len(mets),
    }


def main():
    ctx=b.load_problem();rids,ep,reward,cost,incoming,outgoing=ctx
    epset=set(ep)
    starts={
      "runner":find_state(RUNNER,reward,epset),
      "ground":find_state(GROUND,reward,epset),
    }
    rows=[]
    for name,st in starts.items():
      for A in AS:
        for seed in SEEDS:
          r=run_one(name,st,A,seed,ctx)
          rows.append(r);print("ROW",json.dumps(r),flush=True)

    agg={}
    for name in starts:
      for A in AS:
        rr=[r for r in rows if r["start"]==name and r["A"]==A]
        scores=[r["best_read_hp"] for r in rr if r["best_read_hp"] is not None]
        agg[f"{name}|A={A:g}"]={
          "runs":len(rr),
          "primary_hp":sum(r["primary_valid"] for r in rr),
          "primary_ground":sum(r["primary_ground"] for r in rr),
          "any_read_hp":sum(r["any_read_hp"] for r in rr),
          "any_read_ground":sum(r["any_read_ground"] for r in rr),
          "best_hp":max(scores,default=None),
          "median_best_hp":sorted(scores)[len(scores)//2] if scores else None,
          "mean_cycles":sum(r["mean_read_cycles"] for r in rr)/len(rr),
        }
    result={"aggregate":agg,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)


if __name__=="__main__":
    main()
