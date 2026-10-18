"""Vanilla OpenJij reverse-SQA diagnostic from the CHM13 runner-up HP.

No custom updater. Uses SQASampler SingleSpinFlip only.

OpenJij accepts arbitrary custom schedules as long as each 0 <= s <= 1; it does
not enforce monotonicity. This allows a reverse-anneal-like schedule:

    s=1 -> s_min -> pause -> s=1

initialized from the known runner-up classical HP.

This is a controlled barrier diagnostic, not an oracle solver:
- the runner-up is used only as a benchmark starting state;
- the certified ground score is used only after sampling to label success.

Hamiltonian:
    H_weight + H_degree + H_count
(rank-free, no acyclicity guidance)

Compare forward baseline and reverse schedules.
"""
from __future__ import annotations
import json,time
from pathlib import Path
import openjij as oj

import benchmark_chm13_standard_sqa_acyclicity_baseline as b
from benchmark_chm13_projected_edge_hybrid import build_bqm_count
from diagnose_chm13_runnerup_ground_geometry import normalized_paths,score,state

OUT=Path("debug/qubo/chm13_reverse_sqa_runnerup_20260930.json")
GROUND=1_343_093
RUNNER=1_338_209
READS=8
SEEDS=tuple(202610100+i for i in range(16))
BETA=5.0
GAMMA=1.0
P=8
TOTAL_STEPS=2000

CONFIGS={
 "forward":None,
 "rev_095":0.95,
 "rev_090":0.90,
 "rev_080":0.80,
 "rev_070":0.70,
 "rev_060":0.60,
 "rev_050":0.50,
}


def reverse_schedule(smin,total=TOTAL_STEPS):
    # 20% ramp down, 30% dwell, 50% ramp back to s=1.
    nd=max(1,int(total*0.20))
    npause=max(1,int(total*0.30))
    nu=max(1,total-nd-npause)
    sch=[]
    for k in range(nd):
        f=(k+1)/nd
        s=1.0-(1.0-smin)*f
        sch.append((s,BETA,1))
    for _ in range(npause):
        sch.append((smin,BETA,1))
    for k in range(nu):
        f=(k+1)/nu
        s=smin+(1.0-smin)*f
        sch.append((min(1.0,s),BETA,1))
    return sch


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


def runner_state(reward,epset):
    for p in normalized_paths():
        if score(p,reward)==RUNNER:
            st=state(p)
            if st<=epset:return st
    raise RuntimeError("runner-up not found")


def run(config,seed,ctx):
    rids,ep,reward,bqm,runner=ctx
    vs,idx,q=make_qubo(bqm)
    ini={idx[v]:0 for v in vs}
    for e in ep:ini[idx[e]]=int(e in runner)

    kw={
      "beta":BETA,"gamma":GAMMA,"trotter":P,
      "num_reads":READS,"seed":seed,"initial_state":ini,
      "updater":"single spin flip",
    }
    smin=CONFIGS[config]
    if smin is None:
        kw["num_sweeps"]=TOTAL_STEPS
    else:
        kw["schedule"]=reverse_schedule(smin)

    t=time.perf_counter()
    resp=oj.SQASampler().sample_qubo(q,**kw)
    sec=time.perf_counter()-t

    labels=list(resp.variables)
    read_metrics=[]
    for arr,en in zip(resp.record.sample,resp.record.energy):
        sm={vs[int(lab)]:int(arr[col]) for col,lab in enumerate(labels)}
        sel={e for e in ep if sm.get(e,0)}
        met=b.graph_metrics(rids,sel,b.project_rank(rids,sel),reward)
        read_metrics.append({
          "valid_path":bool(met["valid_path"]),
          "path_score":met["path_score"],
          "selected_edges":met["selected_edges"],
          "degree_conflicts":met["degree_conflicts"],
          "cycle_count":met["cycle_count"],
          "energy":float(en+bqm.offset),
        })

    first=resp.first
    sm={vs[i]:int(x) for i,x in first.sample.items()}
    psel={e for e in ep if sm.get(e,0)}
    pm=b.graph_metrics(rids,psel,b.project_rank(rids,psel),reward)

    return {
      "config":config,"smin":smin,"seed":seed,"seconds":sec,
      "primary_valid":bool(pm["valid_path"]),
      "primary_score":pm["path_score"],
      "primary_ground":pm["path_score"]==GROUND,
      "any_read_hp":any(x["valid_path"] for x in read_metrics),
      "any_read_ground":any(x["path_score"]==GROUND for x in read_metrics),
      "best_read_hp":max((x["path_score"] for x in read_metrics if x["valid_path"]),default=None),
      "mean_cycles":sum(x["cycle_count"] for x in read_metrics)/len(read_metrics),
      "mean_degree_conflicts":sum(x["degree_conflicts"] for x in read_metrics)/len(read_metrics),
    }


def main():
    rids,ep,reward,cost,incoming,outgoing=b.load_problem()
    dummy={r:0 for r in rids}
    bqm=build_bqm_count(
      ep,cost,incoming,outgoing,dummy,len(rids),
      degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)
    runner=runner_state(reward,set(ep))
    ctx=(rids,ep,reward,bqm,runner)

    rows=[]
    for config in CONFIGS:
      for seed in SEEDS:
        rr=run(config,seed,ctx)
        rows.append(rr);print("ROW",json.dumps(rr),flush=True)

    agg={}
    for config in CONFIGS:
        rr=[r for r in rows if r["config"]==config]
        scores=[r["best_read_hp"] for r in rr if r["best_read_hp"] is not None]
        agg[config]={
          "runs":len(rr),
          "primary_hp":sum(r["primary_valid"] for r in rr),
          "primary_ground":sum(r["primary_ground"] for r in rr),
          "any_read_hp":sum(r["any_read_hp"] for r in rr),
          "any_read_ground":sum(r["any_read_ground"] for r in rr),
          "best_hp":max(scores,default=None),
          "median_best_hp":sorted(scores)[len(scores)//2] if scores else None,
          "mean_cycles":sum(r["mean_cycles"] for r in rr)/len(rr),
          "mean_degree_conflicts":sum(r["mean_degree_conflicts"] for r in rr)/len(rr),
        }
    result={"configs":CONFIGS,"aggregate":agg,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":
    main()
