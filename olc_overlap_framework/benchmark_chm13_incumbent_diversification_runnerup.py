"""Incumbent-diversification diagnostic on the exact runner-up.

All inner solves are vanilla OpenJij SQASampler SingleSpinFlip.
No custom updater. No ground early stop.

Start from the exact runner-up HP (score 1,338,209) only as a controlled
diagnostic state. The perturbation does NOT use the ground state.

Base Hamiltonian:
    H0 = H_weight + H_degree + H_count

Diversification probes:
1) uniform incumbent repulsion
       H_div = lambda * sum_{e in incumbent} x_e
   This gives every alternative path a small reward proportional to how many
   incumbent edges it replaces.

2) low-reward-weighted incumbent repulsion
       H_div = lambda * sum_{e in incumbent} q_e x_e
   where q_e is larger for lower-reward incumbent edges.
   This uses only the known overlap objective, not the optimal path.

The incumbent state is used only as warm start and to define the repulsion.
Certified ground is checked offline.

24 seeds/config, 8 reads/seed, 2000 sweeps/read.
"""
from __future__ import annotations
import json,time
from pathlib import Path
import openjij as oj

from benchmark_chm13_projected_edge_hybrid import load_problem,graph_metrics,project_rank,build_bqm_count
from diagnose_chm13_runnerup_ground_geometry import normalized_paths,score,state

OUT=Path("debug/qubo/chm13_incumbent_diversification_runnerup_20260930.json")
RUNNER=1_338_209
GROUND=1_343_093
SEEDS=tuple(202610200+i for i in range(24))
READS=8
SWEEPS=2000
CONFIGS={
 "none":("none",0.0),
 "uniform_002":("uniform",0.002),
 "uniform_005":("uniform",0.005),
 "uniform_010":("uniform",0.010),
 "uniform_020":("uniform",0.020),
 "uniform_050":("uniform",0.050),
 "lowreward_010":("lowreward",0.010),
 "lowreward_020":("lowreward",0.020),
 "lowreward_050":("lowreward",0.050),
}


def find_runner(reward,epset):
    for p in normalized_paths():
        if score(p,reward)==RUNNER:
            st=state(p)
            if st<=epset:return st
    raise RuntimeError("runner not found")


def make_bqm(cfg,rids,ep,reward,cost,incoming,outgoing,runner):
    dummy={r:0 for r in rids}
    bqm=build_bqm_count(
      ep,cost,incoming,outgoing,dummy,len(rids),
      degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)
    kind,lam=CONFIGS[cfg]
    if kind=="none":
        return bqm
    if kind=="uniform":
        for e in runner:bqm.add_linear(e,lam)
        return bqm
    if kind=="lowreward":
        vals=[reward[e] for e in runner]
        lo=min(vals);hi=max(vals);span=max(1,hi-lo)
        for e in runner:
            q=(hi-reward[e])/span
            bqm.add_linear(e,lam*q)
        return bqm
    raise ValueError(kind)


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


def run_one(cfg,seed,ctx,runner):
    rids,ep,reward,cost,incoming,outgoing=ctx
    bqm=make_bqm(cfg,rids,ep,reward,cost,incoming,outgoing,runner)
    vs,idx,q=make_qubo(bqm)
    ini={idx[v]:0 for v in vs}
    for e in ep:ini[idx[e]]=int(e in runner)

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
        m=graph_metrics(rids,sel,project_rank(rids,sel),reward)
        mets.append(m)
    hp=[m["path_score"] for m in mets if m["valid_path"]]
    return {
      "config":cfg,"seed":seed,"seconds":sec,
      "any_hp":bool(hp),
      "any_ground":any(x==GROUND for x in hp),
      "best_hp":max(hp,default=None),
      "runner_returned":any(x==RUNNER for x in hp),
      "mean_cycles":sum(m["cycle_count"] for m in mets)/len(mets),
    }


def main():
    ctx=load_problem();rids,ep,reward,*_=ctx
    runner=find_runner(reward,set(ep))
    rows=[]
    for cfg in CONFIGS:
      for seed in SEEDS:
        r=run_one(cfg,seed,ctx,runner)
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
      }
    result={"configs":CONFIGS,"aggregate":agg,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":
    main()
