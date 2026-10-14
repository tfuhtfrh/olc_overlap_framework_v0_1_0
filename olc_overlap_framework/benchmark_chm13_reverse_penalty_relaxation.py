"""Reverse-SQA with temporary constraint-penalty relaxation.

Controlled CHM13 runner-up -> ground diagnostic.

The unrelaxed reverse-SQA schedule s=1->0.5->1 perfectly retained the runner-up
in 16/16 runs but never reached ground, indicating that the standard
SingleSpinFlip path is locally locked.

This experiment keeps the same vanilla SQASampler updater and reverse schedule,
but rescales the constraint penalties:

    A_degree = 288 * lambda
    A_count  = 32  * lambda

while keeping the overlap objective unchanged.

Since both runner-up and ground are feasible HPs, both have zero constraint
penalty at every lambda; their objective ordering is unchanged. Relaxation only
lowers the energy of infeasible intermediate states.

lambda sweep:
    1, 0.5, 0.25, 0.10, 0.05, 0.02, 0.01

Schedule:
    beta=5, gamma=1, P=8
    s=1 -> 0.5 -> pause -> 1
    2000 update steps
    16 seeds, 8 reads/seed.

No custom updater. Ground score is offline evaluation only.
"""
from __future__ import annotations
import json,time,os
from pathlib import Path
import openjij as oj

import benchmark_chm13_reverse_sqa_runnerup as rev
import benchmark_chm13_standard_sqa_acyclicity_baseline as b
from benchmark_chm13_projected_edge_hybrid import build_bqm_count

GROUND=1_343_093
RUNNER=1_338_209
LAMBDAS=(1.0,0.5,0.25,0.10,0.05,0.02,0.01)
SEEDS=tuple(202610200+i for i in range(16))
READS=8
SMIN=0.50


def run_lambda(lam,seed):
    rids,ep,reward,cost,incoming,outgoing=b.load_problem()
    dummy={r:0 for r in rids}
    bqm=build_bqm_count(
      ep,cost,incoming,outgoing,dummy,len(rids),
      degree_conflict_penalty=288.0*lam,
      count_penalty=32.0*lam,
      order_penalty=0.0)
    runner=rev.runner_state(reward,set(ep))
    vs,idx,q=rev.make_qubo(bqm)
    ini={idx[v]:0 for v in vs}
    for e in ep:ini[idx[e]]=int(e in runner)

    t=time.perf_counter()
    resp=oj.SQASampler().sample_qubo(
      q,beta=rev.BETA,gamma=rev.GAMMA,trotter=rev.P,
      num_reads=READS,seed=seed,initial_state=ini,
      updater="single spin flip",schedule=rev.reverse_schedule(SMIN))
    sec=time.perf_counter()-t

    labels=list(resp.variables);rmet=[]
    for arr,en in zip(resp.record.sample,resp.record.energy):
        sm={vs[int(lab)]:int(arr[col]) for col,lab in enumerate(labels)}
        sel={e for e in ep if sm.get(e,0)}
        met=b.graph_metrics(rids,sel,b.project_rank(rids,sel),reward)
        rmet.append({
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
      "lambda":lam,"seed":seed,"seconds":sec,
      "primary_valid":bool(pm["valid_path"]),
      "primary_score":pm["path_score"],
      "primary_ground":pm["path_score"]==GROUND,
      "any_read_hp":any(x["valid_path"] for x in rmet),
      "any_read_ground":any(x["path_score"]==GROUND for x in rmet),
      "best_read_hp":max((x["path_score"] for x in rmet if x["valid_path"]),default=None),
      "mean_selected":sum(x["selected_edges"] for x in rmet)/len(rmet),
      "mean_degree_conflicts":sum(x["degree_conflicts"] for x in rmet)/len(rmet),
      "mean_cycles":sum(x["cycle_count"] for x in rmet)/len(rmet),
    }


def run_worker(lam):
    rows=[]
    for seed in SEEDS:
        r=run_lambda(lam,seed)
        rows.append(r);print("ROW",json.dumps(r),flush=True)
    scores=[r["best_read_hp"] for r in rows if r["best_read_hp"] is not None]
    agg={
      "lambda":lam,"runs":len(rows),
      "primary_hp":sum(r["primary_valid"] for r in rows),
      "primary_ground":sum(r["primary_ground"] for r in rows),
      "any_read_hp":sum(r["any_read_hp"] for r in rows),
      "any_read_ground":sum(r["any_read_ground"] for r in rows),
      "best_hp":max(scores,default=None),
      "median_best_hp":sorted(scores)[len(scores)//2] if scores else None,
      "mean_selected":sum(r["mean_selected"] for r in rows)/len(rows),
      "mean_degree_conflicts":sum(r["mean_degree_conflicts"] for r in rows)/len(rows),
      "mean_cycles":sum(r["mean_cycles"] for r in rows)/len(rows),
    }
    return {"aggregate":agg,"rows":rows}


def main():
    if "LAMBDA" in os.environ:
        lam=float(os.environ["LAMBDA"])
        result=run_worker(lam)
        out=Path(f"debug/qubo/chm13_reverse_penalty_lambda_{lam:g}_20260930.json")
    else:
        allr={}
        for lam in LAMBDAS:
            allr[str(lam)]=run_worker(lam)
        result=allr
        out=Path("debug/qubo/chm13_reverse_penalty_relaxation_20260930.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    if "aggregate" in result:
        print("AGGREGATE",json.dumps(result["aggregate"]),flush=True)

if __name__=="__main__":
    main()
