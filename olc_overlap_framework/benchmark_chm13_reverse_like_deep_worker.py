"""Deep reverse-like SQA scan from the exact runner-up."""
from __future__ import annotations
import json, os
from pathlib import Path
import benchmark_chm13_reverse_like_sqa_runnerup as b

SMIN=float(os.environ["SMIN"])
LABEL=os.environ["LABEL"]
OUT=Path(f"debug/qubo/chm13_reverse_like_deep_{LABEL}_20260930.json")

def custom_schedule(smin):
    # 500 down + 1500 pause + 500 up = 2500 updater steps.
    s0=0.95;sf=0.995;n=100;sch=[]
    for k in range(n):
        f=(k+1)/n
        sch.append((float(s0+(smin-s0)*f),b.BETA,5))
    sch.append((float(smin),b.BETA,1500))
    for k in range(n):
        f=(k+1)/n
        sch.append((float(smin+(sf-smin)*f),b.BETA,5))
    return sch

def run_one(seed,ctx,runner,bqm,vs,idx,q):
    rids,ep,reward,*_=ctx
    ini={idx[v]:0 for v in vs}
    for e in ep:ini[idx[e]]=int(e in runner)
    resp=b.oj.SQASampler().sample_qubo(
      q,beta=b.BETA,gamma=b.GAMMA,trotter=b.P,num_reads=b.READS,seed=seed,
      initial_state=ini,updater="single spin flip",
      schedule=custom_schedule(SMIN))
    labels=list(resp.variables);mets=[]
    for arr in resp.record.sample:
        sm={vs[int(lab)]:int(arr[col]) for col,lab in enumerate(labels)}
        sel={e for e in ep if sm.get(e,0)}
        m=b.graph_metrics(rids,sel,b.project_rank(rids,sel),reward)
        mets.append(m)
    hp=[m["path_score"] for m in mets if m["valid_path"]]
    return {
      "seed":seed,
      "any_hp":bool(hp),
      "any_ground":any(x==b.GROUND for x in hp),
      "runner_returned":any(x==b.RUNNER for x in hp),
      "best_hp":max(hp,default=None),
      "mean_cycles":sum(m["cycle_count"] for m in mets)/len(mets),
      "mean_degree_conflicts":sum(m["degree_conflicts"] for m in mets)/len(mets),
    }

def main():
    ctx=b.load_problem();rids,ep,reward,*_=ctx
    runner=b.find_runner(reward,set(ep))
    bqm=b.make_bqm(ctx);vs,idx,q=b.make_qubo(bqm)
    rows=[run_one(seed,ctx,runner,bqm,vs,idx,q) for seed in b.SEEDS]
    for r in rows: print("ROW",json.dumps(r),flush=True)
    hp=[r["best_hp"] for r in rows if r["best_hp"] is not None]
    agg={
      "smin":SMIN,"runs":len(rows),
      "hp_runs":sum(r["any_hp"] for r in rows),
      "ground_runs":sum(r["any_ground"] for r in rows),
      "runner_return_runs":sum(r["runner_returned"] for r in rows),
      "best_hp":max(hp,default=None),
      "median_best_hp":sorted(hp)[len(hp)//2] if hp else None,
      "mean_cycles":sum(r["mean_cycles"] for r in rows)/len(rows),
      "mean_degree_conflicts":sum(r["mean_degree_conflicts"] for r in rows)/len(rows),
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"aggregate":agg,"rows":rows},indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":
    main()
