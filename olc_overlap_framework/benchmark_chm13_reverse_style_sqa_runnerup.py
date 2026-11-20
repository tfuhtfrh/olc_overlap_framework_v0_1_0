"""Reverse-style standard-SQA diagnostic from the exact CHM13 runner-up.

Purpose
-------
Test incumbent-centered annealing protocol without changing the QUBO and
without changing the OpenJij SingleSpinFlip updater.

The exact runner-up HP (score 1,338,209) is used only as a controlled warm
start. Certified ground 1,343,093 is checked offline.

OpenJij accepts arbitrary custom s schedules as long as 0 <= s <= 1; it does
not require monotonicity. We therefore use a reverse-style schedule

    s_hi=0.999 -> s_min -> pause -> s_hi=0.999

rather than starting exactly at s=1, where log(tanh(beta*gamma*(1-s)/P))
becomes singular/infinite.

All schedules have exactly 2000 MC steps/read, beta=5, gamma=1, P=8.
24 seeds/config, 8 reads/seed.

Controls:
- forward_default: ordinary OpenJij quartic forward SQA warm-started from runner.
- reverse schedules with s_min in {0.9,0.8,0.7,0.6,0.5,0.4}.
- two pause allocations for selected depths.

No ground-score early stop, no custom updater, no rank/reachability term.
"""
from __future__ import annotations

import json, time
from pathlib import Path
import openjij as oj

from benchmark_chm13_projected_edge_hybrid import (
    load_problem, graph_metrics, project_rank, build_bqm_count,
)
from diagnose_chm13_runnerup_ground_geometry import normalized_paths, score, state

OUT=Path("debug/qubo/chm13_reverse_style_sqa_runnerup_20260930.json")
RUNNER=1_338_209
GROUND=1_343_093
READS=8
STEPS=2000
BETA=5.0
GAMMA=1.0
P=8
SEEDS=tuple(202610300+i for i in range(24))

CONFIGS={
    "forward_default":{"kind":"forward"},
    "rev_090_p50":{"kind":"reverse","smin":0.90,"pause":0.50},
    "rev_080_p50":{"kind":"reverse","smin":0.80,"pause":0.50},
    "rev_070_p50":{"kind":"reverse","smin":0.70,"pause":0.50},
    "rev_060_p50":{"kind":"reverse","smin":0.60,"pause":0.50},
    "rev_050_p50":{"kind":"reverse","smin":0.50,"pause":0.50},
    "rev_040_p50":{"kind":"reverse","smin":0.40,"pause":0.50},
    "rev_080_p25":{"kind":"reverse","smin":0.80,"pause":0.25},
    "rev_060_p25":{"kind":"reverse","smin":0.60,"pause":0.25},
    "rev_040_p25":{"kind":"reverse","smin":0.40,"pause":0.25},
}


def find_runner(reward,epset):
    for p in normalized_paths():
        if score(p,reward)==RUNNER:
            st=state(p)
            if st<=epset:
                return st
    raise RuntimeError("runner-up not found")


def make_qubo(bqm):
    vs=list(bqm.variables)
    idx={v:i for i,v in enumerate(vs)}
    q={}
    for v,b in bqm.linear.items():
        if b:
            q[(idx[v],idx[v])]=q.get((idx[v],idx[v]),0.0)+float(b)
    for (u,v),b in bqm.quadratic.items():
        if not b: continue
        i,j=idx[u],idx[v]
        if i>j: i,j=j,i
        q[(i,j)]=q.get((i,j),0.0)+float(b)
    return vs,idx,q


def reverse_schedule(smin,pause_frac):
    shi=0.999
    pause_steps=int(round(STEPS*pause_frac))
    move_steps=STEPS-pause_steps
    down_steps=move_steps//2
    up_steps=move_steps-down_steps

    sch=[]
    # one_mc_step=1 at each schedule point, total exactly STEPS.
    for k in range(down_steps):
        f=(k+1)/max(1,down_steps)
        s=shi+(smin-shi)*f
        sch.append((float(s),BETA,1))
    for _ in range(pause_steps):
        sch.append((float(smin),BETA,1))
    for k in range(up_steps):
        f=(k+1)/max(1,up_steps)
        s=smin+(shi-smin)*f
        sch.append((float(s),BETA,1))
    assert len(sch)==STEPS
    return sch


def run_one(cfg_name,seed,ctx,runner):
    rids,ep,reward,cost,incoming,outgoing=ctx
    dummy={r:0 for r in rids}
    bqm=build_bqm_count(
        ep,cost,incoming,outgoing,dummy,len(rids),
        degree_conflict_penalty=288.0,
        count_penalty=32.0,
        order_penalty=0.0,
    )
    vs,idx,q=make_qubo(bqm)
    ini={idx[v]:0 for v in vs}
    for e in ep:
        ini[idx[e]]=int(e in runner)

    cfg=CONFIGS[cfg_name]
    kwargs=dict(
        beta=BETA,gamma=GAMMA,trotter=P,
        num_reads=READS,seed=seed,
        initial_state=ini,updater="single spin flip",
    )
    if cfg["kind"]=="forward":
        kwargs["num_sweeps"]=STEPS
    else:
        kwargs["schedule"]=reverse_schedule(cfg["smin"],cfg["pause"])

    t=time.perf_counter()
    resp=oj.SQASampler().sample_qubo(q,**kwargs)
    sec=time.perf_counter()-t

    labels=list(resp.variables)
    mets=[]
    for arr,en in zip(resp.record.sample,resp.record.energy):
        sm={vs[int(lab)]:int(arr[col]) for col,lab in enumerate(labels)}
        sel={e for e in ep if sm.get(e,0)}
        m=graph_metrics(rids,sel,project_rank(rids,sel),reward)
        mets.append({
            "valid_path":bool(m["valid_path"]),
            "path_score":m["path_score"],
            "selected_edges":m["selected_edges"],
            "degree_conflicts":m["degree_conflicts"],
            "cycle_count":m["cycle_count"],
            "energy":float(en+bqm.offset),
        })

    hp=[m["path_score"] for m in mets if m["valid_path"]]
    return {
        "config":cfg_name,"seed":seed,"seconds":sec,
        "any_hp":bool(hp),
        "any_ground":any(x==GROUND for x in hp),
        "runner_returned":any(x==RUNNER for x in hp),
        "best_hp":max(hp,default=None),
        "mean_cycles":sum(m["cycle_count"] for m in mets)/len(mets),
        "mean_degree_conflicts":sum(m["degree_conflicts"] for m in mets)/len(mets),
        "mean_selected_edges":sum(m["selected_edges"] for m in mets)/len(mets),
    }


def main():
    ctx=load_problem()
    rids,ep,reward,*_=ctx
    runner=find_runner(reward,set(ep))
    rows=[]
    for cfg in CONFIGS:
        for seed in SEEDS:
            r=run_one(cfg,seed,ctx,runner)
            rows.append(r)
            print("ROW",json.dumps(r),flush=True)

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
            "mean_selected_edges":sum(r["mean_selected_edges"] for r in rr)/len(rr),
            "mean_seconds":sum(r["seconds"] for r in rr)/len(rr),
        }

    result={
        "runner":RUNNER,"ground":GROUND,
        "beta":BETA,"gamma":GAMMA,"P":P,
        "steps_per_read":STEPS,"reads":READS,
        "configs":CONFIGS,"aggregate":agg,"rows":rows,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)


if __name__=="__main__":
    main()
