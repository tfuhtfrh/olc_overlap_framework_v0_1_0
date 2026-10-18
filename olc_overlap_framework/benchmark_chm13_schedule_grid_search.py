"""Coarse-to-fine beta/Gamma schedule search for structured SQA diagnostics.

Representative tasks:
- Stage 2: degree-correct cyclic checkpoint -> any Hamilton path via endpoint transfer.
- Stage 3: runner-up -> certified ground via R3.

Coarse grid:
  beta in {3,4,5,6}
  Gamma_start in {2,3,4}
  Gamma_floor in {0.2,0.3,0.5,0.8}
  cluster length L in {1,2,4}, with L=8 only as a baseline control.

All schedules are geometric in Gamma and fixed-beta in the coarse pass.
Each configuration uses 4 seeds and 2000 attempts for screening.

Scoring emphasizes:
1) target-hit rate,
2) late-stage acceptance (last quarter),
3) smaller cluster length,
4) faster first target hit.

This is a PIMC/SQA sampler study; it does not claim real-time QA dynamics.
"""
from __future__ import annotations

import json
from pathlib import Path

import benchmark_chm13_schedule_cluster_freezing as base

OUT=Path("debug/qubo/chm13_schedule_grid_search_20260930.json")
BETAS=(3.0,4.0,5.0,6.0)
GSTARTS=(2.0,3.0,4.0)
GFLOORS=(0.2,0.3,0.5,0.8)
LENGTHS=(1,2,4)
SEEDS=tuple(202609500+i for i in range(4))
ATTEMPTS=2000


def cfg_name(beta,g0,g1):
    return f"b{beta:g}_g{g0:g}_{g1:g}"


def aggregate(rows):
    agg={}
    groups=sorted({(r["kind"],r["schedule"],r["cluster_length"]) for r in rows})
    for kind,sched,L in groups:
        rr=[r for r in rows if r["kind"]==kind and r["schedule"]==sched and r["cluster_length"]==L]
        hits=[r for r in rr if r["ever_any_target"]]
        allhits=[r for r in rr if r["ever_all_target"]]
        first=[r["first_any_target"] for r in hits]
        agg[f"{kind}|{sched}|L={L}"]={
          "runs":len(rr),
          "any_target_hits":len(hits),
          "P_any":len(hits)/len(rr),
          "all_slice_hits":len(allhits),
          "P_all":len(allhits)/len(rr),
          "mean_acceptance":sum(r["acceptance"] for r in rr)/len(rr),
          "mean_late_acceptance":sum(r["late_acceptance"] for r in rr)/len(rr),
          "mean_final_target_slices":sum(r["final_target_slices"] for r in rr)/len(rr),
          "median_first_any":sorted(first)[len(first)//2] if first else None,
          "K0":rr[0]["K0"],"K1":rr[0]["K1"],
        }
    return agg


def rank_configs(agg,kind,L):
    items=[]
    for k,v in agg.items():
        if not k.startswith(f"{kind}|") or not k.endswith(f"|L={L}"):
            continue
        # Hit rate dominates. Then prefer late mobility and fast hit.
        med=v["median_first_any"] if v["median_first_any"] is not None else 10**9
        score=(v["P_any"],v["P_all"],v["mean_late_acceptance"],-med)
        items.append((score,k,v))
    items.sort(reverse=True)
    return [
      {"key":k,"score_tuple":list(sc),"metrics":v}
      for sc,k,v in items[:10]
    ]


def main():
    old_sched=base.SCHEDULES
    old_len=base.LENGTHS
    old_seeds=base.SEEDS
    old_attempts=base.ATTEMPTS
    try:
        schedules={}
        for beta in BETAS:
            for g0 in GSTARTS:
                for g1 in GFLOORS:
                    if g1>=g0:continue
                    schedules[cfg_name(beta,g0,g1)]={
                      "beta0":beta,"beta1":beta,"g0":g0,"g1":g1,
                    }
        # Add previous baseline as control.
        schedules["baseline"]={"beta0":4.0,"beta1":4.0,"g0":3.0,"g1":0.03}
        base.SCHEDULES=schedules
        base.LENGTHS=LENGTHS
        base.SEEDS=SEEDS
        base.ATTEMPTS=ATTEMPTS

        contexts={"stage2":base.stage2_context(),"stage3":base.stage3_context()}
        rows=[]
        for kind in ("stage2","stage3"):
            for sched in schedules:
                for L in LENGTHS:
                    for seed in SEEDS:
                        r=base.run(kind,sched,L,seed,contexts[kind])
                        rows.append(r)
                        print("ROW",json.dumps(r),flush=True)

        # L=8 control only for baseline and a few promising hand-picked schedules.
        controls={
          "baseline":schedules["baseline"],
          cfg_name(4.0,3.0,0.5):schedules[cfg_name(4.0,3.0,0.5)],
          cfg_name(5.0,3.0,0.5):schedules[cfg_name(5.0,3.0,0.5)],
        }
        base.SCHEDULES=controls
        for kind in ("stage2","stage3"):
            for sched in controls:
                for seed in SEEDS:
                    r=base.run(kind,sched,8,seed,contexts[kind])
                    rows.append(r)
                    print("ROW",json.dumps(r),flush=True)

        agg=aggregate(rows)
        result={
          "grid":{
            "betas":BETAS,"gamma_starts":GSTARTS,"gamma_floors":GFLOORS,
            "cluster_lengths":LENGTHS,"seeds":SEEDS,"attempts":ATTEMPTS,
          },
          "top":{
            "stage2_L1":rank_configs(agg,"stage2",1),
            "stage2_L2":rank_configs(agg,"stage2",2),
            "stage2_L4":rank_configs(agg,"stage2",4),
            "stage3_L1":rank_configs(agg,"stage3",1),
            "stage3_L2":rank_configs(agg,"stage3",2),
            "stage3_L4":rank_configs(agg,"stage3",4),
          },
          "aggregate":agg,
          "rows":rows,
        }
        OUT.parent.mkdir(parents=True,exist_ok=True)
        OUT.write_text(json.dumps(result,indent=2)+"\n")
        print("TOP",json.dumps(result["top"]),flush=True)
    finally:
        base.SCHEDULES=old_sched
        base.LENGTHS=old_len
        base.SEEDS=old_seeds
        base.ATTEMPTS=old_attempts


if __name__=="__main__":
    main()
