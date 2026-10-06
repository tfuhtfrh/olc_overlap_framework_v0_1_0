"""Fine beta/Gamma search around the promising structured-SQA region.

The coarse sweep localized useful schedules near:
- Stage 2: beta ~ 5-6, Gamma_start ~ 3-4, Gamma_floor ~ 0.5-0.8
- Stage 3: beta ~ 6, Gamma_start ~ 4, Gamma_floor ~ 0.6-0.8

This script tests a smaller hand-selected neighborhood with 8 seeds/config and
2500 attempts. Cluster lengths L=1,2,4 are retained; no L=8 is used in the
search itself.

Ranking criteria:
Stage 2:
  any-HP hit rate -> all-slice HP hit rate -> final target-slice count ->
  late acceptance -> speed.
Stage 3:
  any-ground hit rate -> all-slice ground hit rate -> final ground-slice count
  -> late acceptance -> speed.
"""
from __future__ import annotations
import json
from pathlib import Path

import benchmark_chm13_schedule_cluster_freezing as base

OUT=Path("debug/qubo/chm13_schedule_fine_search_20260930.json")
SEEDS=tuple(202609600+i for i in range(8))
ATTEMPTS=2500
LENGTHS=(1,2,4)

STAGE2_CFGS={
  "s2_b5.5_g3_0.5":(5.5,3.0,0.5),
  "s2_b5.5_g3.5_0.6":(5.5,3.5,0.6),
  "s2_b6_g3_0.4":(6.0,3.0,0.4),
  "s2_b6_g3_0.5":(6.0,3.0,0.5),
  "s2_b6_g3_0.6":(6.0,3.0,0.6),
  "s2_b6_g3.5_0.5":(6.0,3.5,0.5),
  "s2_b6_g3.5_0.6":(6.0,3.5,0.6),
  "s2_b6.5_g3_0.5":(6.5,3.0,0.5),
  "s2_b6.5_g3_0.6":(6.5,3.0,0.6),
  "s2_b6.5_g3.5_0.7":(6.5,3.5,0.7),
  "s2_b7_g3_0.6":(7.0,3.0,0.6),
  "s2_b7_g3.5_0.8":(7.0,3.5,0.8),
}

STAGE3_CFGS={
  "s3_b5.5_g3.5_0.6":(5.5,3.5,0.6),
  "s3_b5.5_g4_0.8":(5.5,4.0,0.8),
  "s3_b6_g3.5_0.6":(6.0,3.5,0.6),
  "s3_b6_g4_0.6":(6.0,4.0,0.6),
  "s3_b6_g4_0.8":(6.0,4.0,0.8),
  "s3_b6_g4_1.0":(6.0,4.0,1.0),
  "s3_b6.5_g4_0.6":(6.5,4.0,0.6),
  "s3_b6.5_g4_0.8":(6.5,4.0,0.8),
  "s3_b6.5_g4_1.0":(6.5,4.0,1.0),
  "s3_b7_g4_0.8":(7.0,4.0,0.8),
  "s3_b7_g4_1.0":(7.0,4.0,1.0),
  "s3_b7_g5_1.0":(7.0,5.0,1.0),
}


def to_schedule(t):
    b,g0,g1=t
    return {"beta0":b,"beta1":b,"g0":g0,"g1":g1}


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
          "any_hits":len(hits),"P_any":len(hits)/len(rr),
          "all_hits":len(allhits),"P_all":len(allhits)/len(rr),
          "mean_acceptance":sum(r["acceptance"] for r in rr)/len(rr),
          "mean_late_acceptance":sum(r["late_acceptance"] for r in rr)/len(rr),
          "mean_final_target_slices":sum(r["final_target_slices"] for r in rr)/len(rr),
          "mean_max_target_slices":sum(r["max_target_slices"] for r in rr)/len(rr),
          "median_first_any":sorted(first)[len(first)//2] if first else None,
          "K0":rr[0]["K0"],"K1":rr[0]["K1"],
          "beta":rr[0]["beta0"],"gamma0":rr[0]["gamma0"],"gamma1":rr[0]["gamma1"],
        }
    return agg


def top_for(agg,kind,L):
    a=[]
    for k,v in agg.items():
        if not k.startswith(f"{kind}|") or not k.endswith(f"|L={L}"):continue
        med=v["median_first_any"] if v["median_first_any"] is not None else 10**9
        sc=(v["P_any"],v["P_all"],v["mean_final_target_slices"],
            v["mean_late_acceptance"],-med)
        a.append((sc,k,v))
    a.sort(reverse=True)
    return [{"key":k,"score":list(sc),"metrics":v} for sc,k,v in a[:8]]


def main():
    old_sched,old_len,old_seeds,old_att=base.SCHEDULES,base.LENGTHS,base.SEEDS,base.ATTEMPTS
    try:
        contexts={"stage2":base.stage2_context(),"stage3":base.stage3_context()}
        rows=[]
        base.LENGTHS=LENGTHS;base.SEEDS=SEEDS;base.ATTEMPTS=ATTEMPTS

        base.SCHEDULES={k:to_schedule(v) for k,v in STAGE2_CFGS.items()}
        for sched in base.SCHEDULES:
            for L in LENGTHS:
                for seed in SEEDS:
                    r=base.run("stage2",sched,L,seed,contexts["stage2"])
                    rows.append(r);print("ROW",json.dumps(r),flush=True)

        base.SCHEDULES={k:to_schedule(v) for k,v in STAGE3_CFGS.items()}
        for sched in base.SCHEDULES:
            for L in LENGTHS:
                for seed in SEEDS:
                    r=base.run("stage3",sched,L,seed,contexts["stage3"])
                    rows.append(r);print("ROW",json.dumps(r),flush=True)

        agg=aggregate(rows)
        top={
          "stage2_L1":top_for(agg,"stage2",1),
          "stage2_L2":top_for(agg,"stage2",2),
          "stage2_L4":top_for(agg,"stage2",4),
          "stage3_L1":top_for(agg,"stage3",1),
          "stage3_L2":top_for(agg,"stage3",2),
          "stage3_L4":top_for(agg,"stage3",4),
        }
        result={"top":top,"aggregate":agg,"rows":rows}
        OUT.parent.mkdir(parents=True,exist_ok=True)
        OUT.write_text(json.dumps(result,indent=2)+"\n")
        print("TOP",json.dumps(top),flush=True)
    finally:
        base.SCHEDULES,base.LENGTHS,base.SEEDS,base.ATTEMPTS=old_sched,old_len,old_seeds,old_att


if __name__=="__main__":
    main()
