"""Confirmatory validation of tuned Stage-2 / Stage-3 schedules.

Use 24 seeds and 4000 attempts on a small set of schedules selected from the
coarse and fine searches. No whole-worldline L=8 update is used.

Stage 2 candidates:
- beta=6.0, Gamma 3.5->0.6, L=2
- beta=6.5, Gamma 3.0->0.6, L=2
- beta=6.5, Gamma 3.0->0.6, L=1

Stage 3 candidates:
- beta=5.5, Gamma 4.0->0.8, L=4
- beta=6.0, Gamma 4.0->1.0, L=4
- beta=6.5, Gamma 4.0->0.8, L=1
- beta=6.5, Gamma 4.0->1.0, L=2

Baseline short-cluster controls are included.
"""
from __future__ import annotations
import json
from pathlib import Path
import benchmark_chm13_schedule_cluster_freezing as base

OUT=Path("debug/qubo/chm13_schedule_tuned_validation_20260930.json")
SEEDS=tuple(202609700+i for i in range(24))
ATTEMPTS=4000

CASES=[
 ("stage2","s2_tuned_L2_a",2,6.0,3.5,0.6),
 ("stage2","s2_tuned_L2_b",2,6.5,3.0,0.6),
 ("stage2","s2_tuned_L1",1,6.5,3.0,0.6),
 ("stage2","s2_baseline_L1",1,4.0,3.0,0.03),
 ("stage2","s2_baseline_L2",2,4.0,3.0,0.03),
 ("stage3","s3_tuned_L4_a",4,5.5,4.0,0.8),
 ("stage3","s3_tuned_L4_b",4,6.0,4.0,1.0),
 ("stage3","s3_tuned_L1",1,6.5,4.0,0.8),
 ("stage3","s3_tuned_L2",2,6.5,4.0,1.0),
 ("stage3","s3_baseline_L1",1,4.0,3.0,0.03),
 ("stage3","s3_baseline_L4",4,4.0,3.0,0.03),
]


def main():
    old_sched,old_att,old_seeds=base.SCHEDULES,base.ATTEMPTS,base.SEEDS
    try:
        contexts={"stage2":base.stage2_context(),"stage3":base.stage3_context()}
        base.ATTEMPTS=ATTEMPTS;base.SEEDS=SEEDS
        rows=[]
        for kind,name,L,b,g0,g1 in CASES:
            base.SCHEDULES={name:{"beta0":b,"beta1":b,"g0":g0,"g1":g1}}
            for seed in SEEDS:
                r=base.run(kind,name,L,seed,contexts[kind])
                rows.append(r);print("ROW",json.dumps(r),flush=True)

        agg={}
        for kind,name,L,b,g0,g1 in CASES:
            rr=[r for r in rows if r["kind"]==kind and r["schedule"]==name]
            hits=[r for r in rr if r["ever_any_target"]]
            allhits=[r for r in rr if r["ever_all_target"]]
            first=[r["first_any_target"] for r in hits]
            agg[name]={
              "kind":kind,"L":L,"beta":b,"gamma0":g0,"gamma1":g1,
              "K0":rr[0]["K0"],"K1":rr[0]["K1"],
              "runs":len(rr),"any_hits":len(hits),"P_any":len(hits)/len(rr),
              "all_hits":len(allhits),"P_all":len(allhits)/len(rr),
              "mean_acceptance":sum(r["acceptance"] for r in rr)/len(rr),
              "mean_late_acceptance":sum(r["late_acceptance"] for r in rr)/len(rr),
              "mean_final_target_slices":sum(r["final_target_slices"] for r in rr)/len(rr),
              "mean_max_target_slices":sum(r["max_target_slices"] for r in rr)/len(rr),
              "median_first_any":sorted(first)[len(first)//2] if first else None,
            }

        result={"aggregate":agg,"rows":rows}
        OUT.parent.mkdir(parents=True,exist_ok=True)
        OUT.write_text(json.dumps(result,indent=2)+"\n")
        print("AGGREGATE",json.dumps(agg),flush=True)
    finally:
        base.SCHEDULES,base.ATTEMPTS,base.SEEDS=old_sched,old_att,old_seeds


if __name__=="__main__":
    main()
