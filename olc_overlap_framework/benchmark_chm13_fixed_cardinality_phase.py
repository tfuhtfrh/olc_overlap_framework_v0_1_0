"""Long-budget follow-up for fixed-cardinality kernels.

Tests whether the failure of the first benchmark was simply insufficient
degree-repair effort, and whether reconnect should be activated only after the
edge set is already close to a path-cover degree profile.
"""
from __future__ import annotations
import json, random
from pathlib import Path

import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import load_problem, project_rank, graph_metrics

OUT=Path("debug/qubo/chm13_fixed_cardinality_phase_benchmark_20260928.json")
BASE=20260928
CERT=1343093
RUNS=4
OUTER=18

# More effort than the initial benchmark.
fc.SWEEPS=1500
fc.READS=2

def run(mode,seed):
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    templates=fc.build_reconnect_templates(ep)
    rng=random.Random(seed)
    ranks={r:i for i,r in enumerate(rng.sample(rids,len(rids)))}
    selected=None
    prev_deg=None
    rows=[]
    for it in range(OUTER):
        op=fc.ORDER_SCHEDULE[it % len(fc.ORDER_SCHEDULE)]
        if mode=="exchange_long":
            kernel="exchange_only"
        elif mode=="time_phase":
            kernel="exchange_only" if it<12 else "static_50"
        elif mode=="threshold_phase":
            kernel="static_50" if (prev_deg is not None and prev_deg<=2) else "exchange_only"
        else:
            raise ValueError(mode)
        bqm=fc.build_bqm_fixed_cardinality(
            ep,cost,incoming,outgoing,ranks,
            degree_conflict_penalty=288.0,order_penalty=op)
        selected,sec,en,stats=fc.sample_fixed_cardinality(
            bqm,ep,templates,len(rids)-1,selected,seed+1000*it,kernel)
        ranks=project_rank(rids,selected)
        met=graph_metrics(rids,selected,ranks,reward)
        prev_deg=met["degree_conflicts"]
        row={"mode":mode,"seed":seed,"iteration":it,"kernel":kernel,
             "order_penalty":op,"seconds":sec,**stats,**met}
        rows.append(row)
        print("ROW",json.dumps(row),flush=True)
    feas=[r for r in rows if r["valid_path"]]
    best=max(feas,key=lambda r:r["path_score"]) if feas else None
    return rows,{
      "mode":mode,"seed":seed,"ever_feasible":bool(feas),
      "first_feasible_iteration":next((r["iteration"] for r in rows if r["valid_path"]),None),
      "best_score":None if best is None else best["path_score"],
      "score_gap":None if best is None else CERT-best["path_score"],
      "min_degree_conflicts":min(r["degree_conflicts"] for r in rows),
      "final_degree_conflicts":rows[-1]["degree_conflicts"],
      "final_cycle_count":rows[-1]["cycle_count"],
    }

def main():
    rows=[];summ=[]
    modes=("exchange_long","time_phase","threshold_phase")
    for mi,mode in enumerate(modes):
        for i in range(RUNS):
            rr,ss=run(mode,BASE+100000*mi+10000*i)
            rows.extend(rr);summ.append(ss)
            print("SUMMARY",json.dumps(ss),flush=True)
    agg={}
    for mode in modes:
        ss=[s for s in summ if s["mode"]==mode]
        fs=[s for s in ss if s["ever_feasible"]]
        agg[mode]={
          "runs":len(ss),"feasible_runs":len(fs),
          "best_score":max((s["best_score"] for s in fs),default=None),
          "min_degree_conflicts":min(s["min_degree_conflicts"] for s in ss),
        }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"aggregate":agg,"summaries":summ,"rows":rows},indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":
    main()
