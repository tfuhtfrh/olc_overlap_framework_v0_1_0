"""Reachability penalty normalization under a probe-only SQA hybrid.

Purpose
-------
Current reachability guidance uses a fixed per-edge coefficient A=0.02.
The number |R(x)| of cycle-closing candidate edges changes across outer
iterations, so the total available guidance can vary substantially.

This benchmark keeps:
- 4 rank-free explorer reads;
- 4 reachability probe reads;
- explorer propagation from rank-free reads only;
- 16 outer iterations;
- vanilla OpenJij SingleSpinFlip only.

Compare:
fixed_002:
    a_e = 0.02.

norm_010:
    a_e = min(0.02, 0.10 / |R|).

norm_020:
    a_e = min(0.02, 0.20 / |R|).

norm_040:
    a_e = min(0.02, 0.40 / |R|).

Thus normalized variants never make an individual edge penalty stronger than
the current 0.02; they only weaken guidance when many edges are simultaneously
classified as cycle-closing.
"""
from __future__ import annotations
import json,random
from pathlib import Path

import benchmark_chm13_rank_expression_screening as rs
from benchmark_chm13_reachability_rank import reachability_bad_edges
from benchmark_chm13_dual_track_rank_hybrid import choose_explorer

OUT=Path("debug/qubo/chm13_reachability_normalization_20260930.json")
GROUND=1_343_093
OUTER=16
SEEDS=tuple(202609900+i for i in range(8))
STARTS=("zero","random")
METHODS={
 "fixed_002":None,
 "norm_010":0.10,
 "norm_020":0.20,
 "norm_040":0.40,
}


def guided_bqm(rids,ep,cost,incoming,outgoing,selected,total_budget):
    q=rs.base_bqm(rids,ep,cost,incoming,outgoing)
    bad=reachability_bad_edges(rids,ep,selected)
    if total_budget is None:
        ae=0.02
    else:
        ae=min(0.02,total_budget/max(1,len(bad)))
    for e in bad:q.add_linear(e,ae)
    return q,len(bad),ae


def run(method,start,seed):
    total_budget=METHODS[method]
    rids,ep,reward,cost,incoming,outgoing=rs.load_problem();N=len(rids)
    rng=random.Random(seed)
    base=rs.base_bqm(rids,ep,cost,incoming,outgoing)
    selected=None;inc=None;rows=[]

    for it in range(OUTER):
        bs=seed+10000*it
        prop=[];allc=[];branches=[];sec=0.0
        cc,dt=rs.sample_branch(
          base,base,ep,rids,reward,selected,start,rng,bs,4)
        prop+=cc;allc+=cc;sec+=dt;branches.append("none:4")

        if selected is None:
            cc,dt=rs.sample_branch(
              base,base,ep,rids,reward,selected,start,rng,bs+5000,4)
            allc+=cc;sec+=dt;nbad=0;ae=0.0;branches.append("none_probe:4")
        else:
            gb,nbad,ae=guided_bqm(
              rids,ep,cost,incoming,outgoing,selected,total_budget)
            cc,dt=rs.sample_branch(
              gb,base,ep,rids,reward,selected,start,rng,bs+5000,4)
            allc+=cc;sec+=dt;branches.append("reach_probe:4")

        hp=[c["metrics"]["path_score"] for c in allc if c["metrics"]["valid_path"]]
        if hp:
            cur=max(hp);inc=cur if inc is None else max(inc,cur)

        chosen=choose_explorer(prop,N)
        selected=set(chosen["selected"])
        met=chosen["metrics"]
        row={
          "method":method,"start":start,"seed":seed,"iteration":it,
          "reachability_edges":nbad,"per_edge_A":ae,
          "candidate_hp_count":len(hp),"candidate_best_hp":max(hp,default=None),
          "incumbent_score":inc,"seconds":sec,
          "explorer_base_energy":chosen["base_energy"],**met,
        }
        rows.append(row);print("ROW",json.dumps(row),flush=True)

    return rows,{
      "method":method,"start":start,"seed":seed,
      "incumbent_score":inc,"ground_ever":inc==GROUND,
      "final_explorer_valid":rows[-1]["valid_path"],
      "mean_reach_edges":sum(r["reachability_edges"] for r in rows)/len(rows),
      "mean_per_edge_A":sum(r["per_edge_A"] for r in rows)/len(rows),
    }


def main():
    rows=[];summ=[]
    for method in METHODS:
      for start in STARTS:
        for si,bs in enumerate(SEEDS):
          seed=bs+10000*si
          rr,ss=run(method,start,seed);rows+=rr;summ.append(ss)
          print("SUMMARY",json.dumps(ss),flush=True)

    agg={}
    for method in METHODS:
      for start in STARTS:
        aa=[s for s in summ if s["method"]==method and s["start"]==start]
        inc=[s["incumbent_score"] for s in aa if s["incumbent_score"] is not None]
        agg[f"{method}|{start}"]={
          "runs":len(aa),"hp_incumbent_runs":len(inc),
          "ground_ever":sum(s["ground_ever"] for s in aa),
          "best_incumbent":max(inc,default=None),
          "median_incumbent":sorted(inc)[len(inc)//2] if inc else None,
          "final_explorer_hp":sum(s["final_explorer_valid"] for s in aa),
          "mean_reach_edges":sum(s["mean_reach_edges"] for s in aa)/len(aa),
          "mean_per_edge_A":sum(s["mean_per_edge_A"] for s in aa)/len(aa),
        }
    result={"methods":METHODS,"outer":OUTER,"reads_per_outer":8,"aggregate":agg,"summaries":summ,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":
    main()
