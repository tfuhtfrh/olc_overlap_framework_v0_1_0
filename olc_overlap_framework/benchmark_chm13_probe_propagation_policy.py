"""Propagation-policy screening for the standard-SQA hybrid.

All inner solves are vanilla OpenJij SQASampler SingleSpinFlip.
No custom updater, no ground oracle, no early stop.

Fixed per-outer budget: 8 standard SQA reads.
Fixed outer budget: 16.

Methods
-------
pool_reach4:
    4 no-rank reads + 4 reachability-guided reads from current explorer.
    All 8 samples are eligible to become next explorer, selected by rank-free H0.

probe_reach4:
    Same 4+4 sampling, but ONLY no-rank samples can become next explorer.
    Reachability samples can update the best-feasible incumbent but cannot steer
    the explorer trajectory.

probe_reach2:
    6 no-rank + 2 reachability probe reads; explorer from no-rank only.

incumbent_probe:
    4 no-rank reads from explorer -> eligible for propagation;
    2 reachability reads from explorer -> incumbent-only probes;
    2 reachability reads warm-started from current incumbent when one exists
      (otherwise 2 extra no-rank explorer reads) -> incumbent-only probes.

This tests whether rank guidance is better treated as an observation/probe
channel rather than as a state-transition mechanism.
"""
from __future__ import annotations
import json, random
from pathlib import Path

import benchmark_chm13_rank_expression_screening as rs
from benchmark_chm13_reachability_rank import reachability_bad_edges
from benchmark_chm13_dual_track_rank_hybrid import choose_explorer

OUT=Path("debug/qubo/chm13_probe_propagation_policy_20260930.json")
GROUND=1_343_093
A=0.02
OUTER=16
SEEDS=tuple(202609700+i for i in range(8))
STARTS=("zero","random")
METHODS=("pool_reach4","probe_reach4","probe_reach2","incumbent_probe")


def reach_bqm(rids,ep,cost,incoming,outgoing,selected):
    b=rs.base_bqm(rids,ep,cost,incoming,outgoing)
    for e in reachability_bad_edges(rids,ep,selected):
        b.add_linear(e,A)
    return b


def update_inc(inc,inc_sel,cands):
    best=inc
    best_sel=inc_sel
    for c in cands:
        m=c["metrics"]
        if not m["valid_path"]:
            continue
        sc=m["path_score"]
        if best is None or sc>best:
            best=sc;best_sel=set(c["selected"])
    return best,best_sel


def run(method,start,seed):
    rids,ep,reward,cost,incoming,outgoing=rs.load_problem();N=len(rids)
    rng=random.Random(seed)
    base=rs.base_bqm(rids,ep,cost,incoming,outgoing)
    selected=None;inc=None;inc_sel=None;inc_it=None;rows=[]

    for it in range(OUTER):
        propagation=[];all_cands=[];branches=[];sec=0.0
        bs=seed+10000*it

        if selected is None:
            cc,dt=rs.sample_branch(
              base,base,ep,rids,reward,None,start,rng,bs,8)
            propagation+=cc;all_cands+=cc;sec+=dt;branches.append("none:8")
        elif method=="pool_reach4":
            cc,dt=rs.sample_branch(
              base,base,ep,rids,reward,selected,start,rng,bs,4)
            propagation+=cc;all_cands+=cc;sec+=dt;branches.append("none:4")
            rb=reach_bqm(rids,ep,cost,incoming,outgoing,selected)
            cc,dt=rs.sample_branch(
              rb,base,ep,rids,reward,selected,start,rng,bs+5000,4)
            propagation+=cc;all_cands+=cc;sec+=dt;branches.append("reach:4(pool)")
        elif method=="probe_reach4":
            cc,dt=rs.sample_branch(
              base,base,ep,rids,reward,selected,start,rng,bs,4)
            propagation+=cc;all_cands+=cc;sec+=dt;branches.append("none:4")
            rb=reach_bqm(rids,ep,cost,incoming,outgoing,selected)
            cc,dt=rs.sample_branch(
              rb,base,ep,rids,reward,selected,start,rng,bs+5000,4)
            all_cands+=cc;sec+=dt;branches.append("reach:4(probe)")
        elif method=="probe_reach2":
            cc,dt=rs.sample_branch(
              base,base,ep,rids,reward,selected,start,rng,bs,6)
            propagation+=cc;all_cands+=cc;sec+=dt;branches.append("none:6")
            rb=reach_bqm(rids,ep,cost,incoming,outgoing,selected)
            cc,dt=rs.sample_branch(
              rb,base,ep,rids,reward,selected,start,rng,bs+5000,2)
            all_cands+=cc;sec+=dt;branches.append("reach:2(probe)")
        elif method=="incumbent_probe":
            cc,dt=rs.sample_branch(
              base,base,ep,rids,reward,selected,start,rng,bs,4)
            propagation+=cc;all_cands+=cc;sec+=dt;branches.append("none:4")
            rb=reach_bqm(rids,ep,cost,incoming,outgoing,selected)
            cc,dt=rs.sample_branch(
              rb,base,ep,rids,reward,selected,start,rng,bs+4000,2)
            all_cands+=cc;sec+=dt;branches.append("reach_explorer:2(probe)")
            if inc_sel is not None:
                ib=reach_bqm(rids,ep,cost,incoming,outgoing,inc_sel)
                cc,dt=rs.sample_branch(
                  ib,base,ep,rids,reward,inc_sel,start,rng,bs+7000,2)
                all_cands+=cc;sec+=dt;branches.append("reach_incumbent:2(probe)")
            else:
                cc,dt=rs.sample_branch(
                  base,base,ep,rids,reward,selected,start,rng,bs+7000,2)
                propagation+=cc;all_cands+=cc;sec+=dt;branches.append("none_extra:2")
        else:
            raise ValueError(method)

        old_inc=inc
        inc,inc_sel=update_inc(inc,inc_sel,all_cands)
        if inc!=old_inc:
            inc_it=it

        chosen=choose_explorer(propagation,N)
        selected=set(chosen["selected"])
        met=chosen["metrics"]

        hp_all=[c["metrics"]["path_score"] for c in all_cands if c["metrics"]["valid_path"]]
        hp_prop=[c["metrics"]["path_score"] for c in propagation if c["metrics"]["valid_path"]]
        row={
          "method":method,"start":start,"seed":seed,"iteration":it,
          "branches":branches,
          "all_hp_count":len(hp_all),"all_best_hp":max(hp_all,default=None),
          "prop_hp_count":len(hp_prop),"prop_best_hp":max(hp_prop,default=None),
          "incumbent_score":inc,"incumbent_iteration":inc_it,
          "explorer_base_energy":chosen["base_energy"],"seconds":sec,
          **met,
        }
        rows.append(row);print("ROW",json.dumps(row),flush=True)

    return rows,{
      "method":method,"start":start,"seed":seed,
      "incumbent_score":inc,"ground_ever":inc==GROUND,
      "first_hp_iteration":next((r["iteration"] for r in rows if r["incumbent_score"] is not None),None),
      "final_explorer_valid":rows[-1]["valid_path"],
      "final_explorer_score":rows[-1]["path_score"],
    }


def main():
    rows=[];summ=[]
    for method in METHODS:
      for start in STARTS:
        for si,bs in enumerate(SEEDS):
          seed=bs+10000*si
          rr,ss=run(method,start,seed)
          rows+=rr;summ.append(ss);print("SUMMARY",json.dumps(ss),flush=True)

    agg={}
    for method in METHODS:
      for start in STARTS:
        aa=[s for s in summ if s["method"]==method and s["start"]==start]
        inc=[s["incumbent_score"] for s in aa if s["incumbent_score"] is not None]
        first=[s["first_hp_iteration"] for s in aa if s["first_hp_iteration"] is not None]
        agg[f"{method}|{start}"]={
          "runs":len(aa),
          "hp_incumbent_runs":len(inc),
          "ground_ever":sum(s["ground_ever"] for s in aa),
          "best_incumbent":max(inc,default=None),
          "median_incumbent":sorted(inc)[len(inc)//2] if inc else None,
          "median_first_hp_iteration":sorted(first)[len(first)//2] if first else None,
          "final_explorer_hp":sum(s["final_explorer_valid"] for s in aa),
        }
    result={"A":A,"outer":OUTER,"reads_per_outer":8,"aggregate":agg,"summaries":summ,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":
    main()
