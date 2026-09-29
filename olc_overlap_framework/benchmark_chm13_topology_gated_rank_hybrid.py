"""Topology-gated read allocation for the standard-SQA hybrid.

Every inner solve is vanilla OpenJij SQASampler SingleSpinFlip.
No custom updater, no ground oracle, fixed 8-read budget per outer iteration.

Policy A: gated_reach
- irregular state (wrong count or degree conflicts): 8 no-rank reads
- degree-correct cyclic state: 6 no-rank + 2 reachability reads
- HP explorer state: 6 no-rank + 2 reachability reads
  (incumbent is already archived; guided reads only provide mild topology bias)

Policy B: gated_combo
- irregular: 8 no-rank
- degree-correct cyclic: 4 no-rank + 2 reachability + 2 alternative-rank reads
- HP: 6 no-rank + 2 reachability

Policy C: cyclic_only_combo
- irregular: 8 no-rank
- degree-correct cyclic: 4 no-rank + 2 reachability + 2 alternative-rank
- HP: 8 no-rank

All candidates may update the incumbent. Explorer propagation is restricted to
rank-free candidates so guided branches never directly steer the main state.
"""
from __future__ import annotations
import json,random
from pathlib import Path

import benchmark_chm13_rank_expression_screening as rs
from benchmark_chm13_reachability_rank import reachability_bad_edges
from benchmark_chm13_dual_track_rank_hybrid import choose_explorer
from benchmark_chm13_projected_edge_hybrid import graph_metrics,project_rank

OUT=Path("debug/qubo/chm13_topology_gated_rank_hybrid_20260930.json")
GROUND=1_343_093
A=0.02
OUTER=16
SEEDS=tuple(202609800+i for i in range(8))
STARTS=("zero","random")
METHODS=("gated_reach","gated_combo","cyclic_only_combo")


def reach_bqm(rids,ep,cost,incoming,outgoing,selected):
    b=rs.base_bqm(rids,ep,cost,incoming,outgoing)
    for e in reachability_bad_edges(rids,ep,selected):
        b.add_linear(e,A)
    return b


def update_inc(inc,inc_sel,cands):
    for c in cands:
        m=c["metrics"]
        if m["valid_path"] and (inc is None or m["path_score"]>inc):
            inc=m["path_score"];inc_sel=set(c["selected"])
    return inc,inc_sel


def run(method,start,seed):
    rids,ep,reward,cost,incoming,outgoing=rs.load_problem();N=len(rids)
    rng=random.Random(seed)
    base=rs.base_bqm(rids,ep,cost,incoming,outgoing)
    selected=None;inc=None;inc_sel=None;rows=[];inc_it=None

    for it in range(OUTER):
        bs=seed+10000*it
        propagation=[];allc=[];branches=[];sec=0.0

        if selected is None:
            state_type="initial"
            n0,nreach,nrank=8,0,0
        else:
            m0=graph_metrics(rids,selected,project_rank(rids,selected),reward)
            if m0["selected_edges"]!=N-1 or m0["degree_conflicts"]!=0:
                state_type="irregular"
                n0,nreach,nrank=8,0,0
            elif m0["cycle_count"]>0:
                state_type="cyclic"
                if method=="gated_reach":
                    n0,nreach,nrank=6,2,0
                else:
                    n0,nreach,nrank=4,2,2
            else:
                state_type="hp"
                if method=="cyclic_only_combo":
                    n0,nreach,nrank=8,0,0
                else:
                    n0,nreach,nrank=6,2,0

        if n0:
            cc,dt=rs.sample_branch(
              base,base,ep,rids,reward,selected,start,rng,bs,n0)
            propagation+=cc;allc+=cc;sec+=dt;branches.append(f"none:{n0}")
        if nreach:
            rb=reach_bqm(rids,ep,cost,incoming,outgoing,selected)
            cc,dt=rs.sample_branch(
              rb,base,ep,rids,reward,selected,start,rng,bs+5000,nreach)
            allc+=cc;sec+=dt;branches.append(f"reach:{nreach}(probe)")
        if nrank:
            ranks_list=rs.alt_ranks(rids,selected,nrank,rng)
            for k,ranks in enumerate(ranks_list):
                qb=rs.rank_bqm(rids,ep,cost,incoming,outgoing,ranks,A)
                cc,dt=rs.sample_branch(
                  qb,base,ep,rids,reward,selected,start,rng,bs+7000+1000*k,1)
                allc+=cc;sec+=dt;branches.append(f"rank{k}:1(probe)")

        old=inc
        inc,inc_sel=update_inc(inc,inc_sel,allc)
        if inc!=old:inc_it=it

        chosen=choose_explorer(propagation,N)
        selected=set(chosen["selected"])
        met=chosen["metrics"]

        hp=[c["metrics"]["path_score"] for c in allc if c["metrics"]["valid_path"]]
        row={
          "method":method,"start":start,"seed":seed,"iteration":it,
          "state_type":state_type,"branches":branches,
          "candidate_hp_count":len(hp),"candidate_best_hp":max(hp,default=None),
          "incumbent_score":inc,"incumbent_iteration":inc_it,
          "seconds":sec,"explorer_base_energy":chosen["base_energy"],**met,
        }
        rows.append(row);print("ROW",json.dumps(row),flush=True)

    return rows,{
      "method":method,"start":start,"seed":seed,
      "incumbent_score":inc,"ground_ever":inc==GROUND,
      "final_explorer_valid":rows[-1]["valid_path"],
      "final_explorer_score":rows[-1]["path_score"],
      "cyclic_guided_iterations":sum(r["state_type"]=="cyclic" for r in rows),
      "hp_guided_iterations":sum(r["state_type"]=="hp" for r in rows),
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
          "mean_cyclic_guided_iterations":sum(s["cyclic_guided_iterations"] for s in aa)/len(aa),
          "mean_hp_guided_iterations":sum(s["hp_guided_iterations"] for s in aa)/len(aa),
        }
    result={"A":A,"outer":OUTER,"reads_per_outer":8,"aggregate":agg,"summaries":summ,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":
    main()
