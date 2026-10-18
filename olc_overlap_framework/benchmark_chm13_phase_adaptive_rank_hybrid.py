"""Phase-adaptive standard-SQA rank hybrid.

Empirical motivation
--------------------
- multi-rank mixture gave the highest HP-incumbent coverage in the matched
  screening;
- reachability probes produced certified ground hits while retaining more
  weight exploration;
- strong/static rank did not reliably retain an HP;
- a non-oracle best-feasible incumbent can provide retention externally.

Policy
------
Before any Hamilton path has been observed:
    use rank-free exploration plus multi-hypothesis rank probes when the
    explorer is degree-correct cyclic.

After the first HP incumbent exists:
    use rank-free exploration plus reachability-guided probes.

The explorer state is always propagated by minimum rank-free H0 energy.
The best feasible HP is archived separately and is never used as a warm start
unless it happens to be the explorer itself.

Every inner solve is vanilla OpenJij SQASampler SingleSpinFlip.
No ground-score early stop. Fixed outer budget.
"""
from __future__ import annotations
import json,random
from pathlib import Path

import benchmark_chm13_dual_track_rank_hybrid as d
import benchmark_chm13_rank_expression_screening as rs
from benchmark_chm13_reachability_rank import reachability_bad_edges
from benchmark_chm13_projected_edge_hybrid import graph_metrics,project_rank

OUT=Path("debug/qubo/chm13_phase_adaptive_rank_hybrid_20260930.json")
GROUND=1_343_093
A=0.02
OUTER=16
SEEDS=(202609481,202609482,202609483,202609484)
STARTS=("zero","random")


def reach_bqm(rids,ep,cost,incoming,outgoing,selected):
    b=rs.base_bqm(rids,ep,cost,incoming,outgoing)
    for e in reachability_bad_edges(rids,ep,selected):
        b.add_linear(e,A)
    return b


def run(start,seed):
    rids,ep,reward,cost,incoming,outgoing=rs.load_problem();N=len(rids)
    rng=random.Random(seed)
    base=rs.base_bqm(rids,ep,cost,incoming,outgoing)
    selected=None;inc=None;inc_it=None;rows=[]

    for it in range(OUTER):
        candidates=[];sec=0.0;branches=[];phase="feasibility" if inc is None else "weight"

        if selected is None:
            cc,dt=rs.sample_branch(
              base,base,ep,rids,reward,selected,start,rng,
              seed+10000*it,8)
            candidates+=cc;sec+=dt;branches.append("none:8")
        elif phase=="feasibility":
            prev=graph_metrics(rids,selected,project_rank(rids,selected),reward)
            gated=(prev["selected_edges"]==N-1 and prev["degree_conflicts"]==0 and prev["cycle_count"]>0)
            if not gated:
                cc,dt=rs.sample_branch(
                  base,base,ep,rids,reward,selected,start,rng,
                  seed+10000*it,8)
                candidates+=cc;sec+=dt;branches.append("none:8")
            else:
                cc,dt=rs.sample_branch(
                  base,base,ep,rids,reward,selected,start,rng,
                  seed+10000*it,4)
                candidates+=cc;sec+=dt;branches.append("none:4")
                ranks_list=rs.alt_ranks(rids,selected,4,rng)
                for k,ranks in enumerate(ranks_list):
                    rb=rs.rank_bqm(rids,ep,cost,incoming,outgoing,ranks,A)
                    cc,dt=rs.sample_branch(
                      rb,base,ep,rids,reward,selected,start,rng,
                      seed+10000*it+1000*(k+1),1)
                    candidates+=cc;sec+=dt;branches.append(f"r{k}:1")
        else:
            # Weight phase: keep half the reads completely rank-free and use
            # only sparse reachability guidance on the other half.
            cc,dt=rs.sample_branch(
              base,base,ep,rids,reward,selected,start,rng,
              seed+10000*it,4)
            candidates+=cc;sec+=dt;branches.append("none:4")
            rb=reach_bqm(rids,ep,cost,incoming,outgoing,selected)
            cc,dt=rs.sample_branch(
              rb,base,ep,rids,reward,selected,start,rng,
              seed+10000*it+5000,4)
            candidates+=cc;sec+=dt;branches.append("reach:4")

        hp=[c["metrics"]["path_score"] for c in candidates if c["metrics"]["valid_path"]]
        if hp:
            cur=max(hp)
            if inc is None or cur>inc:
                inc=cur;inc_it=it

        chosen=d.choose_explorer(candidates,N)
        selected=set(chosen["selected"])
        met=chosen["metrics"]

        row={
          "start":start,"seed":seed,"iteration":it,"phase":phase,
          "branches":branches,"seconds":sec,
          "candidate_hp_count":len(hp),"candidate_best_hp":max(hp,default=None),
          "incumbent_score":inc,"incumbent_iteration":inc_it,
          "explorer_base_energy":chosen["base_energy"],**met,
        }
        rows.append(row);print("ROW",json.dumps(row),flush=True)

    return rows,{
      "start":start,"seed":seed,
      "incumbent_score":inc,"ground_ever":inc==GROUND,
      "first_hp_iteration":next((r["iteration"] for r in rows if r["incumbent_score"] is not None),None),
      "final_explorer_valid":rows[-1]["valid_path"],
      "final_explorer_score":rows[-1]["path_score"],
      "final_explorer_ground":rows[-1]["path_score"]==GROUND,
      "weight_phase_iterations":sum(r["phase"]=="weight" for r in rows),
    }


def main():
    rows=[];summ=[]
    for start in STARTS:
      for si,bs in enumerate(SEEDS):
        seed=bs+10000*si
        rr,ss=run(start,seed);rows+=rr;summ.append(ss)
        print("SUMMARY",json.dumps(ss),flush=True)

    agg={}
    for start in STARTS:
      aa=[s for s in summ if s["start"]==start]
      inc=[s["incumbent_score"] for s in aa if s["incumbent_score"] is not None]
      first=[s["first_hp_iteration"] for s in aa if s["first_hp_iteration"] is not None]
      agg[start]={
        "runs":len(aa),"hp_incumbent_runs":len(inc),
        "ground_ever":sum(s["ground_ever"] for s in aa),
        "best_incumbent":max(inc,default=None),
        "median_incumbent":sorted(inc)[len(inc)//2] if inc else None,
        "median_first_hp_iteration":sorted(first)[len(first)//2] if first else None,
        "final_explorer_hp":sum(s["final_explorer_valid"] for s in aa),
        "final_explorer_ground":sum(s["final_explorer_ground"] for s in aa),
        "mean_weight_phase_iterations":sum(s["weight_phase_iterations"] for s in aa)/len(aa),
      }
    result={"A":A,"aggregate":agg,"summaries":summ,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)


if __name__=="__main__":
    main()
