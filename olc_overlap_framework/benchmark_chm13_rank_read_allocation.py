"""Read-allocation screening for the standard-SQA rank hybrid.

All inner calls are vanilla OpenJij SQASampler SingleSpinFlip.
No custom MC updater, no ground oracle, no early stop.

Each outer iteration has exactly 8 standard SQA reads and 16 outer iterations.
The explorer is propagated by minimum rank-free H0 energy. Best valid HP seen
across all reads/outer iterations is archived as a non-oracle incumbent.

Allocations
-----------
none8:
    8 rank-free reads.

reach2:
    6 rank-free + 2 reachability-guided reads.

reach4:
    4 rank-free + 4 reachability-guided reads.

tri_mix:
    4 rank-free + 2 reachability-guided + 2 alternative-rank reads.

rankmix4:
    4 rank-free + four one-read alternative-rank hypotheses, only when current
    explorer is degree-correct cyclic; otherwise 8 rank-free reads.

This directly tests the exploration/guidance tradeoff while holding total SQA
read budget fixed.
"""
from __future__ import annotations
import json, random
from pathlib import Path

import benchmark_chm13_dual_track_rank_hybrid as d
import benchmark_chm13_rank_expression_screening as rs
from benchmark_chm13_reachability_rank import reachability_bad_edges
from benchmark_chm13_projected_edge_hybrid import graph_metrics, project_rank

OUT=Path("debug/qubo/chm13_rank_read_allocation_20260930.json")
GROUND=1_343_093
A=0.02
OUTER=16
SEEDS=tuple(202609600+i for i in range(8))
STARTS=("zero","random")
METHODS=("none8","reach2","reach4","tri_mix","rankmix4")


def reach_bqm(rids,ep,cost,incoming,outgoing,selected):
    b=rs.base_bqm(rids,ep,cost,incoming,outgoing)
    for e in reachability_bad_edges(rids,ep,selected):
        b.add_linear(e,A)
    return b


def add_rank_reads(candidates,sec,branches,n,rids,ep,reward,cost,incoming,outgoing,
                   selected,start,rng,seed,base):
    if n<=0:
        return candidates,sec,branches
    ranks_list=rs.alt_ranks(rids,selected,n,rng)
    for k,ranks in enumerate(ranks_list):
        rb=rs.rank_bqm(rids,ep,cost,incoming,outgoing,ranks,A)
        cc,dt=rs.sample_branch(
          rb,base,ep,rids,reward,selected,start,rng,
          seed+1000*(k+1),1)
        candidates+=cc;sec+=dt;branches.append(f"rank{k}:1")
    return candidates,sec,branches


def run(method,start,seed):
    rids,ep,reward,cost,incoming,outgoing=rs.load_problem();N=len(rids)
    rng=random.Random(seed)
    base=rs.base_bqm(rids,ep,cost,incoming,outgoing)
    selected=None;inc=None;inc_it=None;rows=[]

    for it in range(OUTER):
        candidates=[];sec=0.0;branches=[]
        base_seed=seed+10000*it

        if selected is None or method=="none8":
            cc,dt=rs.sample_branch(
              base,base,ep,rids,reward,selected,start,rng,base_seed,8)
            candidates+=cc;sec+=dt;branches.append("none:8")
        else:
            prev=graph_metrics(rids,selected,project_rank(rids,selected),reward)
            gated=(prev["selected_edges"]==N-1 and prev["degree_conflicts"]==0 and prev["cycle_count"]>0)

            if method=="reach2":
                n0,nr,nrank=6,2,0
            elif method=="reach4":
                n0,nr,nrank=4,4,0
            elif method=="tri_mix":
                n0,nr,nrank=4,2,2
            elif method=="rankmix4":
                if gated:
                    n0,nr,nrank=4,0,4
                else:
                    n0,nr,nrank=8,0,0
            else:
                raise ValueError(method)

            if n0:
                cc,dt=rs.sample_branch(
                  base,base,ep,rids,reward,selected,start,rng,base_seed,n0)
                candidates+=cc;sec+=dt;branches.append(f"none:{n0}")
            if nr:
                rb=reach_bqm(rids,ep,cost,incoming,outgoing,selected)
                cc,dt=rs.sample_branch(
                  rb,base,ep,rids,reward,selected,start,rng,base_seed+5000,nr)
                candidates+=cc;sec+=dt;branches.append(f"reach:{nr}")
            if nrank and gated:
                candidates,sec,branches=add_rank_reads(
                  candidates,sec,branches,nrank,rids,ep,reward,cost,incoming,outgoing,
                  selected,start,rng,base_seed+7000,base)
            elif nrank and not gated:
                # Preserve fixed 8-read budget with rank-free reads when rank
                # hypotheses are not meaningful.
                cc,dt=rs.sample_branch(
                  base,base,ep,rids,reward,selected,start,rng,base_seed+7000,nrank)
                candidates+=cc;sec+=dt;branches.append(f"none_extra:{nrank}")

        hp=[c["metrics"]["path_score"] for c in candidates if c["metrics"]["valid_path"]]
        if hp:
            cur=max(hp)
            if inc is None or cur>inc:
                inc=cur;inc_it=it

        chosen=d.choose_explorer(candidates,N)
        selected=set(chosen["selected"])
        met=chosen["metrics"]

        row={
          "method":method,"start":start,"seed":seed,"iteration":it,
          "branches":branches,"candidate_hp_count":len(hp),
          "candidate_best_hp":max(hp,default=None),
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
      "final_explorer_ground":rows[-1]["path_score"]==GROUND,
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
          "final_explorer_ground":sum(s["final_explorer_ground"] for s in aa),
        }
    result={"A":A,"outer":OUTER,"reads_per_outer":8,"aggregate":agg,"summaries":summ,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":
    main()
