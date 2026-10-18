"""Dual-track standard-SQA hybrid: rank-free explorer + guided probes.

Motivation
----------
Experiments show:
- no-rank explores broadly but does not retain feasible HPs;
- deterministic rank reduces exploration;
- rank dropout / rank mixtures improve HP incidence;
- reachability rank can transiently hit the certified ground;
- increasing rank strength does not reliably retain an HP or ground under a
  fresh forward-SQA episode.

Therefore separate two roles:
1) explorer state: propagated using rank-free base energy only;
2) incumbent archive: best valid HP ever observed, using feasibility and the
   known overlap objective only (no knowledge of the optimum).

Each outer iteration has a fixed total budget of 8 standard SQA reads.
No custom updater.

Methods
-------
none:
    8 rank-free reads.

reach_probe:
    4 rank-free reads + 4 reachability-guided reads (A=0.02) whenever a
    previous explorer state exists; otherwise 8 rank-free reads.

rankmix_probe:
    4 rank-free reads + four one-read alternative projected-rank hypotheses
    when the explorer is degree-correct cyclic; otherwise 8 rank-free reads.

The next explorer state is the candidate with lowest RANK-FREE BQM energy.
Guided branches therefore cannot force the Markov state simply because their
auxiliary rank energy prefers them. They can, however, discover an HP and
update the incumbent.

Fixed 12 outer iterations; no ground-score early stop.
"""
from __future__ import annotations
import json,random,time
from pathlib import Path
import openjij as oj

import benchmark_chm13_rank_expression_screening as rs
from benchmark_chm13_reachability_rank import reachability_bad_edges
from benchmark_chm13_projected_edge_hybrid import graph_metrics,project_rank

OUT=Path("debug/qubo/chm13_dual_track_rank_hybrid_20260930.json")
GROUND=1_343_093
A=0.02
OUTER=12
SEEDS=(202609461,202609462,202609463,202609464)
STARTS=("zero","random")
METHODS=("none","reach_probe","rankmix_probe")


def reach_bqm(rids,ep,cost,incoming,outgoing,selected):
    b=rs.base_bqm(rids,ep,cost,incoming,outgoing)
    for e in reachability_bad_edges(rids,ep,selected):
        b.add_linear(e,A)
    return b


def choose_explorer(candidates,N):
    # H0 energy is the primary propagation quantity. Topology only breaks exact
    # numerical ties, so the explorer remains essentially rank-free.
    return min(
      candidates,
      key=lambda c:(
        c["base_energy"],
        c["metrics"]["degree_conflicts"],
        abs(c["metrics"]["selected_edges"]-(N-1)),
        c["metrics"]["cycle_count"],
      )
    )


def run(method,start,seed):
    rids,ep,reward,cost,incoming,outgoing=rs.load_problem();N=len(rids)
    rng=random.Random(seed)
    base=rs.base_bqm(rids,ep,cost,incoming,outgoing)
    selected=None;rows=[];inc=None;inc_it=None

    for it in range(OUTER):
        candidates=[];sec=0.0;branches=[]

        if method=="none" or selected is None:
            cc,dt=rs.sample_branch(
              base,base,ep,rids,reward,selected,start,rng,
              seed+10000*it,8)
            candidates+=cc;sec+=dt;branches.append("none:8")

        elif method=="reach_probe":
            cc,dt=rs.sample_branch(
              base,base,ep,rids,reward,selected,start,rng,
              seed+10000*it,4)
            candidates+=cc;sec+=dt;branches.append("none:4")
            rb=reach_bqm(rids,ep,cost,incoming,outgoing,selected)
            cc,dt=rs.sample_branch(
              rb,base,ep,rids,reward,selected,start,rng,
              seed+10000*it+5000,4)
            candidates+=cc;sec+=dt;branches.append("reach:4")

        elif method=="rankmix_probe":
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
            raise ValueError(method)

        hp=[c["metrics"]["path_score"] for c in candidates if c["metrics"]["valid_path"]]
        if hp:
            cur=max(hp)
            if inc is None or cur>inc:
                inc=cur;inc_it=it

        chosen=choose_explorer(candidates,N)
        selected=set(chosen["selected"])
        met=chosen["metrics"]

        row={
          "method":method,"start":start,"seed":seed,"iteration":it,
          "branches":branches,"seconds":sec,
          "candidate_hp_count":len(hp),"candidate_best_hp":max(hp,default=None),
          "incumbent_score":inc,"incumbent_iteration":inc_it,
          "explorer_base_energy":chosen["base_energy"],
          **met,
        }
        rows.append(row);print("ROW",json.dumps(row),flush=True)

    return rows,{
      "method":method,"start":start,"seed":seed,
      "incumbent_score":inc,"ground_ever":inc==GROUND,
      "final_explorer_valid":rows[-1]["valid_path"],
      "final_explorer_score":rows[-1]["path_score"],
      "final_explorer_ground":rows[-1]["path_score"]==GROUND,
      "mean_seconds":sum(r["seconds"] for r in rows)/len(rows),
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
        agg[f"{method}|{start}"]={
          "runs":len(aa),"hp_incumbent_runs":len(inc),
          "ground_ever":sum(s["ground_ever"] for s in aa),
          "best_incumbent":max(inc,default=None),
          "median_incumbent":sorted(inc)[len(inc)//2] if inc else None,
          "final_explorer_hp":sum(s["final_explorer_valid"] for s in aa),
          "final_explorer_ground":sum(s["final_explorer_ground"] for s in aa),
          "mean_seconds_per_outer":sum(s["mean_seconds"] for s in aa)/len(aa),
        }
    result={"A":A,"aggregate":agg,"summaries":summ,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)


if __name__=="__main__":
    main()
