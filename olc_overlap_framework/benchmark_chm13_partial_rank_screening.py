"""Partial-order rank screening with vanilla OpenJij SQA inner solver.

Motivation
----------
A projected total rank unnecessarily orders vertices inside a nontrivial SCC.
That arbitrary within-SCC ordering creates the "one backward edge" artifact.

This benchmark treats rank as a partial-order certificate:

- between SCCs: use the condensation DAG topological order;
- inside one SCC: vertices are incomparable, so no rank penalty is applied.

Methods:
- none
- exact_cut: weak current-cycle exact cut A=0.02
- full_rank: weak projected total-rank backward penalty A=0.02
- partial_rank: weak SCC-condensation backward penalty A=0.02
- partial_rank_cut: partial rank + weak exact current-cycle cut

Guidance is gated to states with N-1 edges, zero degree conflict, and cycles.
Every inner solve is vanilla OpenJij SQASampler SingleSpinFlip.
No custom updater. Fixed 12 outer episodes. No ground early stop.
Best feasible incumbent is archived non-oracularly for reporting.
"""
from __future__ import annotations

import json, random, time
from pathlib import Path
import networkx as nx
import openjij as oj

import benchmark_chm13_standard_sqa_acyclicity_baseline as b
from benchmark_chm13_projected_edge_hybrid import build_bqm_count, project_rank

OUT=Path("debug/qubo/chm13_partial_rank_screening_20260930.json")
GROUND=1_343_093
A=0.02
OUTER=12
READS=8
SWEEPS=1400
SEEDS=(202609421,202609422,202609423,202609424)
STARTS=("zero","random")
METHODS=("none","exact_cut","full_rank","partial_rank","partial_rank_cut")


def partial_backward_edges(rids,ep,selected):
    g=nx.DiGraph();g.add_nodes_from(rids);g.add_edges_from(selected or [])
    sccs=list(nx.strongly_connected_components(g))
    c=nx.condensation(g,sccs)
    mapping=c.graph["mapping"]
    topo=list(nx.topological_sort(c))
    pos={ci:i for i,ci in enumerate(topo)}
    bad=set()
    for e in ep:
        u,v=e;cu,cv=mapping[u],mapping[v]
        if cu==cv:
            continue
        if pos[cv] <= pos[cu]:
            bad.add(e)
    return bad


def build_guided(method,rids,ep,cost,incoming,outgoing,selected,gated):
    dummy={r:0 for r in rids}
    base=build_bqm_count(
      ep,cost,incoming,outgoing,dummy,len(rids),
      degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)
    meta=[]
    if not gated or method=="none":
        return base,meta,0,0

    active_cycles=b.cycles(rids,selected)

    if method in ("exact_cut","partial_rank_cut"):
        meta=b.add_exact(base,active_cycles,A)

    if method=="full_rank":
        ranks=project_rank(rids,selected)
        for e in ep:
            if ranks[e[1]]<=ranks[e[0]]:
                base.add_linear(e,A)

    if method in ("partial_rank","partial_rank_cut"):
        bad=partial_backward_edges(rids,ep,selected)
        for e in bad:
            base.add_linear(e,A)
        return base,meta,len(active_cycles),len(bad)

    return base,meta,len(active_cycles),0


def run(method,start,seed):
    rids,ep,reward,cost,incoming,outgoing=b.load_problem();N=len(rids)
    rng=random.Random(seed);selected=None;rows=[];inc=None

    for it in range(OUTER):
        prev=(b.graph_metrics(rids,selected,b.project_rank(rids,selected),reward)
              if selected is not None else None)
        gated=bool(
          selected is not None and prev["selected_edges"]==N-1
          and prev["degree_conflicts"]==0 and prev["cycle_count"]>0)
        bqm,meta,ncyc,nbad=build_guided(
          method,rids,ep,cost,incoming,outgoing,selected,gated)
        cfg={"beta":5.0,"gamma":1.0,"P":8}
        selected,sec,en,rm,nv,nq=b.sample_standard(
          bqm,ep,selected,meta,start,rng,seed+1000*it,cfg,rids,reward,N)
        met=b.graph_metrics(rids,selected,b.project_rank(rids,selected),reward)

        scores=[x["path_score"] for x in rm if x["valid_path"]]
        if met["valid_path"]:scores.append(met["path_score"])
        if scores:
            cur=max(scores)
            inc=cur if inc is None else max(inc,cur)

        row={
          "method":method,"start":start,"seed":seed,"iteration":it,
          "gated":gated,"active_cycles":ncyc,"partial_backward_candidates":nbad,
          "incumbent_score":inc,"seconds":sec,
          **met,
        }
        rows.append(row);print("ROW",json.dumps(row),flush=True)

    return rows,{
      "method":method,"start":start,"seed":seed,
      "incumbent_score":inc,"ground_ever":inc==GROUND,
      "final_valid":rows[-1]["valid_path"],"final_score":rows[-1]["path_score"],
      "final_ground":rows[-1]["path_score"]==GROUND,
      "gated_iterations":sum(r["gated"] for r in rows),
    }


def main():
    rows=[];summ=[]
    for method in METHODS:
      for start in STARTS:
        for si,base_seed in enumerate(SEEDS):
          seed=base_seed+10000*si
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
          "final_hp":sum(s["final_valid"] for s in aa),
          "final_ground":sum(s["final_ground"] for s in aa),
          "mean_gated_iterations":sum(s["gated_iterations"] for s in aa)/len(aa),
        }
    result={"A":A,"aggregate":agg,"summaries":summ,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)


if __name__=="__main__":
    main()
