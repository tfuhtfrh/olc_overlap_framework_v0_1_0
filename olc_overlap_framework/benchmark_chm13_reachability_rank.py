"""Reachability-based partial-order rank for the CHM13 SQA hybrid.

Idea
----
Do not choose a total topological order.

Let C(u) be the SCC of vertex u in the current selected graph and let D be the
condensation DAG. A candidate edge u->v receives a weak linear penalty only if

    C(v) can already reach C(u) in D,

because adding u->v would then close a directed cycle between SCCs.

Properties:
- no arbitrary ordering between incomparable SCCs;
- no arbitrary direction inside an SCC;
- on an existing Hamilton path, backward/chord edges that would close a cycle
  remain weakly penalized, so guidance does not disappear merely because the
  current state is acyclic;
- it does not itself break a cycle inside one SCC, so it is also tested together
  with a weak current exact cycle cut.

All inner solves are vanilla OpenJij SQASampler SingleSpinFlip.
No custom updater. Fixed 12 outer episodes. No ground-score early stop.
Best valid HP is archived non-oracularly.
"""
from __future__ import annotations
import json,random
from pathlib import Path
import networkx as nx

import benchmark_chm13_standard_sqa_acyclicity_baseline as b
from benchmark_chm13_projected_edge_hybrid import build_bqm_count

OUT=Path("debug/qubo/chm13_reachability_rank_20260930.json")
GROUND=1_343_093
A_REACH=0.02
A_CUT=0.02
OUTER=12
SEEDS=(202609431,202609432,202609433,202609434)
STARTS=("zero","random")
METHODS=("none","current_cut","reachability","reachability_cut")


def reachability_bad_edges(rids,ep,selected):
    if selected is None:
        return set()
    g=nx.DiGraph();g.add_nodes_from(rids);g.add_edges_from(selected)
    sccs=list(nx.strongly_connected_components(g))
    c=nx.condensation(g,sccs)
    mp=c.graph["mapping"]

    # Descendant sets in the condensation DAG.
    desc={ci:nx.descendants(c,ci) for ci in c.nodes}
    bad=set()
    for u,v in ep:
        cu,cv=mp[u],mp[v]
        if cu==cv:
            continue
        # Existing path C(v) -> ... -> C(u), so u->v would close a cycle.
        if cu in desc[cv]:
            bad.add((u,v))
    return bad


def build(method,rids,ep,cost,incoming,outgoing,selected):
    dummy={r:0 for r in rids}
    bqm=build_bqm_count(
      ep,cost,incoming,outgoing,dummy,len(rids),
      degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)

    current=b.cycles(rids,selected)
    meta=[]
    if method in ("current_cut","reachability_cut") and current:
        meta=b.add_exact(bqm,current,A_CUT)

    bad=set()
    if method in ("reachability","reachability_cut") and selected is not None:
        bad=reachability_bad_edges(rids,ep,selected)
        for e in bad:
            bqm.add_linear(e,A_REACH)

    return bqm,meta,len(current),len(bad)


def run(method,start,seed):
    rids,ep,reward,cost,incoming,outgoing=b.load_problem();N=len(rids)
    rng=random.Random(seed);selected=None;rows=[];inc=None;inc_it=None

    for it in range(OUTER):
        bqm,meta,ncyc,nbad=build(
          method,rids,ep,cost,incoming,outgoing,selected)
        cfg={"beta":5.0,"gamma":1.0,"P":8}
        selected,sec,en,rm,nv,nq=b.sample_standard(
          bqm,ep,selected,meta,start,rng,seed+1000*it,cfg,rids,reward,N)
        met=b.graph_metrics(rids,selected,b.project_rank(rids,selected),reward)

        scores=[x["path_score"] for x in rm if x["valid_path"]]
        if met["valid_path"]:scores.append(met["path_score"])
        if scores:
            cur=max(scores)
            if inc is None or cur>inc:
                inc=cur;inc_it=it

        row={
          "method":method,"start":start,"seed":seed,"iteration":it,
          "cycle_witnesses":ncyc,"reachability_penalized_edges":nbad,
          "incumbent_score":inc,"incumbent_iteration":inc_it,
          "seconds":sec,**met,
        }
        rows.append(row);print("ROW",json.dumps(row),flush=True)

    return rows,{
      "method":method,"start":start,"seed":seed,
      "incumbent_score":inc,"ground_ever":inc==GROUND,
      "final_valid":rows[-1]["valid_path"],"final_score":rows[-1]["path_score"],
      "final_ground":rows[-1]["path_score"]==GROUND,
      "max_reachability_edges":max(r["reachability_penalized_edges"] for r in rows),
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
          "final_hp":sum(s["final_valid"] for s in aa),
          "final_ground":sum(s["final_ground"] for s in aa),
          "max_reachability_edges":max(s["max_reachability_edges"] for s in aa),
        }
    result={"A_reach":A_REACH,"A_cut":A_CUT,"aggregate":agg,"summaries":summ,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)


if __name__=="__main__":
    main()
