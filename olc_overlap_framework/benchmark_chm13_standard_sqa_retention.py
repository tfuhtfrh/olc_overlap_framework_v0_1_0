"""Standard-SQA retention test for acyclicity guidance.

Motivation
----------
A standard-SQA + current rank-ensemble run reached certified ground at outer
iteration 4, then lost it immediately because an HP has no current cycle and
the ensemble bias vanished.

This benchmark keeps the updater vanilla OpenJij SingleSpinFlip and compares:

1) accum_exact:
   accumulate discovered exact cycle cuts at weak A=0.02. Exact cycle cuts are
   globally valid, so keeping them does not exclude any Hamilton path.

2) sticky_ensemble:
   retain the most recently observed non-empty cycle-ensemble pressure when
   the current state becomes acyclic. This is a heuristic guidance test, not an
   exact acyclicity encoding, because it continues to linearly penalize edges
   from a previously broken cycle.

No ground-score early stop. Fixed 12 outer episodes.
An incumbent best valid HP is archived only for reporting; it never changes
the sampling state or Hamiltonian.
"""
from __future__ import annotations
import json,random,time
from pathlib import Path
import openjij as oj

import benchmark_chm13_standard_sqa_acyclicity_baseline as b

OUT=Path("debug/qubo/chm13_standard_sqa_retention_20260930.json")
SEEDS=(202609311,202609312,202609313,202609314)
STARTS=("zero","random")
METHODS=("accum_exact","sticky_ensemble")
OUTER=12
GROUND=1_343_093
CFG={"beta":5.0,"gamma":1.0,"P":8}


def run(method,start,seed):
    rids,ep,reward,cost,incoming,outgoing=b.load_problem();N=len(rids)
    dummy={r:0 for r in rids};rng=random.Random(seed)
    selected=None;rows=[];history=[];histset=set();sticky=[]
    incumbent=None

    for it in range(OUTER):
        observed=b.cycles(rids,selected)
        if method=="accum_exact":
            for cyc in observed:
                if cyc not in histset:
                    histset.add(cyc);history.append(cyc)
            active=list(history)
        else:
            if observed:sticky=list(observed)
            active=list(sticky)

        bqm=b.build_bqm_count(
          ep,cost,incoming,outgoing,dummy,N,
          degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)
        if method=="accum_exact":
            meta=b.add_exact(bqm,active,b.CUT_A)
        else:
            meta=b.add_ensemble(bqm,active)

        selected,sec,en,rm,nv,nq=b.sample_standard(
          bqm,ep,selected,meta,start,rng,seed+1000*it,CFG,rids,reward,N)
        met=b.graph_metrics(rids,selected,b.project_rank(rids,selected),reward)

        read_hp=[x["path_score"] for x in rm if x["valid_path"]]
        candidates=list(read_hp)
        if met["valid_path"]:candidates.append(met["path_score"])
        if candidates:
            best=max(candidates)
            incumbent=best if incumbent is None else max(incumbent,best)

        row={
          "method":method,"start":start,"seed":seed,"iteration":it,
          "observed_cycles":len(observed),"active_guidance":len(active),
          "primary_valid":bool(met["valid_path"]),"primary_score":met["path_score"],
          "primary_cycles":met["cycle_count"],"any_read_ground":any(x["path_score"]==GROUND for x in rm),
          "incumbent_score":incumbent,"seconds":sec,
        }
        rows.append(row);print("ROW",json.dumps(row),flush=True)

    return rows,{
      "method":method,"start":start,"seed":seed,
      "incumbent_score":incumbent,
      "ground_ever":incumbent==GROUND,
      "final_valid":rows[-1]["primary_valid"],
      "final_score":rows[-1]["primary_score"],
      "final_ground":rows[-1]["primary_score"]==GROUND,
      "max_active_guidance":max(r["active_guidance"] for r in rows),
    }


def main():
    rows=[];summ=[]
    for method in METHODS:
      for start in STARTS:
        for si,base_seed in enumerate(SEEDS):
          seed=base_seed+10000*si
          rr,ss=run(method,start,seed);rows+=rr;summ.append(ss)
          print("SUMMARY",json.dumps(ss),flush=True)

    agg={}
    for method in METHODS:
      for start in STARTS:
        aa=[s for s in summ if s["method"]==method and s["start"]==start]
        inc=[s["incumbent_score"] for s in aa if s["incumbent_score"] is not None]
        agg[f"{method}|{start}"]={
          "runs":len(aa),"ground_ever":sum(s["ground_ever"] for s in aa),
          "final_ground":sum(s["final_ground"] for s in aa),
          "final_hp":sum(s["final_valid"] for s in aa),
          "best_incumbent":max(inc,default=None),
          "max_active_guidance":max(s["max_active_guidance"] for s in aa),
        }
    result={"aggregate":agg,"summaries":summ,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)


if __name__=="__main__":
    main()
