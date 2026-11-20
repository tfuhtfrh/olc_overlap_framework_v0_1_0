"""Parallel worker for combined hybrid validation.

Environment:
  METHOD in {none8,cyclic_combo,combo_inc2,combo_inc2_reach}
  START in {zero,random}

Runs the same 12-seed fixed-budget experiment as
benchmark_chm13_combined_hybrid_validation.py, but for one method/start pair.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import benchmark_chm13_combined_hybrid_validation as b

METHOD=os.environ["METHOD"]
START=os.environ["START"]
OUT=Path(f"debug/qubo/chm13_combined_worker_{METHOD}_{START}_20260930.json")

def main():
    rows=[];summ=[]
    for si,bs in enumerate(b.SEEDS):
        seed=bs+10000*si
        rr,ss=b.run(METHOD,START,seed)
        rows+=rr;summ.append(ss)
        print("SUMMARY",json.dumps(ss),flush=True)

    inc=[s["incumbent_score"] for s in summ if s["incumbent_score"] is not None]
    grounds=[s for s in summ if s["ground_ever"]]
    src={}
    for s in grounds:
        src[s["ground_source"]]=src.get(s["ground_source"],0)+1
    gits=[s["ground_iteration"] for s in grounds if s["ground_iteration"] is not None]
    agg={
      "method":METHOD,"start":START,"runs":len(summ),
      "hp_incumbent_runs":len(inc),
      "ground_ever":len(grounds),
      "ground_sources":src,
      "best_incumbent":max(inc,default=None),
      "median_incumbent":sorted(inc)[len(inc)//2] if inc else None,
      "median_ground_iteration":sorted(gits)[len(gits)//2] if gits else None,
      "final_explorer_hp":sum(s["final_explorer_valid"] for s in summ),
    }
    result={"aggregate":agg,"summaries":summ,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":
    main()
