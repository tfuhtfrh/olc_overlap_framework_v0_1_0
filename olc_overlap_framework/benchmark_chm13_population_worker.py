"""Parallel worker for two-explorer population hybrid."""
from __future__ import annotations
import json,os
from pathlib import Path
import benchmark_chm13_population_hybrid as b

METHOD=os.environ["METHOD"]
START=os.environ["START"]
OUT=Path(f"debug/qubo/chm13_population_worker_{METHOD}_{START}_20260930.json")

def main():
    rows=[];summ=[]
    for si,bs in enumerate(b.SEEDS):
        seed=bs+10000*si
        rr,ss=b.run(METHOD,START,seed)
        rows+=rr;summ.append(ss);print("SUMMARY",json.dumps(ss),flush=True)
    inc=[s["incumbent_score"] for s in summ if s["incumbent_score"] is not None]
    grounds=[s for s in summ if s["ground_ever"]]
    src={}
    for s in grounds:src[s["ground_source"]]=src.get(s["ground_source"],0)+1
    agg={
      "method":METHOD,"start":START,"runs":len(summ),
      "hp_incumbent_runs":len(inc),
      "ground_ever":len(grounds),"ground_sources":src,
      "best_incumbent":max(inc,default=None),
      "median_incumbent":sorted(inc)[len(inc)//2] if inc else None,
      "final_population_hp":sum(s["final_population_hp"] for s in summ),
      "mean_final_population_hamming":sum(s["final_population_hamming"] for s in summ)/len(summ),
    }
    result={"aggregate":agg,"summaries":summ,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":
    main()
