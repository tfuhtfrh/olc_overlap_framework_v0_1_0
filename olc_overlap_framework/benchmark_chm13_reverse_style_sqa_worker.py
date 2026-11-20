"""Parallel worker for reverse-style runner-up SQA diagnostic."""
from __future__ import annotations
import json, os
from pathlib import Path
import benchmark_chm13_reverse_style_sqa_runnerup as b

CFG=os.environ["CONFIG"]
OUT=Path(f"debug/qubo/chm13_reverse_style_worker_{CFG}_20260930.json")

def main():
    ctx=b.load_problem()
    rids,ep,reward,*_=ctx
    runner=b.find_runner(reward,set(ep))
    rows=[]
    for seed in b.SEEDS:
        r=b.run_one(CFG,seed,ctx,runner)
        rows.append(r)
        print("ROW",json.dumps(r),flush=True)
    hp=[r["best_hp"] for r in rows if r["best_hp"] is not None]
    agg={
      "config":CFG,
      "runs":len(rows),
      "hp_runs":sum(r["any_hp"] for r in rows),
      "ground_runs":sum(r["any_ground"] for r in rows),
      "runner_return_runs":sum(r["runner_returned"] for r in rows),
      "best_hp":max(hp,default=None),
      "median_best_hp":sorted(hp)[len(hp)//2] if hp else None,
      "mean_cycles":sum(r["mean_cycles"] for r in rows)/len(rows),
      "mean_degree_conflicts":sum(r["mean_degree_conflicts"] for r in rows)/len(rows),
      "mean_selected_edges":sum(r["mean_selected_edges"] for r in rows)/len(rows),
      "mean_seconds":sum(r["seconds"] for r in rows)/len(rows),
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"aggregate":agg,"rows":rows},indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":
    main()
