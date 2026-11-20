"""Parallel worker for reverse-SQA runner-up benchmark."""
from __future__ import annotations
import json, os
from pathlib import Path
import benchmark_chm13_reverse_sqa_runnerup as b
from benchmark_chm13_projected_edge_hybrid import build_bqm_count

CONFIG=os.environ["CONFIG"]
OUT=Path(f"debug/qubo/chm13_reverse_worker_{CONFIG}_20260930.json")

def main():
    rids,ep,reward,cost,incoming,outgoing=b.b.load_problem()
    dummy={r:0 for r in rids}
    bqm=build_bqm_count(
      ep,cost,incoming,outgoing,dummy,len(rids),
      degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)
    runner=b.runner_state(reward,set(ep))
    ctx=(rids,ep,reward,bqm,runner)

    rows=[]
    for seed in b.SEEDS:
        rr=b.run(CONFIG,seed,ctx)
        rows.append(rr);print("ROW",json.dumps(rr),flush=True)

    scores=[r["best_read_hp"] for r in rows if r["best_read_hp"] is not None]
    agg={
      "config":CONFIG,"runs":len(rows),
      "primary_hp":sum(r["primary_valid"] for r in rows),
      "primary_ground":sum(r["primary_ground"] for r in rows),
      "any_read_hp":sum(r["any_read_hp"] for r in rows),
      "any_read_ground":sum(r["any_read_ground"] for r in rows),
      "best_hp":max(scores,default=None),
      "median_best_hp":sorted(scores)[len(scores)//2] if scores else None,
      "mean_cycles":sum(r["mean_cycles"] for r in rows)/len(rows),
      "mean_degree_conflicts":sum(r["mean_degree_conflicts"] for r in rows)/len(rows),
    }
    result={"aggregate":agg,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":
    main()
