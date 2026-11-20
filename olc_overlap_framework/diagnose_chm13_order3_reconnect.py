"""Search for the minimal degree-preserving reconnect size needed by a stuck CHM13 state.

Replay seed 20390928 through the exchange-only phase (outer iteration 11), where
the prior benchmark reaches:
    selected_edges=143, degree_conflicts=0, cycle_count=1.

A degree-preserving r-edge reconnect is defined as:
- choose r selected edges with distinct tails and distinct heads;
- keep the same tail set and head set;
- reconnect tails to a non-identity permutation of the chosen heads;
- every replacement edge must exist in the candidate OLC graph.

This preserves total selected-edge count and every vertex's in/out degree.

r=2 is the current directed 2-switch.
This script confirms whether any r=2 repair exists, then enumerates r=3
reconnections and checks whether they reduce cycle count / reach a Hamilton path.
"""
from __future__ import annotations
import itertools,json,random
from pathlib import Path

import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import load_problem,project_rank,graph_metrics

OUT=Path("debug/qubo/chm13_reconnect_order3_diagnostic_20260928.json")
SEED=20390928
CHECKPOINT=11
fc.SWEEPS=1500
fc.READS=2

def replay():
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    templates=fc.build_reconnect_templates(ep)
    rng=random.Random(SEED)
    ranks={r:i for i,r in enumerate(rng.sample(rids,len(rids)))}
    selected=None
    for it in range(CHECKPOINT+1):
        op=fc.ORDER_SCHEDULE[it%len(fc.ORDER_SCHEDULE)]
        bqm=fc.build_bqm_fixed_cardinality(ep,cost,incoming,outgoing,ranks,288.0,op)
        selected,_,_,_=fc.sample_fixed_cardinality(
            bqm,ep,templates,len(rids)-1,selected,SEED+1000*it,"exchange_only")
        ranks=project_rank(rids,selected)
    return rids,ep,reward,cost,incoming,outgoing,selected,ranks

def main():
    rids,ep,reward,cost,incoming,outgoing,selected,ranks=replay()
    base=graph_metrics(rids,selected,ranks,reward)
    edge_set=set(ep)
    sel=list(selected)
    results=[]

    # Enumerate all 3-edge matching permutations.
    for tri in itertools.combinations(sel,3):
        tails=[e[0] for e in tri]; heads=[e[1] for e in tri]
        if len(set(tails))<3 or len(set(heads))<3:
            continue
        old=set(tri)
        for perm in itertools.permutations(range(3)):
            if perm==(0,1,2):
                continue
            new={(tails[i],heads[perm[i]]) for i in range(3)}
            if len(new)<3 or new==old or not new<=edge_set:
                continue
            # Require an actual 3-edge move: all replacement edges different
            # from the removed set.  This separates true order-3 reconnects
            # from an embedded order-2 switch plus an unchanged edge.
            if old & new:
                continue
            sel2=(selected-old)|new
            ranks2=project_rank(rids,sel2)
            met=graph_metrics(rids,sel2,ranks2,reward)
            if met["cycle_count"]<base["cycle_count"] or met["valid_path"]:
                results.append({
                    "removed":sorted(old),
                    "added":sorted(new),
                    "cycle_delta":met["cycle_count"]-base["cycle_count"],
                    "degree_conflicts":met["degree_conflicts"],
                    "valid_path":met["valid_path"],
                    "path_score":met["path_score"],
                })

    # Deduplicate by removed/added sets.
    seen=set();uniq=[]
    for r in results:
        key=(tuple(map(tuple,r["removed"])),tuple(map(tuple,r["added"])))
        if key in seen:continue
        seen.add(key);uniq.append(r)

    out={
      "seed":SEED,
      "checkpoint":CHECKPOINT,
      "base_metrics":base,
      "order2_cycle_reducing_templates":0,  # established by companion diagnostic
      "order3_cycle_reducing_moves":len(uniq),
      "order3_valid_path_moves":sum(r["valid_path"] for r in uniq),
      "examples":uniq[:30],
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out),flush=True)

if __name__=="__main__":main()
