"""Classical large-neighborhood repair of the best projected-SQA path.

No reference information is used to choose moves.  Starting from the replayed
best projected-SQA Hamilton path, enumerate feasible contiguous-segment
relocations that preserve a Hamilton path and accept the best positive
edge-weight gain.  Repeat to local optimum.

A relocation removes one contiguous segment from the current path and reinserts
it elsewhere without reversing the segment.  Only boundary adjacencies change.
"""
from __future__ import annotations
import json,random
from pathlib import Path

from benchmark_chm13_projected_edge_sqa import sample_edges_sqa
from benchmark_chm13_projected_edge_hybrid import (
    load_problem,project_rank,graph_metrics,build_bqm_count,
)
from benchmark_chm13_projected_sqa_path_diagnostic import derive_order
from demo_chm13_edge_ordered_path_qubo import DATASET_DIR

BASE_SEED=20260926
RUN=7
STOP_ITER=14
SCHEDULE=[0.0,0.25,0.5,1.0,2.0,4.0]
OUT=Path("debug/qubo/chm13_projected_sqa_segment_repair_20260926.json")


def path_score(order,reward):
    return sum(reward[(u,v)] for u,v in zip(order,order[1:]))

def relocation(order,i,j,insert_pos):
    # segment order[i:j+1], insert into remainder before remainder[insert_pos]
    seg=order[i:j+1]
    rem=order[:i]+order[j+1:]
    return rem[:insert_pos]+seg+rem[insert_pos:]

def feasible_order(order,edge_set):
    return all((u,v) in edge_set for u,v in zip(order,order[1:]))

def best_segment_relocation(order,reward,max_len):
    edge_set=set(reward)
    base=path_score(order,reward)
    best=None
    n=len(order)
    for i in range(n):
        for j in range(i,min(n,i+max_len)):
            seglen=j-i+1
            remn=n-seglen
            for k in range(remn+1):
                # Skip insertion reconstructing same order.
                orig_k=i
                if k==orig_k:
                    continue
                cand=relocation(order,i,j,k)
                if not feasible_order(cand,edge_set):
                    continue
                sc=path_score(cand,reward)
                gain=sc-base
                if gain>0 and (best is None or gain>best["gain"]):
                    best={
                        "i":i,"j":j,"length":seglen,"insert_pos":k,
                        "gain":int(gain),"score":int(sc),
                        "segment":order[i:j+1],
                        "candidate":cand,
                    }
    return best


def replay():
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    rng=random.Random(BASE_SEED+10000*RUN)
    ranks={r:i for i,r in enumerate(rng.sample(rids,len(rids)))}
    selected=set()
    for it in range(STOP_ITER+1):
        op=SCHEDULE[it%len(SCHEDULE)]
        bqm=build_bqm_count(
            ep,cost,incoming,outgoing,ranks,len(rids),
            degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=op
        )
        selected,_,_=sample_edges_sqa(
            bqm,selected if selected else None,BASE_SEED+10000*RUN+it
        )
        ranks=project_rank(rids,selected)
    order=derive_order(rids,selected)
    return rids,reward,order


def main():
    rids,reward,order=replay()
    initial=list(order)
    trace=[]
    for step in range(20):
        move=best_segment_relocation(order,reward,max_len=16)
        if move is None:
            break
        before=path_score(order,reward)
        order=move.pop("candidate")
        trace.append({"step":step,"before":int(before),**move})
        print("MOVE",json.dumps(trace[-1]),flush=True)
    final=path_score(order,reward)
    ref=json.loads((DATASET_DIR/"reference_path.json").read_text())["normalized_nodes"]
    pref={r:i for i,r in enumerate(ref)}
    pfin={r:i for i,r in enumerate(order)}
    result={
        "initial_score":int(path_score(initial,reward)),
        "final_score":int(final),
        "certified_optimum":1343093,
        "reached_certified_optimum":int(final)==1343093,
        "moves":trace,
        "final_order":order,
        "evaluation_only_reference_max_abs_displacement":max(abs(pfin[r]-pref[r]) for r in rids),
        "evaluation_only_exact_reference_order":order==ref,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("RESULT",json.dumps(result),flush=True)


if __name__=="__main__":main()
