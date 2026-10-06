"""Projected edge-only hybrid search with OpenJij SQA as the inner QUBO solver.

The outer loop is classical:
- project a selected edge set to integer latent ranks;
- optionally derive source/sink;
- rebuild an edge-only QUBO under those fixed classical quantities.

Only the M real edge bits are sampled by SQA.  Therefore the overall algorithm
is a classical-quantum-inspired hybrid / alternating optimization algorithm,
not a pure full-Hamiltonian SQA solve.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import openjij as oj

from benchmark_chm13_projected_edge_hybrid import (
    SEED,
    load_problem,
    best_endpoints,
    project_rank,
    graph_metrics,
    build_bqm_source_sink,
    build_bqm_count,
)

OUT=Path("debug/qubo/chm13_projected_edge_sqa_20260926.json")


def sample_edges_sqa(bqm,init,seed):
    vars_=list(bqm.variables)
    idx={v:i for i,v in enumerate(vars_)}
    qubo={}
    for v,bias in bqm.linear.items():
        if bias:
            i=idx[v]
            qubo[(i,i)]=qubo.get((i,i),0.0)+float(bias)
    for (u,v),bias in bqm.quadratic.items():
        if bias:
            i,j=idx[u],idx[v]
            if i>j:i,j=j,i
            qubo[(i,j)]=qubo.get((i,j),0.0)+float(bias)

    sampler=oj.SQASampler()
    kwargs={
        "num_reads":8,
        "num_sweeps":1200,
        "trotter":8,
        "seed":seed,
    }
    if init is not None:
        kwargs["initial_state"]={idx[v]:int(v in init) for v in vars_}
    t0=time.perf_counter()
    resp=sampler.sample_qubo(qubo,**kwargs)
    sec=time.perf_counter()-t0
    first=resp.first
    selected={vars_[i] for i,val in first.sample.items() if int(val)}
    return selected,sec,float(first.energy+bqm.offset)


def run_variant(variant,outer_iters,seed):
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    rng=random.Random(seed)
    ranks={r:i for i,r in enumerate(rng.sample(rids,len(rids)))}
    selected=set()
    source,sink=best_endpoints(rids,selected)
    rows=[]
    schedule=[0.0,0.25,0.5,1.0,2.0,4.0,8.0,16.0,32.0]

    for it in range(outer_iters):
        op=schedule[min(it,len(schedule)-1)]
        if variant=="source_sink":
            if it>0:
                source,sink=best_endpoints(rids,selected)
            bqm=build_bqm_source_sink(
                ep,cost,incoming,outgoing,ranks,source,sink,
                degree_penalty=144.0,
                order_penalty=op,
            )
        elif variant=="count":
            bqm=build_bqm_count(
                ep,cost,incoming,outgoing,ranks,len(rids),
                degree_conflict_penalty=288.0,
                count_penalty=32.0,
                order_penalty=op,
            )
        else:
            raise ValueError(variant)

        selected,sec,energy=sample_edges_sqa(
            bqm,selected if selected else None,seed+it
        )
        ranks=project_rank(rids,selected)
        metrics=graph_metrics(rids,selected,ranks,reward)
        row={
            "backend":"openjij-sqa",
            "variant":variant,
            "iteration":it,
            "order_penalty":op,
            "seconds":sec,
            "inner_qubo_energy":energy,
            "source":source if variant=="source_sink" else None,
            "sink":sink if variant=="source_sink" else None,
            **metrics,
        }
        rows.append(row)
        print(json.dumps(row),flush=True)
        if metrics["valid_path"]:
            break
    return rows


def main():
    rows=[]
    for vi,variant in enumerate(("source_sink","count")):
        print("VARIANT",variant,flush=True)
        rows.extend(run_variant(variant,12,SEED+1000*vi))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"rows":rows},indent=2)+"\n")
    print("\nFINAL")
    for variant in ("source_sink","count"):
        vr=[r for r in rows if r["variant"]==variant]
        print(json.dumps(vr[-1]),flush=True)


if __name__=="__main__":
    main()
