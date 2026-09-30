"""Projected edge-only hybrid search for CHM13.

This deliberately removes rank/comparator/source/sink certificate variables from
the QUBO search.  The QUBO backend sees only real edge-selection bits x_e.

Outer classical projection:
- source/sink (for the 3789 semantics) are chosen optimally for the current
  selected edge set under the one-hot endpoint constraints;
- an integer latent rank is recomputed from the selected digraph using SCC
  decomposition plus a feedback-arc ordering heuristic inside nontrivial SCCs;
- comparator/borrow bits are not represented at all because for fixed integer
  ranks they are deterministic.

Inner QUBO:
- weighted edge objective;
- degree/path constraints (source_sink variant) or at-most-one + count
  constraint (count variant);
- a linear penalty on edges that point backward under the current classical
  rank.

The full algorithm is therefore hybrid block-coordinate search, NOT a pure SQA
or pure QUBO solve, even if the inner QUBO is solved by an annealer.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import dimod
import networkx as nx
from dwave.samplers import TabuSampler

from demo_chm13_edge_ordered_path_qubo import DATASET_DIR, load_chm13_graph

OUT=Path("debug/qubo/chm13_projected_edge_hybrid_20260926.json")
SEED=20260926


def load_problem():
    reads,edges,reward=load_chm13_graph(DATASET_DIR)
    rids=[r.rid for r in reads]
    ridx={r:i for i,r in enumerate(rids)}
    ep=sorted(reward,key=lambda e:(ridx[e[0]],ridx[e[1]]))
    shift=max(reward.values(),default=0.0)
    raw={e:shift-reward[e] for e in ep}
    norm=max(max(raw.values(),default=1.0),1.0)
    cost={e:raw[e]/norm for e in ep}
    incoming={r:[] for r in rids};outgoing={r:[] for r in rids}
    for e in ep:
        outgoing[e[0]].append(e);incoming[e[1]].append(e)
    return rids,ep,reward,cost,incoming,outgoing


def selected_degrees(rids,selected):
    indeg={r:0 for r in rids};outdeg={r:0 for r in rids}
    for u,v in selected:
        outdeg[u]+=1;indeg[v]+=1
    return indeg,outdeg


def best_endpoints(rids,selected):
    indeg,outdeg=selected_degrees(rids,selected)
    # With exactly one source and one sink, local deltas are 2*d-1.
    # For N>1 a Hamilton path needs distinct endpoints.  Choosing them
    # independently can select the same isolated vertex and create the
    # pathological "isolated vertex + cycle on the rest" basin.
    best=None
    for source in rids:
        ds=2*indeg[source]-1
        for sink in rids:
            if sink==source and len(rids)>1:
                continue
            score=ds+(2*outdeg[sink]-1)
            key=(score,source,sink)
            if best is None or key<best[0]:
                best=(key,source,sink)
    return best[1],best[2]


def feedback_order_for_scc(g,nodes):
    """Eades-style ordering; exact for simple cycles up to rotation."""
    sub=g.subgraph(nodes).copy()
    left=[];right=[]
    while sub.number_of_nodes():
        sources=[v for v in sub.nodes if sub.in_degree(v)==0]
        if sources:
            v=min(sources)
            left.append(v);sub.remove_node(v);continue
        sinks=[v for v in sub.nodes if sub.out_degree(v)==0]
        if sinks:
            v=min(sinks)
            right.append(v);sub.remove_node(v);continue
        v=max(sub.nodes,key=lambda q:(sub.out_degree(q)-sub.in_degree(q),str(q)))
        left.append(v);sub.remove_node(v)
    return left+list(reversed(right))


def project_rank(rids,selected):
    g=nx.DiGraph()
    g.add_nodes_from(rids);g.add_edges_from(selected)
    sccs=list(nx.strongly_connected_components(g))
    c=nx.condensation(g,sccs)
    rank_order=[]
    for ci in nx.topological_sort(c):
        members=list(c.nodes[ci]["members"])
        if len(members)==1:
            rank_order.extend(members)
        else:
            rank_order.extend(feedback_order_for_scc(g,members))
    return {r:i for i,r in enumerate(rank_order)}


def graph_metrics(rids,selected,ranks,reward):
    indeg,outdeg=selected_degrees(rids,selected)
    degree_conflicts=sum(max(0,d*(d-1)//2) for d in indeg.values())+sum(max(0,d*(d-1)//2) for d in outdeg.values())
    backward=sum(1 for u,v in selected if ranks[v]<=ranks[u])
    g=nx.DiGraph();g.add_nodes_from(rids);g.add_edges_from(selected)
    cycles=list(nx.simple_cycles(g))
    sources=[r for r in rids if indeg[r]==0 and outdeg[r]<=1]
    sinks=[r for r in rids if outdeg[r]==0 and indeg[r]<=1]
    valid=False;order=[];score=None
    if degree_conflicts==0 and len(selected)==len(rids)-1 and len(sources)==1 and len(sinks)==1 and nx.is_directed_acyclic_graph(g):
        cur=sources[0];seen=set()
        nxt={u:v for u,v in selected}
        while cur not in seen:
            seen.add(cur);order.append(cur)
            if cur==sinks[0]:break
            if cur not in nxt:break
            cur=nxt[cur]
        valid=len(order)==len(rids) and order[-1]==sinks[0]
        if valid:
            score=int(sum(reward[e] for e in zip(order,order[1:])))
    return {
        "selected_edges":len(selected),
        "degree_conflicts":degree_conflicts,
        "backward_selected_edges":backward,
        "cycle_count":len(cycles),
        "derived_sources":len(sources),
        "derived_sinks":len(sinks),
        "valid_path":valid,
        "path_score":score,
    }


def add_square(bqm,coeffs,constant,penalty):
    items=[(v,float(a)) for v,a in coeffs.items() if a]
    bqm.offset+=penalty*constant*constant
    for v,a in items:
        bqm.add_variable(v,penalty*(a*a+2*constant*a))
    for i,(u,a) in enumerate(items):
        for v,b in items[i+1:]:
            bqm.add_interaction(u,v,2*penalty*a*b)


def build_bqm_source_sink(ep,cost,incoming,outgoing,ranks,source,sink,degree_penalty,order_penalty):
    bqm=dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
    for e in ep:
        backward=1.0 if ranks[e[1]]<=ranks[e[0]] else 0.0
        bqm.add_variable(e,cost[e]+order_penalty*backward)
    for r in incoming:
        coeff={e:-1.0 for e in incoming[r]}
        constant=1.0-(1.0 if r==source else 0.0)
        add_square(bqm,coeff,constant,degree_penalty)
        coeff={e:-1.0 for e in outgoing[r]}
        constant=1.0-(1.0 if r==sink else 0.0)
        add_square(bqm,coeff,constant,degree_penalty)
    return bqm


def build_bqm_count(ep,cost,incoming,outgoing,ranks,n,degree_conflict_penalty,count_penalty,order_penalty):
    bqm=dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
    for e in ep:
        backward=1.0 if ranks[e[1]]<=ranks[e[0]] else 0.0
        bqm.add_variable(e,cost[e]+order_penalty*backward)
    for r in incoming:
        for group in (incoming[r],outgoing[r]):
            for i,e1 in enumerate(group):
                for e2 in group[i+1:]:
                    bqm.add_interaction(e1,e2,degree_conflict_penalty)
    add_square(bqm,{e:1.0 for e in ep},-(n-1),count_penalty)
    return bqm


def sample_edges(bqm,init,timeout_ms,seed):
    sampler=TabuSampler()
    kwargs={
        "num_reads":1,
        "timeout":timeout_ms,
        "seed":seed,
        "coefficient_z_first":500,
        "coefficient_z_restart":125,
        "lower_bound_z":100000,
        "num_restarts":30,
    }
    if init is not None:
        kwargs["initial_states"]=[{e:int(e in init) for e in bqm.variables}]
        kwargs["initial_states_generator"]="none"
    ss=sampler.sample(bqm,**kwargs)
    s=ss.first.sample
    return {e for e in bqm.variables if int(s[e])}


def run_variant(variant,outer_iters,seed):
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    rng=random.Random(seed)

    # Stage 0: edge topology without any rank penalty.
    ranks={r:i for i,r in enumerate(rng.sample(rids,len(rids)))}
    selected=set()
    source,sink=best_endpoints(rids,selected)

    rows=[]
    order_schedule=[0.0,0.25,0.5,1.0,2.0,4.0,8.0,16.0,32.0]
    for it in range(outer_iters):
        op=order_schedule[min(it,len(order_schedule)-1)]
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

        t0=time.perf_counter()
        selected=sample_edges(bqm,selected if selected else None,1200,seed+it)
        sec=time.perf_counter()-t0
        ranks=project_rank(rids,selected)
        metrics=graph_metrics(rids,selected,ranks,reward)
        row={
            "variant":variant,
            "iteration":it,
            "order_penalty":op,
            "seconds":sec,
            "source":source if variant=="source_sink" else None,
            "sink":sink if variant=="source_sink" else None,
            **metrics,
        }
        rows.append(row);print(json.dumps(row),flush=True)
        if metrics["valid_path"]:
            break
    return rows


def main():
    all_rows=[]
    for vi,variant in enumerate(("source_sink","count")):
        print("VARIANT",variant,flush=True)
        all_rows.extend(run_variant(variant,20,SEED+1000*vi))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"solver":"edge-only Tabu + classical rank projection","rows":all_rows},indent=2)+"\n")
    print("\nFINAL")
    for variant in ("source_sink","count"):
        vr=[r for r in all_rows if r["variant"]==variant]
        print(json.dumps(vr[-1]),flush=True)


if __name__=="__main__":
    main()
