"""Hybrid rank-expression screening with vanilla OpenJij SQA inner solver.

Goal
----
Preserve the exploratory strength of the no-rank Hamiltonian while using rank
only as a weak topology preference.

No custom MC updater is used. Every inner solve is OpenJij SQASampler with the
built-in SingleSpinFlip updater.

Compared expressions
--------------------
none:
    no rank penalty.

gated_rank:
    a single projected rank with weak backward-edge penalty A=0.02, activated
    only when the current state is degree-correct, has N-1 edges, and is cyclic.

rank_dropout:
    same gated rank, but only half of the standard SQA reads receive the rank
    penalty; half remain completely rank-free. This is a mixture of exploration
    and rank-guided exploitation, not an averaged Hamiltonian.

rank_mixture:
    when gated, build four alternative rank hypotheses by rotating each
    nontrivial SCC ordering. Each hypothesis receives one standard SQA read.
    The candidate pool is therefore a mixture over rank hypotheses rather than
    an average of their backward-edge penalties.

The total read budget is 4 reads per outer iteration for every method in this
screening. Fixed 12 outer iterations. No ground-score early stop.

Candidate propagation and incumbent
------------------------------------
All standard read outputs are pooled. The next warm-start state is selected
without any knowledge of the certified optimum:
  1) prefer a valid Hamilton path; among HPs maximize path score;
  2) otherwise minimize degree conflicts;
  3) then cardinality error |E-(N-1)|;
  4) then cycle count;
  5) then the rank-free base BQM energy.

The best valid HP ever observed is archived as a non-oracle incumbent and is
never fed back into the sampler unless it is also the chosen next state.
"""
from __future__ import annotations

import json, random, time
from pathlib import Path
import networkx as nx
import openjij as oj

from benchmark_chm13_projected_edge_hybrid import (
    load_problem, graph_metrics, project_rank, build_bqm_count,
    feedback_order_for_scc,
)

OUT=Path("debug/qubo/chm13_rank_expression_screening_20260930.json")
GROUND=1_343_093
OUTER=12
SWEEPS=1400
TOTAL_READS=4
A_RANK=0.02
SEEDS=(202609401,202609402,202609403,202609404)
STARTS=("zero","random")
METHODS=("none","gated_rank","rank_dropout","rank_mixture")


def base_bqm(rids,ep,cost,incoming,outgoing):
    dummy={r:0 for r in rids}
    return build_bqm_count(
      ep,cost,incoming,outgoing,dummy,len(rids),
      degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)


def rank_bqm(rids,ep,cost,incoming,outgoing,ranks,A):
    return build_bqm_count(
      ep,cost,incoming,outgoing,ranks,len(rids),
      degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=A)


def make_qubo(bqm):
    vs=list(bqm.variables);idx={v:i for i,v in enumerate(vs)};q={}
    for v,b in bqm.linear.items():
        if b:q[(idx[v],idx[v])]=q.get((idx[v],idx[v]),0.0)+float(b)
    for (u,v),b in bqm.quadratic.items():
        if not b:continue
        i,j=idx[u],idx[v]
        if i>j:i,j=j,i
        q[(i,j)]=q.get((i,j),0.0)+float(b)
    return vs,idx,q


def initial_state(vs,idx,ep,selected,start,rng,N):
    sm={v:0 for v in vs}
    if selected is not None:
        for e in ep:sm[e]=int(e in selected)
    elif start=="random":
        p=(N-1)/len(ep)
        for e in ep:sm[e]=int(rng.random()<p)
    return {idx[v]:int(sm[v]) for v in vs}


def alt_ranks(rids,selected,K,rng):
    """Generate K plausible orders by rotating the deterministic SCC order.

    The condensation DAG order is preserved. Inside each nontrivial SCC, the
    Eades order is cyclically shifted. For a simple directed cycle this exactly
    changes which cycle edge is designated backward without changing the cycle
    itself.
    """
    g=nx.DiGraph();g.add_nodes_from(rids);g.add_edges_from(selected or [])
    sccs=list(nx.strongly_connected_components(g))
    c=nx.condensation(g,sccs)
    comps=[]
    for ci in nx.topological_sort(c):
        members=list(c.nodes[ci]["members"])
        if len(members)<=1:
            comps.append(members)
        else:
            comps.append(list(feedback_order_for_scc(g,members)))

    out=[]
    for k in range(K):
        order=[]
        for members in comps:
            if len(members)<=1:
                order.extend(members)
                continue
            # deterministic coverage plus seed-dependent phase offset
            off=(k + rng.randrange(len(members))) % len(members)
            order.extend(members[off:]+members[:off])
        out.append({v:i for i,v in enumerate(order)})
    return out


def decode_reads(resp,vs,ep,rids,reward,base):
    labels=list(resp.variables)
    ans=[]
    for arr,en in zip(resp.record.sample,resp.record.energy):
        sm={vs[int(lab)]:int(arr[col]) for col,lab in enumerate(labels)}
        sel={e for e in ep if sm.get(e,0)}
        met=graph_metrics(rids,sel,project_rank(rids,sel),reward)
        # Evaluate rank-independent base energy.
        sample={e:int(e in sel) for e in ep}
        e0=float(base.energy(sample))
        ans.append({"selected":sel,"metrics":met,"base_energy":e0})
    return ans


def sample_branch(bqm,base,ep,rids,reward,selected,start,rng,seed,reads):
    vs,idx,q=make_qubo(bqm)
    ini=initial_state(vs,idx,ep,selected,start,rng,len(rids))
    t=time.perf_counter()
    resp=oj.SQASampler().sample_qubo(
      q,beta=5.0,gamma=1.0,trotter=8,
      num_reads=reads,num_sweeps=SWEEPS,seed=seed,
      initial_state=ini,updater="single spin flip")
    sec=time.perf_counter()-t
    return decode_reads(resp,vs,ep,rids,reward,base),sec


def merit(c,N):
    m=c["metrics"]
    if m["valid_path"]:
        return (0, -int(m["path_score"]), 0, 0, c["base_energy"])
    return (
      1,
      int(m["degree_conflicts"]),
      abs(int(m["selected_edges"])-(N-1)),
      int(m["cycle_count"]),
      c["base_energy"],
    )


def run(method,start,seed):
    rids,ep,reward,cost,incoming,outgoing=load_problem();N=len(rids)
    rng=random.Random(seed)
    base=base_bqm(rids,ep,cost,incoming,outgoing)
    selected=None;rows=[];incumbent=None;incumbent_it=None

    for it in range(OUTER):
        prev=graph_metrics(rids,selected,project_rank(rids,selected),reward) if selected is not None else None
        gated=bool(
          selected is not None
          and prev["selected_edges"]==N-1
          and prev["degree_conflicts"]==0
          and prev["cycle_count"]>0
        )

        candidates=[];sec=0.0;branch_desc=[]

        if method=="none" or not gated:
            cc,dt=sample_branch(
              base,base,ep,rids,reward,selected,start,rng,
              seed+10000*it,TOTAL_READS)
            candidates+=cc;sec+=dt;branch_desc.append("none:4")

        elif method=="gated_rank":
            ranks=project_rank(rids,selected)
            rb=rank_bqm(rids,ep,cost,incoming,outgoing,ranks,A_RANK)
            cc,dt=sample_branch(
              rb,base,ep,rids,reward,selected,start,rng,
              seed+10000*it,TOTAL_READS)
            candidates+=cc;sec+=dt;branch_desc.append("rank:4")

        elif method=="rank_dropout":
            # Half no-rank, half one projected-rank branch.
            cc,dt=sample_branch(
              base,base,ep,rids,reward,selected,start,rng,
              seed+10000*it,2)
            candidates+=cc;sec+=dt;branch_desc.append("none:2")
            ranks=project_rank(rids,selected)
            rb=rank_bqm(rids,ep,cost,incoming,outgoing,ranks,A_RANK)
            cc,dt=sample_branch(
              rb,base,ep,rids,reward,selected,start,rng,
              seed+10000*it+5000,2)
            candidates+=cc;sec+=dt;branch_desc.append("rank:2")

        elif method=="rank_mixture":
            ranks_list=alt_ranks(rids,selected,4,rng)
            for k,ranks in enumerate(ranks_list):
                rb=rank_bqm(rids,ep,cost,incoming,outgoing,ranks,A_RANK)
                cc,dt=sample_branch(
                  rb,base,ep,rids,reward,selected,start,rng,
                  seed+10000*it+1000*k,1)
                candidates+=cc;sec+=dt;branch_desc.append(f"r{k}:1")
        else:
            raise ValueError(method)

        # Non-oracle propagation rule shared by all methods.
        chosen=min(candidates,key=lambda c:merit(c,N))
        selected=set(chosen["selected"])
        met=chosen["metrics"]

        hp_scores=[c["metrics"]["path_score"] for c in candidates if c["metrics"]["valid_path"]]
        if hp_scores:
            best=max(hp_scores)
            if incumbent is None or best>incumbent:
                incumbent=best;incumbent_it=it

        row={
          "method":method,"start":start,"seed":seed,"iteration":it,
          "rank_gated":gated,"branches":branch_desc,
          "seconds":sec,"candidate_count":len(candidates),
          "candidate_hp_count":len(hp_scores),
          "candidate_best_hp":max(hp_scores,default=None),
          "incumbent_score":incumbent,"incumbent_iteration":incumbent_it,
          **met,
        }
        rows.append(row);print("ROW",json.dumps(row),flush=True)

    return rows,{
      "method":method,"start":start,"seed":seed,
      "incumbent_score":incumbent,
      "ground_ever":incumbent==GROUND,
      "final_valid":rows[-1]["valid_path"],
      "final_score":rows[-1]["path_score"],
      "final_ground":rows[-1]["path_score"]==GROUND,
      "rank_gated_iterations":sum(r["rank_gated"] for r in rows),
      "mean_seconds":sum(r["seconds"] for r in rows)/len(rows),
    }


def main():
    rows=[];summ=[]
    for method in METHODS:
      for start in STARTS:
        for si,base_seed in enumerate(SEEDS):
          seed=base_seed+10000*si
          rr,ss=run(method,start,seed)
          rows+=rr;summ.append(ss);print("SUMMARY",json.dumps(ss),flush=True)

    agg={}
    for method in METHODS:
      for start in STARTS:
        aa=[s for s in summ if s["method"]==method and s["start"]==start]
        inc=[s["incumbent_score"] for s in aa if s["incumbent_score"] is not None]
        agg[f"{method}|{start}"]={
          "runs":len(aa),
          "hp_incumbent_runs":len(inc),
          "ground_ever":sum(s["ground_ever"] for s in aa),
          "best_incumbent":max(inc,default=None),
          "median_incumbent":sorted(inc)[len(inc)//2] if inc else None,
          "final_hp":sum(s["final_valid"] for s in aa),
          "final_ground":sum(s["final_ground"] for s in aa),
          "mean_rank_gated_iterations":sum(s["rank_gated_iterations"] for s in aa)/len(aa),
          "mean_seconds_per_outer":sum(s["mean_seconds"] for s in aa)/len(aa),
        }
    result={"A_rank":A_RANK,"aggregate":agg,"summaries":summ,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)


if __name__=="__main__":
    main()
