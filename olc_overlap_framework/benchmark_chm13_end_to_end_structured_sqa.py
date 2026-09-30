"""End-to-end CHM13 SQA-assisted ground-state benchmark from zero/random starts.

Goal: certified ground score 1,343,093 without target-edge information.

Global phase
------------
OpenJij P=8 SQA on the full 261-edge binary space with
    H_weight + H_degree + H_count
plus one of two *current-state, weak, scale-adaptive* acyclicity biases:

1) exact_cut:
   exact current-cycle inequality sum_C x_e <= |C|-1 using bounded slack.
   Penalty A is calibrated from the cheapest locally available cycle-reducing
   structured move rather than an arbitrary O(1..10) schedule.

2) rank_ensemble:
   for a simple current cycle of length L, average all equivalent cyclic rank
   rotations, giving (A/L) sum_{e in C} x_e.
   A is calibrated so that removing one cycle edge receives approximately the
   same energy pressure as exact_cut.

The acyclicity term is gated: it is used only when the previous state has
N-1 edges, zero degree conflicts, and at least one cycle.

Feasible refinement
-------------------
Once any Hamilton path is found, keep the rank-free static BQM and apply
symmetric whole-worldline structured proposals restricted to the Hamilton-path
manifold:
- genuine R3 reconnect
- simultaneous disjoint R2-pair reconnect

The exact audit established that the certified ground is the unique local
maximum under these structured feasible moves for all 192 exact Hamilton paths.

This is SQA-assisted hybrid optimization: the OpenJij global phase is SQA; the
whole-worldline structured kernel is a valid nonlocal path-integral MC update
for its static rank-free BQM, but should not be called physical tunneling.
"""
from __future__ import annotations

import json, math, random, time
from pathlib import Path
import networkx as nx
import openjij as oj

import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import (
    load_problem, project_rank, graph_metrics, build_bqm_count, add_square,
)
from diagnose_chm13_move_basis_proposal import build_r3_templates, applicable

OUT=Path("debug/qubo/chm13_end_to_end_structured_sqa_20260929.json")
GROUND=1_343_093
OUTER=20
SQA_READS=8
SQA_SWEEPS=1400
TROTTER=8
BETA_REFINE=4.0
REFINE_ATTEMPTS=12000
SEEDS=(202609291,202609292,202609293)
STARTS=("zero","random")
METHODS=("exact_cut","rank_ensemble")


def current_cycles(rids,selected):
    if not selected:return []
    g=nx.DiGraph();g.add_nodes_from(rids);g.add_edges_from(selected)
    ans=[]
    for comp in nx.strongly_connected_components(g):
        if len(comp)<=1:continue
        sub=g.subgraph(comp).copy()
        try:cyc=nx.find_cycle(sub,orientation="original")
        except nx.NetworkXNoCycle:continue
        ans.append(tuple(sorted((u,v) for u,v,*_ in cyc)))
    return sorted(set(ans))


def bounded_weights(u):
    ws=[];rem=int(u);p=1
    while rem>0:
        w=min(p,rem);ws.append(w);rem-=w;p*=2
    return ws


def encode_value(weights,value):
    value=int(value);bits=[0]*len(weights)
    for i in range(len(weights)-1,-1,-1):
        if weights[i]<=value:
            bits[i]=1;value-=weights[i]
    if value:
        from itertools import product
        for cand in product((0,1),repeat=len(weights)):
            if sum(w*b for w,b in zip(weights,cand))==value:return list(cand)
        raise ValueError((weights,value))
    return bits


def add_exact_cuts(bqm,cycles,A):
    meta=[]
    for ci,cyc in enumerate(cycles):
        L=len(cyc); ws=bounded_weights(L-1)
        coeff={e:1.0 for e in cyc}; labels=[]
        for bi,w in enumerate(ws):
            lab=("cut_slack",ci,bi);labels.append(lab);coeff[lab]=float(w)
        add_square(bqm,coeff,-float(L-1),float(A))
        meta.append({"cycle":cyc,"labels":labels,"weights":ws})
    return meta


def add_rank_ensemble(bqm,cycles,A_total):
    for cyc in cycles:
        if not cyc:continue
        w=float(A_total)/len(cyc)
        for e in cyc:bqm.add_linear(e,w)
    return []


def warm_sample(bqm,ep,selected,meta,start_mode,rng,N):
    sm={v:0 for v in bqm.variables}
    if selected is not None:
        for e in ep:sm[e]=int(e in selected)
    elif start_mode=="random":
        # Random full-space start with expected cardinality N-1.
        p=(N-1)/len(ep)
        for e in ep:sm[e]=int(rng.random()<p)
    # zero mode is already all-zero
    if selected is not None:
        for item in meta:
            chosen=sum(int(e in selected) for e in item["cycle"])
            val=max(0,len(item["cycle"])-1-chosen)
            bits=encode_value(item["weights"],val)
            for lab,b in zip(item["labels"],bits):sm[lab]=b
    return sm


def sample_openjij(bqm,ep,selected,meta,start_mode,rng,seed,N):
    vs=list(bqm.variables);idx={v:i for i,v in enumerate(vs)};q={}
    for v,b in bqm.linear.items():
        if b:q[(idx[v],idx[v])]=q.get((idx[v],idx[v]),0.0)+float(b)
    for (u,v),b in bqm.quadratic.items():
        if not b:continue
        i,j=idx[u],idx[v]
        if i>j:i,j=j,i
        q[(i,j)]=q.get((i,j),0.0)+float(b)
    ini=warm_sample(bqm,ep,selected,meta,start_mode,rng,N)
    kw={
      "num_reads":SQA_READS,"num_sweeps":SQA_SWEEPS,
      "trotter":TROTTER,"seed":seed,
      "initial_state":{idx[v]:int(ini[v]) for v in vs},
    }
    t=time.perf_counter()
    resp=oj.SQASampler().sample_qubo(q,**kw)
    sec=time.perf_counter()-t
    sm={vs[i]:int(x) for i,x in resp.first.sample.items()}
    sel={e for e in ep if sm.get(e,0)}
    return sel,sec,len(vs),len(bqm.quadratic)


def state_metrics(rids,selected,reward):
    ranks=project_rank(rids,selected)
    return graph_metrics(rids,selected,ranks,reward)


def array_rankfree(ep,cost,incoming,outgoing):
    dummy={u:0 for e in ep for u in e}
    bqm=fc.build_bqm_fixed_cardinality(
      ep,cost,incoming,outgoing,dummy,
      degree_conflict_penalty=288.0,order_penalty=0.0)
    return fc.ArrayBQM.from_bqm(bqm,ep)


def local_repair_scale(rids,ep,reward,selected,abqm,r2,r3):
    """Estimate the smallest classical barrier to reduce current cycle count.

    No target path is used. Search only locally available R2/R3 moves that
    preserve fixed cardinality and do not introduce degree conflicts.
    """
    base=state_metrics(rids,selected,reward)
    if base["degree_conflicts"]!=0 or base["cycle_count"]<=0:
        return 0.02
    idx={e:i for i,e in enumerate(ep)}
    st=frozenset(idx[e] for e in selected)
    x=[1 if i in st else 0 for i in range(len(ep))]
    deltas=[]
    for temps in (r2,r3):
        for tpl in temps:
            nb=applicable(st,tpl)
            if nb is None:continue
            ns={ep[i] for i in nb}
            met=state_metrics(rids,ns,reward)
            if met["degree_conflicts"]==0 and met["cycle_count"]<base["cycle_count"]:
                flips=tuple(sorted(st.symmetric_difference(nb)))
                deltas.append(abqm.delta_flipset(x,flips))
    if not deltas:
        return 0.05
    barrier=max(0.0,min(deltas))
    # Small margin, then clip to avoid landscape domination.
    return min(0.5,max(0.02,1.25*barrier+0.005))


def apply_r2pair(st,t1,t2):
    f1=set(t1[0]+t1[1]);f2=set(t2[0]+t2[1])
    if f1&f2:return None
    n1=applicable(st,t1);n2=applicable(st,t2)
    if n1 is None or n2 is None:return None
    return frozenset(set(st).symmetric_difference(f1|f2))


def feasible_worldline_refine(rids,ep,reward,selected,abqm,r2,r3,seed):
    """Static symmetric whole-worldline structured MC on HP manifold."""
    idx={e:i for i,e in enumerate(ep)}
    st=frozenset(idx[e] for e in selected)
    x=[1 if i in st else 0 for i in range(len(ep))]
    rng=random.Random(seed)

    # Static disjoint R2-pair proposal list. The proposal itself is an
    # involution; invalid/non-HP applications are null proposals.
    pairs=[]
    for a in range(len(r2)):
        fa=set(r2[a][0]+r2[a][1])
        for b in range(a+1,len(r2)):
            fb=set(r2[b][0]+r2[b][1])
            if not fa&fb:pairs.append((a,b))

    accepted=0;eligible=0
    initial_score=state_metrics(rids,{ep[i] for i in st},reward)["path_score"]
    best_score=initial_score
    for t in range(REFINE_ATTEMPTS):
        if best_score==GROUND:
            return {ep[i] for i in st},{
              "ground_hit":True,"attempts":t,"accepted":accepted,
              "initial_score":initial_score,"final_score":best_score,
            }
        # Give R3 substantial proposal mass; it is the exact runner-up->ground move.
        if rng.random()<0.6:
            tpl=rng.choice(r3)
            nb=applicable(st,tpl)
            fam="r3"
        else:
            a,b=rng.choice(pairs)
            nb=apply_r2pair(st,r2[a],r2[b])
            fam="r2_pair"
        if nb is None:continue
        ns={ep[i] for i in nb}
        met=state_metrics(rids,ns,reward)
        if not met["valid_path"]:continue
        eligible+=1
        flips=tuple(sorted(st.symmetric_difference(nb)))
        de=abqm.delta_flipset(x,flips)
        # Whole-worldline action: Delta S = beta * Delta H; Trotter part cancels.
        if de<=0 or rng.random()<math.exp(-min(700.0,BETA_REFINE*de)):
            for i in flips:x[i]^=1
            st=nb;accepted+=1
            best_score=met["path_score"]
    final={ep[i] for i in st}
    return final,{
      "ground_hit":best_score==GROUND,"attempts":REFINE_ATTEMPTS,
      "accepted":accepted,"eligible":eligible,
      "initial_score":initial_score,"final_score":best_score,
    }


def run(method,start_mode,seed):
    rids,ep,reward,cost,incoming,outgoing=load_problem();N=len(rids)
    rng=random.Random(seed)
    selected=None
    r2=fc.build_reconnect_templates(ep);r3=build_r3_templates(ep)
    rankfree=array_rankfree(ep,cost,incoming,outgoing)
    rows=[];first_hp=None;refine=None

    for it in range(OUTER):
        prev=state_metrics(rids,selected,reward) if selected is not None else None
        cycles=current_cycles(rids,selected)
        gated=(
          selected is not None and len(selected)==N-1 and prev["degree_conflicts"]==0
          and prev["cycle_count"]>0
        )
        exactA=local_repair_scale(rids,ep,reward,selected,rankfree,r2,r3) if gated else 0.0

        dummy={r:0 for r in rids}
        bqm=build_bqm_count(
          ep,cost,incoming,outgoing,dummy,N,
          degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)
        meta=[]
        applied_A=0.0
        if gated and method=="exact_cut":
            applied_A=exactA
            meta=add_exact_cuts(bqm,cycles,applied_A)
        elif gated and method=="rank_ensemble":
            # Removing one edge from an L-cycle reduces averaged pressure by A/L.
            L=max(len(c) for c in cycles)
            applied_A=min(4.0,L*exactA)
            meta=add_rank_ensemble(bqm,cycles,applied_A)

        selected,sec,nv,nq=sample_openjij(
          bqm,ep,selected,meta,start_mode,rng,seed+1000*it,N)
        met=state_metrics(rids,selected,reward)
        row={
          "method":method,"start":start_mode,"seed":seed,"iteration":it,
          "acyclicity_gated":gated,"acyclicity_A":applied_A,
          "active_cycles":len(cycles) if gated else 0,
          "variables":nv,"quadratic":nq,"seconds":sec,**met,
        }
        rows.append(row);print("ROW",json.dumps(row),flush=True)

        if met["valid_path"]:
            first_hp=met["path_score"]
            selected,refine=feasible_worldline_refine(
              rids,ep,reward,selected,rankfree,r2,r3,seed+900000+it)
            final=state_metrics(rids,selected,reward)
            break
    else:
        final=state_metrics(rids,selected,reward)

    return rows,{
      "method":method,"start":start_mode,"seed":seed,
      "global_found_hp":first_hp is not None,
      "first_hp_score":first_hp,
      "refine":refine,
      "final_score":final["path_score"],
      "ground_hit":final["path_score"]==GROUND,
      "final_valid_path":final["valid_path"],
      "outer_iterations":len(rows),
    }


def main():
    rows=[];summ=[]
    for method in METHODS:
        for start in STARTS:
            for si,base in enumerate(SEEDS):
                # common seeds across methods/start modes for comparison
                seed=base+10000*si
                rr,ss=run(method,start,seed)
                rows+=rr;summ.append(ss);print("SUMMARY",json.dumps(ss),flush=True)
    agg={}
    for method in METHODS:
        for start in STARTS:
            aa=[s for s in summ if s["method"]==method and s["start"]==start]
            agg[f"{method}|{start}"]={
              "runs":len(aa),
              "global_hp_hits":sum(s["global_found_hp"] for s in aa),
              "ground_hits":sum(s["ground_hit"] for s in aa),
              "best_final_score":max((s["final_score"] or -1 for s in aa),default=None),
              "first_hp_scores":[s["first_hp_score"] for s in aa],
              "refine_ground_hits":sum(bool(s["refine"] and s["refine"]["ground_hit"]) for s in aa),
            }
    result={"aggregate":agg,"summaries":summ,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)


if __name__=="__main__":main()
