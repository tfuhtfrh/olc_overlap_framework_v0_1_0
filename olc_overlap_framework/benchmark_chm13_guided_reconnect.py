"""Guided reconnect proposal benchmark for CHM13 complex144.

Goal: increase the probability of drawing a structurally useful reconnect while
preserving a correct Metropolis-Hastings transition.

Late phase compares:
1) static_uniform: choose from all 41 static templates; ineligible -> null.
   This is the original symmetric reconnect.
2) eligible_uniform_mh: choose uniformly only from currently eligible
   reconnects.  Use Hastings factor |E(x)|/|E(x')|.
3) component_biased_mh: among eligible reconnects, give larger proposal weight
   to templates whose TWO currently selected edges lie in DIFFERENT weak
   components.  In a degree-correct fixed-cardinality state (one path + cycles),
   such a reconnect is exactly the geometry that merges components and reduces
   the cycle count by one.  All eligible templates retain positive weight so
   reverse proposals remain possible.  Exact Hastings ratio is included.

Important geometry:
with N vertices, N-1 selected edges and indegree/outdegree <=1, the selected
graph consists of exactly ONE directed path component plus zero or more cycles.
A directed 2-switch preserves the entire degree vector:
- two selected edges in the path -> path + one extra cycle;
- path edge + cycle edge -> absorbs the cycle into the path;
- edges from two different cycles -> merges those cycles;
- two edges from one cycle -> can split it.
Thus "different selected weak components" is the natural cycle-reducing target.
"""
from __future__ import annotations
import json, math, random, time
from pathlib import Path

import networkx as nx
import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import load_problem, project_rank, graph_metrics

OUT=Path("debug/qubo/chm13_guided_reconnect_20260928.json")
BASE=20260928
CERT=1343093
RUNS=4
OUTER=18
TROTTER=8
SWEEPS=1500
READS=2
BETA=4.0
GAMMA_MAX=3.0
GAMMA_MIN=0.03
CONFIGS=("static_uniform","eligible_uniform_mh","component_biased_mh")
BIAS=8.0

def exchange1(x,rng):
    ones=[i for i,v in enumerate(x) if v]
    zeros=[i for i,v in enumerate(x) if not v]
    return (rng.choice(ones),rng.choice(zeros))

def eligible_info(x,templates,ep,mode):
    """Return [(template_index, flips, proposal_weight, cross_component)]."""
    selected_edges=[ep[i] for i,v in enumerate(x) if v]
    comp={}
    if mode=="component_biased_mh":
        g=nx.Graph()
        g.add_nodes_from({v for e in ep for v in e})
        g.add_edges_from(selected_edges)
        for ci,nodes in enumerate(nx.connected_components(g)):
            for v in nodes: comp[v]=ci

    out=[]
    for ti,(side1,side2) in enumerate(templates):
        a=all(x[i] for i in side1) and all(not x[i] for i in side2)
        b=all(x[i] for i in side2) and all(not x[i] for i in side1)
        if not (a or b): continue
        selected_side=side1 if a else side2
        removed=[ep[i] for i in selected_side]
        cross=False
        if mode=="component_biased_mh":
            cross=comp.get(removed[0][0])!=comp.get(removed[1][0])
        weight=BIAS if (mode=="component_biased_mh" and cross) else 1.0
        out.append((ti,tuple(side1+side2),weight,cross))
    return out

def pick_weighted(items,rng):
    z=rng.random()*sum(x[2] for x in items);a=0.0
    for item in items:
        a+=item[2]
        if z<=a:return item
    return items[-1]

def mh_action(abqm,slices,s,flips,kt):
    dclass=abqm.delta_flipset(slices[s],flips)
    dtrot=fc.trotter_delta(
      slices[s],slices[(s-1)%TROTTER],slices[(s+1)%TROTTER],flips,kt)
    return (BETA/TROTTER)*dclass+dtrot

def attempt_exchange(abqm,slices,s,kt,rng):
    flips=exchange1(slices[s],rng)
    da=mh_action(abqm,slices,s,flips,kt)
    if da<=0 or rng.random()<math.exp(-min(700.0,da)):
        for i in flips:slices[s][i]^=1
        return True
    return False

def attempt_reconnect(abqm,slices,s,kt,rng,templates,ep,mode,stats):
    stats["attempted"]+=1
    cross=False
    if mode=="static_uniform":
        ti=rng.randrange(len(templates))
        side1,side2=templates[ti]
        x=slices[s]
        a=all(x[i] for i in side1) and all(not x[i] for i in side2)
        b=all(x[i] for i in side2) and all(not x[i] for i in side1)
        if not(a or b): return False
        stats["eligible"]+=1
        flips=tuple(side1+side2)
        qratio=1.0
        # diagnostic cross-component classification
        info=eligible_info(x,[templates[ti]],ep,"component_biased_mh")
        if info and info[0][3]:stats["cross_component_draws"]+=1
    else:
        info=eligible_info(slices[s],templates,ep,mode)
        if not info:return False
        stats["eligible"]+=1
        item=pick_weighted(info,rng)
        ti,flips,wf,cross=item
        if cross:stats["cross_component_draws"]+=1
        zf=sum(x[2] for x in info)
        qf=wf/zf

        # Exact reverse proposal probability in proposed state.
        xp=slices[s][:]
        for i in flips:xp[i]^=1
        rev=eligible_info(xp,templates,ep,mode)
        match=[x for x in rev if x[0]==ti]
        if not match:
            raise RuntimeError("involution disappeared from reverse eligible set")
        wr=match[0][2];zr=sum(x[2] for x in rev)
        qr=wr/zr
        qratio=qr/qf

    da=mh_action(abqm,slices,s,flips,kt)
    acc=min(1.0,math.exp(-min(700.0,da))*qratio) if da>0 else min(1.0,math.exp(min(700.0,-da))*qratio)
    if rng.random()<acc:
        for i in flips:slices[s][i]^=1
        stats["accepted"]+=1
        if cross:stats["accepted_cross_component"]+=1
        return True
    return False

def one_read(abqm,templates,ep,k_select,init,seed,mode,outer_it):
    rng=random.Random(seed);m=len(abqm.h)
    if init is None:
        slices=[fc.random_fixed_state(m,k_select,rng) for _ in range(TROTTER)]
    else:
        base=[1 if i in init else 0 for i in range(m)]
        slices=[base[:] for _ in range(TROTTER)]
    stats={
      "exchange":{"attempted":0,"accepted":0},
      "reconnect":{"attempted":0,"eligible":0,"accepted":0,
                   "cross_component_draws":0,"accepted_cross_component":0},
    }
    for sw in range(SWEEPS):
        frac=sw/max(1,SWEEPS-1)
        gamma=GAMMA_MAX*((GAMMA_MIN/GAMMA_MAX)**frac)
        xarg=max(1e-12,min(50.0,BETA*gamma/TROTTER))
        kt=-0.5*math.log(max(1e-300,math.tanh(xarg)))
        order=list(range(TROTTER));rng.shuffle(order)
        for s in order:
            stats["exchange"]["attempted"]+=1
            if attempt_exchange(abqm,slices,s,kt,rng):
                stats["exchange"]["accepted"]+=1
            if outer_it>=12:
                attempt_reconnect(abqm,slices,s,kt,rng,templates,ep,mode,stats["reconnect"])
    best=min(slices,key=abqm.energy)
    return best,abqm.energy(best),stats

def sample(bqm,ep,templates,k_select,init_edges,seed,mode,outer_it):
    abqm=fc.ArrayBQM.from_bqm(bqm,ep)
    init=None
    if init_edges is not None:
        idx={e:i for i,e in enumerate(ep)};init={idx[e] for e in init_edges}
    total={
      "exchange":{"attempted":0,"accepted":0},
      "reconnect":{"attempted":0,"eligible":0,"accepted":0,
                   "cross_component_draws":0,"accepted_cross_component":0},
    }
    best=None;t0=time.perf_counter()
    for rd in range(READS):
        x,en,st=one_read(abqm,templates,ep,k_select,init,seed+100003*rd,mode,outer_it)
        for family in total:
            for key in total[family]:total[family][key]+=st[family][key]
        if best is None or en<best[1]:best=(x,en)
    return {ep[i] for i,v in enumerate(best[0]) if v},time.perf_counter()-t0,best[1],total

def run(mode,seed):
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    templates=fc.build_reconnect_templates(ep)
    rng=random.Random(seed);ranks={r:i for i,r in enumerate(rng.sample(rids,len(rids)))}
    selected=None;rows=[]
    for it in range(OUTER):
        op=fc.ORDER_SCHEDULE[it%len(fc.ORDER_SCHEDULE)]
        bqm=fc.build_bqm_fixed_cardinality(ep,cost,incoming,outgoing,ranks,288.0,op)
        selected,sec,en,st=sample(bqm,ep,templates,len(rids)-1,selected,seed+1000*it,mode,it)
        ranks=project_rank(rids,selected);met=graph_metrics(rids,selected,ranks,reward)
        row={"mode":mode,"seed":seed,"iteration":it,"seconds":sec,
             "order_penalty":op,"move_stats":st,**met}
        rows.append(row);print("ROW",json.dumps(row),flush=True)
    feas=[r for r in rows if r["valid_path"]];best=max(feas,key=lambda r:r["path_score"]) if feas else None
    return rows,{"mode":mode,"seed":seed,"ever_feasible":bool(feas),
      "best_score":None if best is None else best["path_score"],
      "ground_hit":bool(best and best["path_score"]==CERT),
      "first_feasible_iteration":next((r["iteration"] for r in rows if r["valid_path"]),None),
      "min_degree_conflicts":min(r["degree_conflicts"] for r in rows),
      "final_degree_conflicts":rows[-1]["degree_conflicts"],
      "final_cycle_count":rows[-1]["cycle_count"]}

def main():
    rows=[];summ=[]
    for mi,mode in enumerate(CONFIGS):
        for i in range(RUNS):
            rr,ss=run(mode,BASE+100000*mi+10000*i)
            rows.extend(rr);summ.append(ss);print("SUMMARY",json.dumps(ss),flush=True)
    agg={}
    for mode in CONFIGS:
        ss=[s for s in summ if s["mode"]==mode];fs=[s for s in ss if s["ever_feasible"]]
        agg[mode]={"runs":len(ss),"feasible_runs":len(fs),"ground_hits":sum(s["ground_hit"] for s in ss),
          "best_score":max((s["best_score"] for s in fs),default=None),
          "min_degree_conflicts":min(s["min_degree_conflicts"] for s in ss)}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"aggregate":agg,"summaries":summ,"rows":rows},indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":main()
