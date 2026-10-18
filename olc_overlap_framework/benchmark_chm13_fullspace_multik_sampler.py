"""Full-space multi-k path-integral sampler benchmark for CHM13 complex144.

Priority route: keep the full binary state space and the original count penalty,
then improve only the Monte-Carlo transition kernel.

Target energy:
  H = H_weight + H_degree_conflict + H_count + H_order(rank)

Every slice may leave the 143-edge manifold.  The sampler therefore remains on
the full {0,1}^261 space.  Symmetric multi-bit proposals are used only as MCMC
accelerators for the same path-integral target:

- single: uniformly flip one edge bit (changes cardinality; guarantees full-space
  connectivity);
- exchange: uniformly choose one selected and one unselected edge and flip both;
- reconnect: static directed 2-switch template, chosen uniformly from a fixed
  template list; ineligible templates are null proposals.

All proposal-mixture weights depend only on sweep/outer iteration, never on the
current state, so each instantaneous proposal kernel is symmetric.

Configurations:
1. single_only
2. single_exchange
3. dynamic_full
   - early: mostly single-bit to permit cardinality/degree repair
   - later: reduce single-bit probability, increase exchange
   - topology phase (outer >=12): enable reconnect
"""
from __future__ import annotations
import json, math, random, time
from pathlib import Path

import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import (
    load_problem, project_rank, graph_metrics, build_bqm_count
)

OUT=Path("debug/qubo/chm13_fullspace_multik_sampler_20260928.json")
BASE=20260928
CERT=1343093
RUNS=4
OUTER=18
TROTTER=8
SWEEPS=1200
READS=2
BETA=4.0
GAMMA_MAX=3.0
GAMMA_MIN=0.03
CONFIGS=("single_only","single_exchange","dynamic_full")

def schedule(config,outer_it,frac):
    if config=="single_only":
        return {"single":1.0,"exchange":0.0,"reconnect":0.0}
    if config=="single_exchange":
        return {"single":0.5,"exchange":0.5,"reconnect":0.0}
    if config=="dynamic_full":
        if outer_it<8:
            # retain strong cardinality-changing exploration early
            ps=0.75-0.25*frac
            return {"single":ps,"exchange":1-ps,"reconnect":0.0}
        if outer_it<12:
            ps=0.35-0.20*frac
            return {"single":ps,"exchange":1-ps,"reconnect":0.0}
        # topology phase: keep a small single-bit channel for full-space
        # ergodicity, but emphasize cardinality-preserving topology changes.
        ps=0.15
        pr=0.45
        return {"single":ps,"exchange":1-ps-pr,"reconnect":pr}
    raise ValueError(config)

def choose_kernel(probs,rng):
    z=rng.random();a=0.0
    for k in ("single","exchange","reconnect"):
        a+=probs[k]
        if z<=a:return k
    return "reconnect"

def propose_single(m,rng):
    return (rng.randrange(m),),True

def propose_exchange(x,rng):
    ones=[i for i,v in enumerate(x) if v]
    zeros=[i for i,v in enumerate(x) if not v]
    if not ones or not zeros:return (),False
    return (rng.choice(ones),rng.choice(zeros)),True

def mh(abqm,slices,s,flips,kt,rng):
    dclass=abqm.delta_flipset(slices[s],flips)
    dtrot=fc.trotter_delta(
      slices[s],slices[(s-1)%TROTTER],slices[(s+1)%TROTTER],flips,kt)
    da=(BETA/TROTTER)*dclass+dtrot
    if da<=0 or rng.random()<math.exp(-min(700.0,da)):
        for i in flips:slices[s][i]^=1
        return True
    return False

def one_read(abqm,templates,init,seed,config,outer_it):
    rng=random.Random(seed);m=len(abqm.h)
    if init is None:
        # Start near the known hard-count target but DO NOT constrain it.
        # Every slice begins with a different random 143-edge state.
        slices=[fc.random_fixed_state(m,143,rng) for _ in range(TROTTER)]
    else:
        base=[1 if i in init else 0 for i in range(m)]
        slices=[base[:] for _ in range(TROTTER)]

    stats={k:{"attempted":0,"eligible":0,"accepted":0} for k in ("single","exchange","reconnect")}
    card_min=10**9;card_max=-1
    for sw in range(SWEEPS):
        frac=sw/max(1,SWEEPS-1)
        gamma=GAMMA_MAX*((GAMMA_MIN/GAMMA_MAX)**frac)
        xarg=max(1e-12,min(50.0,BETA*gamma/TROTTER))
        kt=-0.5*math.log(max(1e-300,math.tanh(xarg)))
        probs=schedule(config,outer_it,frac)
        order=list(range(TROTTER));rng.shuffle(order)
        for s in order:
            kernel=choose_kernel(probs,rng);stats[kernel]["attempted"]+=1
            if kernel=="single":
                flips,ok=propose_single(m,rng)
            elif kernel=="exchange":
                flips,ok=propose_exchange(slices[s],rng)
            else:
                flips,ok=fc.propose_reconnect(slices[s],templates,rng)
            if not ok:continue
            stats[kernel]["eligible"]+=1
            if mh(abqm,slices,s,flips,kt,rng):stats[kernel]["accepted"]+=1
        cards=[sum(x) for x in slices]
        card_min=min(card_min,min(cards));card_max=max(card_max,max(cards))
    best=min(slices,key=abqm.energy)
    return best,abqm.energy(best),stats,card_min,card_max

def sample(bqm,ep,templates,init_edges,seed,config,outer_it):
    abqm=fc.ArrayBQM.from_bqm(bqm,ep)
    init=None
    if init_edges is not None:
        idx={e:i for i,e in enumerate(ep)};init={idx[e] for e in init_edges}
    total={k:{"attempted":0,"eligible":0,"accepted":0} for k in ("single","exchange","reconnect")}
    best=None;gmin=10**9;gmax=-1;t0=time.perf_counter()
    for rd in range(READS):
        x,en,st,cmin,cmax=one_read(abqm,templates,init,seed+100003*rd,config,outer_it)
        gmin=min(gmin,cmin);gmax=max(gmax,cmax)
        for k in total:
            for z in total[k]:total[k][z]+=st[k][z]
        if best is None or en<best[1]:best=(x,en)
    selected={ep[i] for i,v in enumerate(best[0]) if v}
    return selected,time.perf_counter()-t0,best[1],total,gmin,gmax

def run(config,seed):
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    templates=fc.build_reconnect_templates(ep)
    rng=random.Random(seed)
    ranks={r:i for i,r in enumerate(rng.sample(rids,len(rids)))}
    selected=None;rows=[]
    for it in range(OUTER):
        op=fc.ORDER_SCHEDULE[it%len(fc.ORDER_SCHEDULE)]
        bqm=build_bqm_count(
          ep,cost,incoming,outgoing,ranks,len(rids),
          degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=op)
        selected,sec,en,st,cmin,cmax=sample(
          bqm,ep,templates,selected,seed+1000*it,config,it)
        ranks=project_rank(rids,selected);met=graph_metrics(rids,selected,ranks,reward)
        row={"config":config,"seed":seed,"iteration":it,"seconds":sec,
             "order_penalty":op,"inner_energy":en,"move_stats":st,
             "cardinality_seen_min":cmin,"cardinality_seen_max":cmax,**met}
        rows.append(row);print("ROW",json.dumps(row),flush=True)
    feas=[r for r in rows if r["valid_path"]]
    best=max(feas,key=lambda r:r["path_score"]) if feas else None
    return rows,{
      "config":config,"seed":seed,"ever_feasible":bool(feas),
      "first_feasible_iteration":next((r["iteration"] for r in rows if r["valid_path"]),None),
      "best_score":None if best is None else best["path_score"],
      "ground_hit":bool(best and best["path_score"]==CERT),
      "min_abs_count_error":min(abs(r["selected_edges"]-143) for r in rows),
      "min_degree_conflicts":min(r["degree_conflicts"] for r in rows),
      "final_selected_edges":rows[-1]["selected_edges"],
      "final_degree_conflicts":rows[-1]["degree_conflicts"],
      "final_cycle_count":rows[-1]["cycle_count"],
    }

def main():
    rows=[];summ=[]
    for ci,cfg in enumerate(CONFIGS):
        for i in range(RUNS):
            rr,ss=run(cfg,BASE+100000*ci+10000*i)
            rows.extend(rr);summ.append(ss);print("SUMMARY",json.dumps(ss),flush=True)
    agg={}
    for cfg in CONFIGS:
        ss=[s for s in summ if s["config"]==cfg];fs=[s for s in ss if s["ever_feasible"]]
        agg[cfg]={
          "runs":len(ss),"feasible_runs":len(fs),"ground_hits":sum(s["ground_hit"] for s in ss),
          "best_score":max((s["best_score"] for s in fs),default=None),
          "min_degree_conflicts":min(s["min_degree_conflicts"] for s in ss),
          "min_abs_count_error":min(s["min_abs_count_error"] for s in ss),
        }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"aggregate":agg,"summaries":summ,"rows":rows},indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":main()
