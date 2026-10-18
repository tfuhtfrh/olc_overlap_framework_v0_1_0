"""Multi-k fixed-cardinality exchange benchmark for CHM13 complex144.

Extends the successful fixed-cardinality phase sampler:
  Phase A (outer iters 0..11): exchange-only degree repair.
  Phase B (outer iters 12..17): 50/50 exchange + directed reconnect.

The only change across configurations is the exchange move-size distribution.
A k-exchange chooses uniformly:
  - k currently selected edges,
  - k currently unselected edges,
and flips all 2k bits.  At fixed cardinality this proposal is symmetric:
  q(x->x') = p_k / [C(143,k) C(118,k)] = q(x'->x).

Configurations:
  k1        : {1:1.0}
  k12       : {1:0.7, 2:0.3}
  k123      : {1:0.6, 2:0.3, 3:0.1}
  k123_wide : {1:0.4, 2:0.4, 3:0.2}

This remains a fixed-cardinality constrained path-integral sampler, not an
exact simulation of the standard independent transverse-field driver.
"""
from __future__ import annotations
import json, math, random, time
from pathlib import Path

import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import load_problem, project_rank, graph_metrics

OUT=Path("debug/qubo/chm13_multik_exchange_benchmark_20260928.json")
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

CONFIGS={
  "k1": {1:1.0},
  "k12": {1:0.7,2:0.3},
  "k123": {1:0.6,2:0.3,3:0.1},
  "k123_wide": {1:0.4,2:0.4,3:0.2},
}

def choose_k(dist,rng):
    z=rng.random(); acc=0.0
    for k,p in sorted(dist.items()):
        acc+=p
        if z<=acc:
            return k
    return max(dist)

def propose_exchange_k(x,k,rng):
    ones=[i for i,v in enumerate(x) if v]
    zeros=[i for i,v in enumerate(x) if not v]
    if len(ones)<k or len(zeros)<k:
        return (),False
    off=rng.sample(ones,k)
    on=rng.sample(zeros,k)
    return tuple(off+on),True

def one_read(abqm,templates,k_select,init,seed,kdist,use_reconnect):
    rng=random.Random(seed)
    m=len(abqm.h)
    if init is None:
        slices=[fc.random_fixed_state(m,k_select,rng) for _ in range(TROTTER)]
    else:
        base=[1 if i in init else 0 for i in range(m)]
        assert sum(base)==k_select
        slices=[base[:] for _ in range(TROTTER)]

    stats={
      "attempts":{"k1":0,"k2":0,"k3":0,"reconnect":0},
      "eligible":{"k1":0,"k2":0,"k3":0,"reconnect":0},
      "accepted":{"k1":0,"k2":0,"k3":0,"reconnect":0},
    }

    for sw in range(SWEEPS):
        frac=sw/max(1,SWEEPS-1)
        gamma=GAMMA_MAX*((GAMMA_MIN/GAMMA_MAX)**frac)
        xarg=max(1e-12,min(50.0,BETA*gamma/TROTTER))
        kt=-0.5*math.log(max(1e-300,math.tanh(xarg)))

        order=list(range(TROTTER)); rng.shuffle(order)
        for s in order:
            do_rec=use_reconnect and rng.random()<0.5
            if do_rec:
                key="reconnect"; stats["attempts"][key]+=1
                flips,ok=fc.propose_reconnect(slices[s],templates,rng)
            else:
                k=choose_k(kdist,rng)
                key=f"k{k}"; stats["attempts"][key]+=1
                flips,ok=propose_exchange_k(slices[s],k,rng)
            if not ok:
                continue
            stats["eligible"][key]+=1

            dclass=abqm.delta_flipset(slices[s],flips)
            dtrot=fc.trotter_delta(
                slices[s],slices[(s-1)%TROTTER],slices[(s+1)%TROTTER],
                flips,kt)
            daction=(BETA/TROTTER)*dclass+dtrot
            if daction<=0.0 or rng.random()<math.exp(-min(700.0,daction)):
                for i in flips:
                    slices[s][i]^=1
                stats["accepted"][key]+=1

    best=min(slices,key=abqm.energy)
    assert sum(best)==k_select
    return best,abqm.energy(best),stats

def sample(bqm,ep,templates,k_select,init_edges,seed,kdist,use_reconnect):
    abqm=fc.ArrayBQM.from_bqm(bqm,ep)
    init=None
    if init_edges is not None:
        idx={e:i for i,e in enumerate(ep)}
        init={idx[e] for e in init_edges}
        assert len(init)==k_select

    total={
      "attempts":{"k1":0,"k2":0,"k3":0,"reconnect":0},
      "eligible":{"k1":0,"k2":0,"k3":0,"reconnect":0},
      "accepted":{"k1":0,"k2":0,"k3":0,"reconnect":0},
    }
    best=None
    t0=time.perf_counter()
    for rd in range(READS):
        x,en,st=one_read(abqm,templates,k_select,init,seed+100003*rd,kdist,use_reconnect)
        for bucket in total:
            for key in total[bucket]:
                total[bucket][key]+=st[bucket][key]
        if best is None or en<best[1]:
            best=(x,en)
    sec=time.perf_counter()-t0
    selected={ep[i] for i,v in enumerate(best[0]) if v}
    return selected,sec,best[1],total

def run(config,seed):
    kdist=CONFIGS[config]
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    templates=fc.build_reconnect_templates(ep)
    rng=random.Random(seed)
    ranks={r:i for i,r in enumerate(rng.sample(rids,len(rids)))}
    selected=None
    rows=[]
    for it in range(OUTER):
        op=fc.ORDER_SCHEDULE[it%len(fc.ORDER_SCHEDULE)]
        bqm=fc.build_bqm_fixed_cardinality(
            ep,cost,incoming,outgoing,ranks,
            degree_conflict_penalty=288.0,order_penalty=op)
        use_reconnect=(it>=12)
        selected,sec,en,stats=sample(
            bqm,ep,templates,len(rids)-1,selected,seed+1000*it,kdist,use_reconnect)
        ranks=project_rank(rids,selected)
        met=graph_metrics(rids,selected,ranks,reward)
        row={"config":config,"seed":seed,"iteration":it,
             "phase":"exchange+reconnect" if use_reconnect else "exchange",
             "order_penalty":op,"seconds":sec,"inner_energy":en,
             **stats,**met}
        rows.append(row)
        print("ROW",json.dumps(row),flush=True)

    feas=[r for r in rows if r["valid_path"]]
    best=max(feas,key=lambda r:r["path_score"]) if feas else None
    return rows,{
      "config":config,"seed":seed,"ever_feasible":bool(feas),
      "first_feasible_iteration":next((r["iteration"] for r in rows if r["valid_path"]),None),
      "best_score":None if best is None else best["path_score"],
      "score_gap":None if best is None else CERT-best["path_score"],
      "ground_hit":bool(best and best["path_score"]==CERT),
      "min_degree_conflicts":min(r["degree_conflicts"] for r in rows),
      "final_degree_conflicts":rows[-1]["degree_conflicts"],
      "final_cycle_count":rows[-1]["cycle_count"],
    }

def main():
    rows=[];summ=[]
    for ci,config in enumerate(CONFIGS):
        for i in range(RUNS):
            rr,ss=run(config,BASE+100000*ci+10000*i)
            rows.extend(rr);summ.append(ss)
            print("SUMMARY",json.dumps(ss),flush=True)
    agg={}
    for config in CONFIGS:
        ss=[s for s in summ if s["config"]==config]
        fs=[s for s in ss if s["ever_feasible"]]
        agg[config]={
          "runs":len(ss),
          "feasible_runs":len(fs),
          "P_feasible":len(fs)/len(ss),
          "ground_hits":sum(s["ground_hit"] for s in ss),
          "best_score":max((s["best_score"] for s in fs),default=None),
          "min_degree_conflicts":min(s["min_degree_conflicts"] for s in ss),
        }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"aggregate":agg,"summaries":summ,"rows":rows},indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":
    main()
