"""Late repeated k>1 exchange benchmark on the hard fixed-cardinality manifold.

Motivation:
- uniform k=2/k=3 exchange throughout the run had low acceptance;
- k=1 exchange is useful early for degree repair;
- once k=1 acceptance collapses, several independent larger-k attempts may
  provide a coordinated escape without using "sample many and take the best".

Each extra attempt is a separate symmetric Metropolis step.  No candidate is
preselected by energy, so detailed balance of each kernel is preserved.

Schedule:
- outer 0..7: one k=1 exchange attempt per slice/sweep;
- outer 8..11: late degree-repair phase, with extra independent k2/k3 attempts;
- outer 12..17: one k1 exchange OR reconnect (50/50), as in the successful
  exchange->reconnect experiment.

Native proposal counts are reported because the enhanced schedules do more work.
"""
from __future__ import annotations
import json, math, random, time
from pathlib import Path

import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import load_problem, project_rank, graph_metrics

OUT=Path("debug/qubo/chm13_late_multik_retries_20260928.json")
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

# (extra k2 attempts, extra k3 attempts) during outer 8..11.
CONFIGS={
  "baseline": (0,0),
  "late_k2x2_k3x4": (2,4),
  "late_k2x4_k3x8": (4,8),
}

def exchange_k(x,k,rng):
    ones=[i for i,v in enumerate(x) if v]
    zeros=[i for i,v in enumerate(x) if not v]
    off=rng.sample(ones,k); on=rng.sample(zeros,k)
    return tuple(off+on)

def mh_attempt(abqm,slices,s,flips,kt,rng):
    dclass=abqm.delta_flipset(slices[s],flips)
    dtrot=fc.trotter_delta(
        slices[s],slices[(s-1)%TROTTER],slices[(s+1)%TROTTER],
        flips,kt)
    daction=(BETA/TROTTER)*dclass+dtrot
    if daction<=0 or rng.random()<math.exp(-min(700.0,daction)):
        for i in flips:slices[s][i]^=1
        return True
    return False

def one_read(abqm,templates,k_select,init,seed,config,outer_it):
    rng=random.Random(seed); m=len(abqm.h)
    if init is None:
        slices=[fc.random_fixed_state(m,k_select,rng) for _ in range(TROTTER)]
    else:
        base=[1 if i in init else 0 for i in range(m)]
        slices=[base[:] for _ in range(TROTTER)]
    stats={k:{"attempted":0,"accepted":0} for k in ("k1","k2","k3","reconnect")}

    e2,e3=CONFIGS[config]
    for sw in range(SWEEPS):
        frac=sw/max(1,SWEEPS-1)
        gamma=GAMMA_MAX*((GAMMA_MIN/GAMMA_MAX)**frac)
        xarg=max(1e-12,min(50.0,BETA*gamma/TROTTER))
        kt=-0.5*math.log(max(1e-300,math.tanh(xarg)))
        order=list(range(TROTTER));rng.shuffle(order)
        for s in order:
            if outer_it>=12:
                # Topology phase: same simple 50/50 mixture used previously.
                if rng.random()<0.5:
                    stats["reconnect"]["attempted"]+=1
                    flips,ok=fc.propose_reconnect(slices[s],templates,rng)
                    if ok and mh_attempt(abqm,slices,s,flips,kt,rng):
                        stats["reconnect"]["accepted"]+=1
                else:
                    stats["k1"]["attempted"]+=1
                    if mh_attempt(abqm,slices,s,exchange_k(slices[s],1,rng),kt,rng):
                        stats["k1"]["accepted"]+=1
                continue

            # Always retain the robust k1 degree-repair step.
            stats["k1"]["attempted"]+=1
            if mh_attempt(abqm,slices,s,exchange_k(slices[s],1,rng),kt,rng):
                stats["k1"]["accepted"]+=1

            # Only late in degree repair, make several INDEPENDENT k>1 trials.
            if outer_it>=8:
                for _ in range(e2):
                    stats["k2"]["attempted"]+=1
                    if mh_attempt(abqm,slices,s,exchange_k(slices[s],2,rng),kt,rng):
                        stats["k2"]["accepted"]+=1
                for _ in range(e3):
                    stats["k3"]["attempted"]+=1
                    if mh_attempt(abqm,slices,s,exchange_k(slices[s],3,rng),kt,rng):
                        stats["k3"]["accepted"]+=1

    best=min(slices,key=abqm.energy)
    return best,abqm.energy(best),stats

def sample(bqm,ep,templates,k_select,init_edges,seed,config,outer_it):
    abqm=fc.ArrayBQM.from_bqm(bqm,ep)
    init=None
    if init_edges is not None:
        idx={e:i for i,e in enumerate(ep)};init={idx[e] for e in init_edges}
    total={k:{"attempted":0,"accepted":0} for k in ("k1","k2","k3","reconnect")}
    best=None;t0=time.perf_counter()
    for rd in range(READS):
        x,en,st=one_read(abqm,templates,k_select,init,seed+100003*rd,config,outer_it)
        for k in total:
            for z in total[k]:total[k][z]+=st[k][z]
        if best is None or en<best[1]:best=(x,en)
    return {ep[i] for i,v in enumerate(best[0]) if v},time.perf_counter()-t0,best[1],total

def run(config,seed):
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    templates=fc.build_reconnect_templates(ep)
    rng=random.Random(seed)
    ranks={r:i for i,r in enumerate(rng.sample(rids,len(rids)))}
    selected=None;rows=[]
    for it in range(OUTER):
        op=fc.ORDER_SCHEDULE[it%len(fc.ORDER_SCHEDULE)]
        bqm=fc.build_bqm_fixed_cardinality(
            ep,cost,incoming,outgoing,ranks,degree_conflict_penalty=288.0,
            order_penalty=op)
        selected,sec,en,st=sample(
            bqm,ep,templates,len(rids)-1,selected,seed+1000*it,config,it)
        ranks=project_rank(rids,selected);met=graph_metrics(rids,selected,ranks,reward)
        row={"config":config,"seed":seed,"iteration":it,"seconds":sec,
             "order_penalty":op,"inner_energy":en,"move_stats":st,**met}
        rows.append(row);print("ROW",json.dumps(row),flush=True)
    feas=[r for r in rows if r["valid_path"]]
    best=max(feas,key=lambda r:r["path_score"]) if feas else None
    native=sum(v["attempted"] for r in rows for v in r["move_stats"].values())
    return rows,{
      "config":config,"seed":seed,"ever_feasible":bool(feas),
      "best_score":None if best is None else best["path_score"],
      "ground_hit":bool(best and best["path_score"]==CERT),
      "first_feasible_iteration":next((r["iteration"] for r in rows if r["valid_path"]),None),
      "min_degree_conflicts":min(r["degree_conflicts"] for r in rows),
      "final_degree_conflicts":rows[-1]["degree_conflicts"],
      "final_cycle_count":rows[-1]["cycle_count"],
      "native_proposals":native,
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
          "mean_native_proposals":sum(s["native_proposals"] for s in ss)/len(ss),
        }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"aggregate":agg,"summaries":summ,"rows":rows},indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":main()
