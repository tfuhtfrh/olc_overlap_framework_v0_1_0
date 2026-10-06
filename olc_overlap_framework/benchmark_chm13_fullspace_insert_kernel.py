"""Full-space structured cardinality-changing kernel benchmark.

Adds a symmetric 1<->2 path-insertion/removal proposal to the full-space
sampler.  A static template is

    {a->b} <-> {a->c, c->b}

for distinct a,b,c when all three candidate edges exist.

Choosing uniformly from a STATIC template set and flipping all three bits only
when exactly one side is occupied is an involution, hence a symmetric proposal.

This move changes selected-edge count by +/-1 while preserving the external
in/out degree of a and b.  It is designed specifically for the full-space
regime where previous runs tended to stop at 139-142 edges despite degree=0.

Compare count penalties 32 and 64.
"""
from __future__ import annotations
import json, math, random, time
from pathlib import Path

import benchmark_chm13_fixed_cardinality_kernels as fc
import benchmark_chm13_fullspace_multik_sampler as fs
from benchmark_chm13_projected_edge_hybrid import (
    load_problem,project_rank,graph_metrics,build_bqm_count
)

OUT=Path("debug/qubo/chm13_fullspace_insert_kernel_20260928.json")
BASE=20260928
CERT=1343093
RUNS=3
OUTER=18
TROTTER=8
SWEEPS=1200
READS=2
BETA=4.0
GAMMA_MAX=3.0
GAMMA_MIN=0.03

CONFIGS={
  "base_count32": (32.0,False),
  "insert_count32": (32.0,True),
  "insert_count64": (64.0,True),
}

def build_insert_templates(ep):
    idx={e:i for i,e in enumerate(ep)}
    out={}
    for e in ep:out.setdefault(e[0],[]).append(e[1])
    templates=set()
    for (a,b),i in idx.items():
        for c in out.get(a,()):
            if c in (a,b):continue
            e2=(a,c);e3=(c,b)
            if e3 not in idx:continue
            side1=(i,)
            side2=tuple(sorted((idx[e2],idx[e3])))
            if i in side2 or len(set(side2))<2:continue
            templates.add((side1,side2))
    return tuple(sorted(templates))

def propose_insert(x,templates,rng):
    if not templates:return (),False
    side1,side2=rng.choice(templates)
    a=all(x[i] for i in side1) and all(not x[i] for i in side2)
    b=all(x[i] for i in side2) and all(not x[i] for i in side1)
    if a or b:return tuple(side1+side2),True
    return (),False

def schedule(outer_it,frac,use_insert):
    if outer_it<8:
        return {"single":0.7,"exchange":0.3,"reconnect":0.0,"insert":0.0}
    if outer_it<12:
        # Cardinality repair becomes important after degree profile improves.
        return {"single":0.25,"exchange":0.45,"reconnect":0.0,
                "insert":0.30 if use_insert else 0.0}
    # topology/refinement phase
    if use_insert:
        return {"single":0.10,"exchange":0.20,"reconnect":0.40,"insert":0.30}
    return {"single":0.15,"exchange":0.40,"reconnect":0.45,"insert":0.0}

def choose(probs,rng):
    s=sum(probs.values());z=rng.random()*s;a=0
    for k in ("single","exchange","reconnect","insert"):
        a+=probs[k]
        if z<=a:return k
    return "insert"

def mh(abqm,slices,s,flips,kt,rng):
    dclass=abqm.delta_flipset(slices[s],flips)
    dtrot=fc.trotter_delta(slices[s],slices[(s-1)%TROTTER],
                           slices[(s+1)%TROTTER],flips,kt)
    da=(BETA/TROTTER)*dclass+dtrot
    if da<=0 or rng.random()<math.exp(-min(700.0,da)):
        for i in flips:slices[s][i]^=1
        return True
    return False

def one_read(abqm,reconnect_templates,insert_templates,init,seed,use_insert,outer_it):
    rng=random.Random(seed);m=len(abqm.h)
    if init is None:
        slices=[fc.random_fixed_state(m,143,rng) for _ in range(TROTTER)]
    else:
        base=[1 if i in init else 0 for i in range(m)]
        slices=[base[:] for _ in range(TROTTER)]
    stats={k:{"attempted":0,"eligible":0,"accepted":0}
           for k in ("single","exchange","reconnect","insert")}
    for sw in range(SWEEPS):
        frac=sw/max(1,SWEEPS-1)
        gamma=GAMMA_MAX*((GAMMA_MIN/GAMMA_MAX)**frac)
        xarg=max(1e-12,min(50.0,BETA*gamma/TROTTER))
        kt=-0.5*math.log(max(1e-300,math.tanh(xarg)))
        probs=schedule(outer_it,frac,use_insert)
        order=list(range(TROTTER));rng.shuffle(order)
        for s in order:
            k=choose(probs,rng);stats[k]["attempted"]+=1
            if k=="single":
                flips,ok=((rng.randrange(m),),True)
            elif k=="exchange":
                flips,ok=fs.propose_exchange(slices[s],rng)
            elif k=="reconnect":
                flips,ok=fc.propose_reconnect(slices[s],reconnect_templates,rng)
            else:
                flips,ok=propose_insert(slices[s],insert_templates,rng)
            if not ok:continue
            stats[k]["eligible"]+=1
            if mh(abqm,slices,s,flips,kt,rng):stats[k]["accepted"]+=1
    best=min(slices,key=abqm.energy)
    return best,abqm.energy(best),stats

def sample(bqm,ep,rt,it,init_edges,seed,use_insert,outer_it):
    abqm=fc.ArrayBQM.from_bqm(bqm,ep);idx={e:i for i,e in enumerate(ep)}
    init=None if init_edges is None else {idx[e] for e in init_edges}
    total={k:{"attempted":0,"eligible":0,"accepted":0}
           for k in ("single","exchange","reconnect","insert")}
    best=None;t0=time.perf_counter()
    for rd in range(READS):
        x,en,st=one_read(abqm,rt,it,init,seed+100003*rd,use_insert,outer_it)
        for k in total:
            for z in total[k]:total[k][z]+=st[k][z]
        if best is None or en<best[1]:best=(x,en)
    return {ep[i] for i,v in enumerate(best[0]) if v},time.perf_counter()-t0,best[1],total

def run(cfg,seed):
    count_penalty,use_insert=CONFIGS[cfg]
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    rt=fc.build_reconnect_templates(ep);it=build_insert_templates(ep)
    rng=random.Random(seed);ranks={r:i for i,r in enumerate(rng.sample(rids,len(rids)))}
    selected=None;rows=[]
    for outer in range(OUTER):
        op=fc.ORDER_SCHEDULE[outer%len(fc.ORDER_SCHEDULE)]
        bqm=build_bqm_count(ep,cost,incoming,outgoing,ranks,len(rids),
            degree_conflict_penalty=288.0,count_penalty=count_penalty,order_penalty=op)
        selected,sec,en,st=sample(bqm,ep,rt,it,selected,seed+1000*outer,use_insert,outer)
        ranks=project_rank(rids,selected);met=graph_metrics(rids,selected,ranks,reward)
        row={"config":cfg,"seed":seed,"iteration":outer,"count_penalty":count_penalty,
             "insert_templates":len(it),"seconds":sec,"move_stats":st,**met}
        rows.append(row);print("ROW",json.dumps(row),flush=True)
    feas=[r for r in rows if r["valid_path"]];best=max(feas,key=lambda r:r["path_score"]) if feas else None
    return rows,{
      "config":cfg,"seed":seed,"ever_feasible":bool(feas),
      "best_score":None if best is None else best["path_score"],
      "ground_hit":bool(best and best["path_score"]==CERT),
      "first_feasible_iteration":next((r["iteration"] for r in rows if r["valid_path"]),None),
      "min_count_error":min(abs(r["selected_edges"]-143) for r in rows),
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
          "min_count_error":min(s["min_count_error"] for s in ss),
          "min_degree_conflicts":min(s["min_degree_conflicts"] for s in ss),
        }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"aggregate":agg,"summaries":summ,"rows":rows},indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":main()
