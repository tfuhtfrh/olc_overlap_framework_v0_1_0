"""MC proposal-mass benchmark for symmetric source/sink transfer templates.

Sink transfer:
    (a->v) <-> (b->v)     [same head]
moves the unique out-degree defect (sink) between a and b.

Source transfer:
    (u->a) <-> (u->b)     [same tail]
moves the unique in-degree defect (source) between a and b.

Static uniform selection from these unordered templates is symmetric because
each accepted proposal is the same 2-bit involution in both directions.
"""
from __future__ import annotations

import json, math, random
from pathlib import Path

import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import graph_metrics, project_rank
from diagnose_chm13_move_basis_proposal import replay_stuck_checkpoint

OUT=Path("debug/qubo/chm13_endpoint_transfer_mc_20260929.json")
BASE=20260929
ATTEMPTS=8000
SEEDS=tuple(BASE+i for i in range(24))
BETAS=(4.0,1.0,0.25)


def templates_by_shared_endpoint(ep):
    by_head={};by_tail={}
    for i,(u,v) in enumerate(ep):
        by_head.setdefault(v,[]).append(i)
        by_tail.setdefault(u,[]).append(i)
    def pairs(groups):
        out=set()
        for ids in groups.values():
            for a in range(len(ids)):
                for b in range(a+1,len(ids)):
                    out.add((ids[a],ids[b]))
        return tuple(sorted(out))
    return pairs(by_head),pairs(by_tail)


def propose_template(x,templates,rng):
    i,j=rng.choice(templates)
    if x[i]==x[j]:
        return (),False
    return (i,j),True


def propose_exchange(x,rng):
    ones=[i for i,v in enumerate(x) if v]
    zeros=[i for i,v in enumerate(x) if not v]
    return (rng.choice(ones),rng.choice(zeros)),True


def run(config,beta,seed,ctx):
    rids,ep,reward,abqm,x0,head,tail=ctx
    x=x0[:];rng=random.Random(seed)
    accepted=eligible=0
    first_hp=None;best_score=None
    for t in range(ATTEMPTS):
        if config=="uniform_exchange":
            flips,ok=propose_exchange(x,rng)
        elif config=="sink_transfer":
            flips,ok=propose_template(x,head,rng)
        elif config=="endpoint_union":
            pool=head+tail
            flips,ok=propose_template(x,pool,rng)
        else:raise ValueError(config)
        if not ok:continue
        eligible+=1
        de=abqm.delta_flipset(x,flips)
        if de<=0 or rng.random()<math.exp(-min(700,beta*de)):
            for i in flips:x[i]^=1
            accepted+=1
            sel={ep[i] for i,v in enumerate(x) if v}
            ranks=project_rank(rids,sel)
            met=graph_metrics(rids,sel,ranks,reward)
            if met["valid_path"]:
                if first_hp is None:first_hp=t
                best_score=met["path_score"] if best_score is None else max(best_score,met["path_score"])
                # endpoint repair benchmark: first HP is enough
                break
    return {
      "config":config,"beta":beta,"seed":seed,"attempts":ATTEMPTS,
      "eligible":eligible,"accepted":accepted,
      "first_hp_attempt":first_hp,"hp_hit":first_hp is not None,
      "best_score":best_score,
    }


def main():
    rids,ep,reward,cost,incoming,outgoing,selected,ranks,op=replay_stuck_checkpoint()
    bqm=fc.build_bqm_fixed_cardinality(
        ep,cost,incoming,outgoing,project_rank(rids,selected),
        degree_conflict_penalty=288.0,order_penalty=0.0)
    abqm=fc.ArrayBQM.from_bqm(bqm,ep)
    x0=[1 if e in selected else 0 for e in ep]
    head,tail=templates_by_shared_endpoint(ep)
    ctx=(rids,ep,reward,abqm,x0,head,tail)

    rows=[]
    for config in ("uniform_exchange","sink_transfer","endpoint_union"):
        for beta in BETAS:
            for seed in SEEDS:
                r=run(config,beta,seed,ctx);rows.append(r)
                print("ROW",json.dumps(r),flush=True)
    agg={}
    for config in ("uniform_exchange","sink_transfer","endpoint_union"):
        for beta in BETAS:
            rr=[r for r in rows if r["config"]==config and r["beta"]==beta]
            hits=[r for r in rr if r["hp_hit"]]
            agg[f"{config}|beta={beta}"]={
              "runs":len(rr),"hp_hits":len(hits),
              "P_hp":len(hits)/len(rr),
              "median_first_hp":(
                sorted(r["first_hp_attempt"] for r in hits)[len(hits)//2]
                if hits else None),
              "best_score":max((r["best_score"] for r in hits),default=None),
              "mean_eligible":sum(r["eligible"] for r in rr)/len(rr),
              "mean_accepted":sum(r["accepted"] for r in rr)/len(rr),
            }
    result={
      "same_head_templates":len(head),
      "same_tail_templates":len(tail),
      "all_exchange_pairs":sum(x0)*(len(x0)-sum(x0)),
      "aggregate":agg,"rows":rows,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(result["aggregate"]),flush=True)


if __name__=="__main__":main()
