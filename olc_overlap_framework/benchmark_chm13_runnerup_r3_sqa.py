"""Direct path-integral dynamics test for the CHM13 runner-up -> ground R3 move.

The exact runner-up differs from the certified ground state by one genuine R3
alternating 6-cycle (3 removed + 3 added edges).

Start all P=8 slices at the runner-up on the rank-free fixed-cardinality BQM.
Compare:
- single_slice_r3: apply an eligible static R3 template on one slice;
- whole_worldline_r3: apply the same eligible R3 involution to every slice.

Templates are static and sampled uniformly, so proposals are symmetric.
For the whole-worldline move the Trotter term is invariant exactly.
"""
from __future__ import annotations

import json, math, random
from pathlib import Path

import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import (
    load_problem, project_rank, graph_metrics,
)
from diagnose_chm13_move_basis_proposal import build_r3_templates, applicable
from diagnose_chm13_runnerup_ground_geometry import normalized_paths, score, state

OUT=Path("debug/qubo/chm13_runnerup_r3_sqa_20260929.json")
GROUND=1_343_093
RUNNER=1_338_209
P=8
BETA=4.0
GMAX=3.0
GMIN=0.03
ATTEMPTS=4000
SEEDS=tuple(2026092900+i for i in range(24))


def kt(frac):
    gamma=GMAX*((GMIN/GMAX)**frac)
    x=max(1e-12,min(50.0,BETA*gamma/P))
    return -0.5*math.log(max(1e-300,math.tanh(x)))


def trotter_delta(sl,prev,nxt,flips,K):
    dd=0
    for i in flips:
        old=sl[i]
        dd+=(1-2*(old!=prev[i]))
        dd+=(1-2*(old!=nxt[i]))
    return 2.0*K*dd


def selected_from_x(x,ep):
    return {ep[i] for i,v in enumerate(x) if v}


def run(mode,seed,ctx):
    rids,ep,reward,abqm,x0,r3,ground_ids=ctx
    slices=[x0[:] for _ in range(P)]
    rng=random.Random(seed)
    eligible=accepted=0
    first_ground=None;first_hp=None
    ground_proposed=ground_accepted=0
    min_action=None
    for t in range(ATTEMPTS):
        frac=t/max(1,ATTEMPTS-1)
        K=kt(frac)
        tpl=rng.choice(r3)
        if mode=="single_slice_r3":
            s=rng.randrange(P)
            st=frozenset(i for i,v in enumerate(slices[s]) if v)
            nb=applicable(st,tpl)
            if nb is None:continue
            eligible+=1
            flips=tuple(sorted(st.symmetric_difference(nb)))
            is_ground=(nb==ground_ids)
            if is_ground:ground_proposed+=1
            dc=abqm.delta_flipset(slices[s],flips)
            da=(BETA/P)*dc+trotter_delta(
                slices[s],slices[(s-1)%P],slices[(s+1)%P],flips,K)
            min_action=da if min_action is None else min(min_action,da)
            if da<=0 or rng.random()<math.exp(-min(700,da)):
                for i in flips:slices[s][i]^=1
                accepted+=1
                if is_ground:ground_accepted+=1
                sel=selected_from_x(slices[s],ep)
                met=graph_metrics(rids,sel,project_rank(rids,sel),reward)
                if met["valid_path"] and first_hp is None:first_hp=t
                if met["path_score"]==GROUND:
                    first_ground=t;break
        elif mode=="whole_worldline_r3":
            states=[frozenset(i for i,v in enumerate(sl) if v) for sl in slices]
            # Require the same static involution applicable in all replicas.
            nbs=[applicable(st,tpl) for st in states]
            if any(nb is None for nb in nbs):continue
            flips=tuple(sorted(states[0].symmetric_difference(nbs[0])))
            if any(tuple(sorted(st.symmetric_difference(nb)))!=flips for st,nb in zip(states,nbs)):
                continue
            eligible+=1
            is_ground=all(nb==ground_ids for nb in nbs)
            if is_ground:ground_proposed+=1
            dc=sum(abqm.delta_flipset(sl,flips) for sl in slices)
            da=(BETA/P)*dc
            min_action=da if min_action is None else min(min_action,da)
            if da<=0 or rng.random()<math.exp(-min(700,da)):
                for sl in slices:
                    for i in flips:sl[i]^=1
                accepted+=1
                if is_ground:ground_accepted+=1
                sel=selected_from_x(slices[0],ep)
                met=graph_metrics(rids,sel,project_rank(rids,sel),reward)
                if met["valid_path"] and first_hp is None:first_hp=t
                if met["path_score"]==GROUND:
                    first_ground=t;break
        else:raise ValueError(mode)

    return {
      "mode":mode,"seed":seed,"attempts":ATTEMPTS,
      "eligible":eligible,"accepted":accepted,
      "ground_proposed":ground_proposed,"ground_accepted":ground_accepted,
      "first_hp_attempt":first_hp,"first_ground_attempt":first_ground,
      "ground_hit":first_ground is not None,
      "minimum_seen_action_delta":min_action,
    }


def main():
    rids,ep,reward,cost,incoming,outgoing=load_problem();epset=set(ep)
    rows=[]
    for p in normalized_paths():
        sc=score(p,reward)
        if sc in (RUNNER,GROUND):
            st=state(p)
            if st<=epset:rows.append((sc,p,st))
    runner=next(st for sc,_,st in rows if sc==RUNNER)
    ground=next(st for sc,_,st in rows if sc==GROUND)

    ranks=project_rank(rids,runner)
    bqm=fc.build_bqm_fixed_cardinality(
        ep,cost,incoming,outgoing,ranks,
        degree_conflict_penalty=288.0,order_penalty=0.0)
    abqm=fc.ArrayBQM.from_bqm(bqm,ep)
    idx={e:i for i,e in enumerate(ep)}
    x0=[1 if e in runner else 0 for e in ep]
    ground_ids=frozenset(idx[e] for e in ground)
    r3=build_r3_templates(ep)

    # static local geometry at runner-up
    runner_ids=frozenset(idx[e] for e in runner)
    elig=[]
    for ti,tpl in enumerate(r3):
        nb=applicable(runner_ids,tpl)
        if nb is not None:
            elig.append({
              "template":ti,
              "ground":nb==ground_ids,
              "flip_count":len(runner_ids.symmetric_difference(nb)),
            })

    ctx=(rids,ep,reward,abqm,x0,r3,ground_ids)
    outrows=[]
    for mode in ("single_slice_r3","whole_worldline_r3"):
        for seed in SEEDS:
            r=run(mode,seed,ctx);outrows.append(r);print("ROW",json.dumps(r),flush=True)

    agg={}
    for mode in ("single_slice_r3","whole_worldline_r3"):
        rr=[r for r in outrows if r["mode"]==mode];hits=[r for r in rr if r["ground_hit"]]
        agg[mode]={
          "runs":len(rr),"ground_hits":len(hits),"P_ground":len(hits)/len(rr),
          "median_first_ground":(
            sorted(r["first_ground_attempt"] for r in hits)[len(hits)//2]
            if hits else None),
          "mean_eligible":sum(r["eligible"] for r in rr)/len(rr),
          "mean_accepted":sum(r["accepted"] for r in rr)/len(rr),
          "ground_proposals_total":sum(r["ground_proposed"] for r in rr),
          "ground_accepts_total":sum(r["ground_accepted"] for r in rr),
          "min_action_delta":min(
            r["minimum_seen_action_delta"] for r in rr
            if r["minimum_seen_action_delta"] is not None),
        }

    result={
      "r3_templates":len(r3),
      "runner_eligible_r3":len(elig),
      "runner_ground_r3_templates":[r for r in elig if r["ground"]],
      "aggregate":agg,"rows":outrows,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(result["aggregate"]),flush=True)


if __name__=="__main__":main()
