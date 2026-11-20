"""SQA-compatible endpoint-transfer kernel diagnostic on CHM13 stuck state.

Compare the same static symmetric same-head sink-transfer templates with:
1) single-slice 2-bit swaps (subject to Trotter freezing);
2) whole-worldline 2-bit swaps, applying the same involution to every slice.

For the whole-worldline move, each affected bit is flipped on all Trotter
slices, so every imaginary-time equality/disagreement is unchanged and
Delta S_Trotter = 0 exactly.  The proposal is symmetric because the template
set is static and the same two-bit involution is its own reverse.
"""
from __future__ import annotations

import json, math, random
from pathlib import Path

import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import graph_metrics, project_rank
from benchmark_chm13_endpoint_transfer_mc import templates_by_shared_endpoint
from diagnose_chm13_move_basis_proposal import replay_stuck_checkpoint

OUT=Path("debug/qubo/chm13_endpoint_transfer_sqa_20260929.json")
P=8
BETA=4.0
GAMMA_MAX=3.0
GAMMA_MIN=0.03
ATTEMPTS=8000
SEEDS=tuple(202609290+i for i in range(24))


def kt_for(frac):
    gamma=GAMMA_MAX*((GAMMA_MIN/GAMMA_MAX)**frac)
    xarg=max(1e-12,min(50.0,BETA*gamma/P))
    return -0.5*math.log(max(1e-300,math.tanh(xarg)))


def trotter_delta(slice_x,prev_x,next_x,flips,kt):
    dd=0
    for i in flips:
        old=slice_x[i]
        dd+=(1-2*(old!=prev_x[i]))
        dd+=(1-2*(old!=next_x[i]))
    return 2.0*kt*dd


def hp_metrics(rids,ep,reward,x):
    sel={ep[i] for i,v in enumerate(x) if v}
    ranks=project_rank(rids,sel)
    return graph_metrics(rids,sel,ranks,reward)


def run(mode,seed,ctx):
    rids,ep,reward,abqm,x0,head=ctx
    slices=[x0[:] for _ in range(P)]
    rng=random.Random(seed)
    eligible=accepted=0
    first_hp=None
    min_accept_action=None
    for t in range(ATTEMPTS):
        frac=t/max(1,ATTEMPTS-1)
        kt=kt_for(frac)
        tpl=rng.choice(head)
        i,j=tpl
        if mode=="single_slice":
            s=rng.randrange(P)
            if slices[s][i]==slices[s][j]:
                continue
            eligible+=1
            flips=(i,j)
            dc=abqm.delta_flipset(slices[s],flips)
            dt=trotter_delta(
                slices[s],slices[(s-1)%P],slices[(s+1)%P],flips,kt)
            da=(BETA/P)*dc+dt
            if min_accept_action is None or da<min_accept_action:
                min_accept_action=da
            if da<=0 or rng.random()<math.exp(-min(700,da)):
                slices[s][i]^=1;slices[s][j]^=1;accepted+=1
                met=hp_metrics(rids,ep,reward,slices[s])
                if met["valid_path"]:
                    first_hp=t;break
        elif mode=="whole_worldline":
            # Fixed-cardinality-safe only when the pair is complementary in
            # every slice.  The global flip then preserves each slice's count.
            if any(sl[i]==sl[j] for sl in slices):
                continue
            eligible+=1
            dsum=sum(abqm.delta_flipset(sl,(i,j)) for sl in slices)
            da=(BETA/P)*dsum  # exact Trotter delta is zero
            if min_accept_action is None or da<min_accept_action:
                min_accept_action=da
            if da<=0 or rng.random()<math.exp(-min(700,da)):
                for sl in slices:
                    sl[i]^=1;sl[j]^=1
                accepted+=1
                met=hp_metrics(rids,ep,reward,slices[0])
                if met["valid_path"]:
                    first_hp=t;break
        else:raise ValueError(mode)
    return {
      "mode":mode,"seed":seed,"attempts":ATTEMPTS,
      "eligible":eligible,"accepted":accepted,
      "hp_hit":first_hp is not None,"first_hp_attempt":first_hp,
      "minimum_seen_action_delta":min_accept_action,
    }


def main():
    rids,ep,reward,cost,incoming,outgoing,selected,ranks,op=replay_stuck_checkpoint()
    bqm=fc.build_bqm_fixed_cardinality(
        ep,cost,incoming,outgoing,project_rank(rids,selected),
        degree_conflict_penalty=288.0,order_penalty=0.0)
    abqm=fc.ArrayBQM.from_bqm(bqm,ep)
    x0=[1 if e in selected else 0 for e in ep]
    head,_=templates_by_shared_endpoint(ep)
    ctx=(rids,ep,reward,abqm,x0,head)
    rows=[]
    for mode in ("single_slice","whole_worldline"):
        for seed in SEEDS:
            r=run(mode,seed,ctx);rows.append(r);print("ROW",json.dumps(r),flush=True)
    agg={}
    for mode in ("single_slice","whole_worldline"):
        rr=[r for r in rows if r["mode"]==mode];hits=[r for r in rr if r["hp_hit"]]
        agg[mode]={
          "runs":len(rr),"hp_hits":len(hits),"P_hp":len(hits)/len(rr),
          "median_first_hp":(
            sorted(r["first_hp_attempt"] for r in hits)[len(hits)//2]
            if hits else None),
          "mean_accepted":sum(r["accepted"] for r in rr)/len(rr),
          "mean_eligible":sum(r["eligible"] for r in rr)/len(rr),
          "min_action_delta":min(
            (r["minimum_seen_action_delta"] for r in rr
             if r["minimum_seen_action_delta"] is not None),default=None),
        }
    result={
      "same_head_templates":len(head),"P":P,"beta":BETA,
      "gamma_max":GAMMA_MAX,"gamma_min":GAMMA_MIN,
      "aggregate":agg,"rows":rows,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)


if __name__=="__main__":main()
