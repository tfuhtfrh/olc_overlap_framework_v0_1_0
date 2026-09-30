"""Schedule x cluster-length freezing diagnostic for Stage 2 / Stage 3.

Purpose
-------
Test whether milder temperature / transverse-field schedules reduce late-stage
Trotter freezing enough that structured updates no longer need to flip the
entire worldline.

Representative states:
- Stage 2: degree-correct path+cycle checkpoint, same-head endpoint transfer.
- Stage 3: exact runner-up Hamilton path, structured R3 reconnect toward ground.

For each representative state, use the SAME static structured proposal family
but apply an accepted proposal to a contiguous imaginary-time cluster of length
L in {1,2,4,8}; P=8. L=8 is the previous whole-worldline update.

Schedules vary both beta and Gamma:
- baseline: beta 4->4, Gamma 3->0.03
- warm:     beta 1->2, Gamma 1->0.10
- moderate: beta 1->4, Gamma 0.5->0.10
- hotweak:  beta 0.5->2, Gamma 0.5->0.20

This is a path-integral MC diagnostic, not a real-time QA simulation.
"""
from __future__ import annotations

import json, math, random
from pathlib import Path

import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import load_problem, graph_metrics, project_rank
from benchmark_chm13_endpoint_transfer_mc import templates_by_shared_endpoint
from diagnose_chm13_move_basis_proposal import replay_stuck_checkpoint, build_r3_templates, applicable
from diagnose_chm13_runnerup_ground_geometry import normalized_paths, score, state

OUT=Path("debug/qubo/chm13_schedule_cluster_freezing_20260929.json")
P=8
ATTEMPTS=3000
SEEDS=tuple(202609300+i for i in range(8))
GROUND=1_343_093
RUNNER=1_338_209

SCHEDULES={
  "baseline":{"beta0":4.0,"beta1":4.0,"g0":3.0,"g1":0.03},
  "warm":{"beta0":1.0,"beta1":2.0,"g0":1.0,"g1":0.10},
  "moderate":{"beta0":1.0,"beta1":4.0,"g0":0.5,"g1":0.10},
  "hotweak":{"beta0":0.5,"beta1":2.0,"g0":0.5,"g1":0.20},
}
LENGTHS=(1,2,4,8)


def geom(a,b,f):
    if a<=0 or b<=0:return a+(b-a)*f
    return a*((b/a)**f)


def schedule_vals(cfg,frac):
    beta=geom(cfg["beta0"],cfg["beta1"],frac)
    gamma=geom(cfg["g0"],cfg["g1"],frac)
    x=max(1e-12,min(50.0,beta*gamma/P))
    K=-0.5*math.log(max(1e-300,math.tanh(x)))
    return beta,gamma,K


def cluster_indices(start,L):
    return tuple((start+d)%P for d in range(L))


def trotter_delta_cluster(slices,cluster,flips,K):
    C=set(cluster)
    d=0.0
    for m in range(P):
        n=(m+1)%P
        if (m in C)==(n in C):
            continue
        fm=(m in C); fn=(n in C)
        for i in flips:
            old_dis=(slices[m][i]!=slices[n][i])
            a=(1-slices[m][i]) if fm else slices[m][i]
            b=(1-slices[n][i]) if fn else slices[n][i]
            new_dis=(a!=b)
            d += 2.0*K*(int(new_dis)-int(old_dis))
    return d


def apply_cluster(slices,cluster,flips):
    for m in cluster:
        for i in flips:
            slices[m][i]^=1


def selected(ep,x):
    return {ep[i] for i,v in enumerate(x) if v}


def stage2_context():
    rids,ep,reward,cost,incoming,outgoing,sel,ranks,op=replay_stuck_checkpoint()
    bqm=fc.build_bqm_fixed_cardinality(
        ep,cost,incoming,outgoing,project_rank(rids,sel),
        degree_conflict_penalty=288.0,order_penalty=0.0)
    abqm=fc.ArrayBQM.from_bqm(bqm,ep)
    x0=[1 if e in sel else 0 for e in ep]
    heads,_=templates_by_shared_endpoint(ep)
    return rids,ep,reward,abqm,x0,heads


def stage3_context():
    rids,ep,reward,cost,incoming,outgoing=load_problem();epset=set(ep)
    rows=[]
    for p in normalized_paths():
        sc=score(p,reward)
        if sc in (RUNNER,GROUND):
            st=state(p)
            if st<=epset:rows.append((sc,st))
    runner=next(st for sc,st in rows if sc==RUNNER)
    ground=next(st for sc,st in rows if sc==GROUND)
    bqm=fc.build_bqm_fixed_cardinality(
        ep,cost,incoming,outgoing,project_rank(rids,runner),
        degree_conflict_penalty=288.0,order_penalty=0.0)
    abqm=fc.ArrayBQM.from_bqm(bqm,ep)
    x0=[1 if e in runner else 0 for e in ep]
    idx={e:i for i,e in enumerate(ep)}
    ground_ids=frozenset(idx[e] for e in ground)
    r3=build_r3_templates(ep)
    return rids,ep,reward,abqm,x0,r3,ground_ids


def is_stage2_target(rids,ep,reward,x):
    s=selected(ep,x)
    m=graph_metrics(rids,s,project_rank(rids,s),reward)
    return bool(m["valid_path"])


def is_stage3_target(ep,x,ground_ids):
    return frozenset(i for i,v in enumerate(x) if v)==ground_ids


def propose_stage2(slices,cluster,templates,rng):
    i,j=rng.choice(templates)
    for m in cluster:
        if slices[m][i]==slices[m][j]:
            return None
    return (i,j)


def propose_stage3(slices,cluster,templates,rng):
    tpl=rng.choice(templates)
    flips=None
    for m in cluster:
        st=frozenset(i for i,v in enumerate(slices[m]) if v)
        nb=applicable(st,tpl)
        if nb is None:return None
        ff=tuple(sorted(st.symmetric_difference(nb)))
        if flips is None:flips=ff
        elif ff!=flips:return None
    return flips


def run(kind,sched_name,L,seed,ctx):
    cfg=SCHEDULES[sched_name]
    if kind=="stage2":
        rids,ep,reward,abqm,x0,templates=ctx
        ground_ids=None
    else:
        rids,ep,reward,abqm,x0,templates,ground_ids=ctx

    slices=[x0[:] for _ in range(P)]
    rng=random.Random(seed)
    eligible=accepted=0
    elig_late=acc_late=0
    first_any=None;first_all=None
    max_target_slices=0
    action_samples=[]
    for t in range(ATTEMPTS):
        frac=t/max(1,ATTEMPTS-1)
        beta,gamma,K=schedule_vals(cfg,frac)
        cluster=cluster_indices(rng.randrange(P),L)
        flips=(propose_stage2(slices,cluster,templates,rng)
               if kind=="stage2" else
               propose_stage3(slices,cluster,templates,rng))
        if flips is None:continue
        eligible+=1
        if frac>=0.75:elig_late+=1

        dc=sum(abqm.delta_flipset(slices[m],flips) for m in cluster)
        dt=trotter_delta_cluster(slices,cluster,flips,K)
        da=(beta/P)*dc+dt
        if len(action_samples)<200:
            action_samples.append(da)
        if da<=0 or rng.random()<math.exp(-min(700.0,da)):
            apply_cluster(slices,cluster,flips)
            accepted+=1
            if frac>=0.75:acc_late+=1

            if kind=="stage2":
                hits=[is_stage2_target(rids,ep,reward,sl) for sl in slices]
            else:
                hits=[is_stage3_target(ep,sl,ground_ids) for sl in slices]
            nh=sum(hits);max_target_slices=max(max_target_slices,nh)
            if nh and first_any is None:first_any=t
            if nh==P and first_all is None:first_all=t

    if kind=="stage2":
        final_hits=[is_stage2_target(rids,ep,reward,sl) for sl in slices]
    else:
        final_hits=[is_stage3_target(ep,sl,ground_ids) for sl in slices]

    b0,g0,K0=schedule_vals(cfg,0.0)
    b1,g1,K1=schedule_vals(cfg,1.0)
    return {
      "kind":kind,"schedule":sched_name,"cluster_length":L,"seed":seed,
      "attempts":ATTEMPTS,"eligible":eligible,"accepted":accepted,
      "acceptance":accepted/eligible if eligible else 0.0,
      "eligible_last_quarter":elig_late,"accepted_last_quarter":acc_late,
      "late_acceptance":acc_late/elig_late if elig_late else 0.0,
      "first_any_target":first_any,"first_all_target":first_all,
      "ever_any_target":first_any is not None,
      "ever_all_target":first_all is not None,
      "max_target_slices":max_target_slices,
      "final_target_slices":sum(final_hits),
      "beta0":b0,"beta1":b1,"gamma0":g0,"gamma1":g1,
      "K0":K0,"K1":K1,
      "sampled_action_min":min(action_samples,default=None),
      "sampled_action_median":(
        sorted(action_samples)[len(action_samples)//2] if action_samples else None),
    }


def main():
    contexts={"stage2":stage2_context(),"stage3":stage3_context()}
    rows=[]
    for kind in ("stage2","stage3"):
        for sched in SCHEDULES:
            for L in LENGTHS:
                for seed in SEEDS:
                    r=run(kind,sched,L,seed,contexts[kind])
                    rows.append(r);print("ROW",json.dumps(r),flush=True)

    agg={}
    for kind in ("stage2","stage3"):
      for sched in SCHEDULES:
        for L in LENGTHS:
          rr=[r for r in rows if r["kind"]==kind and r["schedule"]==sched and r["cluster_length"]==L]
          anyhits=[r for r in rr if r["ever_any_target"]]
          allhits=[r for r in rr if r["ever_all_target"]]
          agg[f"{kind}|{sched}|L={L}"]={
            "runs":len(rr),
            "any_target_hits":len(anyhits),
            "all_slice_hits":len(allhits),
            "mean_acceptance":sum(r["acceptance"] for r in rr)/len(rr),
            "mean_late_acceptance":sum(r["late_acceptance"] for r in rr)/len(rr),
            "mean_final_target_slices":sum(r["final_target_slices"] for r in rr)/len(rr),
            "median_first_any":(
              sorted(r["first_any_target"] for r in anyhits)[len(anyhits)//2]
              if anyhits else None),
            "K0":rr[0]["K0"],"K1":rr[0]["K1"],
          }

    result={"schedules":SCHEDULES,"aggregate":agg,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)


if __name__=="__main__":
    main()
