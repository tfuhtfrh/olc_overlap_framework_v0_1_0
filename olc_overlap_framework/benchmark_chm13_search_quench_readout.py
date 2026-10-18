"""Structured short-cluster search followed by generic single-bit quench.

Purpose
-------
Test whether tuned finite-Gamma Stage-2/3 search can remove the need for
whole-worldline updates *and* survive a topology-agnostic final quench.

Search:
- Stage 2 representative cyclic checkpoint:
    beta=6.5, Gamma 3.0->0.6, L=2, same-head endpoint-transfer proposals.
- Stage 3 runner-up:
    beta=6.0, Gamma 4.0->1.0, L=4, R3 proposals.

Quench:
- no endpoint/R2/R3 proposals;
- generic single-bit path-integral Metropolis updates only;
- full-space weight+degree+count BQM;
- Gamma floor -> 0.03 at fixed beta;
- compare rapid/medium/slow quench lengths.

P=8 throughout. Final readout statistics report:
- target slices before quench;
- target slices after quench;
- random-slice target probability = target_slices / P;
- all-slice consensus.

This is an SQA/PIMC sampler diagnostic, not real-time QA dynamics.
"""
from __future__ import annotations

import json, math, random
from pathlib import Path

import benchmark_chm13_fixed_cardinality_kernels as fc
import benchmark_chm13_schedule_cluster_freezing as diag
from benchmark_chm13_projected_edge_hybrid import (
    load_problem, graph_metrics, project_rank, build_bqm_count,
)
from benchmark_chm13_endpoint_transfer_mc import templates_by_shared_endpoint
from diagnose_chm13_move_basis_proposal import replay_stuck_checkpoint, build_r3_templates, applicable
from diagnose_chm13_runnerup_ground_geometry import normalized_paths, score, state

OUT=Path("debug/qubo/chm13_search_quench_readout_20260930.json")
P=8
GROUND=1_343_093
RUNNER=1_338_209
SEARCH_ATTEMPTS=4000
QUENCH_ATTEMPTS=(250,1000,4000)
SEEDS=tuple(202609800+i for i in range(24))


def schedule_vals(beta,g0,g1,frac):
    gamma=g0*((g1/g0)**frac)
    x=max(1e-12,min(50.0,beta*gamma/P))
    K=-0.5*math.log(max(1e-300,math.tanh(x)))
    return gamma,K


def cluster_indices(start,L):
    return tuple((start+d)%P for d in range(L))


def trotter_delta_cluster(slices,cluster,flips,K):
    C=set(cluster);d=0.0
    for m in range(P):
        n=(m+1)%P
        if (m in C)==(n in C):continue
        fm=(m in C);fn=(n in C)
        for i in flips:
            old=(slices[m][i]!=slices[n][i])
            a=(1-slices[m][i]) if fm else slices[m][i]
            b=(1-slices[n][i]) if fn else slices[n][i]
            new=(a!=b)
            d += 2.0*K*(int(new)-int(old))
    return d


def apply_cluster(slices,cluster,flips):
    for m in cluster:
        for i in flips:slices[m][i]^=1


def selected(ep,x):
    return {ep[i] for i,v in enumerate(x) if v}


def stage2_ctx():
    rids,ep,reward,cost,incoming,outgoing,sel,ranks,op=replay_stuck_checkpoint()
    bqm_fixed=fc.build_bqm_fixed_cardinality(
        ep,cost,incoming,outgoing,project_rank(rids,sel),
        degree_conflict_penalty=288.0,order_penalty=0.0)
    abqm_fixed=fc.ArrayBQM.from_bqm(bqm_fixed,ep)
    dummy={r:0 for r in rids}
    bqm_full=build_bqm_count(
        ep,cost,incoming,outgoing,dummy,len(rids),
        degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)
    abqm_full=fc.ArrayBQM.from_bqm(bqm_full,ep)
    x0=[1 if e in sel else 0 for e in ep]
    heads,_=templates_by_shared_endpoint(ep)
    return rids,ep,reward,abqm_fixed,abqm_full,x0,heads


def stage3_ctx():
    rids,ep,reward,cost,incoming,outgoing=load_problem();epset=set(ep)
    rows=[]
    for p in normalized_paths():
        sc=score(p,reward)
        if sc in (RUNNER,GROUND):
            st=state(p)
            if st<=epset:rows.append((sc,st))
    runner=next(st for sc,st in rows if sc==RUNNER)
    ground=next(st for sc,st in rows if sc==GROUND)
    bqm_fixed=fc.build_bqm_fixed_cardinality(
        ep,cost,incoming,outgoing,project_rank(rids,runner),
        degree_conflict_penalty=288.0,order_penalty=0.0)
    abqm_fixed=fc.ArrayBQM.from_bqm(bqm_fixed,ep)
    dummy={r:0 for r in rids}
    bqm_full=build_bqm_count(
        ep,cost,incoming,outgoing,dummy,len(rids),
        degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)
    abqm_full=fc.ArrayBQM.from_bqm(bqm_full,ep)
    x0=[1 if e in runner else 0 for e in ep]
    idx={e:i for i,e in enumerate(ep)}
    ground_ids=frozenset(idx[e] for e in ground)
    r3=build_r3_templates(ep)
    return rids,ep,reward,abqm_fixed,abqm_full,x0,r3,ground_ids


def target_flags(kind,ctx,slices):
    if kind=="stage2":
        rids,ep,reward,*_=ctx
        out=[]
        for sl in slices:
            s=selected(ep,sl)
            m=graph_metrics(rids,s,project_rank(rids,s),reward)
            out.append(bool(m["valid_path"]))
        return out
    rids,ep,reward,abf,abfull,x0,r3,ground_ids=ctx
    return [frozenset(i for i,v in enumerate(sl) if v)==ground_ids for sl in slices]


def structured_search(kind,ctx,seed):
    if kind=="stage2":
        rids,ep,reward,abqm,abfull,x0,templates=ctx
        beta,g0,g1,L=6.5,3.0,0.6,2
    else:
        rids,ep,reward,abqm,abfull,x0,templates,ground_ids=ctx
        beta,g0,g1,L=6.0,4.0,1.0,4

    slices=[x0[:] for _ in range(P)]
    rng=random.Random(seed)
    eligible=accepted=0
    for t in range(SEARCH_ATTEMPTS):
        frac=t/max(1,SEARCH_ATTEMPTS-1)
        gamma,K=schedule_vals(beta,g0,g1,frac)
        cluster=cluster_indices(rng.randrange(P),L)

        if kind=="stage2":
            i,j=rng.choice(templates)
            if any(slices[m][i]==slices[m][j] for m in cluster):
                continue
            flips=(i,j)
        else:
            tpl=rng.choice(templates);flips=None;ok=True
            for m in cluster:
                st=frozenset(i for i,v in enumerate(slices[m]) if v)
                nb=applicable(st,tpl)
                if nb is None:
                    ok=False;break
                ff=tuple(sorted(st.symmetric_difference(nb)))
                if flips is None:flips=ff
                elif ff!=flips:
                    ok=False;break
            if not ok:continue

        eligible+=1
        dc=sum(abqm.delta_flipset(slices[m],flips) for m in cluster)
        dt=trotter_delta_cluster(slices,cluster,flips,K)
        da=(beta/P)*dc+dt
        if da<=0 or rng.random()<math.exp(-min(700.0,da)):
            apply_cluster(slices,cluster,flips);accepted+=1

    flags=target_flags(kind,ctx,slices)
    return slices,{
      "beta":beta,"gamma_start":g0,"gamma_floor":g1,"L":L,
      "eligible":eligible,"accepted":accepted,
      "target_slices":sum(flags),"all_target":all(flags),
    }


def single_spin_quench(kind,ctx,slices_in,seed,q_attempts):
    if kind=="stage2":
        rids,ep,reward,abfixed,abqm,x0,templates=ctx
        beta,g0=6.5,0.6
    else:
        rids,ep,reward,abfixed,abqm,x0,templates,ground_ids=ctx
        beta,g0=6.0,1.0
    g1=0.03
    slices=[sl[:] for sl in slices_in]
    rng=random.Random(seed)
    accepted=0
    for t in range(q_attempts):
        frac=t/max(1,q_attempts-1)
        gamma,K=schedule_vals(beta,g0,g1,frac)
        m=rng.randrange(P);i=rng.randrange(len(slices[m]))
        dc=abqm.delta_flipset(slices[m],(i,))
        dt=diag.trotter_delta_cluster(slices,(m,),(i,),K)
        da=(beta/P)*dc+dt
        if da<=0 or rng.random()<math.exp(-min(700.0,da)):
            slices[m][i]^=1;accepted+=1

    flags=target_flags(kind,ctx,slices)
    # Fixed slice 0 is a non-postselected readout proxy.
    return slices,{
      "attempts":q_attempts,"accepted":accepted,
      "target_slices":sum(flags),
      "random_slice_target_probability":sum(flags)/P,
      "slice0_target":bool(flags[0]),
      "all_target":all(flags),
    }


def main():
    contexts={"stage2":stage2_ctx(),"stage3":stage3_ctx()}
    rows=[]
    for kind in ("stage2","stage3"):
        for seed in SEEDS:
            search_slices,sd=structured_search(kind,contexts[kind],seed)
            before=target_flags(kind,contexts[kind],search_slices)
            for qa in QUENCH_ATTEMPTS:
                _,qd=single_spin_quench(
                    kind,contexts[kind],search_slices,seed+1000000+qa,qa)
                row={
                  "kind":kind,"seed":seed,
                  "search":sd,
                  "pre_quench_target_slices":sum(before),
                  "pre_quench_slice0_target":bool(before[0]),
                  "quench":qd,
                }
                rows.append(row);print("ROW",json.dumps(row),flush=True)

    agg={}
    for kind in ("stage2","stage3"):
        for qa in QUENCH_ATTEMPTS:
            rr=[r for r in rows if r["kind"]==kind and r["quench"]["attempts"]==qa]
            agg[f"{kind}|quench={qa}"]={
              "runs":len(rr),
              "pre_any_hits":sum(r["pre_quench_target_slices"]>0 for r in rr),
              "pre_mean_target_slices":sum(r["pre_quench_target_slices"] for r in rr)/len(rr),
              "post_any_hits":sum(r["quench"]["target_slices"]>0 for r in rr),
              "post_slice0_hits":sum(r["quench"]["slice0_target"] for r in rr),
              "post_all_hits":sum(r["quench"]["all_target"] for r in rr),
              "post_mean_target_slices":sum(r["quench"]["target_slices"] for r in rr)/len(rr),
              "mean_random_slice_target_probability":sum(
                    r["quench"]["random_slice_target_probability"] for r in rr)/len(rr),
              "mean_quench_accepts":sum(r["quench"]["accepted"] for r in rr)/len(rr),
            }
    result={"aggregate":agg,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)


if __name__=="__main__":
    main()
