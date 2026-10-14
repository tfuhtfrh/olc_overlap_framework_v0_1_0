"""End-to-end structured SQA v3: short clusters, no whole-worldline topology moves.

Differences from v2
-------------------
- Stage 2 topology repair uses P=8 explicit Trotter slices and L=2 contiguous
  structured clusters with tuned beta/Gamma schedule.
- Stage 3 feasible refinement uses P=8 and L=4 clusters.
- No L=P whole-worldline structured update.
- No ground-score early stop.
- Final state is fixed slice 0, not the best slice and not postselected by
  feasibility/score.
- Ground score is used only after the run as an offline certificate.

Stage 2 schedule:
    beta=6.5, Gamma 3.0 -> 0.6, L=2.
Stage 3 schedule:
    beta=6.0, Gamma 4.0 -> 1.0, L=4.

A rapid generic single-spin quench was diagnosed separately and accepted zero
moves under the full degree/count BQM, so v3 treats the finite-Gamma slice
marginal as the search output and records fixed-slice readout. A hardware-style
final quench remains a separate schedule-control question.

This remains SQA-assisted hybrid optimization: the Hamiltonian is standard
classical QUBO plus transverse-field PIMC representation, while the MC proposal
kernel uses graph-structured moves.
"""
from __future__ import annotations

import json, math, random
from pathlib import Path

import benchmark_chm13_end_to_end_structured_sqa as base
import benchmark_chm13_end_to_end_structured_sqa_v2 as v2
import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_endpoint_transfer_mc import templates_by_shared_endpoint
from diagnose_chm13_move_basis_proposal import applicable

OUT=Path("debug/qubo/chm13_end_to_end_structured_sqa_v3_20260930.json")
P=8
STAGE2_ATTEMPTS=4000
STAGE3_ATTEMPTS=5000
S2_BETA=6.5
S2_G0=3.0
S2_G1=0.6
S2_L=2
S3_BETA=6.0
S3_G0=4.0
S3_G1=1.0
S3_L=4


def sched(beta,g0,g1,t,T):
    f=t/max(1,T-1)
    gamma=g0*((g1/g0)**f)
    x=max(1e-12,min(50.0,beta*gamma/P))
    K=-0.5*math.log(max(1e-300,math.tanh(x)))
    return gamma,K


def cluster(start,L):
    return tuple((start+d)%P for d in range(L))


def dtrot(slices,C,flips,K):
    C=set(C);d=0.0
    for m in range(P):
        n=(m+1)%P
        if (m in C)==(n in C):continue
        fm=m in C;fn=n in C
        for i in flips:
            old=(slices[m][i]!=slices[n][i])
            a=(1-slices[m][i]) if fm else slices[m][i]
            b=(1-slices[n][i]) if fn else slices[n][i]
            new=(a!=b)
            d += 2.0*K*(int(new)-int(old))
    return d


def flip_cluster(slices,C,flips):
    for m in C:
        for i in flips:slices[m][i]^=1


def edges_from_x(ep,x):
    return {ep[i] for i,v in enumerate(x) if v}


def metric(rids,ep,reward,x):
    s=edges_from_x(ep,x)
    return base.state_metrics(rids,s,reward)


def cycle_penalty_ids(method,st,cycles,exactA):
    if not cycles:return 0.0
    if method=="exact_cut":
        return exactA*sum(all(i in st for i in cyc) for cyc in cycles)
    if method=="rank_ensemble":
        Lmax=max(len(c) for c in cycles)
        Atotal=min(4.0,Lmax*exactA)
        return sum(Atotal*sum(i in st for i in cyc)/len(cyc) for cyc in cycles)
    return 0.0


def stage2_short_cluster(
    method,rids,ep,reward,selected,abqm,r2,r3,exactA,seed
):
    idx={e:i for i,e in enumerate(ep)}
    x0=[1 if e in selected else 0 for e in ep]
    slices=[x0[:] for _ in range(P)]
    cycles_edges=base.current_cycles(rids,selected)
    cycles=[tuple(idx[e] for e in cyc) for cyc in cycles_edges]
    heads,tails=templates_by_shared_endpoint(ep)
    rng=random.Random(seed)
    accepted=eligible=0
    any_hp_ever=False

    for t in range(STAGE2_ATTEMPTS):
        gamma,K=sched(S2_BETA,S2_G0,S2_G1,t,STAGE2_ATTEMPTS)
        C=cluster(rng.randrange(P),S2_L)
        z=rng.random();fam=None;flips=None

        if z<0.25:
            pair=rng.choice(heads);fam="sink_transfer"
            if all(slices[m][pair[0]]!=slices[m][pair[1]] for m in C):
                flips=pair
        elif z<0.50:
            pair=rng.choice(tails);fam="source_transfer"
            if all(slices[m][pair[0]]!=slices[m][pair[1]] for m in C):
                flips=pair
        elif z<0.75:
            tpl=rng.choice(r2);fam="r2"
            ff=None;ok=True
            for m in C:
                st=frozenset(i for i,v in enumerate(slices[m]) if v)
                nb=applicable(st,tpl)
                if nb is None:ok=False;break
                cur=tuple(sorted(st.symmetric_difference(nb)))
                if ff is None:ff=cur
                elif ff!=cur:ok=False;break
            if ok:flips=ff
        else:
            tpl=rng.choice(r3);fam="r3"
            ff=None;ok=True
            for m in C:
                st=frozenset(i for i,v in enumerate(slices[m]) if v)
                nb=applicable(st,tpl)
                if nb is None:ok=False;break
                cur=tuple(sorted(st.symmetric_difference(nb)))
                if ff is None:ff=cur
                elif ff!=cur:ok=False;break
            if ok:flips=ff

        if not flips:continue

        # Candidate must remain fixed-cardinality and degree-correct in every
        # affected slice. This keeps Stage 2 on its intended topology manifold.
        new_states=[];ok=True
        for m in C:
            y=slices[m][:]
            for i in flips:y[i]^=1
            mm=metric(rids,ep,reward,y)
            if sum(y)!=len(rids)-1 or mm["degree_conflicts"]!=0:
                ok=False;break
            new_states.append((m,y))
        if not ok:continue

        eligible+=1
        dclass=0.0
        for m,y in new_states:
            oldst=frozenset(i for i,v in enumerate(slices[m]) if v)
            newst=frozenset(i for i,v in enumerate(y) if v)
            dclass += abqm.delta_flipset(slices[m],tuple(flips))
            dclass += (
              cycle_penalty_ids(method,newst,cycles,exactA)
              -cycle_penalty_ids(method,oldst,cycles,exactA)
            )
        da=(S2_BETA/P)*dclass+dtrot(slices,C,flips,K)
        if da<=0 or rng.random()<math.exp(-min(700.0,da)):
            flip_cluster(slices,C,flips);accepted+=1
            if any(metric(rids,ep,reward,sl)["valid_path"] for sl in slices):
                any_hp_ever=True

    final_metrics=[metric(rids,ep,reward,sl) for sl in slices]
    return slices[0],{
      "accepted":accepted,"eligible":eligible,
      "any_hp_ever":any_hp_ever,
      "final_hp_slices":sum(m["valid_path"] for m in final_metrics),
      "slice0_valid_path":final_metrics[0]["valid_path"],
      "slice0_score":final_metrics[0]["path_score"],
      "beta":S2_BETA,"gamma0":S2_G0,"gamma1":S2_G1,"L":S2_L,
    }


def apply_r2_pair(st,t1,t2):
    f1=set(t1[0]+t1[1]);f2=set(t2[0]+t2[1])
    if f1&f2:return None
    n1=applicable(st,t1);n2=applicable(st,t2)
    if n1 is None or n2 is None:return None
    return frozenset(set(st).symmetric_difference(f1|f2))


def stage3_short_cluster(rids,ep,reward,selected,abqm,r2,r3,seed):
    idx={e:i for i,e in enumerate(ep)}
    x0=[1 if e in selected else 0 for e in ep]
    slices=[x0[:] for _ in range(P)]
    rng=random.Random(seed)
    pairs=[]
    for a in range(len(r2)):
        fa=set(r2[a][0]+r2[a][1])
        for b in range(a+1,len(r2)):
            fb=set(r2[b][0]+r2[b][1])
            if not fa&fb:pairs.append((a,b))
    accepted=eligible=0
    best_score_seen=max(metric(rids,ep,reward,sl)["path_score"] or -1 for sl in slices)

    for t in range(STAGE3_ATTEMPTS):
        gamma,K=sched(S3_BETA,S3_G0,S3_G1,t,STAGE3_ATTEMPTS)
        C=cluster(rng.randrange(P),S3_L)
        use_r3=rng.random()<0.6
        ff=None;ok=True
        for m in C:
            st=frozenset(i for i,v in enumerate(slices[m]) if v)
            if use_r3:
                nb=applicable(st,rng.choice(r3)) if m==C[0] else None
                # Need one static template for all slices; handled below.
                ok=False
                break
        if use_r3:
            tpl=rng.choice(r3);ff=None;ok=True
            for m in C:
                st=frozenset(i for i,v in enumerate(slices[m]) if v)
                nb=applicable(st,tpl)
                if nb is None:ok=False;break
                cur=tuple(sorted(st.symmetric_difference(nb)))
                if ff is None:ff=cur
                elif ff!=cur:ok=False;break
        else:
            a,b=rng.choice(pairs);ff=None;ok=True
            for m in C:
                st=frozenset(i for i,v in enumerate(slices[m]) if v)
                nb=apply_r2_pair(st,r2[a],r2[b])
                if nb is None:ok=False;break
                cur=tuple(sorted(st.symmetric_difference(nb)))
                if ff is None:ff=cur
                elif ff!=cur:ok=False;break
        if not ok or not ff:continue

        # Hard feasible-path manifold retained in Stage 3 for this ablation.
        newm=[];valid=True
        for m in C:
            y=slices[m][:]
            for i in ff:y[i]^=1
            mm=metric(rids,ep,reward,y)
            if not mm["valid_path"]:
                valid=False;break
            newm.append((m,y,mm))
        if not valid:continue

        eligible+=1
        dc=sum(abqm.delta_flipset(slices[m],ff) for m,_,_ in newm)
        da=(S3_BETA/P)*dc+dtrot(slices,C,ff,K)
        if da<=0 or rng.random()<math.exp(-min(700.0,da)):
            flip_cluster(slices,C,ff);accepted+=1
            best_score_seen=max(
              best_score_seen,
              max((mm["path_score"] or -1) for _,_,mm in newm)
            )

    fm=[metric(rids,ep,reward,sl) for sl in slices]
    return slices[0],{
      "accepted":accepted,"eligible":eligible,
      "final_ground_slices":sum(m["path_score"]==base.GROUND for m in fm),
      "slice0_score":fm[0]["path_score"],
      "slice0_ground":fm[0]["path_score"]==base.GROUND,
      "best_score_seen_posthoc":best_score_seen,
      "beta":S3_BETA,"gamma0":S3_G0,"gamma1":S3_G1,"L":S3_L,
    }


def run(method,start_mode,seed):
    rids,ep,reward,cost,incoming,outgoing=base.load_problem();N=len(rids)
    rng=random.Random(seed)
    selected=None
    r2=fc.build_reconnect_templates(ep);r3=base.build_r3_templates(ep)
    rankfree=base.array_rankfree(ep,cost,incoming,outgoing)
    rows=[];s2_diags=[];s3_diag=None

    for it in range(base.OUTER):
        prev=base.state_metrics(rids,selected,reward) if selected is not None else None
        cycles=base.current_cycles(rids,selected)
        gated=(
          selected is not None and len(selected)==N-1 and prev["degree_conflicts"]==0
          and prev["cycle_count"]>0
        )
        exactA=base.local_repair_scale(rids,ep,reward,selected,rankfree,r2,r3) if gated else 0.0
        dummy={r:0 for r in rids}
        bqm=base.build_bqm_count(
          ep,cost,incoming,outgoing,dummy,N,
          degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)
        meta=[];applied_A=0.0
        if gated and method=="exact_cut":
            applied_A=exactA;meta=base.add_exact_cuts(bqm,cycles,applied_A)
        elif gated and method=="rank_ensemble":
            L=max(len(c) for c in cycles)
            applied_A=min(4.0,L*exactA)
            meta=base.add_rank_ensemble(bqm,cycles,applied_A)

        selected,sec,nv,nq=base.sample_openjij(
          bqm,ep,selected,meta,start_mode,rng,seed+1000*it,N)
        met=base.state_metrics(rids,selected,reward)

        s2=None
        if len(selected)==N-1 and met["degree_conflicts"]==0 and met["cycle_count"]>0:
            localA=base.local_repair_scale(rids,ep,reward,selected,rankfree,r2,r3)
            x0,s2=stage2_short_cluster(
              method,rids,ep,reward,selected,rankfree,r2,r3,
              localA,seed+500000+it)
            selected=edges_from_x(ep,x0);s2_diags.append(s2)
            met=base.state_metrics(rids,selected,reward)

        rows.append({
          "method":method,"start":start_mode,"seed":seed,"iteration":it,
          "seconds":sec,"variables":nv,"quadratic":nq,
          "stage2":s2,**met,
        })

        if met["valid_path"]:
            x0,s3_diag=stage3_short_cluster(
              rids,ep,reward,selected,rankfree,r2,r3,seed+900000+it)
            selected=edges_from_x(ep,x0)
            break

    final=base.state_metrics(rids,selected,reward)
    return rows,{
      "method":method,"start":start_mode,"seed":seed,
      "stage2_episodes":len(s2_diags),
      "stage2_slice0_hp_hits":sum(d["slice0_valid_path"] for d in s2_diags),
      "stage3":s3_diag,
      "final_valid_path":final["valid_path"],
      "final_score":final["path_score"],
      "ground_hit":final["path_score"]==base.GROUND,
      "outer_iterations":len(rows),
    }


def main():
    # Validation batch: both acyclicity choices, zero/random, six seeds.
    old_methods,old_starts,old_seeds=base.METHODS,base.STARTS,base.SEEDS
    try:
        base.METHODS=("exact_cut","rank_ensemble")
        base.STARTS=("zero","random")
        base.SEEDS=(202609901,202609902,202609903,202609904,202609905,202609906)
        rows=[];summ=[]
        for method in base.METHODS:
            for start in base.STARTS:
                for si,b in enumerate(base.SEEDS):
                    seed=b+10000*si
                    rr,ss=run(method,start,seed)
                    rows+=rr;summ.append(ss)
                    print("SUMMARY",json.dumps(ss),flush=True)
        agg={}
        for method in base.METHODS:
            for start in base.STARTS:
                aa=[s for s in summ if s["method"]==method and s["start"]==start]
                agg[f"{method}|{start}"]={
                  "runs":len(aa),
                  "final_hp_hits":sum(s["final_valid_path"] for s in aa),
                  "ground_hits":sum(s["ground_hit"] for s in aa),
                  "stage2_episodes":sum(s["stage2_episodes"] for s in aa),
                  "stage2_slice0_hp_hits":sum(s["stage2_slice0_hp_hits"] for s in aa),
                  "stage3_runs":sum(s["stage3"] is not None for s in aa),
                  "stage3_slice0_ground_hits":sum(
                    bool(s["stage3"] and s["stage3"]["slice0_ground"]) for s in aa),
                  "best_final_score":max((s["final_score"] or -1 for s in aa),default=None),
                }
        result={"aggregate":agg,"summaries":summ,"rows":rows}
        OUT.parent.mkdir(parents=True,exist_ok=True)
        OUT.write_text(json.dumps(result,indent=2)+"\n")
        print("AGGREGATE",json.dumps(agg),flush=True)
    finally:
        base.METHODS,base.STARTS,base.SEEDS=old_methods,old_starts,old_seeds


if __name__=="__main__":
    main()
