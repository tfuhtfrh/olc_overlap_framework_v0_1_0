"""End-to-end structured SQA v2: add cyclic whole-worldline topology repair.

This extends benchmark_chm13_end_to_end_structured_sqa.py by inserting a
structured repair episode whenever OpenJij SQA reaches the exact
fixed-cardinality/degree manifold but remains cyclic.

Repair proposals (all static symmetric involutions):
- same-head sink-transfer 2-bit swap
- same-tail source-transfer 2-bit swap
- R2 reconnect
- R3 reconnect

The repair episode uses the same weak current-cycle bias as the global method:
- exact_cut: effective penalty A for retaining a complete detected cycle
- rank_ensemble: A/L per selected edge of the detected cycle

All structured moves are whole-worldline updates, so their Trotter interaction
change is exactly zero. Stop the repair episode when a Hamilton path is reached,
then use the feasible R3/R2-pair refinement from v1.
"""
from __future__ import annotations

import json, math, random
from pathlib import Path

import benchmark_chm13_end_to_end_structured_sqa as base
from benchmark_chm13_endpoint_transfer_mc import templates_by_shared_endpoint
from diagnose_chm13_move_basis_proposal import applicable

OUT=Path("debug/qubo/chm13_end_to_end_structured_sqa_v2_20260929.json")
REPAIR_ATTEMPTS=6000
BETA_REPAIR=4.0


def toggle_pair(st,pair):
    i,j=pair
    if (i in st)==(j in st):return None
    return frozenset(set(st).symmetric_difference((i,j)))


def cycle_penalty(method,st,cycles,exactA):
    if not cycles:return 0.0
    if method=="exact_cut":
        return exactA*sum(all(ei in st for ei in cyc) for cyc in cycles)
    if method=="rank_ensemble":
        Lmax=max(len(c) for c in cycles)
        Atotal=min(4.0,Lmax*exactA)
        return sum(Atotal*sum(ei in st for ei in cyc)/len(cyc) for cyc in cycles)
    return 0.0


def cyclic_worldline_repair(method,rids,ep,reward,selected,abqm,r2,r3,exactA,seed):
    idx={e:i for i,e in enumerate(ep)}
    cycles_edges=base.current_cycles(rids,selected)
    cycles=[tuple(idx[e] for e in cyc) for cyc in cycles_edges]
    st=frozenset(idx[e] for e in selected)
    x=[1 if i in st else 0 for i in range(len(ep))]
    heads,tails=templates_by_shared_endpoint(ep)
    rng=random.Random(seed)
    accepted=eligible=0
    accepted_by_family={"sink_transfer":0,"source_transfer":0,"r2":0,"r3":0}
    initial=base.state_metrics(rids,selected,reward)
    effective_cycle_pressure=(
      exactA if method=="exact_cut" else
      (min(4.0,max((len(c) for c in cycles_edges),default=0)*exactA)
       if method=="rank_ensemble" else 0.0)
    )

    for t in range(REPAIR_ATTEMPTS):
        met=base.state_metrics(rids,{ep[i] for i in st},reward)
        if met["valid_path"]:
            return {ep[i] for i in st},{
              "hp_hit":True,"attempts":t,"accepted":accepted,"eligible":eligible,
              "initial_cycles":initial["cycle_count"],"final_cycles":0,
              "final_score":met["path_score"],
              "method":method,"local_exact_scale":exactA,
              "effective_cycle_pressure":effective_cycle_pressure,
              "accepted_by_family":accepted_by_family,
            }

        z=rng.random()
        if z<0.25:
            nb=toggle_pair(st,rng.choice(heads));fam="sink_transfer"
        elif z<0.50:
            nb=toggle_pair(st,rng.choice(tails));fam="source_transfer"
        elif z<0.75:
            nb=applicable(st,rng.choice(r2));fam="r2"
        else:
            nb=applicable(st,rng.choice(r3));fam="r3"
        if nb is None:continue

        ns={ep[i] for i in nb}
        nm=base.state_metrics(rids,ns,reward)
        # Stay on the fixed-cardinality, degree-correct manifold.
        if nm["degree_conflicts"]!=0 or len(ns)!=len(rids)-1:continue
        eligible+=1
        flips=tuple(sorted(st.symmetric_difference(nb)))
        de=abqm.delta_flipset(x,flips)
        de+=cycle_penalty(method,nb,cycles,exactA)-cycle_penalty(method,st,cycles,exactA)
        # Whole-worldline: Delta S_Trotter=0.
        if de<=0 or rng.random()<math.exp(-min(700.0,BETA_REPAIR*de)):
            for i in flips:x[i]^=1
            st=nb;accepted+=1
            accepted_by_family[fam]+=1

    final={ep[i] for i in st}
    fm=base.state_metrics(rids,final,reward)
    return final,{
      "hp_hit":fm["valid_path"],"attempts":REPAIR_ATTEMPTS,
      "accepted":accepted,"eligible":eligible,
      "initial_cycles":initial["cycle_count"],"final_cycles":fm["cycle_count"],
      "final_score":fm["path_score"],
      "method":method,"local_exact_scale":exactA,
      "effective_cycle_pressure":effective_cycle_pressure,
      "accepted_by_family":accepted_by_family,
    }


def run(method,start_mode,seed):
    rids,ep,reward,cost,incoming,outgoing=base.load_problem();N=len(rids)
    rng=random.Random(seed)
    selected=None
    r2=base.fc.build_reconnect_templates(ep);r3=base.build_r3_templates(ep)
    rankfree=base.array_rankfree(ep,cost,incoming,outgoing)
    rows=[];first_hp=None;refine=None;repairs=[]

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
            applied_A=exactA
            meta=base.add_exact_cuts(bqm,cycles,applied_A)
        elif gated and method=="rank_ensemble":
            L=max(len(c) for c in cycles)
            applied_A=min(4.0,L*exactA)
            meta=base.add_rank_ensemble(bqm,cycles,applied_A)

        selected,sec,nv,nq=base.sample_openjij(
          bqm,ep,selected,meta,start_mode,rng,seed+1000*it,N)
        met=base.state_metrics(rids,selected,reward)

        repair=None
        # Immediately repair a degree-correct cyclic result with structured
        # whole-worldline moves and locally calibrated cycle pressure.
        if len(selected)==N-1 and met["degree_conflicts"]==0 and met["cycle_count"]>0:
            localA=base.local_repair_scale(rids,ep,reward,selected,rankfree,r2,r3)
            selected,repair=cyclic_worldline_repair(
              method,rids,ep,reward,selected,rankfree,r2,r3,
              localA,seed+500000+it)
            repairs.append(repair)
            met=base.state_metrics(rids,selected,reward)

        row={
          "method":method,"start":start_mode,"seed":seed,"iteration":it,
          "acyclicity_gated":gated,"acyclicity_A":applied_A,
          "variables":nv,"quadratic":nq,"seconds":sec,
          "repair":repair,**met,
        }
        rows.append(row);print("ROW",json.dumps(row),flush=True)

        if met["valid_path"]:
            first_hp=met["path_score"]
            selected,refine=base.feasible_worldline_refine(
              rids,ep,reward,selected,rankfree,r2,r3,seed+900000+it)
            final=base.state_metrics(rids,selected,reward)
            break
    else:
        final=base.state_metrics(rids,selected,reward)

    return rows,{
      "method":method,"start":start_mode,"seed":seed,
      "global_or_repair_found_hp":first_hp is not None,
      "first_hp_score":first_hp,
      "repair_episodes":len(repairs),
      "repair_hp_hits":sum(r["hp_hit"] for r in repairs),
      "refine":refine,
      "final_score":final["path_score"],
      "ground_hit":final["path_score"]==base.GROUND,
      "final_valid_path":final["valid_path"],
      "outer_iterations":len(rows),
    }


def main():
    rows=[];summ=[]
    for method in base.METHODS:
        for start in base.STARTS:
            for si,b in enumerate(base.SEEDS):
                seed=b+10000*si
                rr,ss=run(method,start,seed)
                rows+=rr;summ.append(ss);print("SUMMARY",json.dumps(ss),flush=True)
    agg={}
    for method in base.METHODS:
        for start in base.STARTS:
            aa=[s for s in summ if s["method"]==method and s["start"]==start]
            agg[f"{method}|{start}"]={
              "runs":len(aa),
              "hp_hits":sum(s["global_or_repair_found_hp"] for s in aa),
              "ground_hits":sum(s["ground_hit"] for s in aa),
              "repair_episodes":sum(s["repair_episodes"] for s in aa),
              "repair_hp_hits":sum(s["repair_hp_hits"] for s in aa),
              "refine_ground_hits":sum(bool(s["refine"] and s["refine"]["ground_hit"]) for s in aa),
              "best_final_score":max((s["final_score"] or -1 for s in aa),default=None),
            }
    result={"aggregate":agg,"summaries":summ,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":main()
