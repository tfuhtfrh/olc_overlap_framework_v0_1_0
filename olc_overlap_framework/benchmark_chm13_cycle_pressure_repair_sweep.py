"""Isolated cycle-pressure scale sweep for structured whole-worldline repair.

Generate three degree-correct cyclic states from one zero-start OpenJij SQA
iteration. For each state, estimate the local cycle-repair scale A0 from the
cheapest R2/R3 cycle-reducing move, then compare:

- none
- exact_cut at 0.5*A0, 1*A0, 2*A0
- rank_ensemble at 0.5*A0, 1*A0, 2*A0

Multiple repair RNG seeds are used per state. This isolates the effect of
acyclicity pressure from the global SQA and feasible-refinement phases.
"""
from __future__ import annotations
import json, random
from pathlib import Path

import benchmark_chm13_end_to_end_structured_sqa as base
import benchmark_chm13_end_to_end_structured_sqa_v2 as v2

OUT=Path("debug/qubo/chm13_cycle_pressure_repair_sweep_20260929.json")
STATE_SEEDS=(202609291,202619292,202629293)
REPS=8
FACTORS=(0.5,1.0,2.0)


def make_state(seed):
    rids,ep,reward,cost,incoming,outgoing=base.load_problem();N=len(rids)
    rng=random.Random(seed)
    dummy={r:0 for r in rids}
    bqm=base.build_bqm_count(
      ep,cost,incoming,outgoing,dummy,N,
      degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)
    sel,sec,nv,nq=base.sample_openjij(
      bqm,ep,None,[],"zero",rng,seed,N)
    return (rids,ep,reward,cost,incoming,outgoing,sel)


def main():
    old=v2.REPAIR_ATTEMPTS
    v2.REPAIR_ATTEMPTS=3000
    rows=[]
    try:
        for si,seed in enumerate(STATE_SEEDS):
            rids,ep,reward,cost,incoming,outgoing,sel=make_state(seed)
            met=base.state_metrics(rids,sel,reward)
            r2=base.fc.build_reconnect_templates(ep);r3=base.build_r3_templates(ep)
            abqm=base.array_rankfree(ep,cost,incoming,outgoing)
            A0=base.local_repair_scale(rids,ep,reward,sel,abqm,r2,r3)
            state_info={
              "state_seed":seed,"selected_edges":len(sel),
              "degree_conflicts":met["degree_conflicts"],
              "cycle_count":met["cycle_count"],"A0":A0,
            }
            print("STATE",json.dumps(state_info),flush=True)
            configs=[("none",0.0)]
            for f in FACTORS:
                configs.append(("exact_cut",A0*f))
                configs.append(("rank_ensemble",A0*f))
            for method,A in configs:
                for rep in range(REPS):
                    final,diag=v2.cyclic_worldline_repair(
                      method,rids,ep,reward,set(sel),abqm,r2,r3,A,
                      seed+100000*rep+1000*int(A*10000)+(0 if method=="none" else (1 if method=="exact_cut" else 2)))
                    row={
                      **state_info,"method":method,
                      "factor":(A/A0 if A0 and method!="none" else 0.0),
                      "A_used":A,"rep":rep,**diag,
                    }
                    rows.append(row);print("ROW",json.dumps(row),flush=True)
    finally:
        v2.REPAIR_ATTEMPTS=old

    agg={}
    keys=sorted({(r["method"],r["factor"]) for r in rows})
    for method,factor in keys:
        rr=[r for r in rows if r["method"]==method and r["factor"]==factor]
        hits=[r for r in rr if r["hp_hit"]]
        attempts=[r["attempts"] for r in hits]
        agg[f"{method}|x{factor:g}"]={
          "runs":len(rr),"hp_hits":len(hits),"P_hp":len(hits)/len(rr),
          "median_attempts":sorted(attempts)[len(attempts)//2] if attempts else None,
          "mean_accepted":sum(r["accepted"] for r in rr)/len(rr),
          "mean_local_A0":sum(r["local_exact_scale"] for r in rr)/len(rr),
        }
    result={"aggregate":agg,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":main()
