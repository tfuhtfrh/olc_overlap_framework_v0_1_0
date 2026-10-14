"""Combined standard-SQA hybrid validation with source attribution.

All inner solves are vanilla OpenJij SQASampler SingleSpinFlip.
No custom updater. No ground oracle. No early stop.

Core design
-----------
- Explorer state is propagated ONLY from rank-free reads.
- Topology probes are activated only for degree-correct cyclic explorer states.
- A best-feasible incumbent is archived independently.
- Optional incumbent probes exploit the best HP without replacing the explorer.
- Every outer iteration uses exactly 8 standard SQA reads.

Methods
-------
none8:
    8 no-rank explorer reads.

cyclic_combo:
    irregular / HP / initial: 8 no-rank.
    degree-correct cyclic:
        4 no-rank explorer
        +2 reachability probes
        +2 alternative-rank probes.

combo_inc2:
    irregular / initial: 8 no-rank.
    cyclic before an incumbent exists:
        4 no-rank +2 reach +2 alternative-rank.
    cyclic after an incumbent exists:
        4 no-rank +1 reach +1 alternative-rank +2 incumbent probes.
    HP explorer after incumbent exists:
        6 no-rank +2 incumbent probes.

combo_inc2_reach:
    same as combo_inc2, except the two incumbent probes use a reachability
    Hamiltonian built from the incumbent HP rather than H0.

Incumbent source attribution
----------------------------
Every candidate carries a source label:
    no_rank, reach, rank, incumbent_h0, incumbent_reach.

Whenever the incumbent improves, the source label is recorded. This allows
ground discoveries to be attributed to the branch that actually produced them.

Fixed resources:
- 16 outer iterations;
- 8 reads/outer;
- beta=5, gamma=1, P=8, 1400 sweeps/read;
- 12 zero-start and 12 random-start seeds per method in screening.
"""
from __future__ import annotations

import json, random
from pathlib import Path

import benchmark_chm13_rank_expression_screening as rs
from benchmark_chm13_reachability_rank import reachability_bad_edges
from benchmark_chm13_dual_track_rank_hybrid import choose_explorer
from benchmark_chm13_projected_edge_hybrid import graph_metrics, project_rank

OUT=Path("debug/qubo/chm13_combined_hybrid_validation_20260930.json")
GROUND=1_343_093
A=0.02
OUTER=16
SEEDS=tuple(202610000+i for i in range(12))
STARTS=("zero","random")
METHODS=("none8","cyclic_combo","combo_inc2","combo_inc2_reach")


def reach_bqm(rids,ep,cost,incoming,outgoing,selected):
    b=rs.base_bqm(rids,ep,cost,incoming,outgoing)
    for e in reachability_bad_edges(rids,ep,selected):
        b.add_linear(e,A)
    return b


def sample_tagged(bqm,base,ep,rids,reward,selected,start,rng,seed,reads,source):
    cc,dt=rs.sample_branch(
      bqm,base,ep,rids,reward,selected,start,rng,seed,reads)
    for c in cc:
        c["source"]=source
    return cc,dt


def update_inc(inc,inc_sel,inc_src,inc_it,cands,it):
    for c in cands:
        m=c["metrics"]
        if not m["valid_path"]:
            continue
        sc=m["path_score"]
        if inc is None or sc>inc:
            inc=sc
            inc_sel=set(c["selected"])
            inc_src=c["source"]
            inc_it=it
    return inc,inc_sel,inc_src,inc_it


def explorer_type(rids,selected,reward,N):
    if selected is None:
        return "initial", None
    m=graph_metrics(rids,selected,project_rank(rids,selected),reward)
    if m["selected_edges"]!=N-1 or m["degree_conflicts"]!=0:
        return "irregular",m
    if m["cycle_count"]>0:
        return "cyclic",m
    return "hp",m


def run(method,start,seed):
    rids,ep,reward,cost,incoming,outgoing=rs.load_problem()
    N=len(rids)
    rng=random.Random(seed)
    base=rs.base_bqm(rids,ep,cost,incoming,outgoing)
    selected=None
    inc=None;inc_sel=None;inc_src=None;inc_it=None
    rows=[]
    improvements=[]

    for it in range(OUTER):
        state_type,pre=explorer_type(rids,selected,reward,N)
        bs=seed+10000*it
        prop=[];allc=[];branches=[];sec=0.0

        def add(cands,dt,label,propagate=False):
            nonlocal sec
            allc.extend(cands);sec+=dt;branches.append(label)
            if propagate:prop.extend(cands)

        if method=="none8":
            cc,dt=sample_tagged(base,base,ep,rids,reward,selected,start,rng,bs,8,"no_rank")
            add(cc,dt,"none:8",True)

        elif state_type in ("initial","irregular"):
            cc,dt=sample_tagged(base,base,ep,rids,reward,selected,start,rng,bs,8,"no_rank")
            add(cc,dt,"none:8",True)

        elif method=="cyclic_combo":
            if state_type=="cyclic":
                cc,dt=sample_tagged(base,base,ep,rids,reward,selected,start,rng,bs,4,"no_rank")
                add(cc,dt,"none:4",True)
                rb=reach_bqm(rids,ep,cost,incoming,outgoing,selected)
                cc,dt=sample_tagged(rb,base,ep,rids,reward,selected,start,rng,bs+4000,2,"reach")
                add(cc,dt,"reach:2",False)
                ranks=rs.alt_ranks(rids,selected,2,rng)
                for k,rk in enumerate(ranks):
                    qb=rs.rank_bqm(rids,ep,cost,incoming,outgoing,rk,A)
                    cc,dt=sample_tagged(qb,base,ep,rids,reward,selected,start,rng,bs+6000+1000*k,1,"rank")
                    add(cc,dt,f"rank{k}:1",False)
            else:
                cc,dt=sample_tagged(base,base,ep,rids,reward,selected,start,rng,bs,8,"no_rank")
                add(cc,dt,"none:8",True)

        elif method in ("combo_inc2","combo_inc2_reach"):
            if state_type=="cyclic" and inc_sel is None:
                cc,dt=sample_tagged(base,base,ep,rids,reward,selected,start,rng,bs,4,"no_rank")
                add(cc,dt,"none:4",True)
                rb=reach_bqm(rids,ep,cost,incoming,outgoing,selected)
                cc,dt=sample_tagged(rb,base,ep,rids,reward,selected,start,rng,bs+4000,2,"reach")
                add(cc,dt,"reach:2",False)
                ranks=rs.alt_ranks(rids,selected,2,rng)
                for k,rk in enumerate(ranks):
                    qb=rs.rank_bqm(rids,ep,cost,incoming,outgoing,rk,A)
                    cc,dt=sample_tagged(qb,base,ep,rids,reward,selected,start,rng,bs+6000+1000*k,1,"rank")
                    add(cc,dt,f"rank{k}:1",False)

            elif state_type=="cyclic" and inc_sel is not None:
                cc,dt=sample_tagged(base,base,ep,rids,reward,selected,start,rng,bs,4,"no_rank")
                add(cc,dt,"none:4",True)
                rb=reach_bqm(rids,ep,cost,incoming,outgoing,selected)
                cc,dt=sample_tagged(rb,base,ep,rids,reward,selected,start,rng,bs+4000,1,"reach")
                add(cc,dt,"reach:1",False)
                ranks=rs.alt_ranks(rids,selected,1,rng)
                qb=rs.rank_bqm(rids,ep,cost,incoming,outgoing,ranks[0],A)
                cc,dt=sample_tagged(qb,base,ep,rids,reward,selected,start,rng,bs+5000,1,"rank")
                add(cc,dt,"rank:1",False)

                if method=="combo_inc2_reach":
                    ib=reach_bqm(rids,ep,cost,incoming,outgoing,inc_sel)
                    src="incumbent_reach"
                else:
                    ib=base
                    src="incumbent_h0"
                cc,dt=sample_tagged(ib,base,ep,rids,reward,inc_sel,start,rng,bs+7000,2,src)
                add(cc,dt,f"{src}:2",False)

            elif state_type=="hp" and inc_sel is not None:
                cc,dt=sample_tagged(base,base,ep,rids,reward,selected,start,rng,bs,6,"no_rank")
                add(cc,dt,"none:6",True)
                if method=="combo_inc2_reach":
                    ib=reach_bqm(rids,ep,cost,incoming,outgoing,inc_sel)
                    src="incumbent_reach"
                else:
                    ib=base
                    src="incumbent_h0"
                cc,dt=sample_tagged(ib,base,ep,rids,reward,inc_sel,start,rng,bs+7000,2,src)
                add(cc,dt,f"{src}:2",False)

            else:
                cc,dt=sample_tagged(base,base,ep,rids,reward,selected,start,rng,bs,8,"no_rank")
                add(cc,dt,"none:8",True)
        else:
            raise ValueError(method)

        old=inc
        inc,inc_sel,inc_src,inc_it=update_inc(
          inc,inc_sel,inc_src,inc_it,allc,it)
        if inc!=old:
            improvements.append({"iteration":it,"score":inc,"source":inc_src})

        # Explorer is always rank-free.
        chosen=choose_explorer(prop,N)
        selected=set(chosen["selected"])
        met=chosen["metrics"]

        hp=[c["metrics"]["path_score"] for c in allc if c["metrics"]["valid_path"]]
        src_counts={}
        for c in allc:
            src_counts[c["source"]]=src_counts.get(c["source"],0)+1

        row={
          "method":method,"start":start,"seed":seed,"iteration":it,
          "state_type":state_type,"branches":branches,"source_counts":src_counts,
          "candidate_hp_count":len(hp),"candidate_best_hp":max(hp,default=None),
          "incumbent_score":inc,"incumbent_source":inc_src,
          "incumbent_iteration":inc_it,"seconds":sec,
          "explorer_base_energy":chosen["base_energy"],**met,
        }
        rows.append(row);print("ROW",json.dumps(row),flush=True)

    return rows,{
      "method":method,"start":start,"seed":seed,
      "incumbent_score":inc,"ground_ever":inc==GROUND,
      "ground_source":inc_src if inc==GROUND else None,
      "ground_iteration":inc_it if inc==GROUND else None,
      "improvements":improvements,
      "final_explorer_valid":rows[-1]["valid_path"],
      "final_explorer_score":rows[-1]["path_score"],
    }


def main():
    rows=[];summ=[]
    for method in METHODS:
      for start in STARTS:
        for si,bs in enumerate(SEEDS):
          seed=bs+10000*si
          rr,ss=run(method,start,seed)
          rows+=rr;summ.append(ss);print("SUMMARY",json.dumps(ss),flush=True)

    agg={}
    for method in METHODS:
      for start in STARTS:
        aa=[s for s in summ if s["method"]==method and s["start"]==start]
        inc=[s["incumbent_score"] for s in aa if s["incumbent_score"] is not None]
        grounds=[s for s in aa if s["ground_ever"]]
        src={}
        for s in grounds:
            src[s["ground_source"]]=src.get(s["ground_source"],0)+1
        gits=[s["ground_iteration"] for s in grounds if s["ground_iteration"] is not None]
        agg[f"{method}|{start}"]={
          "runs":len(aa),
          "hp_incumbent_runs":len(inc),
          "ground_ever":len(grounds),
          "ground_sources":src,
          "best_incumbent":max(inc,default=None),
          "median_incumbent":sorted(inc)[len(inc)//2] if inc else None,
          "median_ground_iteration":sorted(gits)[len(gits)//2] if gits else None,
          "final_explorer_hp":sum(s["final_explorer_valid"] for s in aa),
        }

    result={"A":A,"outer":OUTER,"reads_per_outer":8,
            "aggregate":agg,"summaries":summ,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":
    main()
