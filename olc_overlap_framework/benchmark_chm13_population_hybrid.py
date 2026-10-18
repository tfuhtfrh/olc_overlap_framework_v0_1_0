"""Population / multi-explorer standard-SQA hybrid for CHM13.

Motivation
----------
Single-explorer hybrids now find HPs very reliably but repeatedly converge to
the same runner-up basin. Reverse-SQA and penalty relaxation did not cross the
final runner-up -> ground barrier with vanilla SingleSpinFlip.

This benchmark tests whether keeping TWO rank-free explorer trajectories
preserves basin diversity without increasing the SQA read budget.

All inner solves:
- OpenJij SQASampler;
- SingleSpinFlip only;
- beta=5, gamma=1, P=8;
- 1400 sweeps/read.

Fixed resource budget:
- 8 standard reads / outer iteration;
- 16 outer iterations;
- 12 zero-start + 12 random-start seeds/method.

Methods
-------
single8:
    one explorer, 8 rank-free reads.

beam2_energy:
    two explorers, 4 rank-free reads each.
    Next explorers are the two lowest-H0 distinct rank-free samples.

beam2_diverse:
    two explorers, 4 rank-free reads each.
    Primary = lowest-H0 sample.
    Secondary = most Hamming-distant state from primary among the six
    lowest-H0 distinct samples (tie -> lower H0).

beam2_combo:
    two explorers, each gets:
      - if degree-correct cyclic: 2 no-rank propagation reads
        +1 reachability probe +1 alternative-rank probe;
      - otherwise: 4 no-rank propagation reads.
    Next explorers use the same diversity rule as beam2_diverse.
    Guided samples can update incumbent but never propagate explorers.

Best feasible HP is archived non-oracularly.
"""
from __future__ import annotations
import json, random
from pathlib import Path

import benchmark_chm13_rank_expression_screening as rs
from benchmark_chm13_reachability_rank import reachability_bad_edges
from benchmark_chm13_projected_edge_hybrid import graph_metrics,project_rank

OUT=Path("debug/qubo/chm13_population_hybrid_20260930.json")
GROUND=1_343_093
A=0.02
OUTER=16
SEEDS=tuple(202610300+i for i in range(12))
STARTS=("zero","random")
METHODS=("single8","beam2_energy","beam2_diverse","beam2_combo")


def reach_bqm(rids,ep,cost,incoming,outgoing,selected):
    q=rs.base_bqm(rids,ep,cost,incoming,outgoing)
    for e in reachability_bad_edges(rids,ep,selected):
        q.add_linear(e,A)
    return q


def sample_tagged(bqm,base,ep,rids,reward,selected,start,rng,seed,reads,source,parent):
    cc,dt=rs.sample_branch(
      bqm,base,ep,rids,reward,selected,start,rng,seed,reads)
    for c in cc:
        c["source"]=source
        c["parent"]=parent
    return cc,dt


def state_key(c,ep):
    return tuple(sorted(c["selected"]))


def unique_by_state(cands,ep):
    best={}
    for c in cands:
        k=state_key(c,ep)
        if k not in best or c["base_energy"]<best[k]["base_energy"]:
            best[k]=c
    return list(best.values())


def symdiff(a,b):
    return len(set(a["selected"]).symmetric_difference(set(b["selected"])))


def choose_population(cands,ep,method,K=2):
    u=unique_by_state(cands,ep)
    if not u:
        raise RuntimeError("no propagation candidates")
    u.sort(key=lambda c:c["base_energy"])
    if K==1:
        return [u[0]]
    if len(u)==1:
        return [u[0],u[0]]

    primary=u[0]
    if method=="beam2_energy":
        return [primary,u[1]]

    top=u[:min(6,len(u))]
    sec=max(
      top[1:],
      key=lambda c:(symdiff(primary,c),-c["base_energy"])
    )
    return [primary,sec]


def update_inc(inc,inc_src,inc_it,cands,it):
    for c in cands:
        m=c["metrics"]
        if m["valid_path"] and (inc is None or m["path_score"]>inc):
            inc=m["path_score"];inc_src=c["source"];inc_it=it
    return inc,inc_src,inc_it


def classify(rids,selected,reward,N):
    if selected is None:return "initial"
    m=graph_metrics(rids,selected,project_rank(rids,selected),reward)
    if m["selected_edges"]!=N-1 or m["degree_conflicts"]!=0:
        return "irregular"
    if m["cycle_count"]>0:return "cyclic"
    return "hp"


def run(method,start,seed):
    rids,ep,reward,cost,incoming,outgoing=rs.load_problem();N=len(rids)
    rng=random.Random(seed)
    base=rs.base_bqm(rids,ep,cost,incoming,outgoing)
    explorers=[None]
    inc=None;inc_src=None;inc_it=None
    rows=[]

    for it in range(OUTER):
        prop=[];allc=[];branches=[];sec=0.0

        if method=="single8":
            parent=explorers[0]
            cc,dt=sample_tagged(
              base,base,ep,rids,reward,parent,start,rng,
              seed+10000*it,8,"no_rank",0)
            prop+=cc;allc+=cc;sec+=dt;branches.append("p0:none:8")
        elif len(explorers)==1 and explorers[0] is None:
            # First round: create the population from one common start.
            cc,dt=sample_tagged(
              base,base,ep,rids,reward,None,start,rng,
              seed+10000*it,8,"no_rank",0)
            prop+=cc;allc+=cc;sec+=dt;branches.append("init:none:8")
        else:
            for pi,parent in enumerate(explorers[:2]):
                bs=seed+10000*it+3000*pi
                typ=classify(rids,parent,reward,N)
                if method=="beam2_combo" and typ=="cyclic":
                    cc,dt=sample_tagged(
                      base,base,ep,rids,reward,parent,start,rng,bs,2,
                      "no_rank",pi)
                    prop+=cc;allc+=cc;sec+=dt;branches.append(f"p{pi}:none:2")

                    rb=reach_bqm(rids,ep,cost,incoming,outgoing,parent)
                    cc,dt=sample_tagged(
                      rb,base,ep,rids,reward,parent,start,rng,bs+1000,1,
                      "reach",pi)
                    allc+=cc;sec+=dt;branches.append(f"p{pi}:reach:1")

                    ranks=rs.alt_ranks(rids,parent,1,rng)
                    qb=rs.rank_bqm(rids,ep,cost,incoming,outgoing,ranks[0],A)
                    cc,dt=sample_tagged(
                      qb,base,ep,rids,reward,parent,start,rng,bs+2000,1,
                      "rank",pi)
                    allc+=cc;sec+=dt;branches.append(f"p{pi}:rank:1")
                else:
                    cc,dt=sample_tagged(
                      base,base,ep,rids,reward,parent,start,rng,bs,4,
                      "no_rank",pi)
                    prop+=cc;allc+=cc;sec+=dt;branches.append(f"p{pi}:none:4")

        old=inc
        inc,inc_src,inc_it=update_inc(inc,inc_src,inc_it,allc,it)

        if method=="single8":
            chosen=choose_population(prop,ep,"beam2_energy",K=1)
        elif method=="beam2_energy":
            chosen=choose_population(prop,ep,"beam2_energy",K=2)
        else:
            chosen=choose_population(prop,ep,"beam2_diverse",K=2)

        explorers=[set(c["selected"]) for c in chosen]
        p_metrics=[
          graph_metrics(rids,x,project_rank(rids,x),reward)
          for x in explorers
        ]
        pair_distance=(
          len(explorers[0].symmetric_difference(explorers[1]))
          if len(explorers)>1 else 0
        )
        hp=[c["metrics"]["path_score"] for c in allc if c["metrics"]["valid_path"]]

        row={
          "method":method,"start":start,"seed":seed,"iteration":it,
          "branches":branches,"seconds":sec,
          "candidate_hp_count":len(hp),"candidate_best_hp":max(hp,default=None),
          "incumbent_score":inc,"incumbent_source":inc_src,
          "incumbent_iteration":inc_it,
          "population_size":len(explorers),"population_hamming":pair_distance,
          "population_metrics":p_metrics,
        }
        rows.append(row);print("ROW",json.dumps(row),flush=True)

    return rows,{
      "method":method,"start":start,"seed":seed,
      "incumbent_score":inc,"ground_ever":inc==GROUND,
      "ground_source":inc_src if inc==GROUND else None,
      "ground_iteration":inc_it if inc==GROUND else None,
      "final_population_hp":sum(m["valid_path"] for m in rows[-1]["population_metrics"]),
      "final_population_hamming":rows[-1]["population_hamming"],
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
        for s in grounds:src[s["ground_source"]]=src.get(s["ground_source"],0)+1
        agg[f"{method}|{start}"]={
          "runs":len(aa),"hp_incumbent_runs":len(inc),
          "ground_ever":len(grounds),"ground_sources":src,
          "best_incumbent":max(inc,default=None),
          "median_incumbent":sorted(inc)[len(inc)//2] if inc else None,
          "final_population_hp":sum(s["final_population_hp"] for s in aa),
          "mean_final_population_hamming":sum(s["final_population_hamming"] for s in aa)/len(aa),
        }
    result={"aggregate":agg,"summaries":summ,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":
    main()
