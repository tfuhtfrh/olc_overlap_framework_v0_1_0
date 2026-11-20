"""Population-size ablation for rank-free standard-SQA hybrid.

Fixed total budget:
- 8 standard OpenJij SQASampler reads / outer iteration;
- 16 outer iterations;
- beta=5, gamma=1, P=8, 1400 sweeps/read;
- no rank/reachability/custom updater.

Compare explorer population size K:
- K=1: 8 reads from one explorer;
- K=2: 4 reads per explorer;
- K=4: 2 reads per explorer.

After each outer iteration, pool only rank-free outputs and keep the K
lowest-H0 DISTINCT states as next explorers. If fewer than K unique states are
available, repeat the last selected state.

Best feasible HP is archived non-oracularly.

Environment worker mode:
  K in {1,2,4}
  START in {zero,random}
"""
from __future__ import annotations
import json,os,random
from pathlib import Path

import benchmark_chm13_rank_expression_screening as rs
from benchmark_chm13_projected_edge_hybrid import graph_metrics,project_rank

GROUND=1_343_093
OUTER=16
SEEDS=tuple(202610500+i for i in range(24))
READ_BUDGET=8


def sample_tagged(base,ep,rids,reward,selected,start,rng,seed,reads,parent):
    cc,dt=rs.sample_branch(
      base,base,ep,rids,reward,selected,start,rng,seed,reads)
    for c in cc:
        c["parent"]=parent
    return cc,dt


def key(c):
    return tuple(sorted(c["selected"]))


def choose_k(cands,K):
    best={}
    for c in cands:
        kk=key(c)
        if kk not in best or c["base_energy"]<best[kk]["base_energy"]:
            best[kk]=c
    u=sorted(best.values(),key=lambda c:c["base_energy"])
    if not u:raise RuntimeError("no candidates")
    out=u[:K]
    while len(out)<K:out.append(out[-1])
    return out


def run(K,start,seed):
    rids,ep,reward,cost,incoming,outgoing=rs.load_problem();N=len(rids)
    rng=random.Random(seed)
    base=rs.base_bqm(rids,ep,cost,incoming,outgoing)
    explorers=[None]
    inc=None;inc_it=None;rows=[]

    for it in range(OUTER):
        cands=[];sec=0.0;branches=[]
        if explorers==[None]:
            cc,dt=sample_tagged(
              base,ep,rids,reward,None,start,rng,seed+10000*it,
              READ_BUDGET,0)
            cands+=cc;sec+=dt;branches.append(f"init:{READ_BUDGET}")
        else:
            reads_each=READ_BUDGET//K
            assert reads_each*K==READ_BUDGET
            for pi,parent in enumerate(explorers):
                cc,dt=sample_tagged(
                  base,ep,rids,reward,parent,start,rng,
                  seed+10000*it+2000*pi,reads_each,pi)
                cands+=cc;sec+=dt;branches.append(f"p{pi}:{reads_each}")

        hp=[c["metrics"]["path_score"] for c in cands if c["metrics"]["valid_path"]]
        if hp:
            cur=max(hp)
            if inc is None or cur>inc:
                inc=cur;inc_it=it

        chosen=choose_k(cands,K)
        explorers=[set(c["selected"]) for c in chosen]
        pm=[graph_metrics(rids,x,project_rank(rids,x),reward) for x in explorers]
        # Mean pairwise Hamming distance.
        ds=[]
        for a in range(K):
            for b in range(a+1,K):
                ds.append(len(explorers[a].symmetric_difference(explorers[b])))
        mean_d=sum(ds)/len(ds) if ds else 0.0

        row={
          "K":K,"start":start,"seed":seed,"iteration":it,
          "branches":branches,"seconds":sec,
          "candidate_hp_count":len(hp),"candidate_best_hp":max(hp,default=None),
          "incumbent_score":inc,"incumbent_iteration":inc_it,
          "population_metrics":pm,"mean_pairwise_hamming":mean_d,
        }
        rows.append(row);print("ROW",json.dumps(row),flush=True)

    return rows,{
      "K":K,"start":start,"seed":seed,
      "incumbent_score":inc,"ground_ever":inc==GROUND,
      "ground_iteration":inc_it if inc==GROUND else None,
      "final_population_hp":sum(m["valid_path"] for m in rows[-1]["population_metrics"]),
      "final_mean_pairwise_hamming":rows[-1]["mean_pairwise_hamming"],
    }


def main():
    K=int(os.environ.get("K","2"))
    start=os.environ.get("START","zero")
    rows=[];summ=[]
    for si,bs in enumerate(SEEDS):
        seed=bs+10000*si
        rr,ss=run(K,start,seed)
        rows+=rr;summ.append(ss);print("SUMMARY",json.dumps(ss),flush=True)

    inc=[s["incumbent_score"] for s in summ if s["incumbent_score"] is not None]
    grounds=[s for s in summ if s["ground_ever"]]
    gits=[s["ground_iteration"] for s in grounds if s["ground_iteration"] is not None]
    agg={
      "K":K,"start":start,"runs":len(summ),
      "hp_incumbent_runs":len(inc),
      "ground_ever":len(grounds),
      "best_incumbent":max(inc,default=None),
      "median_incumbent":sorted(inc)[len(inc)//2] if inc else None,
      "median_ground_iteration":sorted(gits)[len(gits)//2] if gits else None,
      "final_population_hp":sum(s["final_population_hp"] for s in summ),
      "mean_final_pairwise_hamming":sum(s["final_mean_pairwise_hamming"] for s in summ)/len(summ),
    }
    out=Path(f"debug/qubo/chm13_population_size_K{K}_{start}_20260930.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"aggregate":agg,"summaries":summ,"rows":rows},indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)

if __name__=="__main__":
    main()
