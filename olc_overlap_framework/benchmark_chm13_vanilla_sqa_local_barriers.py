"""Vanilla OpenJij SQA on two known CHM13 local barriers.

No custom updater. No ground early stop.

Diagnostics:
1) Stage-2 stuck state: degree-correct 143-edge path+cycle checkpoint.
   Compare no acyclicity term vs weak exact current-cycle cut A=0.02.
   Target: any valid Hamilton path.

2) Stage-3 runner-up HP score 1,338,209.
   Use the rank-free full weight+degree+count BQM.
   Target: certified HP ground 1,343,093.
   Note: this BQM does not globally encode acyclicity, so this is a dynamics
   diagnostic, not a complete Hamilton-path Hamiltonian.

Each standard SQA read returns OpenJij's lowest-classical-energy Trotter slice.
We report both SampleSet.first and whether any of 8 standard reads hits target.
"""
from __future__ import annotations
import json,time
from pathlib import Path
import openjij as oj

from benchmark_chm13_projected_edge_hybrid import (
    load_problem,graph_metrics,project_rank,build_bqm_count,
)
from benchmark_chm13_end_to_end_structured_sqa import add_exact_cuts
from diagnose_chm13_move_basis_proposal import replay_stuck_checkpoint
from diagnose_chm13_runnerup_ground_geometry import normalized_paths,score,state

OUT=Path("debug/qubo/chm13_vanilla_sqa_local_barriers_20260930.json")
GROUND=1_343_093
RUNNER=1_338_209
READS=8
SWEEPS=2000
SEEDS=tuple(202609950+i for i in range(24))
CONFIGS={
 "default":{"beta":5.0,"gamma":1.0,"P":8},
 "p16":{"beta":6.0,"gamma":2.0,"P":16},
}


def cycles_from_selected(rids,selected):
    import networkx as nx
    if not selected:return []
    g=nx.DiGraph();g.add_nodes_from(rids);g.add_edges_from(selected)
    ans=[]
    for comp in nx.strongly_connected_components(g):
        if len(comp)<=1:continue
        sub=g.subgraph(comp)
        try:cyc=nx.find_cycle(sub,orientation="original")
        except nx.NetworkXNoCycle:continue
        ans.append(tuple(sorted((u,v) for u,v,*_ in cyc)))
    return ans


def make_qubo(bqm):
    vs=list(bqm.variables);idx={v:i for i,v in enumerate(vs)};q={}
    for v,b in bqm.linear.items():
        if b:q[(idx[v],idx[v])]=q.get((idx[v],idx[v]),0.0)+float(b)
    for (u,v),b in bqm.quadratic.items():
        if not b:continue
        i,j=idx[u],idx[v]
        if i>j:i,j=j,i
        q[(i,j)]=q.get((i,j),0.0)+float(b)
    return vs,idx,q


def encode_slack(meta,selected):
    from itertools import product
    vals={}
    for item in meta:
        chosen=sum(int(e in selected) for e in item["cycle"])
        target=max(0,len(item["cycle"])-1-chosen)
        for cand in product((0,1),repeat=len(item["weights"])):
            if sum(w*b for w,b in zip(item["weights"],cand))==target:
                for lab,b in zip(item["labels"],cand):vals[lab]=b
                break
    return vals


def sample(bqm,ep,selected,meta,cfg,seed,rids,reward):
    vs,idx,q=make_qubo(bqm)
    sm={v:0 for v in vs}
    for e in ep:sm[e]=int(e in selected)
    sm.update(encode_slack(meta,selected))
    ini={idx[v]:int(sm[v]) for v in vs}
    t=time.perf_counter()
    resp=oj.SQASampler().sample_qubo(
      q,beta=cfg["beta"],gamma=cfg["gamma"],trotter=cfg["P"],
      num_reads=READS,num_sweeps=SWEEPS,initial_state=ini,
      updater="single spin flip",seed=seed)
    sec=time.perf_counter()-t

    first=resp.first
    psm={vs[i]:int(x) for i,x in first.sample.items()}
    psel={e for e in ep if psm.get(e,0)}
    pmet=graph_metrics(rids,psel,project_rank(rids,psel),reward)

    labels=list(resp.variables);rmet=[]
    for arr,en in zip(resp.record.sample,resp.record.energy):
        s2={vs[int(lab)]:int(arr[col]) for col,lab in enumerate(labels)}
        sel={e for e in ep if s2.get(e,0)}
        met=graph_metrics(rids,sel,project_rank(rids,sel),reward)
        rmet.append({
          "valid_path":bool(met["valid_path"]),"path_score":met["path_score"],
          "selected_edges":met["selected_edges"],
          "degree_conflicts":met["degree_conflicts"],"cycle_count":met["cycle_count"],
          "energy":float(en+bqm.offset),
        })
    return pmet,rmet,sec


def contexts():
    rids,ep,reward,cost,incoming,outgoing=load_problem();N=len(rids)
    dummy={r:0 for r in rids}

    _,_,_,_,_,_,stuck_sel,_,_=replay_stuck_checkpoint()
    b2=build_bqm_count(
      ep,cost,incoming,outgoing,dummy,N,
      degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)
    cyc=cycles_from_selected(rids,stuck_sel)
    b2cut=b2.copy();meta2=add_exact_cuts(b2cut,cyc,0.02)

    eps=set(ep);runner=None
    for p in normalized_paths():
        if score(p,reward)==RUNNER:
            s=state(p)
            if s<=eps:runner=s;break
    assert runner is not None
    b3=build_bqm_count(
      ep,cost,incoming,outgoing,dummy,N,
      degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)
    return rids,ep,reward,{
      "stage2_none":(b2,stuck_sel,[]),
      "stage2_cut":(b2cut,stuck_sel,meta2),
      "stage3_runner":(b3,runner,[]),
    }


def main():
    rids,ep,reward,ctx=contexts()
    rows=[]
    for name,(bqm,init,meta) in ctx.items():
      for cfgname,cfg in CONFIGS.items():
        for seed in SEEDS:
          pm,rm,sec=sample(bqm,ep,init,meta,cfg,seed,rids,reward)
          if name.startswith("stage2"):
            primary_target=bool(pm["valid_path"])
            any_target=any(x["valid_path"] for x in rm)
          else:
            primary_target=(pm["path_score"]==GROUND)
            any_target=any(x["path_score"]==GROUND for x in rm)
          row={
            "case":name,"config":cfgname,"seed":seed,"seconds":sec,
            "primary_target":primary_target,"any_read_target":any_target,
            "primary_valid_path":bool(pm["valid_path"]),
            "primary_score":pm["path_score"],
            "primary_selected":pm["selected_edges"],
            "primary_degree_conflicts":pm["degree_conflicts"],
            "primary_cycles":pm["cycle_count"],
          }
          rows.append(row);print("ROW",json.dumps(row),flush=True)

    agg={}
    for name in ctx:
      for cfgname in CONFIGS:
        rr=[r for r in rows if r["case"]==name and r["config"]==cfgname]
        scores=[r["primary_score"] for r in rr if r["primary_valid_path"]]
        agg[f"{name}|{cfgname}"]={
          "runs":len(rr),
          "primary_target_hits":sum(r["primary_target"] for r in rr),
          "any_read_target_hits":sum(r["any_read_target"] for r in rr),
          "primary_hp_hits":sum(r["primary_valid_path"] for r in rr),
          "best_primary_hp_score":max(scores,default=None),
          "mean_cycles":sum(r["primary_cycles"] for r in rr)/len(rr),
          "mean_degree_conflicts":sum(r["primary_degree_conflicts"] for r in rr)/len(rr),
          "mean_seconds":sum(r["seconds"] for r in rr)/len(rr),
        }
    result={"aggregate":agg,"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("AGGREGATE",json.dumps(agg),flush=True)


if __name__=="__main__":
    main()
