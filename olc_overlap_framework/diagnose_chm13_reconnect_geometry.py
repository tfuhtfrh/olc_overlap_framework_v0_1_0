"""Diagnose reconnect move geometry and good-move rarity in CHM13.

Replay the exchange-only phase of the fixed-cardinality time-phase experiment,
then enumerate ALL static directed reconnect templates at selected checkpoints.

For each eligible reconnect we classify:
- change in degree conflicts;
- change in cycle count;
- whether it reaches a Hamilton path;
- exact classical BQM energy delta for the NEXT outer iteration;
- whether it is downhill under that BQM.

This quantifies how rare "good" reconnect moves actually are and whether a
component-aware proposal can target them directly.
"""
from __future__ import annotations
import json, random
from pathlib import Path
import networkx as nx

import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import load_problem, project_rank, graph_metrics

OUT=Path("debug/qubo/chm13_reconnect_geometry_diagnostic_20260928.json")
# Match the long-budget time_phase experiment exactly.
fc.SWEEPS=1500
fc.READS=2
SEEDS=[20370928,20380928,20390928]
CHECKPOINT=11  # end of exchange-only phase in prior time_phase benchmark

def apply_template(selected, ep, template):
    side1,side2=template
    set1={ep[i] for i in side1};set2={ep[i] for i in side2}
    if set1<=selected and set2.isdisjoint(selected):
        return (selected-set1)|set2, True, "side1_to_side2"
    if set2<=selected and set1.isdisjoint(selected):
        return (selected-set2)|set1, True, "side2_to_side1"
    return selected,False,None

def component_type_map(rids,selected):
    g=nx.DiGraph();g.add_nodes_from(rids);g.add_edges_from(selected)
    out={}
    for comp in nx.weakly_connected_components(g):
        sub=g.subgraph(comp)
        if len(comp)==1 and sub.number_of_edges()==0:
            typ="isolated"
        elif nx.is_directed_acyclic_graph(sub):
            typ="path_like"
        else:
            typ="cyclic"
        for v in comp:out[v]=typ
    return out

def replay(seed):
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    templates=fc.build_reconnect_templates(ep)
    rng=random.Random(seed)
    ranks={r:i for i,r in enumerate(rng.sample(rids,len(rids)))}
    selected=None
    rows=[]
    for it in range(CHECKPOINT+1):
        op=fc.ORDER_SCHEDULE[it%len(fc.ORDER_SCHEDULE)]
        bqm=fc.build_bqm_fixed_cardinality(
            ep,cost,incoming,outgoing,ranks,
            degree_conflict_penalty=288.0,order_penalty=op)
        selected,sec,en,stats=fc.sample_fixed_cardinality(
            bqm,ep,templates,len(rids)-1,selected,seed+1000*it,"exchange_only")
        ranks=project_rank(rids,selected)
        rows.append(graph_metrics(rids,selected,ranks,reward))
    return rids,ep,reward,cost,incoming,outgoing,templates,selected,ranks,rows

def classify_seed(seed):
    rids,ep,reward,cost,incoming,outgoing,templates,selected,ranks,trace=replay(seed)
    base=graph_metrics(rids,selected,ranks,reward)
    next_it=CHECKPOINT+1
    next_op=fc.ORDER_SCHEDULE[next_it%len(fc.ORDER_SCHEDULE)]
    bqm=fc.build_bqm_fixed_cardinality(
        ep,cost,incoming,outgoing,ranks,288.0,next_op)
    abqm=fc.ArrayBQM.from_bqm(bqm,ep)
    idx={e:i for i,e in enumerate(ep)}
    x=[1 if e in selected else 0 for e in ep]
    base_energy=abqm.energy(x)

    compmap=component_type_map(rids,selected)
    candidates=[]
    for ti,tpl in enumerate(templates):
        sel2,ok,direction=apply_template(selected,ep,tpl)
        if not ok:continue
        ranks2=project_rank(rids,sel2)
        met=graph_metrics(rids,sel2,ranks2,reward)
        x2=[1 if e in sel2 else 0 for e in ep]
        en2=abqm.energy(x2)
        # Which component types did the two selected edges come from?
        side1,side2=tpl
        if direction=="side1_to_side2":
            removed=[ep[i] for i in side1]
        else:
            removed=[ep[i] for i in side2]
        removed_types=sorted([compmap.get(u,"?") for u,v in removed])
        candidates.append({
            "template_index":ti,
            "direction":direction,
            "removed_edges":removed,
            "removed_component_types":removed_types,
            "energy_delta_next_bqm":en2-base_energy,
            "downhill":en2<base_energy-1e-12,
            "cycle_delta":met["cycle_count"]-base["cycle_count"],
            "degree_delta":met["degree_conflicts"]-base["degree_conflicts"],
            "valid_path":met["valid_path"],
            "path_score":met["path_score"],
        })

    summary={
      "seed":seed,
      "checkpoint":CHECKPOINT,
      "base_metrics":base,
      "next_order_penalty":next_op,
      "total_static_templates":len(templates),
      "eligible_templates":len(candidates),
      "downhill_templates":sum(c["downhill"] for c in candidates),
      "cycle_reducing_templates":sum(c["cycle_delta"]<0 for c in candidates),
      "valid_path_templates":sum(c["valid_path"] for c in candidates),
      "path_cycle_reconnects":sum(c["removed_component_types"]==["cyclic","path_like"] for c in candidates),
      "good_probability_if_uniform_static":sum(c["cycle_delta"]<0 for c in candidates)/len(templates),
      "good_probability_if_uniform_eligible":(sum(c["cycle_delta"]<0 for c in candidates)/len(candidates) if candidates else 0),
    }
    return summary,candidates,trace

def main():
    allout=[]
    for seed in SEEDS:
        summary,candidates,trace=classify_seed(seed)
        allout.append({"summary":summary,"candidates":candidates,"trace":trace})
        print("SUMMARY",json.dumps(summary),flush=True)
        good=[c for c in candidates if c["cycle_delta"]<0 or c["valid_path"]]
        print("GOOD",json.dumps(good),flush=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(allout,indent=2)+"\n")

if __name__=="__main__":main()
