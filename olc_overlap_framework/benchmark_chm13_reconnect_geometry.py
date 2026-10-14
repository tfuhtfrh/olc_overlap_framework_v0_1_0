"""Geometry diagnostic for reconnect moves on the known successful CHM13 seed.

Reproduce the time_phase fixed-cardinality run for seed 20370928 up to the
first Hamilton path (historically iteration 9, score 1,326,008), then analyze
the move graph around that concrete path.

Questions:
1. What does a directed 2-switch do geometrically to a Hamilton path?
2. How many of the 41 static reconnect templates are eligible?
3. Can one 2-switch map the path directly to another Hamilton path?
4. How many two-step 2-switch sequences return to a Hamilton path, improve the
   score, or reach the certified optimum?
5. How many true 3-edge degree-preserving reconnects (cyclic head permutations)
   map the current Hamilton path directly to another Hamilton path?

A true 3-edge reconnect selects three current edges
    a_i -> b_i, i=0,1,2
and cyclically permutes the heads:
    a0->b1, a1->b2, a2->b0
or the inverse cycle.
This is a 6-bit degree-preserving move and contains contiguous segment
relocation as a special case.
"""
from __future__ import annotations
import itertools, json, random
from pathlib import Path

import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import load_problem, project_rank, graph_metrics

OUT=Path("debug/qubo/chm13_reconnect_geometry_20260928.json")
SEED=20370928
CERT=1_343_093

fc.SWEEPS=1500
fc.READS=2

def apply_template(selected_idx, template):
    side1,side2=template
    a=all(i in selected_idx for i in side1) and all(i not in selected_idx for i in side2)
    b=all(i in selected_idx for i in side2) and all(i not in selected_idx for i in side1)
    if not (a or b):
        return None
    out=set(selected_idx)
    for i in side1+side2:
        if i in out: out.remove(i)
        else: out.add(i)
    return frozenset(out)

def metrics_for_idx(rids,ep,reward,idxset):
    selected={ep[i] for i in idxset}
    ranks=project_rank(rids,selected)
    return graph_metrics(rids,selected,ranks,reward)

def reproduce_first_path():
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    templates=fc.build_reconnect_templates(ep)
    rng=random.Random(SEED)
    ranks={r:i for i,r in enumerate(rng.sample(rids,len(rids)))}
    selected=None
    history=[]
    first=None
    for it in range(18):
        op=fc.ORDER_SCHEDULE[it % len(fc.ORDER_SCHEDULE)]
        kernel="exchange_only" if it<12 else "static_50"
        bqm=fc.build_bqm_fixed_cardinality(
            ep,cost,incoming,outgoing,ranks,
            degree_conflict_penalty=288.0,order_penalty=op)
        selected,sec,en,stats=fc.sample_fixed_cardinality(
            bqm,ep,templates,len(rids)-1,selected,SEED+1000*it,kernel)
        ranks=project_rank(rids,selected)
        met=graph_metrics(rids,selected,ranks,reward)
        history.append({"iteration":it,"kernel":kernel,"order_penalty":op,**met})
        print("REPRO",json.dumps(history[-1]),flush=True)
        if met["valid_path"] and first is None:
            first=(it,set(selected),met)
            break
    if first is None:
        raise RuntimeError("did not reproduce a Hamilton path")
    return (rids,ep,reward,templates,history,first)

def main():
    rids,ep,reward,templates,history,(it,selected,met)=reproduce_first_path()
    idx={e:i for i,e in enumerate(ep)}
    selected_idx=frozenset(idx[e] for e in selected)
    ep_set=set(ep)

    # One-step 2-switch analysis.
    one=[]
    for ti,t in enumerate(templates):
        nxt=apply_template(selected_idx,t)
        if nxt is None: continue
        mm=metrics_for_idx(rids,ep,reward,nxt)
        one.append({"template":ti,**mm})
    print("ONE_STEP",json.dumps({"eligible":len(one),"rows":one}),flush=True)

    # Two-step 2-switch state graph.  Count ordered static-template sequences,
    # including only actual eligible transitions at both steps.
    two_total=0;two_valid=0;two_improve=0;two_ground=0
    valid_states={}
    for ti,t in enumerate(templates):
        mid=apply_template(selected_idx,t)
        if mid is None: continue
        for tj,u in enumerate(templates):
            fin=apply_template(mid,u)
            if fin is None: continue
            two_total+=1
            mm=metrics_for_idx(rids,ep,reward,fin)
            if mm["valid_path"]:
                two_valid+=1
                key=tuple(sorted(fin))
                valid_states[key]=mm["path_score"]
                if mm["path_score"]>met["path_score"]: two_improve+=1
                if mm["path_score"]==CERT: two_ground+=1
    scores=list(valid_states.values())

    # True 3-edge cyclic head-permutation reconnects directly from the HP.
    sel_edges=list(selected)
    three_candidates=0;three_valid=0;three_improve=0;three_ground=0
    three_states={}
    for triple in itertools.combinations(sel_edges,3):
        tails=[e[0] for e in triple]; heads=[e[1] for e in triple]
        if len(set(tails))<3 or len(set(heads))<3:
            continue
        for perm in ((1,2,0),(2,0,1)):
            new_edges=[(tails[i],heads[perm[i]]) for i in range(3)]
            if len(set(new_edges))<3 or any(e not in ep_set for e in new_edges):
                continue
            if set(new_edges)==set(triple):
                continue
            # Require a genuine 3-edge replacement, not leaving one old edge.
            if any(e in triple for e in new_edges):
                continue
            newsel=set(selected)
            for e in triple:newsel.remove(e)
            if any(e in newsel for e in new_edges):
                continue
            newsel.update(new_edges)
            three_candidates+=1
            ranks=project_rank(rids,newsel)
            mm=graph_metrics(rids,newsel,ranks,reward)
            if mm["valid_path"]:
                three_valid+=1
                key=tuple(sorted(idx[e] for e in newsel))
                three_states[key]=mm["path_score"]
                if mm["path_score"]>met["path_score"]:three_improve+=1
                if mm["path_score"]==CERT:three_ground+=1

    result={
      "seed":SEED,
      "first_hamilton_iteration":it,
      "start_score":met["path_score"],
      "static_2switch_templates":len(templates),
      "one_step_2switch":{
        "eligible":len(one),
        "direct_hamilton_paths":sum(1 for r in one if r["valid_path"]),
        "cycle_counts":[r["cycle_count"] for r in one],
      },
      "two_step_2switch":{
        "eligible_ordered_sequences":two_total,
        "valid_path_sequences":two_valid,
        "improving_valid_sequences":two_improve,
        "ground_sequences":two_ground,
        "unique_valid_paths":len(valid_states),
        "best_score":max(scores) if scores else None,
        "uniform_static_two_draw_probability_valid":two_valid/(len(templates)**2),
        "uniform_static_two_draw_probability_improving":two_improve/(len(templates)**2),
        "uniform_static_two_draw_probability_ground":two_ground/(len(templates)**2),
      },
      "three_edge_reconnect":{
        "genuine_candidate_moves":three_candidates,
        "direct_hamilton_moves":three_valid,
        "improving_direct_hamilton_moves":three_improve,
        "ground_direct_hamilton_moves":three_ground,
        "unique_hamilton_paths":len(three_states),
        "best_score":max(three_states.values()) if three_states else None,
      },
      "history":history,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("RESULT",json.dumps(result),flush=True)

if __name__=="__main__":main()
