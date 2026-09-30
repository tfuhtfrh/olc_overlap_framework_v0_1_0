"""Exact 6-bit transverse-field subspace analysis for CHM13 runner-up -> ground.

Freeze all edge variables outside the six-bit symmetric difference between the
exact runner-up HP and certified ground HP. Enumerate all 64 classical states
under the SAME standard-SQA base QUBO:

    H0 = H_weight + H_degree + H_count

with degree penalty 288 and count penalty 32.

Then exactly diagonalize

    H(g) = H0 - g * sum_i X_i

in the 64-dimensional subspace, where
    g = gamma * (1-s) / s
is the transverse/problem energy ratio after factoring out s from OpenJij's
H(s)=s H0 + gamma(1-s) sum X convention (sign convention is irrelevant for
the spectrum under a basis gauge).

This is a local spectral diagnostic, not a proof of full-QPU dynamics.
"""
from __future__ import annotations

import json, math
from pathlib import Path
import numpy as np

from benchmark_chm13_projected_edge_hybrid import (
    load_problem, graph_metrics, project_rank, build_bqm_count,
)
from diagnose_chm13_runnerup_ground_geometry import (
    normalized_paths, score, state, GROUND, RUNNER,
)

OUT=Path("debug/qubo/chm13_runnerup_ground_6bit_spectrum_20260930.json")


def find_exact_states(reward,epset):
    runner=ground=None
    for p in normalized_paths():
        sc=score(p,reward)
        if sc not in (RUNNER,GROUND):
            continue
        st=state(p)
        if not st<=epset:
            continue
        if sc==RUNNER:
            runner=st
        elif sc==GROUND:
            ground=st
    assert runner is not None and ground is not None
    return runner,ground


def main():
    rids,ep,reward,cost,incoming,outgoing=load_problem()
    runner,ground=find_exact_states(reward,set(ep))
    diff=sorted(runner.symmetric_difference(ground))
    assert len(diff)==6
    didx={e:i for i,e in enumerate(diff)}

    dummy={r:0 for r in rids}
    bqm=build_bqm_count(
        ep,cost,incoming,outgoing,dummy,len(rids),
        degree_conflict_penalty=288.0,
        count_penalty=32.0,
        order_penalty=0.0,
    )

    fixed={e:int(e in runner) for e in ep if e not in didx}
    energies=np.zeros(64,dtype=float)
    rows=[]
    runner_idx=ground_idx=None

    for z in range(64):
        sample=dict(fixed)
        selected={e for e,v in fixed.items() if v}
        bits=[]
        for i,e in enumerate(diff):
            b=(z>>i)&1
            sample[e]=b
            bits.append(b)
            if b:selected.add(e)
        en=float(bqm.energy(sample))
        energies[z]=en
        met=graph_metrics(rids,selected,project_rank(rids,selected),reward)
        if selected==runner:runner_idx=z
        if selected==ground:ground_idx=z
        rows.append({
            "index":z,
            "bits":bits,
            "energy":en,
            "energy_from_runner":None,
            "selected_edges":met["selected_edges"],
            "degree_conflicts":met["degree_conflicts"],
            "cycle_count":met["cycle_count"],
            "valid_path":bool(met["valid_path"]),
            "path_score":met["path_score"],
        })

    assert runner_idx is not None and ground_idx is not None
    er=energies[runner_idx]
    for r in rows:r["energy_from_runner"]=r["energy"]-er

    classical_order=sorted(range(64),key=lambda z:energies[z])
    top_classical=[rows[z] for z in classical_order[:12]]

    # Hypercube transverse adjacency.
    X=np.zeros((64,64),dtype=float)
    for z in range(64):
        for i in range(6):
            zz=z^(1<<i)
            X[z,zz]=1.0

    # Shift the diagonal to improve conditioning.
    e0=float(energies.min())
    D=np.diag(energies-e0)

    # Dense scan covers OpenJij reverse depths:
    # g=(1-s)/s for gamma=1:
    # s=.8 -> .25, .5 -> 1, .3 -> 2.333, .05 -> 19.
    gs=np.concatenate([
        np.linspace(0.0,0.5,101),
        np.linspace(0.5,3.0,126)[1:],
        np.linspace(3.0,20.0,171)[1:],
    ])

    scan=[]
    min_gap=(float("inf"),None)
    for g in gs:
        H=D-g*X
        vals,vecs=np.linalg.eigh(H)
        gap=float(vals[1]-vals[0])
        if gap<min_gap[0]:min_gap=(gap,float(g))
        psi=vecs[:,0]
        probs=np.abs(psi)**2
        scan.append({
            "g":float(g),
            "gap01":gap,
            "ground_prob_runner":float(probs[runner_idx]),
            "ground_prob_target_ground":float(probs[ground_idx]),
            "dominant_classical_index":int(np.argmax(probs)),
            "dominant_probability":float(np.max(probs)),
        })

    # Report values at reverse depths actually tested.
    probe_s=[0.95,0.90,0.80,0.70,0.60,0.50,0.40,0.30,0.25,0.20,0.15,0.10,0.05]
    probes=[]
    for s in probe_s:
        g=(1.0-s)/s
        j=min(range(len(scan)),key=lambda k:abs(scan[k]["g"]-g))
        row=dict(scan[j])
        row["s"]=s
        row["g_exact"]=g
        probes.append(row)

    # Identify the lowest classical cyclic degree-correct state.
    degree_correct_cyclic=[
        r for r in rows
        if r["selected_edges"]==len(rids)-1
        and r["degree_conflicts"]==0
        and r["cycle_count"]>0
    ]
    degree_correct_cyclic.sort(key=lambda r:r["energy"])

    result={
        "diff_edges":diff,
        "runner_index":runner_idx,
        "ground_index":ground_idx,
        "runner_energy":float(energies[runner_idx]),
        "target_ground_energy":float(energies[ground_idx]),
        "target_ground_minus_runner":float(energies[ground_idx]-energies[runner_idx]),
        "classical_global_min_index":int(classical_order[0]),
        "classical_global_min":rows[classical_order[0]],
        "lowest_degree_correct_cyclic":degree_correct_cyclic[0] if degree_correct_cyclic else None,
        "top_classical_states":top_classical,
        "minimum_gap01":min_gap[0],
        "minimum_gap_g":min_gap[1],
        "reverse_depth_probes":probes,
        "scan":scan,
    }

    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("SUMMARY",json.dumps({
        k:result[k] for k in [
          "runner_energy","target_ground_energy",
          "target_ground_minus_runner","classical_global_min_index",
          "classical_global_min","lowest_degree_correct_cyclic",
          "minimum_gap01","minimum_gap_g","reverse_depth_probes"
        ]
    }),flush=True)


if __name__=="__main__":
    main()
