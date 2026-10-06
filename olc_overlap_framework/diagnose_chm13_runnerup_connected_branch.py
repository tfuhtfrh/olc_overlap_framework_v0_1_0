"""Track the transverse-field eigenbranch adiabatically connected to the CHM13 runner-up.

Uses the same frozen six-bit subspace as the local spectrum diagnostic.

At g=0 the runner-up bitstring is an exact classical eigenstate, but it is an
excited state because the target HP is lower by 0.290870109.

We increase g continuously and track the eigenvector with maximum overlap to
the previously selected eigenvector. This approximates the instantaneous
eigenbranch adiabatically connected to the runner-up classical state.

This is a local spectral diagnostic, not a real-time reverse-annealing
simulation.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

import diagnose_chm13_runnerup_ground_6bit_spectrum as base

OUT=Path("debug/qubo/chm13_runnerup_connected_branch_20260930.json")


def main():
    rids,ep,reward,cost,incoming,outgoing=base.load_problem()
    runner,ground=base.find_exact_states(reward,set(ep))
    diff=sorted(runner.symmetric_difference(ground))
    didx={e:i for i,e in enumerate(diff)}

    dummy={r:0 for r in rids}
    bqm=base.build_bqm_count(
        ep,cost,incoming,outgoing,dummy,len(rids),
        degree_conflict_penalty=288.0,count_penalty=32.0,order_penalty=0.0)

    fixed={e:int(e in runner) for e in ep if e not in didx}
    energies=np.zeros(64,dtype=float)
    runner_idx=ground_idx=None
    for z in range(64):
        sample=dict(fixed);selected={e for e,v in fixed.items() if v}
        for i,e in enumerate(diff):
            b=(z>>i)&1;sample[e]=b
            if b:selected.add(e)
        energies[z]=float(bqm.energy(sample))
        if selected==runner:runner_idx=z
        if selected==ground:ground_idx=z
    assert runner_idx is not None and ground_idx is not None

    e0=float(energies.min())
    D=np.diag(energies-e0)
    X=np.zeros((64,64),dtype=float)
    for z in range(64):
        for i in range(6):
            X[z,z^(1<<i)]=1.0

    gs=np.linspace(0.0,20.0,801)
    prev=np.zeros(64);prev[runner_idx]=1.0
    rows=[]
    max_target=(0.0,0.0)
    min_nearest=(float("inf"),0.0)

    for g in gs:
        H=D-g*X
        vals,vecs=np.linalg.eigh(H)
        overlaps=np.abs(vecs.T.conj()@prev)**2
        j=int(np.argmax(overlaps))
        psi=vecs[:,j]
        if np.vdot(prev,psi).real<0:psi=-psi
        probs=np.abs(psi)**2

        if j==0:
            nearest=float(vals[1]-vals[0])
        elif j==len(vals)-1:
            nearest=float(vals[-1]-vals[-2])
        else:
            nearest=float(min(vals[j]-vals[j-1],vals[j+1]-vals[j]))

        pt=float(probs[ground_idx]);pr=float(probs[runner_idx])
        if pt>max_target[0]:max_target=(pt,float(g))
        if nearest<min_nearest[0]:min_nearest=(nearest,float(g))

        rows.append({
          "g":float(g),
          "eigenvalue":float(vals[j]),
          "eigen_rank":j,
          "overlap_continuity":float(overlaps[j]),
          "runner_probability":pr,
          "target_probability":pt,
          "nearest_level_separation":nearest,
          "dominant_basis_index":int(np.argmax(probs)),
          "dominant_probability":float(np.max(probs)),
        })
        prev=psi

    probe_s=[0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.25,0.2,0.15,0.1,0.05]
    probes=[]
    for s in probe_s:
        g=(1-s)/s
        row=min(rows,key=lambda r:abs(r["g"]-g)).copy()
        row["s"]=s;row["g_exact"]=g
        probes.append(row)

    result={
      "runner_index":runner_idx,
      "target_index":ground_idx,
      "runner_classical_energy_from_target":float(energies[runner_idx]-energies[ground_idx]),
      "max_target_probability_on_runner_branch":max_target[0],
      "max_target_probability_g":max_target[1],
      "minimum_nearest_level_separation":min_nearest[0],
      "minimum_nearest_level_separation_g":min_nearest[1],
      "reverse_depth_probes":probes,
      "scan":rows,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("SUMMARY",json.dumps({
      k:result[k] for k in [
        "runner_classical_energy_from_target",
        "max_target_probability_on_runner_branch",
        "max_target_probability_g",
        "minimum_nearest_level_separation",
        "minimum_nearest_level_separation_g",
        "reverse_depth_probes"
      ]
    }),flush=True)


if __name__=="__main__":
    main()
