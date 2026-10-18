"""Local corrected SQA: independent reproducible read seeds and random initial states."""
from __future__ import annotations
import argparse
import concurrent.futures
from dataclasses import asdict
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import openjij as oj
import report_chapter3_benchmark as base
from olc_pipeline.layout_solver import (MissingEdgeHamiltonianConfig, MissingEdgeQUBOHamiltonian,
    WeightedOverlapHamiltonianConfig, WeightedOverlapQUBOHamiltonian,
    EdgePathDAGHamiltonianConfig, EdgePathDAGQUBOHamiltonian,
    EdgeCycleCoverDAGHamiltonianConfig, EdgeCycleCoverDAGQUBOHamiltonian,
    qubo_sample_for_order)

PRESETS = {
    "unit": {"position": [1,1,1,0], "a": [1,1,0], "b": [1,0]},
    "history-pure": {"position": [70,70,55,0], "a": [40,60,0], "b": [40,0]},
    "history-weighted": {"position": [70,70,55,35], "a": [40,60,20], "b": [40,40]},
    "scaled-pure": {"position": [40,40,40,0], "a": [40,40,0], "b": [40,0]},
    "shared-weighted": {"position": [70,70,55,20], "a": [40,60,20], "b": [40,20]},
    "shared40": {"position": [140,140,110,40], "a": [40,60,40], "b": [40,40]},
}


def builder(method, cfg):
    if method == "position":
        r,p,m,w = cfg
        if w:
            return WeightedOverlapQUBOHamiltonian(WeightedOverlapHamiltonianConfig(
                read_once_penalty=r, position_once_penalty=p, missing_edge_penalty=m,
                edge_reward_scale=w, score_mode="overlap_len", normalize_rewards=True))
        return MissingEdgeQUBOHamiltonian(MissingEdgeHamiltonianConfig(r,p,m))
    if method == "a":
        c,d,w = cfg
        return EdgePathDAGQUBOHamiltonian(EdgePathDAGHamiltonianConfig(
            edge_count_penalty=c,degree_penalty=d,edge_reward_scale=w,score_mode="overlap_len"))
    d,w = cfg
    return EdgeCycleCoverDAGQUBOHamiltonian(EdgeCycleCoverDAGHamiltonianConfig(
        degree_penalty=d,edge_reward_scale=w,score_mode="overlap_len"))


def inspect_bits(method, pure, edges, bits):
    return base.evaluate(method,pure,edges,bits)


def certify(method, config, edges, reference_energy, n):
    reward=config[-1]
    if reward == 0:
        assert abs(reference_energy)<1e-6
        return {"certified":True,"lower_bound":0,"proof":"sum of nonnegative constraint penalties"}
    assert all(e.overlap_len >= 0 for e in edges)
    bound=-reward*(n-1)
    attained=abs(reference_energy-bound)<1e-6
    if method == "position":
        # c_j=1+b_j (integers): sum_j c_j*c_(j+1) <= N-1+3*sum_j b_j^2.
        sufficient=config[1]>=3*reward
        proof="position penalty >= 3*max edge reward; bound on adjacent column-count products"
    elif method == "a":
        # k=N-1+d: A_count*d^2 - reward*d >= 0 for integer d if A_count>=reward.
        sufficient=config[0]>=reward
        proof="edge-count penalty >= max edge reward; integer squared count bound"
    else:
        # Endpoint and read-degree square residuals sum to at least 2*abs(k-(N-1)).
        sufficient=2*config[0]>=reward
        proof="degree and endpoint residuals >= 2*absolute edge-count difference"
    return {"certified":attained and sufficient,"lower_bound":bound if sufficient else None,"proof":proof}


def trial(task):
    method,n,seed,preset,reads,sweeps,beta,gamma,data,out = task
    payload,rr,ee = base.load_graph(Path(data)/f"graph-{n}.json")
    config = PRESETS[preset][method]
    model = builder(method,config).build(rr,ee,weight_mode="overlap_len")
    pure = base.build_model(method,rr,ee)
    # Variable layout is identical with/without overlap rewards.
    assert [model.variable_label(i) for i in range(model.num_variables)] == [pure.variable_label(i) for i in range(pure.num_variables)]
    ref = qubo_sample_for_order(model,payload["reference_order"])
    reference_energy = model.energy(ref)
    certificate=certify(method,config,ee,reference_energy,n)
    assert inspect_bits(method,pure,ee,ref)["valid_edge_path"]
    q = {(i,i):v for i,v in enumerate(model.linear)}
    q.update({k:v for k,v in model.quadratic.items() if v})
    bqm = oj.BinaryQuadraticModel.from_qubo(q)
    sampler = oj.SQASampler()
    destination = Path(out)/f"{preset}-{method}-{n}-{seed}"
    destination.mkdir(parents=True,exist_ok=True)
    rng = random.Random(seed ^ 0x5A17C9E3)
    initial_rows, final_rows, results, sampler_seeds = [],[],[],[]
    started = time.perf_counter()
    # Each read is a separate sampler call: a fixed OpenJij seed is otherwise reused internally.
    for index in range(reads):
        sampler_seed = rng.randrange(1,2**31)
        initial = [rng.randrange(2) for _ in range(model.num_variables)]
        response = sampler.sample(bqm,seed=sampler_seed,num_reads=1,num_sweeps=sweeps,
            beta=beta,gamma=gamma,trotter=32,reinitialize_state=True,
            initial_state=dict(enumerate(initial)))
        record = response.first
        bits = [int(record.sample[i]) for i in range(model.num_variables)]
        row = inspect_bits(method,pure,ee,bits)
        energy = model.energy(bits)
        assert abs(energy-(float(record.energy)+model.constant)) < 1e-5
        row.update(energy=energy,energy_gap=energy-reference_energy,
            reference_hit=row["valid_edge_path"] and abs(energy-reference_energy)<1e-6,
            ground_hit=certificate["certified"] and row["valid_edge_path"] and abs(energy-reference_energy)<1e-6,
            constraint_energy=row["energy"], objective_energy=energy-row["energy"],
            read_index=index,sampler_seed=sampler_seed)
        # For weighted runs, reported constraint_energy below is recalculated with actual coefficients.
        r = row["residuals"]
        if method == "position": constraint=config[0]*r["read_once"]+config[1]*r["position_once"]+config[2]*r["missing_edges"]
        elif method == "a": constraint=config[0]*r["edge_count"]+config[1]*r["degree_pairs"]
        else: constraint=config[0]*sum(r.values())
        row.update(constraint_energy=constraint,objective_energy=energy-constraint)
        initial_rows.append(initial);final_rows.append(bits);sampler_seeds.append(sampler_seed);results.append(row)
    seconds = time.perf_counter()-started
    np.savez_compressed(destination/"samples.npz",initial=np.array(initial_rows,dtype=np.uint8),
        samples=np.array(final_rows,dtype=np.uint8),seeds=sampler_seeds)
    with gzip.open(destination/"diagnostics.json.gz","wt",encoding="utf-8") as f:json.dump(results,f)
    model_data={"linear":model.linear,"quadratic":[[i,j,c] for (i,j),c in sorted(model.quadratic.items()) if c],
        "constant":model.constant,"labels":[model.variable_label(i) for i in range(model.num_variables)]}
    base.write_json(destination/"model.json",model_data)
    record={"method":method,"n":n,"seed":seed,"preset":preset,"coefficients":config,
        "num_reads":reads,"num_sweeps":sweeps,"beta":beta,"gamma":gamma,"trotter":32,
        "graph_sha256":base.digest(payload),"model_sha256":base.digest(model_data),
        "variables":model.num_variables,"quadratic_terms":sum(v!=0 for v in model.quadratic.values()),
        "initial_unique":len({tuple(x) for x in initial_rows}),"sample_unique":len({tuple(x) for x in final_rows}),
        "certificate":certificate,"feasible_samples":sum(r["valid_edge_path"] for r in results),
        "ground_samples":sum(r["ground_hit"] for r in results),"reference_energy":reference_energy,
        "valid_trial":any(r["valid_edge_path"] for r in results),"ground_trial":any(r["ground_hit"] for r in results),
        "seconds":seconds,"best":min(results,key=lambda r:r["energy"]),
        "best_valid":min((r for r in results if r["valid_edge_path"]),key=lambda r:r["energy"],default=None)}
    base.write_json(destination/"result.json",record)
    print(json.dumps({k:v for k,v in record.items() if k not in ["best","best_valid"]}),flush=True)
    return record


def run(tasks,workers):
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(trial,tasks))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="chapter3-inputs")
    p.add_argument("--out",default="chapter3-local-pilot")
    p.add_argument("--presets",default="unit,history-pure,history-weighted")
    p.add_argument("--sizes",default="8,20")
    p.add_argument("--methods",default="position,a,b")
    p.add_argument("--seeds",default="900011,900013")
    p.add_argument("--reads",type=int,default=20)
    p.add_argument("--sweeps",type=int,default=1000)
    p.add_argument("--beta",type=float,default=5.0)
    p.add_argument("--gamma",type=float,default=1.0)
    p.add_argument("--workers",type=int,default=3)
    a=p.parse_args()
    settings=vars(a)
    settings["environment"]={k:importlib.metadata.version(k) for k in ["openjij","dimod","numpy","networkx","parasail"]}
    settings["sampler_source_sha256"]=hashlib.sha256(__import__('inspect').getsource(oj.SQASampler._cxxjij_sampling).encode()).hexdigest()
    settings["solver_sha256"]=hashlib.sha256(Path(__import__('olc_pipeline.layout_solver',fromlist=['x']).__file__).read_bytes()).hexdigest()
    base.write_json(Path(a.out)/"run_config.json",settings)
    Path(a.out,'requirements-lock.txt').write_text(subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True))
    tasks=[(m,n,s,c,a.reads,a.sweeps,a.beta,a.gamma,a.data,a.out)
        for c in a.presets.split(",") for n in map(int,a.sizes.split(","))
        for m in a.methods.split(",") for s in map(int,a.seeds.split(","))
        if m != "position" or n <= 20]
    rows=run(tasks,a.workers)
    base.write_json(Path(a.out)/"results.json",rows)

