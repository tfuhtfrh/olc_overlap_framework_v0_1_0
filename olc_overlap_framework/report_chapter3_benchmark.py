"""Chapter 3: pure position / edge-count / endpoint QUBOs, shared inputs and SQA budget."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import time

from olc_pipeline.data import Read, OverlapEdge, PairwiseAlignment
from olc_pipeline.layout_solver import (
    MissingEdgeHamiltonianConfig, MissingEdgeQUBOHamiltonian,
    EdgePathDAGHamiltonianConfig, EdgePathDAGQUBOHamiltonian,
    EdgeCycleCoverDAGHamiltonianConfig, EdgeCycleCoverDAGQUBOHamiltonian,
    qubo_sample_for_order,
)

# Freeze before formal sampling. No fitting to the six formal seeds.
CONFIG = {
    "small_sizes": [8, 12, 16, 20], "edge_sizes": [35, 70, 100],
    "formal_seeds": [40, 41, 42, 43, 44, 45], "pilot_seed": 900001,
    "simulation_seed": 42, "input_shuffle_seed": 710019,
    "read_length": 3000, "read_step": 500, "gc_fraction": 0.65,
    "num_reads": 100, "num_sweeps": 1000, "trotter": 32,
    "beta": 5.0, "gamma": 1.0, "constraint_coefficient": 1.0,
    "edge_reward_scale": 0.0, "auxiliary_terms": [],
    "initialization": "saved random binary vector per trial; shared across reads within trial",
    "output_repair": False, "energy_lower_bound": 0.0,
    "input_pipeline": "existing minimap2 ava-ont + Parasail; no truth-based edge injection",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_model(method, reads, edges):
    p = CONFIG["constraint_coefficient"]
    if method == "position":
        builder = MissingEdgeQUBOHamiltonian(MissingEdgeHamiltonianConfig(p, p, p))
    elif method == "a":
        builder = EdgePathDAGQUBOHamiltonian(EdgePathDAGHamiltonianConfig(
            edge_count_penalty=p, degree_penalty=p, edge_reward_scale=0,
            score_mode="overlap_len", require_hamiltonian_path=True))
    elif method == "b":
        builder = EdgeCycleCoverDAGQUBOHamiltonian(EdgeCycleCoverDAGHamiltonianConfig(
            degree_penalty=p, edge_reward_scale=0, score_mode="overlap_len"))
    else:
        raise ValueError(method)
    return builder.build(reads, edges, weight_mode="overlap_len")


def evaluate(method, model, edges, bits):
    """Inspect actual bits. Never project an invalid position assignment or repair edges."""
    ids = model.read_ids
    n = len(ids)
    allowed = {(e.left_id, e.right_id) for e in edges}
    residual = {}
    if method == "position":
        rows = [sum(bits[model.variable_index(i, j)] for j in range(n)) for i in range(n)]
        cols = [sum(bits[model.variable_index(i, j)] for i in range(n)) for j in range(n)]
        residual["read_once"] = sum((x-1)**2 for x in rows)
        residual["position_once"] = sum((x-1)**2 for x in cols)
        residual["missing_edges"] = sum(
            bits[model.variable_index(i, j)] * bits[model.variable_index(k, j+1)]
            for j in range(n-1) for i in range(n) for k in range(n)
            if (ids[i], ids[k]) not in allowed)
        order = ([ids[next(i for i in range(n) if bits[model.variable_index(i, j)])]
                  for j in range(n)] if not residual["read_once"] and not residual["position_once"] else [])
        pairs = list(zip(order, order[1:]))
    else:
        pairs = [pair for i, pair in enumerate(model.edge_pairs) if bits[i]]
        incoming = {v: sum(w == v for u, w in pairs) for v in ids}
        outgoing = {v: sum(u == v for u, w in pairs) for v in ids}
        if method == "a":
            residual["edge_count"] = (len(pairs)-(n-1))**2
            residual["degree_pairs"] = sum(x*(x-1)//2 for x in [*incoming.values(), *outgoing.values()])
        else:
            s = {v: bits[model.source_variable_index(v)] for v in ids}
            t = {v: bits[model.sink_variable_index(v)] for v in ids}
            residual["read_in"] = sum((1-s[v]-incoming[v])**2 for v in ids)
            residual["read_out"] = sum((1-t[v]-outgoing[v])**2 for v in ids)
            residual["source_count"] = (sum(s.values())-1)**2
            residual["sink_count"] = (sum(t.values())-1)**2
    incoming = {v: sum(w == v for u, w in pairs) for v in ids}
    outgoing = {v: sum(u == v for u, w in pairs) for v in ids}
    sources = [v for v in ids if incoming[v] == 0]
    sinks = [v for v in ids if outgoing[v] == 0]
    degrees_ok = all(x <= 1 for x in [*incoming.values(), *outgoing.values()])
    # Independently traverse the entire selected graph to exclude disconnected cycles.
    order = []
    if len(sources) == len(sinks) == 1 and degrees_ok:
        nxt = {u: v for u, v in pairs}
        v = sources[0]
        while v not in order:
            order.append(v)
            if v not in nxt:
                break
            v = nxt[v]
    valid = (all(x == 0 for x in residual.values()) and degrees_ok and
             len(pairs) == n-1 and len(order) == n and len(set(order)) == n and
             all(pair in allowed for pair in pairs))
    energy = model.energy(bits)
    assert abs(energy-CONFIG["constraint_coefficient"]*sum(residual.values())) < 1e-6
    assert (abs(energy) < 1e-6) == valid
    return {"energy": energy, "residuals": residual, "valid_edge_path": valid,
            "ground_hit": valid and abs(energy) < 1e-6,
            "source_count": len(sources), "sink_count": len(sinks),
            "selected_edge_count": len(pairs), "visited_nodes": len(order),
            "order": order if valid else None}


def load_graph(path):
    payload = json.loads(Path(path).read_text())
    reads = [Read(**r) for r in payload["reads"]]
    edges = []
    for record in payload["edges"]:
        record = dict(record)
        if record.get("alignment") is not None:
            record["alignment"] = PairwiseAlignment(**record["alignment"])
        edges.append(OverlapEdge(**record))
    return payload, reads, edges


def audit(out):
    from types import SimpleNamespace
    reads = [Read(f"r{i}", "ACGT", i, i+4) for i in range(3)]
    edges = [SimpleNamespace(left_id=f"r{i}", right_id=f"r{j}", accepted=True,
                            overlap_len=4-j+i) for i, j in [(0, 1), (1, 2), (0, 2)]]
    rows = []
    for method in ["position", "a", "b"]:
        model = build_model(method, reads, edges)
        count = hits = 0
        for bits in itertools.product((0, 1), repeat=model.num_variables):
            row = evaluate(method, model, edges, bits)
            hits += row["ground_hit"]
            count += 1
        assert hits == 1, (method, hits)
        ref = qubo_sample_for_order(model, [r.rid for r in reads])
        assert evaluate(method, model, edges, ref)["ground_hit"]
        rows.append({"method": method, "states_checked": count, "ground_states": hits})
    write_json(Path(out)/"exhaustive_audit.json", rows)
    print(json.dumps(rows), flush=True)


def prepare(out):
    from types import SimpleNamespace
    from demo_sqa_parameter_sweep import build_overlap_graph, select_layout_inputs
    import networkx as nx
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    audit(out)
    args = SimpleNamespace(genome_len=52500, read_len=CONFIG["read_length"],
        step=CONFIG["read_step"], mismatch_rate=0, ins_rate=0, del_rate=0,
        gc_fraction=CONFIG["gc_fraction"], simulation_seed=CONFIG["simulation_seed"])
    all_reads, all_edges = build_overlap_graph(args)
    manifests = []
    for n in CONFIG["small_sizes"]+CONFIG["edge_sizes"]:
        reads, edges = select_layout_inputs(all_reads, all_edges, n)
        edges = [e for e in edges if e.accepted]
        assert len(reads) == n
        truth = [r.rid for r in sorted(reads, key=lambda r: r.true_start)]
        pairs = {(e.left_id, e.right_id) for e in edges}
        g = nx.DiGraph()
        g.add_nodes_from(truth)
        g.add_edges_from(pairs)
        assert nx.is_directed_acyclic_graph(g)
        assert all(pair in pairs for pair in zip(truth, truth[1:]))
        random.Random(CONFIG["input_shuffle_seed"]+n).shuffle(reads)
        payload = {"reads": [asdict(r) for r in reads], "edges": [asdict(e) for e in edges],
                   "reference_order": truth}
        write_json(out/f"graph-{n}.json", payload)
        models = {}
        for method in ["position", "a", "b"] if n in CONFIG["small_sizes"] else ["a", "b"]:
            m = build_model(method, reads, edges)
            ref = qubo_sample_for_order(m, truth)
            assert evaluate(method, m, edges, ref)["ground_hit"]
            models[method] = {"variables": m.num_variables, "quadratic_terms": sum(v != 0 for v in m.quadratic.values())}
        manifests.append({"n": n, "candidate_edges": len(pairs), "graph_sha256": digest(payload), "models": models})
    write_json(out/"manifest.json", {"config": CONFIG, "config_sha256": digest(CONFIG), "graphs": manifests})
    # Independent-seed full-budget pilot checks the actual sampler interface and raw residuals.
    for method in ["position", "a", "b"]:
        run_trial(out, out/"pilot", method, 8, CONFIG["pilot_seed"])
    write_json(out/"preflight_passed.json", {"passed": True, "config_sha256": digest(CONFIG)})


def run_trial(data, out, method, n, seed):
    import numpy as np
    import openjij as oj
    payload, reads, edges = load_graph(Path(data)/f"graph-{n}.json")
    model = build_model(method, reads, edges)
    initial_seed = seed ^ 0x5A17C9E3
    rng = random.Random(initial_seed)
    initial = [rng.randrange(2) for _ in range(model.num_variables)]
    qubo = {(i, i): bias for i, bias in enumerate(model.linear)}
    qubo.update({pair: coeff for pair, coeff in model.quadratic.items() if coeff})
    destination = Path(out)/f"{method}-{n}-{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    model_data = {"linear": model.linear, "quadratic": [[i,j,c] for (i,j),c in sorted(model.quadratic.items()) if c],
                  "constant": model.constant, "labels": [model.variable_label(i) for i in range(model.num_variables)]}
    write_json(destination/"model.json", model_data)
    write_json(destination/"initial_state.json", {"seed": initial_seed, "bits": initial})
    started = time.perf_counter()
    response = oj.SQASampler().sample_qubo(qubo, seed=seed,
        num_reads=CONFIG["num_reads"], num_sweeps=CONFIG["num_sweeps"],
        trotter=CONFIG["trotter"], beta=CONFIG["beta"], gamma=CONFIG["gamma"],
        reinitialize_state=True, initial_state=dict(enumerate(initial)))
    seconds = time.perf_counter()-started
    columns = list(response.variables)
    samples = np.asarray(response.record.sample, dtype=np.uint8)
    np.savez_compressed(destination/"samples.npz", samples=samples, variables=columns,
        energy_without_constant=response.record.energy, occurrences=response.record.num_occurrences)
    records = []
    for record in response.record:
        bits = [int(record.sample[columns.index(i)]) for i in range(model.num_variables)]
        row = evaluate(method, model, edges, bits)
        assert abs(row["energy"]-(float(record.energy)+model.constant)) < 1e-6
        row["occurrences"] = int(record.num_occurrences)
        records.append(row)
    assert sum(r["occurrences"] for r in records) == CONFIG["num_reads"]
    with gzip.open(destination/"diagnostics.json.gz", "wt", encoding="utf-8") as f:
        json.dump(records, f)
    result = {"method": method, "n": n, "seed": seed, "seconds": seconds,
        "config_sha256": digest(CONFIG), "graph_sha256": digest(payload), "model_sha256": digest(model_data),
        "variables": model.num_variables, "quadratic_terms": sum(c != 0 for c in model.quadratic.values()),
        "returned_samples": sum(r["occurrences"] for r in records),
        "feasible_samples": sum(r["occurrences"] for r in records if r["valid_edge_path"]),
        "valid_trial": any(r["valid_edge_path"] for r in records),
        "ground_trial": any(r["ground_hit"] for r in records),
        "best": min(records, key=lambda r:r["energy"])}
    write_json(destination/"result.json", result)
    print(json.dumps(result), flush=True)


def aggregate(data, out):
    manifest = json.loads((Path(data)/"manifest.json").read_text())
    rows = [json.loads(p.read_text()) for p in Path(out).rglob("result.json")]
    expected = {(method, g["n"], seed) for g in manifest["graphs"] for method in g["models"] for seed in CONFIG["formal_seeds"]}
    keys = [(r["method"],r["n"],r["seed"]) for r in rows]
    assert len(keys) == len(set(keys)) and set(keys) <= expected
    hashes = {g["n"]:g["graph_sha256"] for g in manifest["graphs"]}
    assert all(r["graph_sha256"] == hashes[r["n"]] and r["config_sha256"] == digest(CONFIG) for r in rows)
    summary = []
    for n in CONFIG["small_sizes"]+CONFIG["edge_sizes"]:
        for method in ["position", "a", "b"] if n in CONFIG["small_sizes"] else ["a", "b"]:
            group = [r for r in rows if r["n"] == n and r["method"] == method]
            summary.append({"n":n,"method":method,"completed":len(group),"expected":6,
                "valid_trials":sum(r["valid_trial"] for r in group),
                "ground_trials":sum(r["ground_trial"] for r in group),
                "feasible_samples":sum(r["feasible_samples"] for r in group),
                "seconds":sum(r["seconds"] for r in group)})
    report = {"complete":set(keys)==expected,"missing":sorted(expected-set(keys)),"summary":summary}
    write_json(Path(out)/"comparison.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report["complete"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["audit","prepare","run","aggregate"])
    p.add_argument("--data", type=Path, default=Path("chapter3-inputs"))
    p.add_argument("--out", type=Path, default=Path("chapter3-results"))
    p.add_argument("--method", choices=["position","a","b"])
    p.add_argument("--n", type=int)
    a = p.parse_args()
    if a.command == "audit": audit(a.out)
    elif a.command == "prepare": prepare(a.out)
    elif a.command == "run":
        assert json.loads((a.data/"preflight_passed.json").read_text())["config_sha256"] == digest(CONFIG)
        for seed in CONFIG["formal_seeds"]: run_trial(a.data, a.out, a.method, a.n, seed)
    elif not aggregate(a.data, a.out): sys.exit(2)

