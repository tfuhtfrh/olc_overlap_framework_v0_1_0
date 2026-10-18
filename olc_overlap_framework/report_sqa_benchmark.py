"""Stage-report SQA comparison; no intra-sweep observation or solution repair."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import dimod
import networkx as nx
import openjij as oj

from experimental_chm13_latent_rank_comparator_qubo import (
    build_model, encode_order, audit as rank_audit,
)
from demo_chm13_edge_ordered_path_qubo import DATASET_DIR

# Shared, reviewable experimental configuration.
SEEDS = (2026101801, 2026101802, 2026101803, 2026101804, 2026101805, 2026101806)
PILOT_SEEDS = (2026101811, 2026101812)
PILOT_CYCLE_PENALTIES = (0.01, 0.05, 0.1, 0.5, 1.0, 4.0)
READS = 32
ITERATIONS = 16
SWEEPS_PER_ITERATION = 1300
BETA, GAMMA, TROTTER = 5.0, 1.0, 8
DEGREE_PENALTY, COUNT_PENALTY = 288.0, 32.0


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def environment():
    packages = {}
    for package in ("openjij", "cimod", "dimod", "numpy", "networkx", "dwave-samplers"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "commit": os.environ.get("GITHUB_SHA"),
            "runner_image": os.environ.get("ImageOS"),
            "runner_image_version": os.environ.get("ImageVersion")}


def problem():
    model, rewards, cfg = build_model()
    edges = model.edge_pairs
    shift = max(rewards.values())
    norm = max(shift - rewards[e] for e in edges)
    costs = {e: (shift - rewards[e]) / norm for e in edges}
    return model, rewards, cfg, edges, costs


def rank_bqm(model):
    return dimod.BinaryQuadraticModel(dict(enumerate(model.linear)),
                                     model.quadratic, model.constant, dimod.BINARY)


def edge_bqm(model, edges, costs, cycles=(), cycle_penalty=0.0):
    bqm = dimod.BinaryQuadraticModel({i: costs[e] for i, e in enumerate(edges)}, {}, 0, dimod.BINARY)
    incoming = {v: [] for v in model.read_ids}
    outgoing = {v: [] for v in model.read_ids}
    for i, (u, v) in enumerate(edges):
        outgoing[u].append(i)
        incoming[v].append(i)
    bqm.add_linear_equality_constraint([(i, 1) for i in range(len(edges))],
                                     COUNT_PENALTY, -(len(model.read_ids) - 1))
    for vertex in model.read_ids:
        for incident in (incoming[vertex], outgoing[vertex]):
            for i, j in itertools.combinations(incident, 2):
                bqm.add_quadratic(i, j, DEGREE_PENALTY)
    for cycle in cycles:
        for i in cycle:
            bqm.add_linear(i, cycle_penalty)
    return bqm


def path_metrics(bits, model, edges, rewards):
    selected = [e for i, e in enumerate(edges) if bits[i]]
    indeg = dict.fromkeys(model.read_ids, 0)
    outdeg = dict.fromkeys(model.read_ids, 0)
    nxt = {}
    for u, v in selected:
        indeg[v] += 1
        outdeg[u] += 1
        nxt[u] = v
    conflicts = sum(d * (d - 1) // 2 for d in (*indeg.values(), *outdeg.values()))
    sources = [v for v in model.read_ids if indeg[v] == 0]
    sinks = [v for v in model.read_ids if outdeg[v] == 0]
    order, seen = [], set()
    if not conflicts and len(sources) == len(sinks) == 1:
        v = sources[0]
        while v not in seen:
            order.append(v)
            seen.add(v)
            if v not in nxt:
                break
            v = nxt[v]
    valid = (not conflicts and len(selected) == len(model.read_ids) - 1
             and len(sources) == len(sinks) == 1 and len(order) == len(model.read_ids)
             and order[-1] == sinks[0])
    return {"valid_path": bool(valid), "path_score": int(sum(rewards[e] for e in selected)) if valid else None,
            "selected_edges": len(selected), "sources": len(sources), "sinks": len(sinks),
            "degree_conflicts": conflicts, "edge_count_residual": len(selected) - (len(model.read_ids) - 1)}, order


def current_cycles(bits, model, edges):
    graph = nx.DiGraph()
    graph.add_nodes_from(model.read_ids)
    graph.add_edges_from(e for i, e in enumerate(edges) if bits[i])
    edge_index = {e: i for i, e in enumerate(edges)}
    return sorted(tuple(sorted(edge_index[e] for e in zip(c, c[1:] + c[:1])))
                  for c in nx.simple_cycles(graph))


def sample(bqm, initial, seed, reads, sweeps):
    q = {(i, i): float(bqm.linear[i]) for i in bqm.variables}
    q.update({tuple(sorted((i, j))): float(b) for (i, j), b in bqm.quadratic.items() if b})
    start = time.perf_counter()
    response = oj.SQASampler().sample_qubo(
        q, initial_state={i: int(initial[i]) for i in bqm.variables},
        reinitialize_state=True, num_reads=reads, num_sweeps=sweeps,
        beta=BETA, gamma=GAMMA, trotter=TROTTER, seed=seed)
    seconds = time.perf_counter() - start
    samples = []
    for row in response.data(fields=["sample", "energy", "num_occurrences"], sorted_by=None):
        bits = [int(row.sample[i]) for i in range(len(bqm.variables))]
        energy = float(bqm.energy(bits))
        if abs(energy - (float(row.energy) + bqm.offset)) > 1e-4:
            raise AssertionError("Sampler energy disagrees with constructed QUBO")
        samples.append((bits, energy, int(row.num_occurrences)))
    assert sum(n for _, _, n in samples) == reads
    return samples, seconds


def run(method, seed, cycle_penalty, out, reads=READS, iterations=ITERATIONS, sweeps=SWEEPS_PER_ITERATION):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model, rewards, cfg, edges, costs = problem()
    nvars = model.num_variables if method == "rank" else len(edges)
    initial = [0] * nvars
    cycles = []
    incumbent, best_order, first_feasible = None, None, None
    trace, feasible_reads = [], 0
    calls = 1 if method == "rank" else iterations
    total_start = time.perf_counter()
    write_json(out / "problem.json", {"edges": edges, "weights": [rewards[e] for e in edges],
               "read_ids": model.read_ids, "rank_config": cfg.__dict__, "rank_variables": model.num_variables,
               "rank_quadratic": len(model.quadratic), "cost_normalizer": max(rewards.values()) - min(rewards.values())})
    with gzip.open(out / "samples.jsonl.gz", "wt", encoding="utf-8") as raw:
        for iteration in range(calls):
            bqm = rank_bqm(model) if method == "rank" else edge_bqm(model, edges, costs, cycles, cycle_penalty)
            call_seed = seed + 100000 * iteration
            call_sweeps = iterations * sweeps if method == "rank" else sweeps
            samples, seconds = sample(bqm, initial, call_seed, reads, call_sweeps)
            valid_this_call, fully_feasible = 0, 0
            for read_index, (bits, energy, occurrences) in enumerate(samples):
                metrics, order = path_metrics(bits, model, edges, rewards)
                diagnostic = rank_audit(bits, model, rewards, cfg) if method == "rank" else {}
                raw.write(json.dumps({"iteration": iteration, "read_index": read_index,
                          "occurrences": occurrences, "bits": "".join(map(str, bits)),
                          "energy": energy, "path": metrics, "rank_audit": diagnostic}) + "\n")
                if metrics["valid_path"]:
                    valid_this_call += occurrences
                    feasible_reads += occurrences
                    score = metrics["path_score"]
                    if first_feasible is None:
                        first_feasible = {"iteration": iteration, "returned_sample_index": read_index}
                    if incumbent is None or score > incumbent:
                        incumbent, best_order = score, order
                        write_json(out / "best_path.json", {"score": score, "order": order,
                                   "iteration": iteration, "returned_sample_index": read_index,
                                   "all_qubo_constraints_satisfied": diagnostic.get("valid_path")})
                if diagnostic.get("valid_path"):
                    fully_feasible += occurrences
            # Advancement uses current-H minimum energy; output uses best feasible original weight.
            initial = min(samples, key=lambda row: row[1])[0]
            metrics, _ = path_metrics(initial, model, edges, rewards)
            cycles = current_cycles(initial, model, edges)
            entry = {"iteration": iteration, "call_seed": call_seed, "seconds": seconds,
                     "variables": len(bqm.variables), "quadratic_terms": len(bqm.quadratic),
                     "valid_returned_reads": valid_this_call, "fully_feasible_returned_reads": fully_feasible,
                     "minimum_current_energy": min(en for _, en, _ in samples),
                     "cycle_count_after_call": len(cycles), "explorer": metrics,
                     "best_feasible_score": incumbent}
            trace.append(entry)
            write_json(out / "trace.json", trace)
            print(json.dumps({"method": method, "seed": seed, **entry}), flush=True)
    result = {"method": method, "seed": seed, "reads": reads, "iterations": iterations,
              "sweeps_per_iteration": sweeps, "beta": BETA, "gamma": GAMMA, "trotter": TROTTER,
              "cycle_penalty": cycle_penalty if method == "edge" else None,
              "degree_penalty_edge": DEGREE_PENALTY, "count_penalty_edge": COUNT_PENALTY,
              "initial_state": "all_zero", "subsequent_edge_initialization": "previous_minimum_current_energy",
              "best_score": incumbent, "ever_feasible": incumbent is not None,
              "feasible_returned_reads": feasible_reads, "first_feasible": first_feasible,
              "sampling_seconds": sum(r["seconds"] for r in trace),
              "total_seconds": time.perf_counter() - total_start, "environment": environment()}
    # Reference certificate is read for offline evaluation only after all sampling has finished.
    ref = json.loads((DATASET_DIR / "reference_path.json").read_text())["normalized_nodes"]
    reference_score = int(sum(rewards[e] for e in zip(ref, ref[1:])))
    result.update(reference_score=reference_score, optimum_hit=incumbent == reference_score,
                  relative_gap=None if incumbent is None else (reference_score - incumbent) / reference_score)
    write_json(out / "summary.json", result)
    return result


def preflight(out):
    out = Path(out)
    model, rewards, cfg, edges, costs = problem()
    ref = json.loads((DATASET_DIR / "reference_path.json").read_text())["normalized_nodes"]
    encoded = encode_order(model, ref)
    check = rank_audit(encoded, model, rewards, cfg)
    assert check["valid_path"] and check["comparator_violations"] == 0
    assert model.num_variables == 3789 and len(model.quadratic) == 33376
    assert max(abs(x) for x in model.quadratic.values()) == 576
    base = edge_bqm(model, edges, costs)
    assert len(base.variables) == 261
    ref_bits = encoded[:len(edges)]
    metrics, _ = path_metrics(ref_bits, model, edges, rewards)
    assert metrics["valid_path"] and not current_cycles(ref_bits, model, edges)
    assert abs(base.energy(ref_bits) - sum(costs[e] for e in zip(ref, ref[1:]))) < 1e-6
    # Check the circuit independently for all four-bit truth-table rows.
    for u, v, b, c in itertools.product((0, 1), repeat=4):
        residual = u+b+c+u*b-u*v-b*v-2*u*c-2*b*c+2*v*c
        assert residual >= 0 and ((residual == 0) == (c == int(u+b > v)))
    run("rank", PILOT_SEEDS[0], 0, out / "smoke_rank", reads=2, iterations=1, sweeps=50)
    run("edge", PILOT_SEEDS[0], 0.1, out / "smoke_edge", reads=2, iterations=2, sweeps=50)
    write_json(out / "audit.json", {"reference_audit": check, "edge_reference": metrics,
               "environment": environment(), "dataset_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in DATASET_DIR.iterdir() if p.is_file()}})
    # Disjoint pilot seeds, fixed full iteration count, fewer reads; no optimum oracle in selection.
    candidates = []
    for penalty in PILOT_CYCLE_PENALTIES:
        results = [run("edge", seed, penalty, out / f"pilot_A{penalty}_seed{seed}", reads=8)
                   for seed in PILOT_SEEDS]
        scores = [r["best_score"] for r in results if r["ever_feasible"]]
        candidates.append({"cycle_penalty": penalty, "feasible_runs": len(scores),
                           "median_score": statistics.median(scores) if scores else None})
    best = max(candidates, key=lambda r: (r["feasible_runs"],
               r["median_score"] if r["median_score"] is not None else float("-inf"), -r["cycle_penalty"]))
    write_json(out / "selected_config.json", {"cycle_penalty": best["cycle_penalty"],
               "pilot_candidates": candidates, "pilot_seeds": PILOT_SEEDS,
               "selection_rule": "feasible-run count, then median feasible original weight, then smaller penalty",
               "formal_seeds": SEEDS, "environment": environment()})


def aggregate(root, out):
    results = [json.loads(p.read_text()) for p in Path(root).rglob("summary.json")]
    formal = [r for r in results if r["seed"] in SEEDS and r["reads"] == READS]
    assert len(formal) == 12 and len({(r["method"], r["seed"]) for r in formal}) == 12
    agg = {}
    for method in ("rank", "edge"):
        rows = [r for r in formal if r["method"] == method]
        scores = [r["best_score"] for r in rows if r["ever_feasible"]]
        q = statistics.quantiles(scores, n=4, method="inclusive") if len(scores) >= 2 else None
        agg[method] = {"runs": len(rows), "feasible_runs": len(scores),
                       "optimum_hits": sum(r["optimum_hit"] for r in rows),
                       "best": max(scores) if scores else None,
                       "median_feasible_score": statistics.median(scores) if scores else None,
                       "iqr_feasible_score": q[2]-q[0] if q else None,
                       "median_sampling_seconds": statistics.median(r["sampling_seconds"] for r in rows)}
    write_json(out, {"aggregate": agg, "trials": formal})
    print(json.dumps(agg, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run", "aggregate"))
    parser.add_argument("--method", choices=("rank", "edge"))
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight(args.out)
    elif args.mode == "aggregate":
        aggregate(args.input, args.out)
    else:
        penalty = json.loads(args.config.read_text())["cycle_penalty"]
        run(args.method, SEEDS[args.seed_index], penalty, args.out)
