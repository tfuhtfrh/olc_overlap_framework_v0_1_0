"""Cross-solver benchmark for the CHM13 complex144 edge-ordered QUBO.

This benchmark intentionally keeps the Hamiltonian fixed.  It compares solver
behaviour on the exact same QUBO and reports both raw QUBO energy and structural
Hamilton-path validity.  A second, uniformly rescaled copy of the same QUBO is
also tested; uniform positive scaling preserves the exact minimizers and is
included only as a numerical-conditioning diagnostic.

The existing OpenJij SQA result is read from the repository rather than rerun,
so this script focuses on additional solver families.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from demo_chm13_edge_ordered_path_qubo import (  # noqa: E402
    DATASET_DIR,
    _energy_components,
    load_chm13_graph,
)
from olc_pipeline.layout_solver import (  # noqa: E402
    EdgeOrderedPathHamiltonianConfig,
    EdgeOrderedPathQUBOHamiltonian,
    QUBOLayoutSolver,
    qubo_sample_for_order,
)


SEED = 20260922


def build_problem():
    reads, edges, reward_by_pair = load_chm13_graph(DATASET_DIR)
    hamiltonian = EdgeOrderedPathQUBOHamiltonian(
        EdgeOrderedPathHamiltonianConfig(
            degree_penalty=None,
            order_penalty=None,
            activation_penalty=None,
            edge_cost_scale=1.0,
            cost_mode="shifted_reward",
            score_mode="dp",
            normalize_costs=True,
            include_source_length=False,
        )
    )
    model = hamiltonian.build(reads, edges)

    reference = json.loads(
        (DATASET_DIR / "reference_path.json").read_text(encoding="utf-8")
    )
    reference_order = list(reference["normalized_nodes"])
    reference_sample = qubo_sample_for_order(model, reference_order)
    reference_energy = float(model.energy(reference_sample))

    certificate = json.loads(
        (DATASET_DIR / "weighted_certificate.json").read_text(encoding="utf-8")
    )
    runner = next(
        entry
        for entry in certificate["top_paths"]
        if int(entry["score"]) == int(certificate["runner_up_score"])
    )
    runner_order = [node[:-1] for node in runner["nodes"]]
    runner_sample = qubo_sample_for_order(model, runner_order)
    runner_energy = float(model.energy(runner_sample))

    return {
        "reads": reads,
        "edges": edges,
        "reward_by_pair": reward_by_pair,
        "hamiltonian": hamiltonian,
        "model": model,
        "reference_order": reference_order,
        "reference_sample": reference_sample,
        "reference_energy": reference_energy,
        "reference_score": int(certificate["best_score"]),
        "runner_energy": runner_energy,
        "runner_score": int(certificate["runner_up_score"]),
    }


def evaluate(problem: dict[str, Any], sample: list[int]) -> dict[str, Any]:
    model = problem["model"]
    hamiltonian = problem["hamiltonian"]
    reward_by_pair = problem["reward_by_pair"]
    ref = problem["reference_sample"]

    sample = [int(v) for v in sample]
    order, meta = QUBOLayoutSolver._decode_edge_ordered_path(model, sample)
    energy = float(model.energy(sample))
    components = _energy_components(
        model, hamiltonian, sample, reward_by_pair, 0.0
    )
    selected_pairs = [
        pair
        for pair in model.edge_pairs
        if sample[model.edge_variable_index(*pair)]
    ]
    selected_reward = int(sum(reward_by_pair[pair] for pair in selected_pairs))
    if meta["valid_edge_path"]:
        path_score = int(
            sum(reward_by_pair[pair] for pair in zip(order, order[1:]))
        )
    else:
        path_score = None

    return {
        "energy": energy,
        "energy_above_reference": energy - problem["reference_energy"],
        "ground_hit": abs(energy - problem["reference_energy"]) <= 1e-6,
        "valid_edge_path": bool(meta["valid_edge_path"]),
        "path_score": path_score,
        "score_gap_from_optimum": (
            problem["reference_score"] - path_score if path_score is not None else None
        ),
        "selected_edge_reward_sum": selected_reward,
        "selected_edge_count": int(meta["selected_edge_count"]),
        "selected_source_count": int(meta["selected_source_count"]),
        "selected_sink_count": int(meta["selected_sink_count"]),
        "read_in_constraint_violations": int(meta["read_in_constraint_violations"]),
        "read_out_constraint_violations": int(meta["read_out_constraint_violations"]),
        "void_in_constraint_violation": int(meta["void_in_constraint_violation"]),
        "void_out_constraint_violation": int(meta["void_out_constraint_violation"]),
        "activation_violations": int(meta["activation_violations"]),
        "order_constraint_violations": int(meta["order_constraint_violations"]),
        "position_sequence_valid": bool(meta["position_sequence_valid"]),
        "hamming_from_reference": int(sum(a != b for a, b in zip(sample, ref))),
        "objective_energy": float(components["objective"]),
        "degree_energy": float(components["degree"]),
        "activation_energy": float(components["activation"]),
        "order_energy": float(components["order"]),
        "constraint_energy": float(
            components["degree"] + components["activation"] + components["order"]
        ),
    }


def bqm_for(problem: dict[str, Any], scale: float):
    bqm = problem["model"].to_dimod_bqm()
    if scale != 1.0:
        bqm.scale(scale)
    return bqm


def sampleset_rows(sampleset, n: int):
    rows = []
    for datum in sampleset.data(fields=["sample", "energy"], sorted_by="energy"):
        sample_map = datum.sample
        sample = [int(sample_map.get(i, 0)) for i in range(n)]
        rows.append(sample)
    return rows


def run_sa(problem, scale: float, num_reads: int, num_sweeps: int, seed: int):
    from dwave.samplers import SimulatedAnnealingSampler

    sampler = SimulatedAnnealingSampler()
    bqm = bqm_for(problem, scale)
    t0 = time.perf_counter()
    ss = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
        randomize_order=True,
        beta_schedule_type="geometric",
    )
    dt = time.perf_counter() - t0
    return sampleset_rows(ss, problem["model"].num_variables), dt


def run_tabu(problem, scale: float, num_reads: int, timeout_ms: int, seed: int):
    from dwave.samplers import TabuSampler

    sampler = TabuSampler()
    bqm = bqm_for(problem, scale)
    t0 = time.perf_counter()
    ss = sampler.sample(
        bqm,
        num_reads=num_reads,
        timeout=timeout_ms,
        seed=seed,
    )
    dt = time.perf_counter() - t0
    return sampleset_rows(ss, problem["model"].num_variables), dt


def qubo_polynomial_tensors(problem, scale: float):
    import torch

    model = problem["model"]
    n = model.num_variables
    dtype = torch.float64
    matrix = torch.zeros((n, n), dtype=dtype)
    vector = torch.tensor(model.linear, dtype=dtype)
    for (i, j), coeff in model.quadratic.items():
        value = 0.5 * float(coeff)
        matrix[i, j] = value
        matrix[j, i] = value
    constant = float(model.constant)
    if scale != 1.0:
        matrix *= scale
        vector *= scale
        constant *= scale
    return matrix, vector, constant


def _candidate_vectors(obj, n: int):
    """Return a list of n-bit vectors from a torch/numpy SB result."""
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()
        else:
            arr = np.asarray(obj)
    except Exception:
        arr = np.asarray(obj)

    arr = np.asarray(arr)
    if arr.ndim == 1:
        if arr.size != n:
            raise ValueError(f"SB vector has size {arr.size}, expected {n}")
        return [[int(round(v)) for v in arr.tolist()]]
    if arr.ndim != 2:
        raise ValueError(f"unexpected SB vector shape: {arr.shape}")

    if arr.shape[0] == n:
        arr = arr.T
    elif arr.shape[1] != n:
        raise ValueError(f"unexpected SB vector shape: {arr.shape}")
    return [[int(round(v)) for v in row.tolist()] for row in arr]


def run_sb(
    problem,
    scale: float,
    agents: int,
    max_steps: int,
    seed: int,
    mode: str,
    heated: bool,
):
    import torch
    import simulated_bifurcation as sb

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    matrix, vector, constant = qubo_polynomial_tensors(problem, scale)

    t0 = time.perf_counter()
    # Current API returns (value, vector). best_only=False lets us inspect every
    # agent and score all returned bitstrings with the repository's own QUBO.
    value, vectors = sb.minimize(
        matrix,
        vector,
        constant,
        domain="binary",
        agents=agents,
        max_steps=max_steps,
        best_only=False,
        mode=mode,
        heated=heated,
        early_stopping=True,
        sampling_period=20,
        convergence_threshold=20,
    )
    dt = time.perf_counter() - t0
    del value
    return _candidate_vectors(vectors, problem["model"].num_variables), dt


def summarize_variant(
    problem,
    solver: str,
    variant: str,
    samples: list[list[int]],
    seconds: float,
    params: dict[str, Any],
):
    evaluations = [evaluate(problem, sample) for sample in samples]
    evaluations.sort(
        key=lambda row: (
            not row["valid_edge_path"],
            row["energy"],
        )
    )
    best = min(evaluations, key=lambda row: row["energy"])
    best_feasible = next(
        (row for row in evaluations if row["valid_edge_path"]),
        None,
    )
    return {
        "solver": solver,
        "variant": variant,
        "seconds": seconds,
        "num_returned_samples": len(evaluations),
        "feasible_hits": sum(row["valid_edge_path"] for row in evaluations),
        "ground_hits": sum(row["ground_hit"] for row in evaluations),
        "params": params,
        "best_energy": best["energy"],
        "best_energy_above_reference": best["energy_above_reference"],
        "best_valid_edge_path": best["valid_edge_path"],
        "best_constraint_energy": best["constraint_energy"],
        "best_order_energy": best["order_energy"],
        "best_read_in_violations": best["read_in_constraint_violations"],
        "best_read_out_violations": best["read_out_constraint_violations"],
        "best_activation_violations": best["activation_violations"],
        "best_order_residual_l1": best["order_constraint_violations"],
        "best_hamming_from_reference": best["hamming_from_reference"],
        "best_feasible_energy": (
            best_feasible["energy"] if best_feasible is not None else None
        ),
        "best_feasible_path_score": (
            best_feasible["path_score"] if best_feasible is not None else None
        ),
        "best_feasible_score_gap": (
            best_feasible["score_gap_from_optimum"]
            if best_feasible is not None
            else None
        ),
        "best_evaluation": best,
        "best_feasible_evaluation": best_feasible,
    }


def existing_sqa_row(problem):
    path = (
        HERE
        / "debug"
        / "qubo"
        / "chm13_edge_ordered_path"
        / "sqa_random_late_s050_r8_s2000_t8_seed20260922.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    sqa = data["sqa"]
    ec = sqa["energy_components"]
    return {
        "solver": "OpenJij SQA",
        "variant": "repository-existing",
        "seconds": float(sqa["seconds"]),
        "num_returned_samples": int(sqa["num_reads"]),
        "feasible_hits": 0,
        "ground_hits": 0,
        "params": {
            "num_reads": sqa["num_reads"],
            "num_sweeps": sqa["num_sweeps"],
            "trotter": sqa["trotter"],
            "start_s": sqa["start_s"],
            "seed": sqa["seed"],
        },
        "best_energy": float(sqa["energy"]),
        "best_energy_above_reference": float(sqa["energy"]) - problem["reference_energy"],
        "best_valid_edge_path": bool(sqa["valid_edge_path"]),
        "best_constraint_energy": float(ec["degree"] + ec["activation"] + ec["order"]),
        "best_order_energy": float(ec["order"]),
        "best_read_in_violations": int(sqa["read_in_constraint_violations"]),
        "best_read_out_violations": int(sqa["read_out_constraint_violations"]),
        "best_activation_violations": int(sqa["activation_violations"]),
        "best_order_residual_l1": int(sqa["order_constraint_violations"]),
        "best_hamming_from_reference": int(sqa["hamming_distance_from_reference"]),
        "best_feasible_energy": None,
        "best_feasible_path_score": None,
        "best_feasible_score_gap": None,
        "best_evaluation": None,
        "best_feasible_evaluation": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sa-reads", type=int, default=16)
    parser.add_argument("--sa-sweeps", type=int, default=5000)
    parser.add_argument("--tabu-reads", type=int, default=8)
    parser.add_argument("--tabu-timeout-ms", type=int, default=5000)
    parser.add_argument("--sb-agents", type=int, default=4)
    parser.add_argument("--sb-steps", type=int, default=400)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    problem = build_problem()
    model = problem["model"]
    max_abs = max(
        max(abs(float(v)) for v in model.linear),
        max(abs(float(v)) for v in model.quadratic.values()),
    )
    scale = 1.0 / max_abs

    rows = [existing_sqa_row(problem)]
    errors = []

    tests = [
        (
            "D-Wave SA",
            "raw",
            lambda: run_sa(problem, 1.0, args.sa_reads, args.sa_sweeps, SEED),
            {"num_reads": args.sa_reads, "num_sweeps": args.sa_sweeps, "seed": SEED},
        ),
        (
            "D-Wave SA",
            "uniform-maxabs",
            lambda: run_sa(problem, scale, args.sa_reads, args.sa_sweeps, SEED),
            {
                "num_reads": args.sa_reads,
                "num_sweeps": args.sa_sweeps,
                "seed": SEED,
                "uniform_scale": scale,
            },
        ),
        (
            "D-Wave Tabu",
            "raw",
            lambda: run_tabu(
                problem, 1.0, args.tabu_reads, args.tabu_timeout_ms, SEED
            ),
            {
                "num_reads": args.tabu_reads,
                "timeout_ms_per_read": args.tabu_timeout_ms,
                "seed": SEED,
            },
        ),
        (
            "D-Wave Tabu",
            "uniform-maxabs",
            lambda: run_tabu(
                problem, scale, args.tabu_reads, args.tabu_timeout_ms, SEED
            ),
            {
                "num_reads": args.tabu_reads,
                "timeout_ms_per_read": args.tabu_timeout_ms,
                "seed": SEED,
                "uniform_scale": scale,
            },
        ),
        (
            "Simulated Bifurcation",
            "discrete-uniform-maxabs",
            lambda: run_sb(
                problem,
                scale,
                args.sb_agents,
                args.sb_steps,
                SEED,
                "discrete",
                False,
            ),
            {
                "agents": args.sb_agents,
                "max_steps": args.sb_steps,
                "mode": "discrete",
                "heated": False,
                "seed": SEED,
                "uniform_scale": scale,
            },
        ),
        (
            "Simulated Bifurcation",
            "heated-discrete-uniform-maxabs",
            lambda: run_sb(
                problem,
                scale,
                args.sb_agents,
                args.sb_steps,
                SEED + 1,
                "discrete",
                True,
            ),
            {
                "agents": args.sb_agents,
                "max_steps": args.sb_steps,
                "mode": "discrete",
                "heated": True,
                "seed": SEED + 1,
                "uniform_scale": scale,
            },
        ),
    ]

    for solver, variant, fn, params in tests:
        print(f"RUN {solver} / {variant}", flush=True)
        try:
            samples, seconds = fn()
            row = summarize_variant(
                problem, solver, variant, samples, seconds, params
            )
            rows.append(row)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "solver",
                            "variant",
                            "seconds",
                            "feasible_hits",
                            "ground_hits",
                            "best_energy",
                            "best_energy_above_reference",
                            "best_constraint_energy",
                        )
                    },
                    indent=2,
                ),
                flush=True,
            )
        except Exception as exc:
            import traceback

            err = {
                "solver": solver,
                "variant": variant,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            errors.append(err)
            print(json.dumps(err, indent=2), flush=True)

    metadata = {
        "dataset": "CHM13_complex144",
        "nodes": len(problem["reads"]),
        "edges": len(problem["edges"]),
        "logical_variables": model.num_variables,
        "quadratic_terms": len(model.quadratic),
        "max_abs_linear": max(abs(float(v)) for v in model.linear),
        "max_abs_quadratic": max(abs(float(v)) for v in model.quadratic.values()),
        "uniform_scale": scale,
        "reference_energy": problem["reference_energy"],
        "reference_score": problem["reference_score"],
        "runner_up_energy": problem["runner_energy"],
        "runner_up_score": problem["runner_score"],
        "optimum_margin": problem["runner_energy"] - problem["reference_energy"],
        "note": (
            "Penalty coefficients and relative normalization are intentionally "
            "kept exactly as in the current repository model. uniform-maxabs "
            "multiplies the complete QUBO, including the constant, by one "
            "positive scalar and therefore preserves exact minimizers."
        ),
    }

    payload = {"metadata": metadata, "results": rows, "errors": errors}
    (out / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    flat_fields = [
        "solver",
        "variant",
        "seconds",
        "num_returned_samples",
        "feasible_hits",
        "ground_hits",
        "best_energy",
        "best_energy_above_reference",
        "best_valid_edge_path",
        "best_constraint_energy",
        "best_order_energy",
        "best_read_in_violations",
        "best_read_out_violations",
        "best_activation_violations",
        "best_order_residual_l1",
        "best_hamming_from_reference",
        "best_feasible_energy",
        "best_feasible_path_score",
        "best_feasible_score_gap",
    ]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in flat_fields})

    print("\nFINAL SUMMARY")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
