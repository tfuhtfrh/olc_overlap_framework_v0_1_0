"""Compare production D-Wave TabuSampler on two CHM13 QUBO Hamiltonians.

Models
------
1. current edge-position Hamiltonian (2,637 variables)
2. bounded-coefficient vertex-order/carry Hamiltonian (7,704 variables)

The benchmark uses dwave.samplers.TabuSampler (MST2 multistart tabu), not the
lightweight diagnostic tabu implementation used in earlier exploratory tests.

For the bounded model we test both equal penalties and the diagnostic penalty
balance previously found to give a near-feasible basin.

The weighted objective and graph data are identical across formulations.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from dwave.samplers import TabuSampler

from demo_chm13_edge_ordered_path_qubo import (
    DATASET_DIR,
    _energy_components,
    load_chm13_graph,
)
from experimental_chm13_vertex_carry_qubo import (
    BoundedVertexOrderConfig,
    BoundedVertexOrderModel,
    build_model as build_bounded_model,
    encode_order as encode_bounded_order,
)
from olc_pipeline.layout_solver import (
    EdgeOrderedPathHamiltonianConfig,
    EdgeOrderedPathQUBOHamiltonian,
    QUBOLayoutSolver,
    qubo_sample_for_order,
)


SEED = 20260925


def load_certificate() -> dict[str, Any]:
    return json.loads(
        (DATASET_DIR / "weighted_certificate.json").read_text(encoding="utf-8")
    )


def build_current_problem() -> dict[str, Any]:
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
    certificate = load_certificate()

    return {
        "name": "current_edge_position",
        "reads": reads,
        "edges": edges,
        "reward_by_pair": reward_by_pair,
        "hamiltonian": hamiltonian,
        "model": model,
        "reference_order": reference_order,
        "reference_sample": reference_sample,
        "reference_energy": float(model.energy(reference_sample)),
        "reference_score": int(certificate["reference_score"]),
    }


def build_bounded_problem(
    name: str,
    config: BoundedVertexOrderConfig,
) -> dict[str, Any]:
    model, reward_by_pair, _, _ = build_bounded_model(config)
    reference = json.loads(
        (DATASET_DIR / "reference_path.json").read_text(encoding="utf-8")
    )
    reference_order = list(reference["normalized_nodes"])
    reference_sample = encode_bounded_order(model, reference_order)
    certificate = load_certificate()

    return {
        "name": name,
        "config": config.__dict__,
        "reward_by_pair": reward_by_pair,
        "model": model,
        "reference_order": reference_order,
        "reference_sample": reference_sample,
        "reference_energy": float(model.energy(reference_sample)),
        "reference_score": int(certificate["reference_score"]),
    }


def bqm_from_model(model):
    return model.to_dimod_bqm()


def current_eval(problem: dict[str, Any], sample: list[int]) -> dict[str, Any]:
    model = problem["model"]
    hamiltonian = problem["hamiltonian"]
    reward_by_pair = problem["reward_by_pair"]

    order, meta = QUBOLayoutSolver._decode_edge_ordered_path(model, sample)
    components = _energy_components(
        model, hamiltonian, sample, reward_by_pair, 0.0
    )
    energy = float(model.energy(sample))
    path_score = None
    if meta["valid_edge_path"]:
        path_score = int(
            sum(reward_by_pair[pair] for pair in zip(order, order[1:]))
        )

    return {
        "energy": energy,
        "energy_above_reference": energy - problem["reference_energy"],
        "valid_path": bool(meta["valid_edge_path"]),
        "ground_energy_hit": abs(energy - problem["reference_energy"]) <= 1e-6,
        "path_score": path_score,
        "score_gap": (
            problem["reference_score"] - path_score
            if path_score is not None
            else None
        ),
        "selected_edges": int(meta["selected_edge_count"]),
        "sources": int(meta["selected_source_count"]),
        "sinks": int(meta["selected_sink_count"]),
        "read_in_violations": int(meta["read_in_constraint_violations"]),
        "read_out_violations": int(meta["read_out_constraint_violations"]),
        "void_in_violation": int(meta["void_in_constraint_violation"]),
        "void_out_violation": int(meta["void_out_constraint_violation"]),
        "activation_violations": int(meta["activation_violations"]),
        "order_residual_l1": int(meta["order_constraint_violations"]),
        "constraint_energy": float(
            components["degree"] + components["activation"] + components["order"]
        ),
        "degree_energy": float(components["degree"]),
        "activation_energy": float(components["activation"]),
        "order_energy": float(components["order"]),
        "hamming_from_reference": int(
            sum(a != b for a, b in zip(sample, problem["reference_sample"]))
        ),
    }


def bounded_eval(problem: dict[str, Any], sample: list[int]) -> dict[str, Any]:
    model: BoundedVertexOrderModel = problem["model"]
    reward_by_pair = problem["reward_by_pair"]
    cfg = BoundedVertexOrderConfig(**problem["config"])

    selected = [
        pair for pair in model.edge_pairs if sample[model.edge_index[pair]]
    ]
    sources = [rid for rid in model.read_ids if sample[model.source_index[rid]]]
    sinks = [rid for rid in model.read_ids if sample[model.sink_index[rid]]]

    indeg = {rid: 0 for rid in model.read_ids}
    outdeg = {rid: 0 for rid in model.read_ids}
    next_map: dict[str, str] = {}
    for u, v in selected:
        outdeg[u] += 1
        indeg[v] += 1
        if u not in next_map:
            next_map[u] = v

    read_degree_sq = sum(
        (1 - int(rid in sources) - indeg[rid]) ** 2
        + (1 - int(rid in sinks) - outdeg[rid]) ** 2
        for rid in model.read_ids
    )
    read_degree_l1 = sum(
        abs(1 - int(rid in sources) - indeg[rid])
        + abs(1 - int(rid in sinks) - outdeg[rid])
        for rid in model.read_ids
    )
    void_sq = (1 - len(sources)) ** 2 + (1 - len(sinks)) ** 2

    product_violations = 0
    successor_sq = 0
    successor_l1 = 0
    for pair in model.edge_pairs:
        u, v = pair
        x = int(sample[model.edge_index[pair]])
        for bit in range(model.position_bits):
            pu = int(sample[model.position_index[(u, bit)]])
            pv = int(sample[model.position_index[(v, bit)]])
            a = int(sample[model.left_copy_index[(pair, bit)]])
            b = int(sample[model.right_copy_index[(pair, bit)]])
            product_violations += int(a != x * pu)
            product_violations += int(b != x * pv)

            cin = x if bit == 0 else int(
                sample[model.carry_index[(pair, bit)]]
            )
            cout = 0 if bit == model.position_bits - 1 else int(
                sample[model.carry_index[(pair, bit + 1)]]
            )
            r = a + cin - b - 2 * cout
            successor_sq += r * r
            successor_l1 += abs(r)

    # Structural path validation independent of position auxiliaries.
    valid_degree = read_degree_sq == 0 and void_sq == 0
    order: list[str] = []
    structural_single_path = False
    if valid_degree and len(sources) == 1 and len(sinks) == 1:
        seen = set()
        cur = sources[0]
        while cur not in seen:
            seen.add(cur)
            order.append(cur)
            if cur == sinks[0]:
                break
            if cur not in next_map:
                break
            cur = next_map[cur]
        structural_single_path = (
            len(order) == len(model.read_ids)
            and order[-1] == sinks[0]
            and len(seen) == len(model.read_ids)
        )

    valid_path = (
        structural_single_path
        and product_violations == 0
        and successor_sq == 0
    )

    path_score = None
    if valid_path:
        path_score = int(
            sum(reward_by_pair[pair] for pair in zip(order, order[1:]))
        )

    # Decode positions for diagnostics.
    positions = {}
    for rid in model.read_ids:
        value = 0
        for bit in range(model.position_bits):
            value |= int(sample[model.position_index[(rid, bit)]]) << bit
        positions[rid] = value
    distinct_positions = len(set(positions.values()))

    energy = float(model.energy(sample))
    objective = cfg.edge_cost_scale * sum(
        (max(reward_by_pair.values()) - reward_by_pair[pair])
        / max(
            max(max(reward_by_pair.values()) - reward for reward in reward_by_pair.values()),
            1,
        )
        for pair in selected
    )
    constraint_energy = (
        cfg.degree_penalty * read_degree_sq
        + cfg.void_degree_penalty * void_sq
        + cfg.product_penalty * product_violations
        + cfg.successor_penalty * successor_sq
    )

    return {
        "energy": energy,
        "energy_above_reference": energy - problem["reference_energy"],
        "valid_path": valid_path,
        "ground_energy_hit": abs(energy - problem["reference_energy"]) <= 1e-6,
        "path_score": path_score,
        "score_gap": (
            problem["reference_score"] - path_score
            if path_score is not None
            else None
        ),
        "selected_edges": len(selected),
        "sources": len(sources),
        "sinks": len(sinks),
        "read_degree_residual_sq": int(read_degree_sq),
        "read_degree_residual_l1": int(read_degree_l1),
        "void_residual_sq": int(void_sq),
        "product_violations": int(product_violations),
        "successor_residual_sq": int(successor_sq),
        "successor_residual_l1": int(successor_l1),
        "distinct_positions": int(distinct_positions),
        "constraint_energy": float(constraint_energy),
        "objective_energy": float(objective),
        "hamming_from_reference": int(
            sum(a != b for a, b in zip(sample, problem["reference_sample"]))
        ),
    }


def initial_states(
    kind: str,
    n: int,
    num_reads: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    if kind == "random":
        return None
    if kind == "zero":
        return np.zeros((num_reads, n), dtype=np.int8)
    if kind == "sparse1pct":
        return (rng.random((num_reads, n)) < 0.01).astype(np.int8)
    raise ValueError(kind)


def run_case(
    problem: dict[str, Any],
    evaluator,
    *,
    init_kind: str,
    num_reads: int,
    timeout_ms: int,
    tenure: int | None,
    seed: int,
    coefficient_z_first: int | None,
    coefficient_z_restart: int | None,
    lower_bound_z: int | None,
    num_restarts: int | None,
) -> dict[str, Any]:
    model = problem["model"]
    bqm = bqm_from_model(model)
    rng = np.random.default_rng(seed)
    init = initial_states(init_kind, model.num_variables, num_reads, rng)

    kwargs: dict[str, Any] = {
        "num_reads": num_reads,
        "timeout": timeout_ms,
        "seed": seed,
    }
    if tenure is not None:
        kwargs["tenure"] = tenure
    if coefficient_z_first is not None:
        kwargs["coefficient_z_first"] = coefficient_z_first
    if coefficient_z_restart is not None:
        kwargs["coefficient_z_restart"] = coefficient_z_restart
    if lower_bound_z is not None:
        kwargs["lower_bound_z"] = lower_bound_z
    if num_restarts is not None:
        kwargs["num_restarts"] = num_restarts
    if init is not None:
        kwargs["initial_states"] = init
        kwargs["initial_states_generator"] = "none"

    sampler = TabuSampler()
    t0 = time.perf_counter()
    ss = sampler.sample(bqm, **kwargs)
    seconds = time.perf_counter() - t0

    results = []
    for datum in ss.data(fields=["sample", "energy"], sorted_by="energy"):
        s = [int(datum.sample.get(i, 0)) for i in range(model.num_variables)]
        results.append(evaluator(problem, s))

    results.sort(key=lambda r: r["energy"])
    best = results[0]
    feasible = [r for r in results if r["valid_path"]]
    best_feasible = min(feasible, key=lambda r: r["energy"]) if feasible else None

    return {
        "model": problem["name"],
        "variables": model.num_variables,
        "quadratic_terms": len(model.quadratic),
        "max_abs_linear": max(abs(float(v)) for v in model.linear),
        "max_abs_quadratic": max(abs(float(v)) for v in model.quadratic.values()),
        "reference_energy": problem["reference_energy"],
        "init": init_kind,
        "num_reads": num_reads,
        "timeout_ms_per_read": timeout_ms,
        "tenure": tenure,
        "coefficient_z_first": coefficient_z_first,
        "coefficient_z_restart": coefficient_z_restart,
        "lower_bound_z": lower_bound_z,
        "num_restarts": num_restarts,
        "seed": seed,
        "seconds": seconds,
        "feasible_hits": len(feasible),
        "ground_energy_hits": sum(r["ground_energy_hit"] for r in results),
        "best": best,
        "best_feasible": best_feasible,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--short-timeout-ms", type=int, default=5000)
    parser.add_argument("--long-timeout-ms", type=int, default=20000)
    parser.add_argument("--short-reads", type=int, default=4)
    parser.add_argument("--long-reads", type=int, default=4)
    parser.add_argument("--coefficient-z-first", type=int)
    parser.add_argument("--coefficient-z-restart", type=int)
    parser.add_argument("--lower-bound-z", type=int)
    parser.add_argument("--num-restarts", type=int)
    args = parser.parse_args()

    problems = [
        (
            build_current_problem(),
            current_eval,
        ),
        (
            build_bounded_problem(
                "bounded_equal144",
                BoundedVertexOrderConfig(
                    degree_penalty=144.0,
                    void_degree_penalty=144.0,
                    successor_penalty=144.0,
                    product_penalty=144.0,
                    edge_cost_scale=1.0,
                ),
            ),
            bounded_eval,
        ),
        (
            build_bounded_problem(
                "bounded_balanced",
                BoundedVertexOrderConfig(
                    degree_penalty=1000.0,
                    void_degree_penalty=2000.0,
                    successor_penalty=72.0,
                    product_penalty=144.0,
                    edge_cost_scale=1.0,
                ),
            ),
            bounded_eval,
        ),
    ]

    all_runs = []
    for problem, evaluator in problems:
        print("\nMODEL", problem["name"], flush=True)
        print(
            json.dumps(
                {
                    "variables": problem["model"].num_variables,
                    "quadratic_terms": len(problem["model"].quadratic),
                    "reference_energy": problem["reference_energy"],
                },
                indent=2,
            ),
            flush=True,
        )

        for offset, init_kind in enumerate(("zero", "sparse1pct", "random")):
            run = run_case(
                problem,
                evaluator,
                init_kind=init_kind,
                num_reads=args.short_reads,
                timeout_ms=args.short_timeout_ms,
                tenure=None,
                seed=SEED + offset,
                coefficient_z_first=args.coefficient_z_first,
                coefficient_z_restart=args.coefficient_z_restart,
                lower_bound_z=args.lower_bound_z,
                num_restarts=args.num_restarts,
            )
            all_runs.append(run)
            print(json.dumps(run, indent=2), flush=True)

        # Longer budget only for the initialization that earlier diagnostics
        # indicated is most relevant to these sparse ground states.
        run = run_case(
            problem,
            evaluator,
            init_kind="zero",
            num_reads=args.long_reads,
            timeout_ms=args.long_timeout_ms,
            tenure=None,
            seed=SEED + 100,
            coefficient_z_first=args.coefficient_z_first,
            coefficient_z_restart=args.coefficient_z_restart,
            lower_bound_z=args.lower_bound_z,
            num_restarts=args.num_restarts,
        )
        run["budget_label"] = "long_zero"
        all_runs.append(run)
        print(json.dumps(run, indent=2), flush=True)

    payload = {
        "solver": "dwave.samplers.TabuSampler",
        "algorithm": "MST2 multistart tabu",
        "seed_base": SEED,
        "runs": all_runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\nWROTE", args.output, flush=True)


if __name__ == "__main__":
    main()
