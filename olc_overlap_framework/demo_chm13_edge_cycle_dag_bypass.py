"""Run the legacy edge-cycle DAG QUBO on the cyclic CHM13 graph.

The DAG validation is bypassed only inside this test harness.  The Hamiltonian
itself is unchanged, so read-only subtours are deliberately possible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from scipy.optimize import linear_sum_assignment

from demo_chm13_edge_ordered_path_qubo import DATASET_DIR, load_chm13_graph
from olc_pipeline.layout_solver import (
    EdgeCycleCoverDAGHamiltonianConfig,
    EdgeCycleCoverDAGQUBOHamiltonian,
    EdgePathDAGQUBOHamiltonian,
    OpenJijSQAConfig,
    OpenJijSimulatedQuantumAnnealer,
    QUBOLayoutSolver,
    qubo_sample_for_order,
)


# Experiment parameters are kept here for direct editing.
DEGREE_PENALTY = 100.0
EDGE_REWARD_SCALE = 1.0
EDGE_SELECTION_PENALTY = 0.0
SCORE_MODE = "dp"
NORMALIZE_REWARDS = True
SQA_NUM_READS = 8
SQA_NUM_SWEEPS = 2000
SQA_TROTTER = 8
SQA_SEED = 20260922
SQA_START_S = 0.5

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "debug"
    / "qubo"
    / "chm13_edge_cycle_dag_bypass"
)
VOID = "__void__"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solve-sqa", action="store_true")
    parser.add_argument("--num-reads", type=int, default=SQA_NUM_READS)
    parser.add_argument("--num-sweeps", type=int, default=SQA_NUM_SWEEPS)
    parser.add_argument("--trotter", type=int, default=SQA_TROTTER)
    parser.add_argument("--seed", type=int, default=SQA_SEED)
    parser.add_argument("--start-s", type=float, default=SQA_START_S)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _single_flip_diagnostics(model, sample: list[int]) -> dict[str, float | int]:
    adjacency: list[dict[int, float]] = [
        {} for _ in range(model.num_variables)
    ]
    for (left, right), coefficient in model.quadratic.items():
        adjacency[left][right] = coefficient
        adjacency[right][left] = coefficient
    deltas = []
    for index in range(model.num_variables):
        local = model.linear[index] + sum(
            coefficient * sample[other]
            for other, coefficient in adjacency[index].items()
        )
        deltas.append(local if sample[index] == 0 else -local)
    return {
        "minimum_delta": min(deltas, default=0.0),
        "downhill_flips": sum(delta < -1e-9 for delta in deltas),
        "flat_flips": sum(abs(delta) <= 1e-9 for delta in deltas),
    }


def _cycle_components(model, sample: list[int]) -> list[list[str]]:
    successors: dict[str, list[str]] = {
        read_id: [] for read_id in model.read_ids
    }
    successors[VOID] = []
    for edge_pair in model.edge_pairs:
        if sample[model.edge_variable_index(*edge_pair)]:
            successors[edge_pair[0]].append(edge_pair[1])
    for read_id in model.read_ids:
        if sample[model.source_variable_index(read_id)]:
            successors[VOID].append(read_id)
        if sample[model.sink_variable_index(read_id)]:
            successors[read_id].append(VOID)

    if any(len(targets) != 1 for targets in successors.values()):
        return []
    successor = {node: targets[0] for node, targets in successors.items()}
    components: list[list[str]] = []
    unseen = set(successor)
    while unseen:
        start = min(unseen)
        path: list[str] = []
        position: dict[str, int] = {}
        current = start
        while current not in position and current in successor:
            position[current] = len(path)
            path.append(current)
            current = successor[current]
        cycle = path[position[current]:] if current in position else path
        components.append(cycle)
        unseen.difference_update(path)
    components.sort(key=lambda component: (VOID not in component, -len(component)))
    return components


def _sample_summary(model, sample: list[int], reward_by_pair) -> dict[str, object]:
    _, metadata = QUBOLayoutSolver._decode_edge_cycle_cover(model, sample)
    selected_edges = [
        pair
        for pair in model.edge_pairs
        if sample[model.edge_variable_index(*pair)]
    ]
    components = _cycle_components(model, sample)
    return {
        "energy": model.energy(sample),
        "valid_single_cycle": metadata["valid_edge_cycle"],
        "degree_feasible_cycle_cover": (
            metadata["read_in_constraint_violations"] == 0
            and metadata["read_out_constraint_violations"] == 0
            and metadata["void_in_constraint_violation"] == 0
            and metadata["void_out_constraint_violation"] == 0
        ),
        "selected_edge_count": metadata["selected_edge_count"],
        "selected_source_count": metadata["selected_source_count"],
        "selected_sink_count": metadata["selected_sink_count"],
        "read_in_constraint_violations": metadata[
            "read_in_constraint_violations"
        ],
        "read_out_constraint_violations": metadata[
            "read_out_constraint_violations"
        ],
        "void_in_constraint_violation": metadata[
            "void_in_constraint_violation"
        ],
        "void_out_constraint_violation": metadata[
            "void_out_constraint_violation"
        ],
        "selected_edge_reward_sum": sum(
            reward_by_pair[pair] for pair in selected_edges
        ),
        "cycle_component_count": len(components),
        "cycle_component_read_sizes": [
            sum(node != VOID for node in component)
            for component in components
        ],
        "void_component_read_count": next((
            sum(node != VOID for node in component)
            for component in components
            if VOID in component
        ), 0),
        "read_only_cycle_components": [
            component for component in components if VOID not in component
        ],
        "single_flip": _single_flip_diagnostics(model, sample),
    }


def _exact_cycle_cover_sample(model, reward_by_pair) -> list[int]:
    import numpy as np

    nodes = [*model.read_ids, VOID]
    index = {node: offset for offset, node in enumerate(nodes)}
    forbidden = 1e9
    costs = np.full((len(nodes), len(nodes)), forbidden, dtype=float)
    reward_max = max(reward_by_pair.values(), default=1)
    for pair, reward in reward_by_pair.items():
        costs[index[pair[0]], index[pair[1]]] = -reward / reward_max
    for read_id in model.read_ids:
        costs[index[VOID], index[read_id]] = 0.0
        costs[index[read_id], index[VOID]] = 0.0

    rows, columns = linear_sum_assignment(costs)
    if any(costs[row, column] >= forbidden for row, column in zip(rows, columns)):
        raise ValueError("no complete cycle cover exists")
    sample = [0] * model.num_variables
    for row, column in zip(rows, columns):
        left_id = nodes[row]
        right_id = nodes[column]
        if left_id == VOID:
            sample[model.source_variable_index(right_id)] = 1
        elif right_id == VOID:
            sample[model.sink_variable_index(left_id)] = 1
        else:
            sample[model.edge_variable_index(left_id, right_id)] = 1
    return sample


def main() -> None:
    args = _parse_args()
    if not 0.0 <= args.start_s < 1.0:
        raise ValueError("start-s must be in [0, 1)")
    reads, edges, reward_by_pair = load_chm13_graph(DATASET_DIR)
    hamiltonian = EdgeCycleCoverDAGQUBOHamiltonian(
        EdgeCycleCoverDAGHamiltonianConfig(
            degree_penalty=DEGREE_PENALTY,
            edge_reward_scale=EDGE_REWARD_SCALE,
            edge_selection_penalty=EDGE_SELECTION_PENALTY,
            score_mode=SCORE_MODE,
            normalize_rewards=NORMALIZE_REWARDS,
        )
    )
    with patch.object(
        EdgePathDAGQUBOHamiltonian,
        "_topological_order",
        return_value=[],
    ):
        model = hamiltonian.build(reads, edges)

    reference = json.loads((DATASET_DIR / "reference_path.json").read_text(
        encoding="utf-8"
    ))
    reference_sample = qubo_sample_for_order(model, reference["normalized_nodes"])
    exact_sample = _exact_cycle_cover_sample(model, reward_by_pair)

    report: dict[str, object] = {
        "dataset": str(DATASET_DIR),
        "dag_check_bypassed": True,
        "reads": len(reads),
        "edges": len(edges),
        "logical_variables": model.num_variables,
        "quadratic_terms": len(model.quadratic),
        "degree_penalty": DEGREE_PENALTY,
        "edge_reward_scale": EDGE_REWARD_SCALE,
        "score_mode": SCORE_MODE,
        "reward_normalizer": max(reward_by_pair.values()),
        "reference": _sample_summary(
            model, reference_sample, reward_by_pair
        ),
        "exact_maximum_weight_cycle_cover": _sample_summary(
            model, exact_sample, reward_by_pair
        ),
    }

    if args.solve_sqa:
        schedule = None
        if args.start_s > 0.0:
            denominator = args.num_sweeps - 1
            schedule = tuple(
                (
                    args.start_s
                    + (1.0 - args.start_s) * offset / denominator,
                    1,
                )
                for offset in range(args.num_sweeps)
            )
        annealer = OpenJijSimulatedQuantumAnnealer(OpenJijSQAConfig(
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            trotter=args.trotter,
            seed=args.seed,
            schedule=schedule,
        ))
        start = perf_counter()
        annealing_result = annealer.solve(model)
        sqa_summary = _sample_summary(
            model, annealing_result.sample, reward_by_pair
        )
        sqa_summary.update({
            "num_reads": args.num_reads,
            "num_sweeps": args.num_sweeps,
            "trotter": args.trotter,
            "seed": args.seed,
            "start_s": args.start_s,
            "seconds": perf_counter() - start,
            "reported_energy": annealing_result.energy,
            "energy_above_exact_cycle_cover": (
                annealing_result.energy - model.energy(exact_sample)
            ),
        })
        report["sqa"] = sqa_summary

    if args.output is not None:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = OUTPUT_DIR / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        report["output"] = str(output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
