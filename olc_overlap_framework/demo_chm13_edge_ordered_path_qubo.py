"""Instantiate the edge-ordered Hamilton-path QUBO on CHM13 complex144.

This is a model/input audit, not an annealing run.  The certified reference
path is encoded only after the QUBO is built so its feasibility and objective
value can be checked independently of the graph construction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

from olc_pipeline.data import OverlapEdge, Read
from olc_pipeline.layout_solver import (
    EdgeOrderedPathHamiltonianConfig,
    EdgeOrderedPathQUBOHamiltonian,
    OpenJijSQAConfig,
    OpenJijSimulatedQuantumAnnealer,
    QUBOLayoutSolver,
    quality_margin_coefficients,
    qubo_sample_for_order,
)


# Experiment parameters are kept here for direct editing.
DATASET_DIR = (
    Path(__file__).resolve().parent.parent
    / "test_data"
    / "CHM13"
    / "CHM13_complex144"
)
COST_MODE = "shifted_reward"
SCORE_MODE = "dp"
NORMALIZE_COSTS = True
INCLUDE_SOURCE_LENGTH = False
EDGE_COST_SCALE = 1.0
SQA_NUM_READS = 1
SQA_NUM_SWEEPS = 100
SQA_TROTTER = 4
SQA_SEED = 20260922
SQA_BETA = None
SQA_GAMMA = None
SQA_START_S = 0.0

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "debug"
    / "qubo"
    / "chm13_edge_ordered_path"
)


def read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current_id: str | None = None
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_id = line[1:].split()[0]
                if current_id in sequences:
                    raise ValueError(f"duplicate FASTA id: {current_id}")
                sequences[current_id] = ""
            elif current_id is None:
                raise ValueError(f"sequence before FASTA header in {path}")
            else:
                sequences[current_id] += line
    return sequences


def load_chm13_graph(
    dataset_dir: Path,
) -> tuple[list[Read], list[OverlapEdge], dict[tuple[str, str], int]]:
    weight_spec = json.loads((dataset_dir / "weight_spec.json").read_text(
        encoding="utf-8"
    ))
    match_coefficient, error_coefficient = quality_margin_coefficients(
        float(weight_spec["identity_threshold"])
    )
    sequences = read_fasta(dataset_dir / "reads.normalized.fasta")
    reads = [Read(read_id, sequence) for read_id, sequence in sequences.items()]

    with (dataset_dir / "nodes.tsv").open(
        newline="", encoding="utf-8"
    ) as handle:
        node_rows = list(csv.DictReader(handle, delimiter="\t"))
    oriented_to_normalized = {
        row["node"]: row["read_id"]
        for row in node_rows
    }
    if len(oriented_to_normalized) != len(sequences):
        raise ValueError("nodes.tsv does not map each normalized read exactly once")
    if set(oriented_to_normalized.values()) != set(sequences):
        raise ValueError("nodes.tsv and reads.normalized.fasta contain different reads")

    edges: list[OverlapEdge] = []
    reward_by_pair: dict[tuple[str, str], int] = {}
    with (dataset_dir / "edges.tsv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            left_id = oriented_to_normalized[row["source"]]
            right_id = oriented_to_normalized[row["target"]]
            weight = int(row["weight"])
            matching_bases = int(row["matching_bases"])
            errors = int(row["errors"])
            alignment_block = int(row["alignment_block"])
            overlap_span = int(row["overlap_span"])
            expected_weight = (
                match_coefficient * matching_bases - error_coefficient * errors
            )
            if weight != expected_weight:
                raise ValueError(f"invalid CHM13 edge weight: {left_id} -> {right_id}")
            if alignment_block != matching_bases + errors:
                raise ValueError(f"invalid alignment block: {left_id} -> {right_id}")

            pair = (left_id, right_id)
            previous = reward_by_pair.get(pair)
            if previous is not None:
                raise ValueError(f"duplicate directed edge in edges.tsv: {pair}")
            reward_by_pair[pair] = weight

            left_length = len(sequences[left_id])
            right_length = len(sequences[right_id])
            edges.append(OverlapEdge(
                left_id=left_id,
                right_id=right_id,
                left_start=max(0, left_length - overlap_span),
                left_end=left_length,
                right_start=0,
                right_end=min(right_length, overlap_span),
                overlap_len=overlap_span,
                shift=max(0, left_length - overlap_span),
                matches=matching_bases,
                mismatches=errors,
                insertions=0,
                deletions=0,
                gaps=0,
                edit_distance=errors,
                error_rate=errors / alignment_block,
                identity=matching_bases / alignment_block,
                dp_score=float(weight),
                weight_dp=float(weight),
                candidate_source="chm13_complex144_edges_tsv",
                accepted=True,
            ))
    return reads, edges, reward_by_pair


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solve-sqa", action="store_true")
    parser.add_argument("--num-reads", type=int, default=SQA_NUM_READS)
    parser.add_argument("--num-sweeps", type=int, default=SQA_NUM_SWEEPS)
    parser.add_argument("--trotter", type=int, default=SQA_TROTTER)
    parser.add_argument("--seed", type=int, default=SQA_SEED)
    parser.add_argument("--beta", type=float, default=SQA_BETA)
    parser.add_argument("--gamma", type=float, default=SQA_GAMMA)
    parser.add_argument("--start-s", type=float, default=SQA_START_S)
    parser.add_argument("--edge-cost-scale", type=float, default=EDGE_COST_SCALE)
    parser.add_argument("--constraint-penalty", type=float)
    parser.add_argument("--explicit-void-penalty", type=float, default=0.0)
    parser.add_argument(
        "--initial-state",
        choices=("random", "reference"),
        default="random",
    )
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


def _energy_components(
    model,
    hamiltonian: EdgeOrderedPathQUBOHamiltonian,
    sample: list[int],
    reward_by_pair: dict[tuple[str, str], int],
    explicit_void_penalty: float = 0.0,
) -> dict[str, float]:
    _, metadata = QUBOLayoutSolver._decode_edge_ordered_path(model, sample)
    selected_edges = [
        pair
        for pair in model.edge_pairs
        if sample[model.edge_variable_index(*pair)]
    ]
    selected_sources = {
        read_id
        for read_id in model.read_ids
        if sample[model.source_variable_index(read_id)]
    }
    selected_sinks = {
        read_id
        for read_id in model.read_ids
        if sample[model.sink_variable_index(read_id)]
    }
    incoming_count = {read_id: 0 for read_id in model.read_ids}
    outgoing_count = {read_id: 0 for read_id in model.read_ids}
    for left_id, right_id in selected_edges:
        outgoing_count[left_id] += 1
        incoming_count[right_id] += 1

    objective = hamiltonian.config.edge_cost_scale * sum(
        (hamiltonian.last_reward_shift - reward_by_pair[pair])
        / hamiltonian.last_cost_normalizer
        for pair in selected_edges
    )
    degree_residual_square = sum(
        (1 - incoming_count[read_id] - int(read_id in selected_sources)) ** 2
        + (1 - outgoing_count[read_id] - int(read_id in selected_sinks)) ** 2
        for read_id in model.read_ids
    )
    degree = hamiltonian.last_degree_penalty * degree_residual_square
    activation = (
        hamiltonian.last_activation_penalty
        * metadata["activation_violations"]
    )
    order = hamiltonian.last_order_penalty * sum(
        residual * residual
        for residual in metadata["order_residuals"].values()
    )
    explicit_void = explicit_void_penalty * (
        (len(selected_sources) - 1) ** 2
        + (len(selected_sinks) - 1) ** 2
    )
    return {
        "objective": objective,
        "degree": degree,
        "activation": activation,
        "order": order,
        "explicit_void": explicit_void,
        "total": objective + degree + activation + order + explicit_void,
    }


def main() -> None:
    args = _parse_args()
    reads, edges, reward_by_pair = load_chm13_graph(DATASET_DIR)
    hamiltonian = EdgeOrderedPathQUBOHamiltonian(
        EdgeOrderedPathHamiltonianConfig(
            degree_penalty=args.constraint_penalty,
            order_penalty=args.constraint_penalty,
            activation_penalty=args.constraint_penalty,
            edge_cost_scale=args.edge_cost_scale,
            cost_mode=COST_MODE,
            score_mode=SCORE_MODE,
            normalize_costs=NORMALIZE_COSTS,
            include_source_length=INCLUDE_SOURCE_LENGTH,
        )
    )
    model = hamiltonian.build(reads, edges)
    if args.explicit_void_penalty < 0.0:
        raise ValueError("explicit-void-penalty must be non-negative")
    if args.explicit_void_penalty > 0.0:
        EdgeOrderedPathQUBOHamiltonian._add_square(
            model,
            {
                model.source_variable_index(read_id): 1.0
                for read_id in model.read_ids
            },
            constant=-1.0,
            penalty=args.explicit_void_penalty,
        )
        EdgeOrderedPathQUBOHamiltonian._add_square(
            model,
            {
                model.sink_variable_index(read_id): 1.0
                for read_id in model.read_ids
            },
            constant=-1.0,
            penalty=args.explicit_void_penalty,
        )

    reference = json.loads((DATASET_DIR / "reference_path.json").read_text(
        encoding="utf-8"
    ))
    reference_order = reference["normalized_nodes"]
    sample = qubo_sample_for_order(model, reference_order)
    decoded_order, decode_metadata = QUBOLayoutSolver._decode_edge_ordered_path(
        model, sample
    )

    reference_score = sum(
        reward_by_pair[pair]
        for pair in zip(reference_order, reference_order[1:])
    )
    weight_spec = json.loads((DATASET_DIR / "weight_spec.json").read_text(
        encoding="utf-8"
    ))
    expected_score = int(weight_spec["reference_path_score"])
    if reference_score != expected_score:
        raise ValueError(
            f"reference score mismatch: {reference_score} != {expected_score}"
        )
    if decoded_order != reference_order or not decode_metadata["valid_edge_path"]:
        raise ValueError("certified reference path is not feasible in the QUBO")

    certificate = json.loads((DATASET_DIR / "weighted_certificate.json").read_text(
        encoding="utf-8"
    ))
    runner_up_entry = next(
        entry
        for entry in certificate["top_paths"]
        if int(entry["score"]) == int(certificate["runner_up_score"])
    )
    runner_up_order = [node[:-1] for node in runner_up_entry["nodes"]]
    runner_up_sample = qubo_sample_for_order(model, runner_up_order)
    runner_up_energy = model.energy(runner_up_sample)

    objective_energy = hamiltonian.config.edge_cost_scale * sum(
        (hamiltonian.last_reward_shift - reward_by_pair[pair])
        / hamiltonian.last_cost_normalizer
        for pair in zip(reference_order, reference_order[1:])
    )
    qubo_energy = model.energy(sample)
    if abs(qubo_energy - objective_energy) > 1e-6:
        raise ValueError(
            f"constraint-free energy mismatch: {qubo_energy} != {objective_energy}"
        )

    incoming_count = {read.rid: 0 for read in reads}
    outgoing_count = {read.rid: 0 for read in reads}
    for edge in edges:
        outgoing_count[edge.left_id] += 1
        incoming_count[edge.right_id] += 1

    report: dict[str, object] = {
        "dataset": str(DATASET_DIR),
        "reads": len(reads),
        "edges": len(edges),
        "zero_indegree_nodes": sum(value == 0 for value in incoming_count.values()),
        "zero_outdegree_nodes": sum(value == 0 for value in outgoing_count.values()),
        "position_bits_per_edge": model.position_bit_count,
        "logical_variables": model.num_variables,
        "linear_terms": len(model.linear),
        "quadratic_terms": len(model.quadratic),
        "maximum_absolute_linear": max(abs(value) for value in model.linear),
        "maximum_absolute_quadratic": max(
            abs(value) for value in model.quadratic.values()
        ),
        "cost_mode": COST_MODE,
        "score_mode": SCORE_MODE,
        "edge_cost_scale": args.edge_cost_scale,
        "explicit_void_penalty": args.explicit_void_penalty,
        "reward_shift": hamiltonian.last_reward_shift,
        "cost_normalizer": hamiltonian.last_cost_normalizer,
        "auto_penalty": hamiltonian.last_degree_penalty,
        "reference_score": reference_score,
        "reference_qubo_energy": qubo_energy,
        "reference_path_valid": decode_metadata["valid_edge_path"],
        "runner_up_score": int(runner_up_entry["score"]),
        "runner_up_qubo_energy": runner_up_energy,
        "qubo_optimum_margin": runner_up_energy - qubo_energy,
        "reference_single_flip": _single_flip_diagnostics(model, sample),
        "reference_energy_components": _energy_components(
            model,
            hamiltonian,
            sample,
            reward_by_pair,
            args.explicit_void_penalty,
        ),
    }

    if args.solve_sqa:
        if not 0.0 <= args.start_s < 1.0:
            raise ValueError("start-s must be in [0, 1)")
        schedule = None
        if args.start_s > 0.0:
            if args.num_sweeps < 2:
                raise ValueError("late-start schedule needs at least two sweeps")
            denominator = args.num_sweeps - 1
            schedule = tuple(
                (
                    args.start_s
                    + (1.0 - args.start_s) * index / denominator,
                    1,
                )
                for index in range(args.num_sweeps)
            )
        initial_state = (
            tuple(sample) if args.initial_state == "reference" else None
        )
        annealer = OpenJijSimulatedQuantumAnnealer(OpenJijSQAConfig(
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            trotter=args.trotter,
            seed=args.seed,
            beta=args.beta,
            gamma=args.gamma,
            schedule=schedule,
            initial_state=initial_state,
        ))
        start = perf_counter()
        annealing_result = annealer.solve(model)
        anneal_seconds = perf_counter() - start
        solved_order, solved_metadata = QUBOLayoutSolver._decode_edge_ordered_path(
            model, annealing_result.sample
        )
        selected_pairs = [
            pair
            for pair in model.edge_pairs
            if annealing_result.sample[model.edge_variable_index(*pair)]
        ]
        selected_reward = sum(reward_by_pair[pair] for pair in selected_pairs)
        traced_path_nodes = 0
        if solved_metadata["cycle_components"]:
            traced_path_nodes = max(
                0, len(solved_metadata["cycle_components"][0]) - 2
            )
        report["sqa"] = {
            "num_reads": args.num_reads,
            "num_sweeps": args.num_sweeps,
            "trotter": args.trotter,
            "seed": args.seed,
            "beta": args.beta,
            "gamma": args.gamma,
            "start_s": args.start_s,
            "initial_state": args.initial_state,
            "seconds": anneal_seconds,
            "energy": annealing_result.energy,
            "recomputed_energy": model.energy(annealing_result.sample),
            "energy_above_reference": annealing_result.energy - qubo_energy,
            "selected_binary_variables": sum(annealing_result.sample),
            "hamming_distance_from_reference": sum(
                left != right
                for left, right in zip(annealing_result.sample, sample)
            ),
            "valid_edge_path": solved_metadata["valid_edge_path"],
            "selected_edge_count": solved_metadata["selected_edge_count"],
            "selected_source_count": solved_metadata["selected_source_count"],
            "selected_sink_count": solved_metadata["selected_sink_count"],
            "traced_path_nodes": traced_path_nodes,
            "read_in_constraint_violations": solved_metadata[
                "read_in_constraint_violations"
            ],
            "read_out_constraint_violations": solved_metadata[
                "read_out_constraint_violations"
            ],
            "void_in_constraint_violation": solved_metadata[
                "void_in_constraint_violation"
            ],
            "void_out_constraint_violation": solved_metadata[
                "void_out_constraint_violation"
            ],
            "activation_violations": solved_metadata["activation_violations"],
            "order_constraint_violations": solved_metadata[
                "order_constraint_violations"
            ],
            "position_sequence_valid": solved_metadata[
                "position_sequence_valid"
            ],
            "single_flip": _single_flip_diagnostics(
                model, annealing_result.sample
            ),
            "energy_components": _energy_components(
                model,
                hamiltonian,
                annealing_result.sample,
                reward_by_pair,
                args.explicit_void_penalty,
            ),
            "selected_edge_reward_sum": selected_reward,
            "decoded_order": solved_order if solved_metadata["valid_edge_path"] else [],
        }

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
