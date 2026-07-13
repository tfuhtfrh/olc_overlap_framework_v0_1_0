"""
Parameter sweep for the edge-variable DAG Hamiltonian.

The overlap graph is built once. Each case varies the edge-count penalty,
degree penalty, and annealer seed while keeping the graph and SQA settings
fixed.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from demo_sqa_parameter_sweep import build_overlap_graph, select_layout_inputs
from olc_pipeline.evaluator import LayoutEvaluator
from olc_pipeline.layout_solver import (
    EdgePathDAGHamiltonianConfig,
    EdgePathDAGQUBOHamiltonian,
    OpenJijSQAConfig,
    OpenJijSimulatedQuantumAnnealer,
    QUBOLayoutSolver,
    qubo_sample_for_order,
)


@dataclass(frozen=True)
class SweepCase:
    max_reads: int
    count_penalty: float
    degree_penalty: float
    seed: int


ROW_FIELDS = [
    "case",
    "reads_used",
    "qubo_variables",
    "count_penalty",
    "degree_penalty",
    "reward_scale",
    "seed",
    "energy",
    "true_energy",
    "energy_gap",
    "ground_hit",
    "valid_edge_path",
    "selected_edge_count",
    "edge_count_violation",
    "in_degree_violations",
    "out_degree_violations",
    "selected_adjacent_correct",
    "selected_jump_correct",
    "selected_wrong",
    "anneal_sec",
    "total_sec",
]


def main() -> None:
    args = parse_args()
    reads, edges = build_overlap_graph(args)

    print(f"simulation_reads,{len(reads)}")
    print(f"genome_len,{args.genome_len}")
    print(f"read_len,{args.read_len}")
    print(f"step,{args.step}")
    print(f"read_counts,{args.read_counts or args.max_reads}")
    print(f"num_reads,{args.num_reads}")
    print(f"num_sweeps,{args.num_sweeps}")
    print(f"trotter,{args.trotter}")

    if args.describe_only:
        describe_scales(reads, edges, args)
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        handle.flush()
        for index, case in enumerate(iter_cases(args), start=1):
            layout_reads, layout_edges = select_layout_inputs(
                reads,
                edges,
                case.max_reads,
            )
            row = run_case(index, case, layout_reads, layout_edges, args)
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            print(format_row(row), flush=True)

    print_summary(rows)
    print(f"csv,{output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan edge-path DAG QUBO parameters.")
    parser.add_argument("--max-reads", type=int, default=35)
    parser.add_argument("--read-counts", default=None)
    parser.add_argument("--genome-len", type=int, default=20_000)
    parser.add_argument("--read-len", type=int, default=3_000)
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--mismatch-rate", type=float, default=0.0)
    parser.add_argument("--ins-rate", type=float, default=0.0)
    parser.add_argument("--del-rate", type=float, default=0.0)
    parser.add_argument("--gc-fraction", type=float, default=0.5)
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--simulation-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annealer-seeds", default="40,41,42,43,44,45,46,47,48,49")
    parser.add_argument("--count-penalties", default="80,100,120")
    parser.add_argument("--degree-penalties", default="105,110,115,120,125,130")
    parser.add_argument("--reward-scale", type=float, default=20.0)
    parser.add_argument("--score-mode", default="overlap_len")
    parser.add_argument("--trotter", type=int, default=32)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--output", default="debug/qubo/edge_path_dag_parameter_sweep.csv")
    return parser.parse_args()


def describe_scales(reads, edges, args) -> None:
    read_counts = (
        parse_int_list(args.read_counts)
        if args.read_counts
        else [args.max_reads]
    )
    print("scale,reads_used,qubo_variables,candidate_edges,quadratic_terms")
    for max_reads in read_counts:
        layout_reads, layout_edges = select_layout_inputs(reads, edges, max_reads)
        hamiltonian = EdgePathDAGQUBOHamiltonian(EdgePathDAGHamiltonianConfig(
            edge_count_penalty=parse_float_list(args.count_penalties)[0],
            degree_penalty=parse_float_list(args.degree_penalties)[0],
            edge_reward_scale=args.reward_scale,
            score_mode=args.score_mode,
            normalize_rewards=True,
            require_hamiltonian_path=True,
        ))
        model = hamiltonian.build(layout_reads, layout_edges, weight_mode=args.score_mode)
        print(
            "scale,"
            f"{len(layout_reads)},"
            f"{model.num_variables},"
            f"{len(layout_edges)},"
            f"{len(model.quadratic)}"
        )


def iter_cases(args: argparse.Namespace):
    read_counts = (
        parse_int_list(args.read_counts)
        if args.read_counts
        else [args.max_reads]
    )
    for max_reads in read_counts:
        for count_penalty in parse_float_list(args.count_penalties):
            for degree_penalty in parse_float_list(args.degree_penalties):
                for seed in parse_int_list(args.annealer_seeds):
                    yield SweepCase(max_reads, count_penalty, degree_penalty, seed)


def run_case(index, case, reads, edges, args) -> dict[str, object]:
    hamiltonian = EdgePathDAGQUBOHamiltonian(EdgePathDAGHamiltonianConfig(
        edge_count_penalty=case.count_penalty,
        degree_penalty=case.degree_penalty,
        edge_reward_scale=args.reward_scale,
        score_mode=args.score_mode,
        normalize_rewards=True,
        require_hamiltonian_path=True,
    ))
    solver = QUBOLayoutSolver(
        hamiltonian=hamiltonian,
        annealer=OpenJijSimulatedQuantumAnnealer(OpenJijSQAConfig(
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            seed=case.seed,
            beta=args.beta,
            trotter=args.trotter,
        )),
    )

    start = perf_counter()
    layout = solver.solve(reads, edges, weight_mode=args.score_mode)
    elapsed = perf_counter() - start
    model = solver.last_model
    assert model is not None

    true_order = [
        read.rid
        for read in sorted(reads, key=lambda item: item.true_start)
    ]
    true_energy = model.energy(qubo_sample_for_order(model, true_order))
    energy_gap = layout.objective_value - true_energy
    selected_counts = selected_edge_truth_counts(
        reads,
        layout.metadata.get("selected_edges", []),
    )

    return {
        "case": index,
        "reads_used": len(reads),
        "qubo_variables": model.num_variables,
        "count_penalty": case.count_penalty,
        "degree_penalty": case.degree_penalty,
        "reward_scale": args.reward_scale,
        "seed": case.seed,
        "energy": layout.objective_value,
        "true_energy": true_energy,
        "energy_gap": energy_gap,
        "ground_hit": abs(energy_gap) < 1e-6,
        "valid_edge_path": layout.metadata.get("valid_edge_path"),
        "selected_edge_count": layout.metadata.get("selected_edge_count"),
        "edge_count_violation": layout.metadata.get("edge_count_violation"),
        "in_degree_violations": layout.metadata.get("in_degree_violations"),
        "out_degree_violations": layout.metadata.get("out_degree_violations"),
        "selected_adjacent_correct": selected_counts["adjacent_correct"],
        "selected_jump_correct": selected_counts["jump_correct"],
        "selected_wrong": selected_counts["wrong"],
        "anneal_sec": layout.metadata.get("anneal_sec"),
        "total_sec": elapsed,
    }


def selected_edge_truth_counts(reads, selected_edges) -> dict[str, int]:
    rank = {
        read.rid: index
        for index, read in enumerate(sorted(reads, key=lambda item: item.true_start))
    }
    counts = {"adjacent_correct": 0, "jump_correct": 0, "wrong": 0}
    for left_id, right_id in selected_edges:
        left_rank = rank.get(left_id)
        right_rank = rank.get(right_id)
        if left_rank is None or right_rank is None or right_rank <= left_rank:
            counts["wrong"] += 1
        elif right_rank == left_rank + 1:
            counts["adjacent_correct"] += 1
        else:
            counts["jump_correct"] += 1
    return counts


def print_summary(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[int, int, float, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(
            int(row["reads_used"]),
            int(row["qubo_variables"]),
            float(row["count_penalty"]),
            float(row["degree_penalty"]),
        )].append(row)

    summaries = []
    for (reads_used, qubo_variables, count_penalty, degree_penalty), group in groups.items():
        summaries.append({
            "reads_used": reads_used,
            "qubo_variables": qubo_variables,
            "count_penalty": count_penalty,
            "degree_penalty": degree_penalty,
            "cases": len(group),
            "valid_hits": sum(row["valid_edge_path"] is True for row in group),
            "ground_hits": sum(row["ground_hit"] is True for row in group),
            "avg_adjacent": sum(int(row["selected_adjacent_correct"]) for row in group) / len(group),
            "avg_edge_violation": sum(int(row["edge_count_violation"]) for row in group) / len(group),
            "avg_degree_violation": sum(
                int(row["in_degree_violations"]) + int(row["out_degree_violations"])
                for row in group
            ) / len(group),
            "avg_sec": sum(float(row["total_sec"]) for row in group) / len(group),
        })

    summaries.sort(key=lambda item: (
        item["reads_used"],
        -item["ground_hits"],
        -item["valid_hits"],
        item["avg_degree_violation"],
        item["avg_edge_violation"],
        -item["avg_adjacent"],
        item["avg_sec"],
    ))
    print("summary,reads_used,qubo_variables,count_penalty,degree_penalty,cases,valid_hits,ground_hits,"
          "avg_adjacent,avg_edge_violation,avg_degree_violation,avg_sec")
    for item in summaries:
        print(
            "summary,"
            f"{item['reads_used']},"
            f"{item['qubo_variables']},"
            f"{item['count_penalty']:.1f},"
            f"{item['degree_penalty']:.1f},"
            f"{item['cases']},"
            f"{item['valid_hits']},"
            f"{item['ground_hits']},"
            f"{item['avg_adjacent']:.3f},"
            f"{item['avg_edge_violation']:.3f},"
            f"{item['avg_degree_violation']:.3f},"
            f"{item['avg_sec']:.3f}"
        )


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def format_row(row: dict[str, object]) -> str:
    return ",".join(format_cell(row[field]) for field in ROW_FIELDS)


def format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
