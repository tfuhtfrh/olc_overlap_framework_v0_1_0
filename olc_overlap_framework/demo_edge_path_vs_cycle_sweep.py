"""
Compare the E+2N path-cover Hamiltonian with the E+2N void-cycle Hamiltonian.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from demo_edge_path_parameter_sweep import parse_float_list, parse_int_list, selected_edge_truth_counts
from demo_sqa_parameter_sweep import build_overlap_graph, select_layout_inputs
from olc_pipeline.layout_solver import (
    EdgeCycleCoverDAGHamiltonianConfig,
    EdgeCycleCoverDAGQUBOHamiltonian,
    EdgePathDAGHamiltonianConfig,
    EdgePathDAGQUBOHamiltonian,
    EdgePathCoverDAGHamiltonianConfig,
    EdgePathCoverDAGQUBOHamiltonian,
    OpenJijSQAConfig,
    OpenJijSimulatedQuantumAnnealer,
    QUBOLayoutSolver,
    qubo_sample_for_order,
)


@dataclass(frozen=True)
class SweepCase:
    model: str
    max_reads: int
    path_count_penalty: float | None
    degree_penalty: float
    isolate_penalty: float | None
    path_break_penalty: float | None
    path_count_cap_penalty: float | None
    seed: int


ROW_FIELDS = [
    "case",
    "model",
    "reads_used",
    "candidate_edges",
    "qubo_variables",
    "quadratic_terms",
    "path_count_penalty",
    "degree_penalty",
    "isolate_penalty",
    "path_break_penalty",
    "path_count_cap_penalty",
    "reward_scale",
    "seed",
    "energy",
    "true_energy",
    "energy_gap",
    "ground_hit",
    "valid_layout",
    "selected_edge_count",
    "selected_source_count",
    "selected_sink_count",
    "edge_count_violation",
    "in_degree_violations",
    "out_degree_violations",
    "read_in_constraint_violations",
    "read_out_constraint_violations",
    "void_in_constraint_violation",
    "void_out_constraint_violation",
    "selected_adjacent_correct",
    "selected_jump_correct",
    "selected_wrong",
    "adjacent_recall",
    "truth_forward_edges",
    "truth_gap_edges",
    "truth_backward_edges",
    "main_path_nodes",
    "main_path_edges",
    "main_path_start_rank",
    "main_path_end_rank",
    "main_path_missing_reads",
    "main_path_max_jump",
    "full_dna_recoverable",
    "prunable_extra_edges",
    "build_sec",
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
    print(f"gc_fraction,{args.gc_fraction}")
    print(f"read_counts,{args.read_counts or args.max_reads}")
    print(f"num_reads,{args.num_reads}")
    print(f"num_sweeps,{args.num_sweeps}")
    print(f"trotter,{args.trotter}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        handle.flush()
        for index, case in enumerate(iter_cases(args), start=1):
            layout_reads, layout_edges = select_layout_inputs(reads, edges, case.max_reads)
            row = run_case(index, case, layout_reads, layout_edges, args)
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            print(format_row(row), flush=True)

    print_summary(rows)
    print(f"csv,{output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare E+2N path-cover and void-cycle QUBO Hamiltonians.")
    parser.add_argument("--models", default="path_cover,cycle")
    parser.add_argument("--max-reads", type=int, default=100)
    parser.add_argument("--read-counts", default=None)
    parser.add_argument("--genome-len", type=int, default=52_500)
    parser.add_argument("--read-len", type=int, default=3_000)
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--mismatch-rate", type=float, default=0.0)
    parser.add_argument("--ins-rate", type=float, default=0.0)
    parser.add_argument("--del-rate", type=float, default=0.0)
    parser.add_argument("--gc-fraction", type=float, default=0.65)
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--simulation-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annealer-seeds", default="40,41")
    parser.add_argument("--path-count-penalties", default="40,50,60,70")
    parser.add_argument("--path-degree-penalties", default="60,80,100")
    parser.add_argument("--path-cover-degree-penalties", default="60,80,100,120")
    parser.add_argument("--path-cover-isolate-penalties", default="80,120,160")
    parser.add_argument("--path-cover-break-penalties", default="0,10")
    parser.add_argument("--path-cover-cap-penalties", default="120")
    parser.add_argument("--cycle-degree-penalties", default="40,60,80,100,120")
    parser.add_argument("--reward-scale", type=float, default=20.0)
    parser.add_argument("--score-mode", default="overlap_len")
    parser.add_argument("--trotter", type=int, default=32)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--output", default="debug/qubo/edge_path_vs_cycle_sweep.csv")
    return parser.parse_args()


def iter_cases(args: argparse.Namespace):
    read_counts = parse_int_list(args.read_counts) if args.read_counts else [args.max_reads]
    models = {item.strip() for item in args.models.split(",") if item.strip()}
    for max_reads in read_counts:
        for seed in parse_int_list(args.annealer_seeds):
            if "path" in models:
                for count_penalty in parse_float_list(args.path_count_penalties):
                    for degree_penalty in parse_float_list(args.path_degree_penalties):
                        yield SweepCase("path", max_reads, count_penalty, degree_penalty, None, None, None, seed)
            if "path_cover" in models:
                for degree_penalty in parse_float_list(args.path_cover_degree_penalties):
                    for isolate_penalty in parse_float_list(args.path_cover_isolate_penalties):
                        for break_penalty in parse_float_list(args.path_cover_break_penalties):
                            for cap_penalty in parse_float_list(args.path_cover_cap_penalties):
                                yield SweepCase(
                                    "path_cover",
                                    max_reads,
                                    None,
                                    degree_penalty,
                                    isolate_penalty,
                                    break_penalty,
                                    cap_penalty,
                                    seed,
                                )
            if "cycle" in models:
                for degree_penalty in parse_float_list(args.cycle_degree_penalties):
                    yield SweepCase("cycle", max_reads, None, degree_penalty, None, None, None, seed)


def run_case(index, case, reads, edges, args) -> dict[str, object]:
    if case.model == "path":
        hamiltonian = EdgePathDAGQUBOHamiltonian(EdgePathDAGHamiltonianConfig(
            edge_count_penalty=case.path_count_penalty,
            degree_penalty=case.degree_penalty,
            edge_reward_scale=args.reward_scale,
            score_mode=args.score_mode,
            normalize_rewards=True,
            require_hamiltonian_path=True,
        ))
    elif case.model == "cycle":
        hamiltonian = EdgeCycleCoverDAGQUBOHamiltonian(EdgeCycleCoverDAGHamiltonianConfig(
            degree_penalty=case.degree_penalty,
            edge_reward_scale=args.reward_scale,
            score_mode=args.score_mode,
            normalize_rewards=True,
        ))
    elif case.model == "path_cover":
        hamiltonian = EdgePathCoverDAGQUBOHamiltonian(EdgePathCoverDAGHamiltonianConfig(
            degree_penalty=case.degree_penalty,
            isolate_penalty=case.isolate_penalty if case.isolate_penalty is not None else 100.0,
            path_break_penalty=case.path_break_penalty if case.path_break_penalty is not None else 0.0,
            max_path_count=2,
            path_count_cap_penalty=case.path_count_cap_penalty if case.path_count_cap_penalty is not None else 100.0,
            edge_reward_scale=args.reward_scale,
            score_mode=args.score_mode,
            normalize_rewards=True,
        ))
    else:
        raise ValueError(f"unknown model: {case.model!r}")

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

    true_order = [read.rid for read in sorted(reads, key=lambda item: item.true_start)]
    true_energy = true_order_energy(model, true_order)
    energy_gap = None if true_energy is None else layout.objective_value - true_energy
    selected_counts = selected_edge_truth_counts(reads, layout.metadata.get("selected_edges", []))
    recoverability = evaluate_recoverability(reads, layout.metadata.get("selected_edges", []))
    target_edges = max(0, len(reads) - 1)
    adjacent_recall = selected_counts["adjacent_correct"] / target_edges if target_edges else 1.0
    valid_layout = (
        layout.metadata.get("valid_edge_cycle")
        if case.model == "cycle"
        else layout.metadata.get("single_path_layout")
    )

    return {
        "case": index,
        "model": case.model,
        "reads_used": len(reads),
        "candidate_edges": len(edges),
        "qubo_variables": model.num_variables,
        "quadratic_terms": len(model.quadratic),
        "path_count_penalty": case.path_count_penalty if case.path_count_penalty is not None else "",
        "degree_penalty": case.degree_penalty,
        "isolate_penalty": case.isolate_penalty if case.isolate_penalty is not None else "",
        "path_break_penalty": case.path_break_penalty if case.path_break_penalty is not None else "",
        "path_count_cap_penalty": case.path_count_cap_penalty if case.path_count_cap_penalty is not None else "",
        "reward_scale": args.reward_scale,
        "seed": case.seed,
        "energy": layout.objective_value,
        "true_energy": true_energy if true_energy is not None else "",
        "energy_gap": energy_gap if energy_gap is not None else "",
        "ground_hit": energy_gap is not None and abs(energy_gap) < 1e-6,
        "valid_layout": valid_layout,
        "selected_edge_count": layout.metadata.get("selected_edge_count", ""),
        "selected_source_count": layout.metadata.get("selected_source_count", ""),
        "selected_sink_count": layout.metadata.get("selected_sink_count", ""),
        "edge_count_violation": layout.metadata.get("edge_count_violation", ""),
        "in_degree_violations": layout.metadata.get("in_degree_violations", ""),
        "out_degree_violations": layout.metadata.get("out_degree_violations", ""),
        "read_in_constraint_violations": layout.metadata.get("read_in_constraint_violations", ""),
        "read_out_constraint_violations": layout.metadata.get("read_out_constraint_violations", ""),
        "void_in_constraint_violation": layout.metadata.get("void_in_constraint_violation", ""),
        "void_out_constraint_violation": layout.metadata.get("void_out_constraint_violation", ""),
        "selected_adjacent_correct": selected_counts["adjacent_correct"],
        "selected_jump_correct": selected_counts["jump_correct"],
        "selected_wrong": selected_counts["wrong"],
        "adjacent_recall": adjacent_recall,
        **recoverability,
        "build_sec": layout.metadata.get("build_sec"),
        "anneal_sec": layout.metadata.get("anneal_sec"),
        "total_sec": elapsed,
    }


def true_order_energy(model, true_order: list[str]) -> float | None:
    try:
        return model.energy(qubo_sample_for_order(model, true_order))
    except ValueError:
        return None


def evaluate_recoverability(reads, selected_edges) -> dict[str, object]:
    reads_sorted = sorted(reads, key=lambda item: item.true_start)
    read_by_id = {read.rid: read for read in reads}
    rank = {read.rid: index for index, read in enumerate(reads_sorted)}
    n = len(reads_sorted)

    truth_forward_edges = 0
    truth_gap_edges = 0
    truth_backward_edges = 0
    contiguous_edges: list[tuple[str, str]] = []
    for left_id, right_id in selected_edges:
        left = read_by_id.get(left_id)
        right = read_by_id.get(right_id)
        if left is None or right is None:
            continue
        if rank[right_id] <= rank[left_id]:
            truth_backward_edges += 1
            continue
        truth_forward_edges += 1
        if left.true_end < right.true_start:
            truth_gap_edges += 1
            continue
        contiguous_edges.append((left_id, right_id))

    best_any = longest_contiguous_path(reads_sorted, contiguous_edges, start_rank=None)
    best_from_start = longest_contiguous_path(reads_sorted, contiguous_edges, start_rank=0)
    full_path = best_from_start if best_from_start and rank[best_from_start[-1]] == n - 1 else []
    main_path = full_path or best_any
    main_edges = list(zip(main_path, main_path[1:]))
    main_ranks = [rank[read_id] for read_id in main_path]
    full_dna_recoverable = bool(full_path)
    main_path_missing_reads = (
        (main_ranks[-1] - main_ranks[0] + 1 - len(main_ranks))
        if main_ranks
        else n
    )
    main_path_max_jump = max(
        (right_rank - left_rank for left_rank, right_rank in zip(main_ranks, main_ranks[1:])),
        default=0,
    )
    return {
        "truth_forward_edges": truth_forward_edges,
        "truth_gap_edges": truth_gap_edges,
        "truth_backward_edges": truth_backward_edges,
        "main_path_nodes": len(main_path),
        "main_path_edges": len(main_edges),
        "main_path_start_rank": main_ranks[0] if main_ranks else "",
        "main_path_end_rank": main_ranks[-1] if main_ranks else "",
        "main_path_missing_reads": main_path_missing_reads,
        "main_path_max_jump": main_path_max_jump,
        "full_dna_recoverable": full_dna_recoverable,
        "prunable_extra_edges": max(0, len(selected_edges) - len(main_edges)) if full_dna_recoverable else "",
    }


def longest_contiguous_path(reads_sorted, edges, start_rank: int | None) -> list[str]:
    rank = {read.rid: index for index, read in enumerate(reads_sorted)}
    outgoing: dict[str, list[str]] = {read.rid: [] for read in reads_sorted}
    for left_id, right_id in edges:
        outgoing[left_id].append(right_id)

    best_path: dict[str, list[str]] = {}
    for read in reversed(reads_sorted):
        path = [read.rid]
        for right_id in sorted(outgoing[read.rid], key=rank.get):
            suffix = best_path.get(right_id, [right_id])
            candidate = [read.rid, *suffix]
            if path_score(candidate, rank) > path_score(path, rank):
                path = candidate
        best_path[read.rid] = path

    if start_rank is not None:
        return best_path.get(reads_sorted[start_rank].rid, [])

    candidates = [path for path in best_path.values() if path]
    if not candidates:
        return []
    return max(candidates, key=lambda path: path_score(path, rank))


def path_score(path: list[str], rank: dict[str, int]) -> tuple[int, int, int]:
    if not path:
        return (-1, -1, 0)
    return (rank[path[-1]] - rank[path[0]], len(path), -rank[path[0]])


def format_row(row: dict[str, object]) -> str:
    return ",".join(str(row[field]) for field in ROW_FIELDS)


def print_summary(rows: list[dict[str, object]]) -> None:
    print("summary,model,reads,variables,quadratic_terms,params,cases,valid_hits,ground_hits,recoverable_hits,avg_recall,avg_adjacent,avg_jump,avg_gap,avg_backward,avg_prunable,avg_sec")
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["model"],
            row["reads_used"],
            row["qubo_variables"],
            row["quadratic_terms"],
            row["path_count_penalty"],
            row["degree_penalty"],
            row["isolate_penalty"],
            row["path_break_penalty"],
            row["path_count_cap_penalty"],
        )
        groups.setdefault(key, []).append(row)
    for key, group in sorted(groups.items(), key=lambda item: (str(item[0][0]), int(item[0][1]), str(item[0][4]), float(item[0][5]))):
        model_name, reads_used, variables, quadratic_terms, count_penalty, degree_penalty, isolate_penalty, break_penalty, cap_penalty = key
        params = f"count={count_penalty};degree={degree_penalty};isolate={isolate_penalty};break={break_penalty};cap={cap_penalty}"
        print(
            "summary,"
            f"{model_name},"
            f"{reads_used},"
            f"{variables},"
            f"{quadratic_terms},"
            f"{params},"
            f"{len(group)},"
            f"{sum(row['valid_layout'] is True for row in group)},"
            f"{sum(row['ground_hit'] is True for row in group)},"
            f"{sum(row['full_dna_recoverable'] is True for row in group)},"
            f"{sum(float(row['adjacent_recall']) for row in group) / len(group):.4f},"
            f"{sum(int(row['selected_adjacent_correct']) for row in group) / len(group):.3f},"
            f"{sum(int(row['selected_jump_correct']) for row in group) / len(group):.3f},"
            f"{sum(int(row['truth_gap_edges']) for row in group) / len(group):.3f},"
            f"{sum(int(row['truth_backward_edges']) for row in group) / len(group):.3f},"
            f"{sum(float(row['prunable_extra_edges'] or 0) for row in group) / len(group):.3f},"
            f"{sum(float(row['total_sec']) for row in group) / len(group):.3f}"
        )


if __name__ == "__main__":
    main()
