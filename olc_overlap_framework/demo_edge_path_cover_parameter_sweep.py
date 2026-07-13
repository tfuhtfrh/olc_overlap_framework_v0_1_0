"""
Parameter sweep for the edge-variable DAG path-cover Hamiltonian.

This model uses one variable per candidate edge plus source/sink marker
variables per read. It removes the dense global edge-count constraint from
edge_path_dag and scans A_in=A_out as one local degree-constraint coefficient.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from demo_edge_path_parameter_sweep import parse_float_list, parse_int_list, selected_edge_truth_counts
from demo_sqa_parameter_sweep import build_overlap_graph, select_layout_inputs
from olc_pipeline.layout_solver import (
    EdgePathCoverDAGHamiltonianConfig,
    EdgePathCoverDAGQUBOHamiltonian,
    OpenJijSQAConfig,
    OpenJijSimulatedQuantumAnnealer,
    OverlapRewardScorer,
    QUBOLayoutSolver,
    qubo_sample_for_order,
)


@dataclass(frozen=True)
class SweepCase:
    max_reads: int
    edge_limit: int | None
    degree_penalty: float
    isolate_penalty: float
    path_break_penalty: float
    path_count_cap_penalty: float
    reward_gamma: float
    seed: int


ROW_FIELDS = [
    "case",
    "reads_used",
    "edge_limit",
    "candidate_edges",
    "qubo_variables",
    "quadratic_terms",
    "degree_penalty",
    "isolate_penalty",
    "path_break_penalty",
    "path_count_cap_penalty",
    "reward_gamma",
    "reward_scale",
    "seed",
    "energy",
    "true_energy",
    "energy_gap",
    "ground_hit",
    "valid_path_cover",
    "single_path_layout",
    "path_count",
    "selected_edge_count",
    "source_constraint_violations",
    "sink_constraint_violations",
    "in_degree_violations",
    "out_degree_violations",
    "isolated_node_count",
    "uncovered_node_count",
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
            layout_reads, layout_edges = select_layout_inputs(reads, edges, case.max_reads)
            layout_edges = limit_edges(layout_reads, layout_edges, case.edge_limit, args.score_mode)
            row = run_case(index, case, layout_reads, layout_edges, args)
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            print(format_row(row), flush=True)

    print_summary(rows)
    print(f"csv,{output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan edge-path-cover DAG QUBO parameters.")
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
    parser.add_argument("--annealer-seeds", default="40,41,42,43,44")
    parser.add_argument("--edge-limits", default=None)
    parser.add_argument("--degree-penalties", default="60,80,100,120,140")
    parser.add_argument("--isolate-penalties", default="80,120,160")
    parser.add_argument("--path-break-penalties", default="0,5,10,20,40")
    parser.add_argument("--path-count-cap-penalties", default="120")
    parser.add_argument("--reward-gammas", default="2.0")
    parser.add_argument("--reward-scale", type=float, default=20.0)
    parser.add_argument("--score-mode", default="overlap_len")
    parser.add_argument("--trotter", type=int, default=32)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--output", default="debug/qubo/edge_path_cover_dag_parameter_sweep.csv")
    return parser.parse_args()


def describe_scales(reads, edges, args) -> None:
    read_counts = parse_int_list(args.read_counts) if args.read_counts else [args.max_reads]
    edge_limits = parse_optional_int_list(args.edge_limits)
    print("scale,reads_used,edge_limit,candidate_edges,qubo_variables,quadratic_terms")
    for max_reads in read_counts:
        layout_reads, layout_edges = select_layout_inputs(reads, edges, max_reads)
        for edge_limit in edge_limits:
            score_mode = effective_score_mode(args.score_mode, parse_float_list(args.reward_gammas)[0])
            limited_edges = limit_edges(layout_reads, layout_edges, edge_limit, score_mode)
            hamiltonian = EdgePathCoverDAGQUBOHamiltonian(EdgePathCoverDAGHamiltonianConfig(
                degree_penalty=parse_float_list(args.degree_penalties)[0],
                isolate_penalty=parse_float_list(args.isolate_penalties)[0],
                path_break_penalty=parse_float_list(args.path_break_penalties)[0],
                max_path_count=2,
                path_count_cap_penalty=parse_float_list(args.path_count_cap_penalties)[0],
                edge_reward_scale=args.reward_scale,
                score_mode=score_mode,
                normalize_rewards=True,
            ))
            model = hamiltonian.build(layout_reads, limited_edges, weight_mode=score_mode)
            print(
                "scale,"
                f"{len(layout_reads)},"
                f"{edge_limit if edge_limit is not None else 'all'},"
                f"{len(limited_edges)},"
                f"{model.num_variables},"
                f"{len(model.quadratic)}"
            )


def iter_cases(args: argparse.Namespace):
    read_counts = parse_int_list(args.read_counts) if args.read_counts else [args.max_reads]
    edge_limits = parse_optional_int_list(args.edge_limits)
    for max_reads in read_counts:
        for edge_limit in edge_limits:
            for degree_penalty in parse_float_list(args.degree_penalties):
                for isolate_penalty in parse_float_list(args.isolate_penalties):
                    for path_break_penalty in parse_float_list(args.path_break_penalties):
                        for path_count_cap_penalty in parse_float_list(args.path_count_cap_penalties):
                            for reward_gamma in parse_float_list(args.reward_gammas):
                                for seed in parse_int_list(args.annealer_seeds):
                                    yield SweepCase(
                                        max_reads=max_reads,
                                        edge_limit=edge_limit,
                                        degree_penalty=degree_penalty,
                                        isolate_penalty=isolate_penalty,
                                        path_break_penalty=path_break_penalty,
                                        path_count_cap_penalty=path_count_cap_penalty,
                                        reward_gamma=reward_gamma,
                                        seed=seed,
                                    )


def run_case(index, case, reads, edges, args) -> dict[str, object]:
    score_mode = effective_score_mode(args.score_mode, case.reward_gamma)
    hamiltonian = EdgePathCoverDAGQUBOHamiltonian(EdgePathCoverDAGHamiltonianConfig(
        degree_penalty=case.degree_penalty,
        isolate_penalty=case.isolate_penalty,
        path_break_penalty=case.path_break_penalty,
        max_path_count=2,
        path_count_cap_penalty=case.path_count_cap_penalty,
        edge_reward_scale=args.reward_scale,
        score_mode=score_mode,
        normalize_rewards=True,
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
    layout = solver.solve(reads, edges, weight_mode=score_mode)
    elapsed = perf_counter() - start
    model = solver.last_model
    assert model is not None

    true_order = [read.rid for read in sorted(reads, key=lambda item: item.true_start)]
    true_energy = true_order_energy(model, true_order)
    energy_gap = None if true_energy is None else layout.objective_value - true_energy
    selected_counts = selected_edge_truth_counts(reads, layout.metadata.get("selected_edges", []))

    return {
        "case": index,
        "reads_used": len(reads),
        "edge_limit": case.edge_limit if case.edge_limit is not None else "all",
        "candidate_edges": len(edges),
        "qubo_variables": model.num_variables,
        "quadratic_terms": len(model.quadratic),
        "degree_penalty": case.degree_penalty,
        "isolate_penalty": case.isolate_penalty,
        "path_break_penalty": case.path_break_penalty,
        "path_count_cap_penalty": case.path_count_cap_penalty,
        "reward_gamma": case.reward_gamma,
        "reward_scale": args.reward_scale,
        "seed": case.seed,
        "energy": layout.objective_value,
        "true_energy": true_energy,
        "energy_gap": energy_gap,
        "ground_hit": energy_gap is not None and abs(energy_gap) < 1e-6,
        "valid_path_cover": layout.metadata.get("valid_path_cover"),
        "single_path_layout": layout.metadata.get("single_path_layout"),
        "path_count": layout.metadata.get("path_count"),
        "selected_edge_count": layout.metadata.get("selected_edge_count"),
        "source_constraint_violations": layout.metadata.get("source_constraint_violations"),
        "sink_constraint_violations": layout.metadata.get("sink_constraint_violations"),
        "in_degree_violations": layout.metadata.get("in_degree_violations"),
        "out_degree_violations": layout.metadata.get("out_degree_violations"),
        "isolated_node_count": layout.metadata.get("isolated_node_count"),
        "uncovered_node_count": len(layout.metadata.get("uncovered_nodes", [])),
        "selected_adjacent_correct": selected_counts["adjacent_correct"],
        "selected_jump_correct": selected_counts["jump_correct"],
        "selected_wrong": selected_counts["wrong"],
        "anneal_sec": layout.metadata.get("anneal_sec"),
        "total_sec": elapsed,
    }


def limit_edges(reads, edges, edge_limit: int | None, score_mode: str):
    if edge_limit is None or edge_limit >= len(edges):
        return edges
    rank = {
        read.rid: index
        for index, read in enumerate(sorted(reads, key=lambda item: item.true_start))
    }
    scorer = OverlapRewardScorer()
    return sorted(
        edges,
        key=lambda edge: (
            -scorer.score(edge, score_mode),
            rank.get(edge.left_id, len(rank)),
            rank.get(edge.right_id, len(rank)),
            edge.left_id,
            edge.right_id,
        ),
    )[:edge_limit]


def true_order_energy(model, true_order):
    try:
        return model.energy(qubo_sample_for_order(model, true_order))
    except ValueError:
        return None


def effective_score_mode(score_mode: str, reward_gamma: float) -> str:
    if score_mode == "overlap_len_power":
        return f"overlap_len_power:{reward_gamma}"
    return score_mode


def print_summary(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[int, int, int, float, float, float, float, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(
            int(row["reads_used"]),
            int(row["candidate_edges"]),
            int(row["qubo_variables"]),
            float(row["degree_penalty"]),
            float(row["isolate_penalty"]),
            float(row["path_break_penalty"]),
            float(row["path_count_cap_penalty"]),
            float(row["reward_gamma"]),
        )].append(row)

    summaries = []
    for (
        reads_used,
        candidate_edges,
        qubo_variables,
        degree_penalty,
        isolate_penalty,
        path_break_penalty,
        path_count_cap_penalty,
        reward_gamma,
    ), group in groups.items():
        summaries.append({
            "reads_used": reads_used,
            "candidate_edges": candidate_edges,
            "qubo_variables": qubo_variables,
            "degree_penalty": degree_penalty,
            "isolate_penalty": isolate_penalty,
            "path_break_penalty": path_break_penalty,
            "path_count_cap_penalty": path_count_cap_penalty,
            "reward_gamma": reward_gamma,
            "cases": len(group),
            "cover_hits": sum(row["valid_path_cover"] is True for row in group),
            "single_hits": sum(row["single_path_layout"] is True for row in group),
            "ground_hits": sum(row["ground_hit"] is True for row in group),
            "avg_path_count": sum(int(row["path_count"]) for row in group) / len(group),
            "avg_adjacent": sum(int(row["selected_adjacent_correct"]) for row in group) / len(group),
            "avg_wrong": sum(int(row["selected_wrong"]) for row in group) / len(group),
            "avg_hard_violation": sum(
                int(row["source_constraint_violations"])
                + int(row["sink_constraint_violations"])
                + int(row["in_degree_violations"])
                + int(row["out_degree_violations"])
                + int(row["isolated_node_count"])
                + int(row["uncovered_node_count"])
                for row in group
            ) / len(group),
            "avg_sec": sum(float(row["total_sec"]) for row in group) / len(group),
        })

    summaries.sort(key=lambda item: (
        item["reads_used"],
        item["candidate_edges"],
        -item["ground_hits"],
        -item["single_hits"],
        -item["cover_hits"],
        item["avg_hard_violation"],
        -item["avg_adjacent"],
        item["avg_wrong"],
        item["avg_path_count"],
        item["avg_sec"],
    ))
    print("summary,reads_used,candidate_edges,qubo_variables,degree_penalty,isolate_penalty,path_break_penalty,path_count_cap_penalty,reward_gamma,"
          "cases,cover_hits,single_hits,ground_hits,avg_path_count,avg_adjacent,avg_wrong,"
          "avg_hard_violation,avg_sec")
    for item in summaries:
        print(
            "summary,"
            f"{item['reads_used']},"
            f"{item['candidate_edges']},"
            f"{item['qubo_variables']},"
            f"{item['degree_penalty']:.1f},"
            f"{item['isolate_penalty']:.1f},"
            f"{item['path_break_penalty']:.1f},"
            f"{item['path_count_cap_penalty']:.1f},"
            f"{item['reward_gamma']:.2f},"
            f"{item['cases']},"
            f"{item['cover_hits']},"
            f"{item['single_hits']},"
            f"{item['ground_hits']},"
            f"{item['avg_path_count']:.3f},"
            f"{item['avg_adjacent']:.3f},"
            f"{item['avg_wrong']:.3f},"
            f"{item['avg_hard_violation']:.3f},"
            f"{item['avg_sec']:.3f}"
        )


def format_row(row: dict[str, object]) -> str:
    return ",".join(format_cell(row[field]) for field in ROW_FIELDS)


def format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def parse_optional_int_list(value: str | None) -> list[int | None]:
    if value is None:
        return [None]
    result: list[int | None] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() in {"all", "none"}:
            result.append(None)
        else:
            result.append(int(item))
    return result or [None]


if __name__ == "__main__":
    main()
