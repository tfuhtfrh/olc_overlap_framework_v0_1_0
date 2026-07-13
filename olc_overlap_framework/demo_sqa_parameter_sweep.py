"""
Parameter sweep for QUBO layout annealing.

This script builds one fixed overlap graph, then runs a small grid over QUBO
penalties and annealer parameters. It is intended to diagnose whether SQA/SA is
entering the valid one-hot layout subspace before tuning layout quality.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

from olc_pipeline.candidate_finder import Minimap2CandidateFinder, Minimap2Config
from olc_pipeline.data import OverlapEdge, Read
from olc_pipeline.evaluator import LayoutEvaluator
from olc_pipeline.layout_solver import (
    DWaveAnnealingConfig,
    DWaveSimulatedAnnealer,
    OpenJijSQAConfig,
    OpenJijSimulatedQuantumAnnealer,
    QUBOLayoutSolver,
    QUBOModel,
    WeightedOverlapHamiltonianConfig,
    WeightedOverlapQUBOHamiltonian,
)
from olc_pipeline.mi_scorer import AlignmentMIScorer
from olc_pipeline.pipeline import OverlapExperimentPipeline
from olc_pipeline.refiner import ParasailOverlapRefiner, RefinerConfig
from olc_pipeline.simulator import RandomReadSimulator, SimulationConfig


@dataclass(frozen=True)
class SweepCase:
    read_penalty: float
    position_penalty: float
    missing_penalty: float
    reward_scale: float
    beta: float | None
    trotter: int | None
    annealer_seed: int


ROW_FIELDS = [
    "case",
    "status",
    "backend",
    "read_penalty",
    "position_penalty",
    "missing_penalty",
    "reward_scale",
    "beta",
    "trotter",
    "annealer_seed",
    "energy",
    "true_energy",
    "energy_gap",
    "valid",
    "read_violations",
    "position_violations",
    "legal_edge_count",
    "legal_true_adjacent_count",
    "missing_edge_count",
    "connected_layout",
    "acceptable_connected_layout",
    "missing_edge_cost",
    "adjacent_correct",
    "wrong_adjacencies",
    "inversions",
    "anneal_sec",
    "total_sec",
]


def main() -> None:
    args = parse_args()
    reads, edges = build_overlap_graph(args)
    sweep_reads, sweep_edges = select_layout_inputs(reads, edges, args.max_reads)

    print(f"reads_used,{len(sweep_reads)}")
    print(f"edges_used,{len(sweep_edges)}")
    print(f"backend,{args.backend}")
    print(f"num_reads,{args.num_reads}")
    print(f"num_sweeps,{args.num_sweeps}")
    print(f"simulation_seed,{effective_simulation_seed(args)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        handle.flush()
        for index, case in enumerate(iter_cases(args), start=1):
            row = run_case(index, case, sweep_reads, sweep_edges, args)
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            print(format_row(row), flush=True)

    print_summary(rows)
    print(f"csv,{output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan QUBO/SQA layout parameters.")
    parser.add_argument("--backend", choices=["openjij-sqa", "dwave-sa"], default="openjij-sqa")
    parser.add_argument("--max-reads", type=int, default=16)
    parser.add_argument("--num-reads", type=int, default=50)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simulation-seed", type=int, default=None)
    parser.add_argument("--annealer-seeds", default=None)
    parser.add_argument("--score-mode", default="overlap_len")
    parser.add_argument("--gc-fraction", type=float, default=0.5)
    parser.add_argument("--read-penalties", default="100")
    parser.add_argument("--position-penalties", default=None)
    parser.add_argument("--missing-penalties", default="50")
    parser.add_argument("--reward-scales", default="30")
    parser.add_argument("--betas", default="none")
    parser.add_argument("--trotters", default="16,32,48,64,96,128")
    parser.add_argument("--acceptable-missing-edges", type=int, default=2)
    parser.add_argument("--output", default="debug/qubo/sqa_parameter_sweep.csv")
    return parser.parse_args()


def build_overlap_graph(args: argparse.Namespace) -> tuple[list[Read], list[OverlapEdge]]:
    simulator = RandomReadSimulator()
    sim_config = SimulationConfig(
        genome_len=getattr(args, "genome_len", 20_000),
        read_len=getattr(args, "read_len", 3_000),
        step=getattr(args, "step", 500),
        mismatch_rate=getattr(args, "mismatch_rate", 0.0),
        ins_rate=getattr(args, "ins_rate", 0.0),
        del_rate=getattr(args, "del_rate", 0.0),
        gc_fraction=getattr(args, "gc_fraction", 0.5),
        seed=effective_simulation_seed(args),
        shuffle_reads=True,
    )
    _, reads = simulator.simulate(sim_config)

    finder = Minimap2CandidateFinder(Minimap2Config(
        preset="ava-ont",
        min_overlap=500,
        max_error_rate_hint=0.30,
        overhang_tolerance=80,
        min_mapq=0,
        debug_dir=Path("debug/minimap2"),
    ))
    refiner = ParasailOverlapRefiner(RefinerConfig(
        min_overlap=500,
        max_error_rate=0.20,
        margin=100,
        match_score=2,
        mismatch_penalty=3,
        gap_open=5,
        gap_extend=1,
    ))
    pipeline = OverlapExperimentPipeline(
        candidate_finder=finder,
        refiner=refiner,
        mi_scorer=AlignmentMIScorer(),
    )
    result = pipeline.run(reads)
    return reads, result.edges


def select_layout_inputs(
    reads: list[Read],
    edges: list[OverlapEdge],
    max_reads: int,
) -> tuple[list[Read], list[OverlapEdge]]:
    selected_reads = sorted(reads, key=lambda read: read.true_start)[:max_reads]
    selected_ids = {read.rid for read in selected_reads}
    selected_edges = [
        edge
        for edge in edges
        if edge.left_id in selected_ids and edge.right_id in selected_ids
    ]
    return selected_reads, selected_edges


def iter_cases(args: argparse.Namespace) -> Iterable[SweepCase]:
    read_penalties = parse_float_list(args.read_penalties)
    if args.position_penalties is None:
        penalty_pairs = [(penalty, penalty) for penalty in read_penalties]
    else:
        penalty_pairs = [
            (read_penalty, position_penalty)
            for read_penalty in read_penalties
            for position_penalty in parse_float_list(args.position_penalties)
        ]
    missing_penalties = parse_float_list(args.missing_penalties)
    reward_scales = parse_float_list(args.reward_scales)
    betas = parse_optional_float_list(args.betas)
    trotters = parse_optional_int_list(args.trotters)
    annealer_seeds = parse_int_list(args.annealer_seeds) if args.annealer_seeds else [args.seed]

    for read_penalty, position_penalty in penalty_pairs:
        for missing_penalty in missing_penalties:
            for reward_scale in reward_scales:
                for beta in betas:
                    for trotter in trotters:
                        for annealer_seed in annealer_seeds:
                            yield SweepCase(
                                read_penalty=read_penalty,
                                position_penalty=position_penalty,
                                missing_penalty=missing_penalty,
                                reward_scale=reward_scale,
                                beta=beta,
                                trotter=trotter,
                                annealer_seed=annealer_seed,
                            )


def run_case(
    index: int,
    case: SweepCase,
    reads: list[Read],
    edges: list[OverlapEdge],
    args: argparse.Namespace,
) -> dict[str, object]:
    hamiltonian = WeightedOverlapQUBOHamiltonian(WeightedOverlapHamiltonianConfig(
        read_once_penalty=case.read_penalty,
        position_once_penalty=case.position_penalty,
        missing_edge_penalty=case.missing_penalty,
        edge_reward_scale=case.reward_scale,
        score_mode=args.score_mode,
        normalize_rewards=True,
    ))
    solver = QUBOLayoutSolver(
        hamiltonian=hamiltonian,
        annealer=build_annealer(args, case),
        polisher=None,
    )

    start = perf_counter()
    try:
        layout = solver.solve(reads, edges, weight_mode=args.score_mode)
        elapsed = perf_counter() - start
        report = LayoutEvaluator().evaluate_order(reads, layout.order)
        connection_stats = layout_connection_stats(reads, layout.order, edges)
        missing_edge_count = connection_stats["missing_edge_count"]
        true_energy = true_order_energy(reads, solver.last_model) if solver.last_model else None
        energy_gap = None if true_energy is None else layout.objective_value - true_energy
        return {
            "case": index,
            "status": "ok",
            "backend": layout.metadata.get("annealer_backend"),
            "read_penalty": case.read_penalty,
            "position_penalty": case.position_penalty,
            "missing_penalty": case.missing_penalty,
            "reward_scale": case.reward_scale,
            "beta": format_optional(case.beta),
            "trotter": format_optional(case.trotter),
            "annealer_seed": case.annealer_seed,
            "energy": layout.objective_value,
            "true_energy": true_energy,
            "energy_gap": energy_gap,
            "valid": layout.metadata.get("valid_binary_layout"),
            "read_violations": layout.metadata.get("read_assignment_violations"),
            "position_violations": layout.metadata.get("position_assignment_violations"),
            "legal_edge_count": connection_stats["legal_edge_count"],
            "legal_true_adjacent_count": connection_stats["legal_true_adjacent_count"],
            "missing_edge_count": missing_edge_count,
            "connected_layout": missing_edge_count == 0,
            "acceptable_connected_layout": missing_edge_count <= args.acceptable_missing_edges,
            "missing_edge_cost": missing_edge_count * case.missing_penalty,
            "adjacent_correct": report.adjacent_correct_in_layout,
            "wrong_adjacencies": report.wrong_adjacencies_in_layout,
            "inversions": report.inversion_count,
            "anneal_sec": layout.metadata.get("anneal_sec"),
            "total_sec": elapsed,
        }
    except RuntimeError as exc:
        elapsed = perf_counter() - start
        return {
            "case": index,
            "status": f"error:{exc}",
            "backend": args.backend,
            "read_penalty": case.read_penalty,
            "position_penalty": case.position_penalty,
            "missing_penalty": case.missing_penalty,
            "reward_scale": case.reward_scale,
            "beta": format_optional(case.beta),
            "trotter": format_optional(case.trotter),
            "annealer_seed": case.annealer_seed,
            "energy": "",
            "true_energy": "",
            "energy_gap": "",
            "valid": "",
            "read_violations": "",
            "position_violations": "",
            "legal_edge_count": "",
            "legal_true_adjacent_count": "",
            "missing_edge_count": "",
            "connected_layout": "",
            "acceptable_connected_layout": "",
            "missing_edge_cost": "",
            "adjacent_correct": "",
            "wrong_adjacencies": "",
            "inversions": "",
            "anneal_sec": "",
            "total_sec": elapsed,
        }


def build_annealer(args: argparse.Namespace, case: SweepCase):
    if args.backend == "openjij-sqa":
        return OpenJijSimulatedQuantumAnnealer(OpenJijSQAConfig(
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            seed=case.annealer_seed,
            beta=case.beta,
            trotter=case.trotter,
        ))
    return DWaveSimulatedAnnealer(DWaveAnnealingConfig(
        num_reads=args.num_reads,
        num_sweeps=args.num_sweeps,
        seed=case.annealer_seed,
    ))


def effective_simulation_seed(args: argparse.Namespace) -> int:
    return args.seed if args.simulation_seed is None else args.simulation_seed


def true_order_energy(reads: list[Read], model: QUBOModel | None) -> float | None:
    if model is None:
        return None
    read_index_by_id = {read_id: idx for idx, read_id in enumerate(model.read_ids)}
    sample = [0] * model.num_variables
    for position_index, read in enumerate(sorted(reads, key=lambda item: item.true_start)):
        read_index = read_index_by_id[read.rid]
        sample[model.variable_index(read_index, position_index)] = 1
    return model.energy(sample)


def layout_connection_stats(
    reads: list[Read],
    order: list[str],
    edges: list[OverlapEdge],
) -> dict[str, int]:
    edge_pairs = {(edge.left_id, edge.right_id) for edge in edges}
    true_rank = {
        read.rid: index
        for index, read in enumerate(sorted(reads, key=lambda item: item.true_start))
    }

    legal_edge_count = 0
    legal_true_adjacent_count = 0
    missing_edge_count = 0
    for left_id, right_id in zip(order, order[1:]):
        is_legal = (left_id, right_id) in edge_pairs
        is_true_adjacent = true_rank.get(right_id, -10**9) == true_rank.get(left_id, 10**9) + 1
        if is_legal:
            legal_edge_count += 1
            if is_true_adjacent:
                legal_true_adjacent_count += 1
        else:
            missing_edge_count += 1

    return {
        "legal_edge_count": legal_edge_count,
        "legal_true_adjacent_count": legal_true_adjacent_count,
        "missing_edge_count": missing_edge_count,
    }


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_optional_float_list(value: str) -> list[float | None]:
    result: list[float | None] = []
    for item in value.split(","):
        text = item.strip().lower()
        if not text:
            continue
        result.append(None if text in {"none", "null", "default"} else float(text))
    return result


def parse_optional_int_list(value: str) -> list[int | None]:
    result: list[int | None] = []
    for item in value.split(","):
        text = item.strip().lower()
        if not text:
            continue
        result.append(None if text in {"none", "null", "default"} else int(text))
    return result


def format_optional(value: object) -> object:
    return "default" if value is None else value


def format_row(row: dict[str, object]) -> str:
    fields = [
        "case",
        "status",
        "read_penalty",
        "position_penalty",
        "missing_penalty",
        "reward_scale",
        "beta",
        "trotter",
        "annealer_seed",
        "energy",
        "true_energy",
        "energy_gap",
        "valid",
        "read_violations",
        "position_violations",
        "legal_edge_count",
        "legal_true_adjacent_count",
        "missing_edge_count",
        "connected_layout",
        "acceptable_connected_layout",
        "missing_edge_cost",
        "adjacent_correct",
        "wrong_adjacencies",
        "inversions",
        "anneal_sec",
        "total_sec",
    ]
    return ",".join(format_cell(row.get(field, "")) for field in fields)


def format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    valid_rows = [row for row in rows if row.get("valid") is True]
    invalid_rows = [row for row in rows if row.get("valid") is False]
    connected_rows = [
        row
        for row in valid_rows
        if row.get("connected_layout") is True
    ]
    acceptable_rows = [
        row
        for row in valid_rows
        if row.get("acceptable_connected_layout") is True
    ]
    print(f"summary,total_cases,{len(rows)}")
    print(f"summary,valid_cases,{len(valid_rows)}")
    print(f"summary,invalid_cases,{len(invalid_rows)}")
    print(f"summary,connected_valid_cases,{len(connected_rows)}")
    print(f"summary,acceptable_connected_valid_cases,{len(acceptable_rows)}")
    if not valid_rows:
        return

    best_connected = sorted(
        connected_rows,
        key=lambda row: (
            -int(row["adjacent_correct"]),
            float(row["energy_gap"]),
            float(row["total_sec"]),
        ),
    )[:5]
    best_acceptable = sorted(
        acceptable_rows,
        key=lambda row: (
            int(row["missing_edge_count"]),
            -int(row["legal_true_adjacent_count"]),
            -int(row["adjacent_correct"]),
            float(row["energy_gap"]),
            float(row["total_sec"]),
        ),
    )[:5]
    best_adjacent = sorted(
        valid_rows,
        key=lambda row: (
            int(row["missing_edge_count"]),
            -int(row["legal_true_adjacent_count"]),
            -int(row["adjacent_correct"]),
            float(row["energy_gap"]),
            float(row["total_sec"]),
        ),
    )[:5]
    best_energy = sorted(
        valid_rows,
        key=lambda row: (
            float(row["energy_gap"]),
            -int(row["adjacent_correct"]),
            float(row["total_sec"]),
        ),
    )[:5]

    print("summary,best_connected_valid")
    for row in best_connected:
        print_summary_row(row)
    print("summary,best_acceptable_connected_valid")
    for row in best_acceptable:
        print_summary_row(row)
    print("summary,best_by_adjacent")
    for row in best_adjacent:
        print_summary_row(row)
    print("summary,best_by_energy_gap")
    for row in best_energy:
        print_summary_row(row)


def print_summary_row(row: dict[str, object]) -> None:
    print(
        "summary_case,"
        f"case={row['case']},"
        f"read_penalty={row['read_penalty']},"
        f"missing_penalty={row['missing_penalty']},"
        f"reward_scale={row['reward_scale']},"
        f"trotter={row['trotter']},"
        f"seed={row['annealer_seed']},"
        f"missing_edges={row['missing_edge_count']},"
        f"legal_edges={row['legal_edge_count']},"
        f"legal_true_adjacent={row['legal_true_adjacent_count']},"
        f"adjacent_correct={row['adjacent_correct']},"
        f"wrong={row['wrong_adjacencies']},"
        f"energy_gap={format_cell(row['energy_gap'])},"
        f"total_sec={format_cell(row['total_sec'])}"
    )


if __name__ == "__main__":
    main()
