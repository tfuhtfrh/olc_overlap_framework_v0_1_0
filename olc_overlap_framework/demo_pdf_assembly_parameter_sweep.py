"""
Parameter sweep for the PDF assembly Hamiltonian with the path A term.

The model keeps the PDF B/C/D terms, but replaces the original cycle-cover A
term with the existing edge-path DAG A term: select N-1 edges and avoid
multiple predecessors/successors for each read.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from demo_edge_path_cover_parameter_sweep import limit_edges, parse_optional_int_list
from demo_edge_path_parameter_sweep import parse_float_list, parse_int_list, selected_edge_truth_counts
from demo_edge_path_vs_cycle_sweep import evaluate_recoverability
from olc_pipeline.candidate_finder import Minimap2CandidateFinder, Minimap2Config
from olc_pipeline.data import OverlapEdge, Read
from olc_pipeline.layout_solver import (
    BinaryAnnealingConfig,
    BinarySimulatedAnnealer,
    OpenJijSQAConfig,
    OpenJijSimulatedQuantumAnnealer,
    OverlapRewardScorer,
    PDFAssemblyHamiltonianConfig,
    PDFAssemblyQUBOHamiltonian,
    QUBOLayoutSolver,
    qubo_sample_for_order,
)
from olc_pipeline.mi_scorer import AlignmentMIScorer
from olc_pipeline.overlap_features import gc_fraction
from olc_pipeline.pipeline import OverlapExperimentPipeline
from olc_pipeline.refiner import ParasailOverlapRefiner, RefinerConfig
from olc_pipeline.simulator import RandomReadSimulator, SimulationConfig


@dataclass(frozen=True)
class SweepCase:
    max_reads: int
    edge_limit: int | None
    degree_penalty: float
    length_penalty: float
    gc_penalty: float
    mi_reward_scale: float
    seed: int


ROW_FIELDS = [
    "case",
    "status",
    "reads_used",
    "edge_limit",
    "candidate_edges",
    "qubo_variables",
    "quadratic_terms",
    "degree_penalty",
    "length_target",
    "length_penalty",
    "gc_target_fraction",
    "gc_penalty",
    "mi_reward_scale",
    "mi_score_mode",
    "mi_min",
    "mi_max",
    "candidate_adjacent_edges",
    "candidate_jump_edges",
    "candidate_wrong_edges",
    "adjacent_score_mean",
    "jump_score_mean",
    "wrong_score_mean",
    "overlap_gc_method",
    "backend",
    "seed",
    "energy",
    "true_path_available",
    "true_path_energy",
    "energy_gap",
    "ground_hit",
    "valid_edge_path",
    "single_path_layout",
    "selected_edge_count",
    "edge_count_violation",
    "in_degree_violations",
    "out_degree_violations",
    "hard_violation",
    "selected_adjacent_correct",
    "selected_jump_correct",
    "selected_wrong",
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
    "anneal_sec",
    "total_sec",
]


def main() -> None:
    args = parse_args()
    genome, reads, edges = build_overlap_graph(args)

    print(f"simulation_reads,{len(reads)}")
    print(f"genome_len,{len(genome)}")
    print(f"read_len,{args.read_len}")
    print(f"step,{args.step}")
    print(f"read_counts,{args.read_counts or ('all' if args.max_reads <= 0 else args.max_reads)}")
    print(f"backend,{args.backend}")
    print(f"num_reads,{args.num_reads}")
    print(f"num_sweeps,{args.num_sweeps}")
    print(f"target_mode,{args.target_mode}")

    if args.describe_only:
        describe_models(genome, reads, edges, args)
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
            layout_edges = limit_edges(layout_reads, layout_edges, case.edge_limit, args.edge_score_mode)
            row = run_case(index, case, genome, layout_reads, layout_edges, args)
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            print(format_row(row), flush=True)

    print_summary(rows)
    print(f"csv,{output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan PDF assembly QUBO parameters with the path A term.")
    parser.add_argument("--max-reads", type=int, default=0, help="Use <=0 to keep all simulated reads.")
    parser.add_argument("--read-counts", default=None)
    parser.add_argument("--edge-limits", default=None)
    parser.add_argument("--genome-len", type=int, default=20_000)
    parser.add_argument("--read-len", type=int, default=3_000)
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--mismatch-rate", type=float, default=0.0)
    parser.add_argument("--ins-rate", type=float, default=0.0)
    parser.add_argument("--del-rate", type=float, default=0.0)
    parser.add_argument("--gc-fraction", type=float, default=0.5)
    parser.add_argument("--simulation-seed", type=int, default=42)
    parser.add_argument("--min-overlap", type=int, default=500)
    parser.add_argument("--max-error-rate-hint", type=float, default=0.30)
    parser.add_argument("--overhang-tolerance", type=int, default=80)
    parser.add_argument("--refiner-max-error-rate", type=float, default=0.20)
    parser.add_argument("--refiner-margin", type=int, default=100)
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--backend", choices=["builtin", "openjij-sqa"], default="builtin")
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--trotter", type=int, default=32)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--annealer-seeds", default="40,41,42,43,44")
    parser.add_argument("--degree-penalties", default="100,300,1000")
    parser.add_argument("--length-penalties", default="0,0.000001,0.00001")
    parser.add_argument("--gc-penalties", default="0,0.000001,0.00001")
    parser.add_argument("--mi-reward-scales", default="0,1")
    parser.add_argument("--length-target", type=float, default=None)
    parser.add_argument("--gc-target-fraction", type=float, default=None)
    parser.add_argument("--target-mode", choices=["selected-span", "genome"], default="selected-span")
    parser.add_argument("--mi-score-mode", default="mi")
    parser.add_argument("--overlap-gc-method", choices=["average", "left", "right"], default="average")
    parser.add_argument("--edge-score-mode", default="overlap_len")
    parser.add_argument("--output", default="debug/qubo/pdf_assembly_path_parameter_sweep.csv")
    return parser.parse_args()


def build_overlap_graph(args: argparse.Namespace) -> tuple[str, list[Read], list[OverlapEdge]]:
    simulator = RandomReadSimulator()
    sim_config = SimulationConfig(
        genome_len=args.genome_len,
        read_len=args.read_len,
        step=args.step,
        mismatch_rate=args.mismatch_rate,
        ins_rate=args.ins_rate,
        del_rate=args.del_rate,
        gc_fraction=args.gc_fraction,
        seed=args.simulation_seed,
        shuffle_reads=True,
    )
    genome, reads = simulator.simulate(sim_config)

    finder = Minimap2CandidateFinder(Minimap2Config(
        preset="ava-ont",
        min_overlap=args.min_overlap,
        max_error_rate_hint=args.max_error_rate_hint,
        overhang_tolerance=args.overhang_tolerance,
        min_mapq=0,
        debug_dir=Path("debug/minimap2"),
    ))
    refiner = ParasailOverlapRefiner(RefinerConfig(
        min_overlap=args.min_overlap,
        max_error_rate=args.refiner_max_error_rate,
        margin=args.refiner_margin,
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
    return genome, reads, result.edges


def select_layout_inputs(
    reads: list[Read],
    edges: list[OverlapEdge],
    max_reads: int,
) -> tuple[list[Read], list[OverlapEdge]]:
    ordered_reads = sorted(reads, key=lambda read: read.true_start)
    selected_reads = ordered_reads if max_reads <= 0 else ordered_reads[:max_reads]
    selected_ids = {read.rid for read in selected_reads}
    selected_edges = [
        edge
        for edge in edges
        if edge.left_id in selected_ids and edge.right_id in selected_ids
    ]
    return selected_reads, selected_edges


def describe_models(genome: str, reads: list[Read], edges: list[OverlapEdge], args: argparse.Namespace) -> None:
    print("scale,reads_used,edge_limit,candidate_edges,qubo_variables,quadratic_terms")
    for max_reads in read_counts(args):
        layout_reads, layout_edges = select_layout_inputs(reads, edges, max_reads)
        for edge_limit in parse_optional_int_list(args.edge_limits):
            limited_edges = limit_edges(layout_reads, layout_edges, edge_limit, args.edge_score_mode)
            hamiltonian = build_hamiltonian(
                genome,
                layout_reads,
                args,
                degree_penalty=parse_float_list(args.degree_penalties)[0],
                length_penalty=parse_float_list(args.length_penalties)[0],
                gc_penalty=parse_float_list(args.gc_penalties)[0],
                mi_reward_scale=parse_float_list(args.mi_reward_scales)[0],
            )
            try:
                model = hamiltonian.build(layout_reads, limited_edges)
                variables = model.num_variables
                quadratic_terms = len(model.quadratic)
            except ValueError as exc:
                variables = "error"
                quadratic_terms = str(exc)
            print(
                "scale,"
                f"{len(layout_reads)},"
                f"{edge_limit if edge_limit is not None else 'all'},"
                f"{len(limited_edges)},"
                f"{variables},"
                f"{quadratic_terms}"
            )


def iter_cases(args: argparse.Namespace):
    edge_limits = parse_optional_int_list(args.edge_limits)
    for max_reads in read_counts(args):
        for edge_limit in edge_limits:
            for degree_penalty in parse_float_list(args.degree_penalties):
                for length_penalty in parse_float_list(args.length_penalties):
                    for gc_penalty in parse_float_list(args.gc_penalties):
                        for mi_reward_scale in parse_float_list(args.mi_reward_scales):
                            for seed in parse_int_list(args.annealer_seeds):
                                yield SweepCase(
                                    max_reads=max_reads,
                                    edge_limit=edge_limit,
                                    degree_penalty=degree_penalty,
                                    length_penalty=length_penalty,
                                    gc_penalty=gc_penalty,
                                    mi_reward_scale=mi_reward_scale,
                                    seed=seed,
                                )


def run_case(
    index: int,
    case: SweepCase,
    genome: str,
    reads: list[Read],
    edges: list[OverlapEdge],
    args: argparse.Namespace,
) -> dict[str, object]:
    hamiltonian = build_hamiltonian(
        genome,
        reads,
        args,
        degree_penalty=case.degree_penalty,
        length_penalty=case.length_penalty,
        gc_penalty=case.gc_penalty,
        mi_reward_scale=case.mi_reward_scale,
    )
    solver = QUBOLayoutSolver(
        hamiltonian=hamiltonian,
        annealer=build_annealer(args, case),
    )

    start = perf_counter()
    try:
        layout = solver.solve(reads, edges)
        elapsed = perf_counter() - start
        model = solver.last_model
        assert model is not None

        true_order = [read.rid for read in sorted(reads, key=lambda item: item.true_start)]
        true_energy = true_path_energy(model, true_order)
        energy_gap = None if true_energy is None else layout.objective_value - true_energy
        selected_edges = layout.metadata.get("selected_edges", [])
        selected_counts = selected_edge_truth_counts(reads, selected_edges)
        recoverability = evaluate_recoverability(reads, selected_edges)
        score_summary = candidate_score_summary(reads, edges, args.mi_score_mode)
        hard_violation = (
            int(layout.metadata.get("edge_count_violation") or 0)
            + int(layout.metadata.get("in_degree_violations") or 0)
            + int(layout.metadata.get("out_degree_violations") or 0)
        )

        return {
            "case": index,
            "status": "ok",
            "reads_used": len(reads),
            "edge_limit": case.edge_limit if case.edge_limit is not None else "all",
            "candidate_edges": len(edges),
            "qubo_variables": model.num_variables,
            "quadratic_terms": len(model.quadratic),
            "degree_penalty": case.degree_penalty,
            "length_target": length_target(genome, reads, args),
            "length_penalty": case.length_penalty,
            "gc_target_fraction": gc_target_fraction(genome, reads, args),
            "gc_penalty": case.gc_penalty,
            "mi_reward_scale": case.mi_reward_scale,
            "mi_score_mode": args.mi_score_mode,
            "mi_min": layout.metadata.get("mi_min", ""),
            "mi_max": layout.metadata.get("mi_max", ""),
            **score_summary,
            "overlap_gc_method": args.overlap_gc_method,
            "backend": layout.metadata.get("annealer_backend"),
            "seed": case.seed,
            "energy": layout.objective_value,
            "true_path_available": true_energy is not None,
            "true_path_energy": true_energy if true_energy is not None else "",
            "energy_gap": energy_gap if energy_gap is not None else "",
            "ground_hit": energy_gap is not None and abs(energy_gap) < 1e-6,
            "valid_edge_path": layout.metadata.get("valid_edge_path"),
            "single_path_layout": layout.metadata.get("single_path_layout"),
            "selected_edge_count": layout.metadata.get("selected_edge_count"),
            "edge_count_violation": layout.metadata.get("edge_count_violation"),
            "in_degree_violations": layout.metadata.get("in_degree_violations"),
            "out_degree_violations": layout.metadata.get("out_degree_violations"),
            "hard_violation": hard_violation,
            "selected_adjacent_correct": selected_counts["adjacent_correct"],
            "selected_jump_correct": selected_counts["jump_correct"],
            "selected_wrong": selected_counts["wrong"],
            **recoverability,
            "anneal_sec": layout.metadata.get("anneal_sec"),
            "total_sec": elapsed,
        }
    except (RuntimeError, ValueError) as exc:
        elapsed = perf_counter() - start
        return error_row(index, case, args, len(reads), len(edges), str(exc), elapsed)


def build_hamiltonian(
    genome: str,
    reads: list[Read],
    args: argparse.Namespace,
    degree_penalty: float,
    length_penalty: float,
    gc_penalty: float,
    mi_reward_scale: float,
) -> PDFAssemblyQUBOHamiltonian:
    return PDFAssemblyQUBOHamiltonian(PDFAssemblyHamiltonianConfig(
        degree_penalty=degree_penalty,
        length_target=length_target(genome, reads, args),
        length_penalty=length_penalty,
        gc_target_fraction=gc_target_fraction(genome, reads, args),
        gc_penalty=gc_penalty,
        overlap_gc_method=args.overlap_gc_method,
        mi_reward_scale=mi_reward_scale,
        mi_score_mode=args.mi_score_mode,
    ))


def build_annealer(args: argparse.Namespace, case: SweepCase):
    if args.backend == "openjij-sqa":
        return OpenJijSimulatedQuantumAnnealer(OpenJijSQAConfig(
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            seed=case.seed,
            beta=args.beta,
            trotter=args.trotter,
        ))
    return BinarySimulatedAnnealer(BinaryAnnealingConfig(
        initial_temperature=10.0,
        final_temperature=0.01,
        cooling_rate=0.95,
        sweeps_per_temperature=max(1, args.num_sweeps // 100),
        seed=case.seed,
        random_restarts=max(1, args.num_reads),
        start_from_valid_permutation=False,
    ))


def true_path_energy(model, true_order: list[str]) -> float | None:
    try:
        return model.energy(qubo_sample_for_order(model, true_order))
    except ValueError:
        return None


def length_target(genome: str, reads: list[Read], args: argparse.Namespace) -> float:
    if args.length_target is not None:
        return float(args.length_target)
    if args.target_mode == "genome":
        return float(len(genome))
    start, end = selected_span(reads)
    return float(end - start)


def gc_target_fraction(genome: str, reads: list[Read], args: argparse.Namespace) -> float:
    if args.gc_target_fraction is not None:
        return float(args.gc_target_fraction)
    if args.target_mode == "genome":
        return gc_fraction(genome)
    start, end = selected_span(reads)
    return gc_fraction(genome[start:end])


def selected_span(reads: list[Read]) -> tuple[int, int]:
    starts = [read.true_start for read in reads if read.true_start >= 0]
    ends = [read.true_end for read in reads if read.true_end >= 0]
    if not starts or not ends:
        return 0, 0
    return min(starts), max(ends)


def read_counts(args: argparse.Namespace) -> list[int]:
    return parse_int_list(args.read_counts) if args.read_counts else [args.max_reads]


def candidate_score_summary(
    reads: list[Read],
    edges: list[OverlapEdge],
    score_mode: str,
) -> dict[str, object]:
    rank = {
        read.rid: index
        for index, read in enumerate(sorted(reads, key=lambda item: item.true_start))
    }
    scores: dict[str, list[float]] = {
        "adjacent": [],
        "jump": [],
        "wrong": [],
    }
    scorer = OverlapRewardScorer()
    for edge in edges:
        left_rank = rank.get(edge.left_id)
        right_rank = rank.get(edge.right_id)
        if left_rank is None or right_rank is None or right_rank <= left_rank:
            category = "wrong"
        elif right_rank == left_rank + 1:
            category = "adjacent"
        else:
            category = "jump"
        scores[category].append(scorer.score(edge, score_mode))

    def mean_or_blank(values: list[float]) -> float | str:
        return sum(values) / len(values) if values else ""

    return {
        "candidate_adjacent_edges": len(scores["adjacent"]),
        "candidate_jump_edges": len(scores["jump"]),
        "candidate_wrong_edges": len(scores["wrong"]),
        "adjacent_score_mean": mean_or_blank(scores["adjacent"]),
        "jump_score_mean": mean_or_blank(scores["jump"]),
        "wrong_score_mean": mean_or_blank(scores["wrong"]),
    }


def error_row(
    index: int,
    case: SweepCase,
    args: argparse.Namespace,
    reads_used: int,
    candidate_edges: int,
    error: str,
    elapsed: float,
) -> dict[str, object]:
    row = {field: "" for field in ROW_FIELDS}
    row.update({
        "case": index,
        "status": f"error:{error}",
        "reads_used": reads_used,
        "edge_limit": case.edge_limit if case.edge_limit is not None else "all",
        "candidate_edges": candidate_edges,
        "degree_penalty": case.degree_penalty,
        "length_penalty": case.length_penalty,
        "gc_penalty": case.gc_penalty,
        "mi_reward_scale": case.mi_reward_scale,
        "mi_score_mode": args.mi_score_mode,
        "overlap_gc_method": args.overlap_gc_method,
        "backend": args.backend,
        "seed": case.seed,
        "total_sec": elapsed,
    })
    return row


def print_summary(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[int, str, float, float, float, float, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["status"] != "ok":
            continue
        groups[(
            int(row["reads_used"]),
            str(row["edge_limit"]),
            float(row["degree_penalty"]),
            float(row["length_penalty"]),
            float(row["gc_penalty"]),
            float(row["mi_reward_scale"]),
            str(row["mi_score_mode"]),
        )].append(row)

    summaries = []
    for (
        reads_used,
        edge_limit,
        degree_penalty,
        length_penalty,
        gc_penalty,
        mi_reward_scale,
        mi_score_mode,
    ), group in groups.items():
        summaries.append({
            "reads_used": reads_used,
            "edge_limit": edge_limit,
            "degree_penalty": degree_penalty,
            "length_penalty": length_penalty,
            "gc_penalty": gc_penalty,
            "mi_reward_scale": mi_reward_scale,
            "mi_score_mode": mi_score_mode,
            "cases": len(group),
            "path_hits": sum(row["valid_edge_path"] is True for row in group),
            "single_path_hits": sum(row["single_path_layout"] is True for row in group),
            "ground_hits": sum(row["ground_hit"] is True for row in group),
            "recoverable_hits": sum(row["full_dna_recoverable"] is True for row in group),
            "avg_hard_violation": sum(int(row["hard_violation"]) for row in group) / len(group),
            "avg_adjacent": sum(int(row["selected_adjacent_correct"]) for row in group) / len(group),
            "avg_jump": sum(int(row["selected_jump_correct"]) for row in group) / len(group),
            "avg_wrong": sum(int(row["selected_wrong"]) for row in group) / len(group),
            "avg_missing_reads": sum(int(row["main_path_missing_reads"]) for row in group) / len(group),
            "avg_prunable_edges": sum(int(row["prunable_extra_edges"] or 0) for row in group) / len(group),
            "avg_sec": sum(float(row["total_sec"]) for row in group) / len(group),
        })

    summaries.sort(key=lambda item: (
        item["reads_used"],
        item["edge_limit"],
        -item["ground_hits"],
        -item["single_path_hits"],
        -item["path_hits"],
        item["avg_hard_violation"],
        -item["avg_adjacent"],
        item["avg_wrong"],
        item["avg_sec"],
    ))
    print("summary,reads_used,edge_limit,degree_penalty,length_penalty,gc_penalty,"
          "mi_reward_scale,mi_score_mode,cases,path_hits,single_path_hits,ground_hits,"
          "recoverable_hits,avg_hard_violation,avg_adjacent,avg_jump,avg_wrong,"
          "avg_missing_reads,avg_prunable_edges,avg_sec")
    for item in summaries:
        print(
            "summary,"
            f"{item['reads_used']},"
            f"{item['edge_limit']},"
            f"{item['degree_penalty']:.6g},"
            f"{item['length_penalty']:.6g},"
            f"{item['gc_penalty']:.6g},"
            f"{item['mi_reward_scale']:.6g},"
            f"{item['mi_score_mode']},"
            f"{item['cases']},"
            f"{item['path_hits']},"
            f"{item['single_path_hits']},"
            f"{item['ground_hits']},"
            f"{item['recoverable_hits']},"
            f"{item['avg_hard_violation']:.3f},"
            f"{item['avg_adjacent']:.3f},"
            f"{item['avg_jump']:.3f},"
            f"{item['avg_wrong']:.3f},"
            f"{item['avg_missing_reads']:.3f},"
            f"{item['avg_prunable_edges']:.3f},"
            f"{item['avg_sec']:.3f}"
        )


def format_row(row: dict[str, object]) -> str:
    return ",".join(format_cell(row[field]) for field in ROW_FIELDS)


def format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
