"""
Compare the old edge-path DAG Hamiltonian against PDF B/C/D ablations.

Each genome length is simulated independently and all reads from that
simulation are used. This avoids mixing the comparison with prefix truncation.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from demo_edge_path_parameter_sweep import parse_float_list, parse_int_list, selected_edge_truth_counts
from demo_edge_path_vs_cycle_sweep import evaluate_recoverability
from demo_pdf_assembly_parameter_sweep import build_overlap_graph
from olc_pipeline.layout_solver import (
    EdgePathDAGHamiltonianConfig,
    EdgePathDAGQUBOHamiltonian,
    OpenJijSQAConfig,
    OpenJijSimulatedQuantumAnnealer,
    PDFAssemblyHamiltonianConfig,
    PDFAssemblyQUBOHamiltonian,
    QUBOLayoutSolver,
    qubo_sample_for_order,
)
from olc_pipeline.overlap_features import gc_fraction


@dataclass(frozen=True)
class ModelCase:
    label: str
    kind: str
    length_penalty: float = 0.0
    gc_penalty: float = 0.0
    mi_reward_scale: float = 0.0


ROW_FIELDS = [
    "case",
    "status",
    "model",
    "reads_used",
    "genome_len",
    "candidate_edges",
    "qubo_variables",
    "quadratic_terms",
    "seed",
    "energy",
    "true_path_available",
    "true_path_energy",
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
    "old_count_penalty",
    "old_degree_penalty",
    "old_reward_scale",
    "pdf_a_penalty",
    "pdf_b_length_penalty",
    "pdf_c_gc_penalty",
    "pdf_d_mi_reward_scale",
    "mi_min",
    "mi_max",
    "anneal_sec",
    "total_sec",
]


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        handle.flush()

        case_index = 0
        for genome_len in parse_int_list(args.genome_lens):
            graph_args = graph_namespace(args, genome_len)
            genome, reads, edges = build_overlap_graph(graph_args)
            print(
                "dataset,"
                f"genome_len={genome_len},"
                f"reads={len(reads)},"
                f"edges={len(edges)},"
                f"gc={gc_fraction(genome):.4f}",
                flush=True,
            )
            for model_case in model_cases(args):
                for seed in parse_int_list(args.annealer_seeds):
                    case_index += 1
                    row = run_case(case_index, model_case, genome, reads, edges, seed, args)
                    rows.append(row)
                    writer.writerow(row)
                    handle.flush()
                    print(format_row(row), flush=True)

    print_summary(rows)
    print(f"csv,{output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare old N-1 Hamiltonian with PDF B/C/D ablations.")
    parser.add_argument("--genome-lens", default="20000,22500,25000")
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
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--trotter", type=int, default=32)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--annealer-seeds", default="40,41")
    parser.add_argument("--old-count-penalty", type=float, default=100.0)
    parser.add_argument("--old-degree-penalty", type=float, default=110.0)
    parser.add_argument("--old-reward-scale", type=float, default=20.0)
    parser.add_argument("--old-score-mode", default="overlap_len")
    parser.add_argument("--pdf-a-penalty", type=float, default=100.0)
    parser.add_argument("--pdf-b-length-penalty", type=float, default=1e-6)
    parser.add_argument("--pdf-c-gc-penalty", type=float, default=1e-4)
    parser.add_argument("--pdf-d-mi-reward-scale", type=float, default=0.01)
    parser.add_argument("--mi-score-mode", default="mi")
    parser.add_argument("--overlap-gc-method", choices=["average", "left", "right"], default="average")
    parser.add_argument("--cases", default="old,A,AB,AC,AD,ABC,ABD,ABCD")
    parser.add_argument("--output", default="debug/qubo/pdf_vs_edge_path_ablation.csv")
    return parser.parse_args()


def graph_namespace(args: argparse.Namespace, genome_len: int) -> SimpleNamespace:
    return SimpleNamespace(
        genome_len=genome_len,
        read_len=args.read_len,
        step=args.step,
        mismatch_rate=args.mismatch_rate,
        ins_rate=args.ins_rate,
        del_rate=args.del_rate,
        gc_fraction=args.gc_fraction,
        simulation_seed=args.simulation_seed,
        min_overlap=args.min_overlap,
        max_error_rate_hint=args.max_error_rate_hint,
        overhang_tolerance=args.overhang_tolerance,
        refiner_max_error_rate=args.refiner_max_error_rate,
        refiner_margin=args.refiner_margin,
    )


def model_cases(args: argparse.Namespace) -> list[ModelCase]:
    requested = {item.strip() for item in args.cases.split(",") if item.strip()}
    all_cases = {
        "old": ModelCase("old_edge_path_dag", "old"),
        "A": ModelCase("pdf_A_only", "pdf"),
        "AB": ModelCase("pdf_A_B", "pdf", length_penalty=args.pdf_b_length_penalty),
        "AC": ModelCase("pdf_A_C", "pdf", gc_penalty=args.pdf_c_gc_penalty),
        "AD": ModelCase("pdf_A_D", "pdf", mi_reward_scale=args.pdf_d_mi_reward_scale),
        "ABC": ModelCase(
            "pdf_A_B_C",
            "pdf",
            length_penalty=args.pdf_b_length_penalty,
            gc_penalty=args.pdf_c_gc_penalty,
        ),
        "ABD": ModelCase(
            "pdf_A_B_D",
            "pdf",
            length_penalty=args.pdf_b_length_penalty,
            mi_reward_scale=args.pdf_d_mi_reward_scale,
        ),
        "ABCD": ModelCase(
            "pdf_A_B_C_D",
            "pdf",
            length_penalty=args.pdf_b_length_penalty,
            gc_penalty=args.pdf_c_gc_penalty,
            mi_reward_scale=args.pdf_d_mi_reward_scale,
        ),
    }
    return [case for key, case in all_cases.items() if key in requested]


def run_case(
    case_index: int,
    model_case: ModelCase,
    genome: str,
    reads,
    edges,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    start = perf_counter()
    try:
        hamiltonian = build_hamiltonian(model_case, genome, args)
        solver = QUBOLayoutSolver(
            hamiltonian=hamiltonian,
            annealer=OpenJijSimulatedQuantumAnnealer(OpenJijSQAConfig(
                num_reads=args.num_reads,
                num_sweeps=args.num_sweeps,
                seed=seed,
                beta=args.beta,
                trotter=args.trotter,
            )),
        )
        layout = solver.solve(reads, edges, weight_mode=args.old_score_mode)
        elapsed = perf_counter() - start
        model = solver.last_model
        assert model is not None
        true_order = [read.rid for read in sorted(reads, key=lambda item: item.true_start)]
        true_energy = true_path_energy(model, true_order)
        energy_gap = None if true_energy is None else layout.objective_value - true_energy
        selected_edges = layout.metadata.get("selected_edges", [])
        selected_counts = selected_edge_truth_counts(reads, selected_edges)
        recoverability = evaluate_recoverability(reads, selected_edges)
        target_edges = max(0, len(reads) - 1)
        adjacent_recall = (
            selected_counts["adjacent_correct"] / target_edges
            if target_edges
            else 1.0
        )
        return {
            "case": case_index,
            "status": "ok",
            "model": model_case.label,
            "reads_used": len(reads),
            "genome_len": len(genome),
            "candidate_edges": len(edges),
            "qubo_variables": model.num_variables,
            "quadratic_terms": len(model.quadratic),
            "seed": seed,
            "energy": layout.objective_value,
            "true_path_available": true_energy is not None,
            "true_path_energy": true_energy if true_energy is not None else "",
            "energy_gap": energy_gap if energy_gap is not None else "",
            "ground_hit": energy_gap is not None and abs(energy_gap) < 1e-6,
            "valid_edge_path": layout.metadata.get("valid_edge_path"),
            "selected_edge_count": layout.metadata.get("selected_edge_count"),
            "edge_count_violation": layout.metadata.get("edge_count_violation"),
            "in_degree_violations": layout.metadata.get("in_degree_violations"),
            "out_degree_violations": layout.metadata.get("out_degree_violations"),
            "selected_adjacent_correct": selected_counts["adjacent_correct"],
            "selected_jump_correct": selected_counts["jump_correct"],
            "selected_wrong": selected_counts["wrong"],
            "adjacent_recall": adjacent_recall,
            **recoverability,
            "old_count_penalty": args.old_count_penalty if model_case.kind == "old" else "",
            "old_degree_penalty": args.old_degree_penalty if model_case.kind == "old" else "",
            "old_reward_scale": args.old_reward_scale if model_case.kind == "old" else "",
            "pdf_a_penalty": args.pdf_a_penalty if model_case.kind == "pdf" else "",
            "pdf_b_length_penalty": model_case.length_penalty if model_case.kind == "pdf" else "",
            "pdf_c_gc_penalty": model_case.gc_penalty if model_case.kind == "pdf" else "",
            "pdf_d_mi_reward_scale": model_case.mi_reward_scale if model_case.kind == "pdf" else "",
            "mi_min": layout.metadata.get("mi_min", ""),
            "mi_max": layout.metadata.get("mi_max", ""),
            "anneal_sec": layout.metadata.get("anneal_sec"),
            "total_sec": elapsed,
        }
    except (RuntimeError, ValueError) as exc:
        elapsed = perf_counter() - start
        row = {field: "" for field in ROW_FIELDS}
        row.update({
            "case": case_index,
            "status": f"error:{exc}",
            "model": model_case.label,
            "reads_used": len(reads),
            "genome_len": len(genome),
            "candidate_edges": len(edges),
            "seed": seed,
            "total_sec": elapsed,
        })
        return row


def build_hamiltonian(model_case: ModelCase, genome: str, args: argparse.Namespace):
    if model_case.kind == "old":
        return EdgePathDAGQUBOHamiltonian(EdgePathDAGHamiltonianConfig(
            edge_count_penalty=args.old_count_penalty,
            degree_penalty=args.old_degree_penalty,
            edge_reward_scale=args.old_reward_scale,
            score_mode=args.old_score_mode,
            normalize_rewards=True,
            require_hamiltonian_path=True,
        ))
    return PDFAssemblyQUBOHamiltonian(PDFAssemblyHamiltonianConfig(
        degree_penalty=args.pdf_a_penalty,
        length_target=float(len(genome)),
        length_penalty=model_case.length_penalty,
        gc_target_fraction=gc_fraction(genome),
        gc_penalty=model_case.gc_penalty,
        overlap_gc_method=args.overlap_gc_method,
        mi_reward_scale=model_case.mi_reward_scale,
        mi_score_mode=args.mi_score_mode,
    ))


def true_path_energy(model, true_order: list[str]) -> float | None:
    try:
        return model.energy(qubo_sample_for_order(model, true_order))
    except ValueError:
        return None


def print_summary(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["status"] != "ok":
            continue
        groups[(int(row["reads_used"]), str(row["model"]))].append(row)

    print(
        "summary,reads_used,model,cases,valid_hits,ground_hits,recoverable_hits,"
        "avg_adjacent_recall,avg_adjacent,avg_jump,avg_wrong,avg_hard_violation,"
        "avg_main_path_nodes,avg_missing_reads,avg_prunable_edges,avg_sec"
    )
    for (reads_used, model), group in sorted(groups.items()):
        hard = [
            int(row["edge_count_violation"])
            + int(row["in_degree_violations"])
            + int(row["out_degree_violations"])
            for row in group
        ]
        print(
            "summary,"
            f"{reads_used},"
            f"{model},"
            f"{len(group)},"
            f"{sum(row['valid_edge_path'] is True for row in group)},"
            f"{sum(row['ground_hit'] is True for row in group)},"
            f"{sum(row['full_dna_recoverable'] is True for row in group)},"
            f"{sum(float(row['adjacent_recall']) for row in group) / len(group):.4f},"
            f"{sum(int(row['selected_adjacent_correct']) for row in group) / len(group):.3f},"
            f"{sum(int(row['selected_jump_correct']) for row in group) / len(group):.3f},"
            f"{sum(int(row['selected_wrong']) for row in group) / len(group):.3f},"
            f"{sum(hard) / len(hard):.3f},"
            f"{sum(int(row['main_path_nodes']) for row in group) / len(group):.3f},"
            f"{sum(int(row['main_path_missing_reads']) for row in group) / len(group):.3f},"
            f"{sum(int(row['prunable_extra_edges'] or 0) for row in group) / len(group):.3f},"
            f"{sum(float(row['total_sec']) for row in group) / len(group):.3f}"
        )


def format_row(row: dict[str, object]) -> str:
    return ",".join(format_cell(row[field]) for field in ROW_FIELDS)


def format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
