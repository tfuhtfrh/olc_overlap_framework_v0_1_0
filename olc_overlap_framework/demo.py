"""
demo.py
Version: 0.1.0

Small executable demo for the OLC overlap framework.
Requires command-line minimap2 to run the default candidate finder.
Parasail is required for semi-global DP overlap refinement.
"""

from __future__ import annotations

from pathlib import Path

from olc_pipeline.simulator import RandomReadSimulator, SimulationConfig
from olc_pipeline.candidate_finder import Minimap2CandidateFinder, Minimap2Config
from olc_pipeline.refiner import ParasailOverlapRefiner, RefinerConfig
from olc_pipeline.mi_scorer import AlignmentMIScorer
from olc_pipeline.pipeline import OverlapExperimentPipeline
from olc_pipeline.evaluator import LayoutEvaluator
from olc_pipeline.layout_solver import (
    BinaryAnnealingConfig,
    BinarySimulatedAnnealer,
    DWaveAnnealingConfig,
    DWaveSimulatedAnnealer,
    MissingEdgeHamiltonianConfig,
    MissingEdgeQUBOHamiltonian,
    PermutationLocalSearchPolisher,
    PermutationPolishConfig,
    QUBOLayoutSolver,
    WeightedOverlapHamiltonianConfig,
    WeightedOverlapQUBOHamiltonian,
)
from olc_pipeline.graph_viz import write_overlap_graph_dot


USE_QUBO_LAYOUT = True
# Keep this small for quick demo runs. Set to None to use every read and every
# edge whose endpoints are present in the selected read set.
QUBO_LAYOUT_MAX_READS = 100
QUBO_READ_ONCE_PENALTY = 500.0
QUBO_POSITION_ONCE_PENALTY = 500.0
QUBO_MISSING_EDGE_PENALTY = 50.0
QUBO_HAMILTONIAN = "weighted_overlap"  # "missing_edge" or "weighted_overlap"
QUBO_SCORE_MODE = "overlap_len"  # "overlap_len", "identity", "dp", "mi", "nmi", "mapq", "matches", "quality"
QUBO_EDGE_REWARD_SCALE = 20.0
QUBO_NORMALIZE_REWARDS = True
QUBO_DWAVE_NUM_READS = 1000
QUBO_DWAVE_NUM_SWEEPS = 1000
QUBO_SEED = 42
WRITE_QUBO_GRAPH_DOT = True
QUBO_GRAPH_DOT_PATH = Path("debug/qubo/overlap_graph.dot")
QUBO_POLISH_LAYOUT = False
QUBO_POLISH_MAX_PASSES = 50
QUBO_POLISH_USE_SEGMENT_MOVES = True
QUBO_POLISH_MAX_SEGMENT_LEN = 12


def fmt_float(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def fmt_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f} s"


def single_read_error_rate(config: SimulationConfig) -> float:
    """Expected edit events per reference base in one simulated read."""
    return config.ins_rate + config.del_rate + (1.0 - config.del_rate) * config.mismatch_rate


def pairwise_overlap_error_rate(config: SimulationConfig) -> float:
    """Approximate expected pairwise edit rate between two independently errored reads."""
    present = 1.0 - config.del_rate
    same_observed_base = (1.0 - config.mismatch_rate) ** 2 + (config.mismatch_rate ** 2) / 3.0
    substitution_disagreement = present * present * (1.0 - same_observed_base)
    one_side_deleted = 2.0 * config.del_rate * present
    one_side_inserted = 2.0 * config.ins_rate
    return substitution_disagreement + one_side_deleted + one_side_inserted


def main() -> None:
    simulator = RandomReadSimulator()
    sim_config = SimulationConfig(
        genome_len=20_000,
        read_len=3_000,
        step=500,
        mismatch_rate=0.00,
        ins_rate=0.00,
        del_rate=0.00,
        seed=42,
        shuffle_reads=True,
    )
    genome, reads = simulator.simulate(sim_config)

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

    print_section("Simulation")
    print_kv("Genome length", f"{len(genome):,} bp")
    print_kv("Reads", len(reads))
    print_kv("Read length / step", f"{sim_config.read_len:,} / {sim_config.step:,} bp")
    print_kv("Single-read error", fmt_pct(single_read_error_rate(sim_config)))
    print_kv("Pairwise expected error", fmt_pct(pairwise_overlap_error_rate(sim_config)))
    print_kv(
        "Error model",
        (
            f"mismatch={fmt_pct(sim_config.mismatch_rate)}, "
            f"ins={fmt_pct(sim_config.ins_rate)}, "
            f"del={fmt_pct(sim_config.del_rate)}"
        ),
    )

    print_section("Overlap Graph")
    print_kv("Raw candidates", len(result.candidates))
    print_kv("Accepted edges", len(result.edges))
    print_edge_report("Minimap2 candidates", result.candidate_report)
    print_edge_report("After DP refinement", result.edge_report)

    print_section("DP Refinement")
    print_refinement_report(result.refinement_report)

    print_section("Minimap2")
    print_kv("Command", " ".join(finder.last_command))
    print_filter_counts(dict(finder.last_filter_counts))
    if finder.last_debug_paf_path is not None:
        print_kv("Debug PAF", finder.last_debug_paf_path)
    print_minimap_stderr_summary(finder.last_stderr)

    print_section("Timings")
    for name, seconds in result.timings.items():
        print_kv(name, f"{seconds:.4f} s")

    print_section("Top Edges By MI")
    for edge in sorted(result.edges, key=lambda e: e.weight_mi or 0.0, reverse=True)[:10]:
        print(
            f"{edge.left_id} -> {edge.right_id} "
            f"L={edge.overlap_len} shift={edge.shift} "
            f"err={edge.error_rate:.3f} nmi={edge.nmi:.3f} "
            f"w_dp={edge.weight_dp:.1f} w_mi={edge.weight_mi:.1f}"
        )

    if USE_QUBO_LAYOUT:
        run_qubo_layout_demo(reads, result.edges)


def print_section(title: str) -> None:
    print(f"\n== {title} ==")


def print_kv(label: str, value: object) -> None:
    print(f"{label:<28} {value}")


def print_edge_report(title: str, report) -> None:
    print(f"\n{title}:")
    print_kv("total", report.total_edges)
    print_kv("adjacent correct", report.adjacent_correct)
    print_kv("jump correct", report.jump_correct)
    print_kv("wrong", report.wrong_edges)
    print_kv("missing adjacent", report.missing_adjacent_edges)
    print_kv("precision", fmt_pct(report.edge_precision))
    print_kv("adjacent recall", fmt_pct(report.adjacent_recall))
    print_kv("shift error mean/median/max", (
        f"{fmt_float(report.mean_shift_error, 2)} / "
        f"{fmt_float(report.median_shift_error, 2)} / "
        f"{report.max_shift_error if report.max_shift_error is not None else 'n/a'} bp"
    ))
    if report.notes:
        print_kv("notes", "; ".join(report.notes))


def print_refinement_report(report) -> None:
    print_kv("mean / median / max error", (
        f"{fmt_pct(report.mean_error_rate)} / "
        f"{fmt_pct(report.median_error_rate)} / "
        f"{fmt_pct(report.max_error_rate)}"
    ))
    print_kv("aggregate error", fmt_pct(report.aggregate_error_rate))
    print_kv("mean identity", fmt_pct(report.mean_identity))
    print_kv("overlap len err mean/med/max", (
        f"{fmt_float(report.mean_overlap_len_error, 2)} / "
        f"{fmt_float(report.median_overlap_len_error, 2)} / "
        f"{report.max_overlap_len_error if report.max_overlap_len_error is not None else 'n/a'} bp"
    ))
    print_kv("edit distance", report.total_edit_distance)
    print_kv("mismatch / ins / del", (
        f"{report.total_mismatches} / {report.total_insertions} / {report.total_deletions}"
    ))
    print_kv("gap truth precision", fmt_pct(report.gap_truth_precision))
    print_kv("truth-supported gaps", (
        f"{report.truth_supported_gap_columns} / {report.gap_columns}"
    ))
    print_kv("unsupported gaps", report.unsupported_gap_columns)
    print_kv("non-gap coord error", fmt_pct(report.non_gap_coord_error_rate))


def print_filter_counts(counts: dict[str, int]) -> None:
    if not counts:
        print_kv("PAF/filter counts", "{}")
        return
    ordered_keys = [
        "raw_records",
        "accepted",
        "skipped_min_overlap",
        "skipped_error_hint",
        "skipped_geometry",
        "skipped_reverse_strand",
        "skipped_mapq",
        "skipped_self",
    ]
    for key in ordered_keys:
        if key in counts:
            print_kv(key.replace("_", " "), counts[key])
    extra = {k: v for k, v in counts.items() if k not in ordered_keys}
    if extra:
        print_kv("other counts", extra)


def print_minimap_stderr_summary(stderr: str) -> None:
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    version = next((line for line in lines if "Version:" in line), None)
    runtime = next((line for line in lines if "Real time:" in line), None)
    if version:
        print_kv("Version", version.split("Version:", 1)[1].strip())
    if runtime:
        print_kv("Runtime", runtime)


def run_qubo_layout_demo(reads, edges) -> None:
    sorted_reads = sorted(reads, key=lambda read: read.true_start)
    if QUBO_LAYOUT_MAX_READS is None:
        layout_reads = sorted_reads
    else:
        layout_reads = sorted_reads[:QUBO_LAYOUT_MAX_READS]
    layout_read_ids = {read.rid for read in layout_reads}
    layout_edges = [
        edge
        for edge in edges
        if edge.left_id in layout_read_ids and edge.right_id in layout_read_ids
    ]

    print_section("QUBO Layout")
    print_kv("Reads used", f"{len(layout_reads)} / {len(reads)}")
    print_kv("Edges used", f"{len(layout_edges)} / {len(edges)}")
    print_qubo_config()
    if WRITE_QUBO_GRAPH_DOT:
        dot_path = write_overlap_graph_dot(
            layout_reads,
            layout_edges,
            QUBO_GRAPH_DOT_PATH,
            score_mode=QUBO_SCORE_MODE,
            normalize_rewards=QUBO_NORMALIZE_REWARDS,
        )
        print_kv("Graph DOT", dot_path)

    hamiltonian = build_qubo_hamiltonian()
    solver = QUBOLayoutSolver(
        hamiltonian=hamiltonian,
        annealer=DWaveSimulatedAnnealer(DWaveAnnealingConfig(
            num_reads=QUBO_DWAVE_NUM_READS,
            num_sweeps=QUBO_DWAVE_NUM_SWEEPS,
            seed=QUBO_SEED,
        )),
        polisher=build_qubo_polisher(),
    )

    try:
        layout = solver.solve(layout_reads, layout_edges, weight_mode=QUBO_SCORE_MODE)
    except RuntimeError as exc:
        print_kv("D-Wave backend", f"unavailable ({exc})")
        print_kv("Fallback", "built-in binary simulated annealer")
        solver = QUBOLayoutSolver(
            hamiltonian=hamiltonian,
            annealer=BinarySimulatedAnnealer(BinaryAnnealingConfig(
                initial_temperature=10.0,
                final_temperature=0.05,
                cooling_rate=0.90,
                sweeps_per_temperature=2,
                seed=QUBO_SEED,
                random_restarts=2,
                swap_move_probability=0.8,
            )),
            polisher=build_qubo_polisher(),
        )
        layout = solver.solve(layout_reads, layout_edges, weight_mode=QUBO_SCORE_MODE)

    report = LayoutEvaluator().evaluate_order(layout_reads, layout.order)
    print_kv("Backend", layout.metadata.get("annealer_backend", "unknown"))
    print_kv("Energy", fmt_float(layout.objective_value, 3))
    if solver.last_model is not None:
        print_kv("True order energy", fmt_float(true_order_energy(layout_reads, solver.last_model), 3))
    print_kv("QUBO build time", fmt_seconds(layout.metadata.get("build_sec")))
    print_kv("SA time", fmt_seconds(layout.metadata.get("anneal_sec")))
    print_kv("Polish time", fmt_seconds(layout.metadata.get("polish_sec")))
    print_kv("Polish improvements", layout.metadata.get("polish_improvements"))
    print_kv("QUBO total time", fmt_seconds(layout.metadata.get("total_sec")))
    print_kv("Valid binary layout", layout.metadata.get("valid_binary_layout"))
    print_kv("Adjacent correct", report.adjacent_correct_in_layout)
    print_kv("Wrong adjacencies", report.wrong_adjacencies_in_layout)
    print_kv("Inversion count", report.inversion_count)
    print_kv("Order", " -> ".join(layout.order))


def print_qubo_config() -> None:
    print_kv("Hamiltonian", QUBO_HAMILTONIAN)
    print_kv("Score mode", QUBO_SCORE_MODE)
    print_kv("Read once penalty", QUBO_READ_ONCE_PENALTY)
    print_kv("Position once penalty", QUBO_POSITION_ONCE_PENALTY)
    print_kv("Missing edge penalty", QUBO_MISSING_EDGE_PENALTY)
    print_kv("Edge reward scale", QUBO_EDGE_REWARD_SCALE)
    print_kv("Normalize rewards", QUBO_NORMALIZE_REWARDS)
    print_kv("D-Wave reads / sweeps", f"{QUBO_DWAVE_NUM_READS} / {QUBO_DWAVE_NUM_SWEEPS}")
    print_kv("Missing-edge effect", f"+{QUBO_MISSING_EDGE_PENALTY:.3f} per adjacent pair not in E")
    print_kv("Edge reward effect", f"-{QUBO_EDGE_REWARD_SCALE:.3f} * normalized_reward per adjacent edge")


def true_order_energy(reads, model) -> float:
    read_index_by_id = {read_id: idx for idx, read_id in enumerate(model.read_ids)}
    sample = [0] * model.num_variables
    for position_index, read in enumerate(sorted(reads, key=lambda item: item.true_start)):
        read_index = read_index_by_id[read.rid]
        sample[model.variable_index(read_index, position_index)] = 1
    return model.energy(sample)


def build_qubo_hamiltonian():
    if QUBO_HAMILTONIAN == "missing_edge":
        return MissingEdgeQUBOHamiltonian(MissingEdgeHamiltonianConfig(
            read_once_penalty=QUBO_READ_ONCE_PENALTY,
            position_once_penalty=QUBO_POSITION_ONCE_PENALTY,
            missing_edge_penalty=QUBO_MISSING_EDGE_PENALTY,
        ))
    if QUBO_HAMILTONIAN == "weighted_overlap":
        return WeightedOverlapQUBOHamiltonian(WeightedOverlapHamiltonianConfig(
            read_once_penalty=QUBO_READ_ONCE_PENALTY,
            position_once_penalty=QUBO_POSITION_ONCE_PENALTY,
            missing_edge_penalty=QUBO_MISSING_EDGE_PENALTY,
            edge_reward_scale=QUBO_EDGE_REWARD_SCALE,
            score_mode=QUBO_SCORE_MODE,
            normalize_rewards=QUBO_NORMALIZE_REWARDS,
        ))
    raise ValueError(f"Unknown QUBO_HAMILTONIAN: {QUBO_HAMILTONIAN!r}")


def build_qubo_polisher():
    if not QUBO_POLISH_LAYOUT:
        return None
    return PermutationLocalSearchPolisher(PermutationPolishConfig(
        max_passes=QUBO_POLISH_MAX_PASSES,
        use_swap_moves=True,
        use_insert_moves=True,
        use_segment_insert_moves=QUBO_POLISH_USE_SEGMENT_MOVES,
        max_segment_len=QUBO_POLISH_MAX_SEGMENT_LEN,
    ))


if __name__ == "__main__":
    main()
