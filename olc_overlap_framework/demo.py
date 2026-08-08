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
    DWaveQPUAnnealer,
    DWaveQPUConfig,
    DWaveSimulatedAnnealer,
    EdgePathDAGHamiltonianConfig,
    EdgePathDAGQUBOHamiltonian,
    EdgeCycleCoverDAGHamiltonianConfig,
    EdgeCycleCoverDAGQUBOHamiltonian,
    EdgePathCoverDAGHamiltonianConfig,
    EdgePathCoverDAGQUBOHamiltonian,
    MissingEdgeHamiltonianConfig,
    MissingEdgeQUBOHamiltonian,
    OpenJijSQAConfig,
    OpenJijSimulatedQuantumAnnealer,
    PDFAssemblyHamiltonianConfig,
    PDFAssemblyQUBOHamiltonian,
    PermutationLocalSearchPolisher,
    PermutationPolishConfig,
    QUBOLayoutSolver,
    WeightedOverlapHamiltonianConfig,
    WeightedOverlapQUBOHamiltonian,
    qubo_sample_for_order,
)
from olc_pipeline.graph_viz import write_overlap_graph_dot
from olc_pipeline.overlap_features import gc_fraction


USE_QUBO_LAYOUT = True
# Keep this small for quick demo runs. Set to None to use every read and every
# edge whose endpoints are present in the selected read set.
QUBO_LAYOUT_MAX_READS = None
QUBO_READ_ONCE_PENALTY = 100.0
QUBO_POSITION_ONCE_PENALTY = 100.0
QUBO_MISSING_EDGE_PENALTY = 120.0
QUBO_HAMILTONIAN = "edge_path_cover_dag"  # "missing_edge", "weighted_overlap", "edge_path_dag", "edge_path_cover_dag", "edge_cycle_cover_dag", or "pdf_assembly"
QUBO_SCORE_MODE = "overlap_len_power"  # "overlap_len", "overlap_len_power", "overlap_len_power2", "overlap_len_power3", "identity", "dp", "mi", "nmi", "mapq", "matches", "quality"
QUBO_SCORE_GAMMA = 3.0
QUBO_EDGE_REWARD_SCALE = 40.0
QUBO_NORMALIZE_REWARDS = True
QUBO_EDGE_COUNT_PENALTY = 100.0
QUBO_EDGE_DEGREE_PENALTY = 120.0
QUBO_EDGE_ISOLATE_PENALTY = 160.0
QUBO_PATH_BREAK_PENALTY = 10.0
QUBO_MAX_PATH_COUNT = 2
QUBO_PATH_COUNT_CAP_PENALTY = 120.0
QUBO_ASSEMBLY_LENGTH_TARGET = None  # None uses the simulated genome length when available.
QUBO_ASSEMBLY_LENGTH_PENALTY = 0.0
QUBO_GC_TARGET_FRACTION = None  # None uses the simulated genome GC fraction when available.
QUBO_GC_PENALTY = 0.0
QUBO_OVERLAP_GC_METHOD = "average"  # "average", "left", or "right"
QUBO_MI_REWARD_SCALE = 0.0
QUBO_MI_SCORE_MODE = "mi"
QUBO_ANNEALER_BACKEND = "openjij_sqa"  # "dwave_sa", "openjij_sqa", or "dwave_qpu"
QUBO_DWAVE_NUM_READS = 1000
QUBO_DWAVE_NUM_SWEEPS = 1000
QUBO_OPENJIJ_NUM_READS = 60
QUBO_OPENJIJ_NUM_SWEEPS = 700
QUBO_OPENJIJ_BETA = None
QUBO_OPENJIJ_TROTTER = 32
QUBO_DWAVE_QPU_NUM_READS = 100
QUBO_DWAVE_QPU_CHAIN_STRENGTH = None
QUBO_DWAVE_QPU_ANNEALING_TIME = None
QUBO_SEED = 44
WRITE_QUBO_GRAPH_DOT = True
QUBO_GRAPH_DOT_PATH = Path("debug/qubo/overlap_graph.dot")
QUBO_POLISH_LAYOUT = False
QUBO_POLISH_MAX_PASSES = 50
QUBO_POLISH_USE_SEGMENT_MOVES = True
QUBO_POLISH_MAX_SEGMENT_LEN = 12
SIM_GC_FRACTION = 0.5

EDGE_PATH_HAMILTONIANS = {
    "edge_path_dag",
    "edge_path_cover_dag",
    "edge_cycle_cover_dag",
    "pdf_assembly",
}
PATH_COVER_HAMILTONIANS = {
    "edge_path_cover_dag",
}


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
        step=400,
        mismatch_rate=0.00,
        ins_rate=0.00,
        del_rate=0.00,
        gc_fraction=SIM_GC_FRACTION,
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
    print_kv("Genome GC", fmt_pct(gc_fraction(genome)))
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
        run_qubo_layout_demo(reads, result.edges, genome=genome)


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


def run_qubo_layout_demo(reads, edges, genome=None) -> None:
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
    print_qubo_config(genome)
    effective_score_mode = qubo_effective_score_mode()
    if WRITE_QUBO_GRAPH_DOT:
        dot_path = write_overlap_graph_dot(
            layout_reads,
            layout_edges,
            QUBO_GRAPH_DOT_PATH,
            score_mode=effective_score_mode,
            normalize_rewards=QUBO_NORMALIZE_REWARDS,
        )
        print_kv("Graph DOT", dot_path)

    hamiltonian = build_qubo_hamiltonian(genome=genome)
    solver = QUBOLayoutSolver(
        hamiltonian=hamiltonian,
        annealer=build_qubo_annealer(),
        polisher=build_qubo_polisher(),
    )

    try:
        layout = solver.solve(layout_reads, layout_edges, weight_mode=effective_score_mode)
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
        layout = solver.solve(layout_reads, layout_edges, weight_mode=effective_score_mode)

    report = LayoutEvaluator().evaluate_order(layout_reads, layout.order)
    selected_edge_report = selected_edge_truth_counts(
        layout_reads,
        layout.metadata.get("selected_edges", []),
    )
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
    if layout.metadata.get("hamiltonian") in EDGE_PATH_HAMILTONIANS:
        print_kv("QUBO variables", layout.metadata.get("variables"))
        if layout.metadata.get("hamiltonian") == "edge_path_dag":
            print_kv(
                "Selected edges",
                f"{layout.metadata.get('selected_edge_count')} / {layout.metadata.get('target_edge_count')}",
            )
            print_kv("Edge count violation", layout.metadata.get("edge_count_violation"))
        elif layout.metadata.get("hamiltonian") == "pdf_assembly":
            print_kv(
                "Selected edges",
                f"{layout.metadata.get('selected_edge_count')} / {layout.metadata.get('target_edge_count')}",
            )
            print_kv("Edge count violation", layout.metadata.get("edge_count_violation"))
        else:
            print_kv(
                "Selected edges",
                f"{layout.metadata.get('selected_edge_count')} / {len(layout_reads) - 1}",
            )
        print_kv("In-degree violations", layout.metadata.get("in_degree_violations"))
        print_kv("Out-degree violations", layout.metadata.get("out_degree_violations"))
        if layout.metadata.get("in_degree_conflicts"):
            print_kv("In-degree conflicts", layout.metadata.get("in_degree_conflicts"))
        if layout.metadata.get("out_degree_conflicts"):
            print_kv("Out-degree conflicts", layout.metadata.get("out_degree_conflicts"))
        print_kv("Selected graph acyclic", layout.metadata.get("selected_graph_acyclic"))
        print_kv("Single path layout", layout.metadata.get("single_path_layout"))
        if layout.metadata.get("hamiltonian") in PATH_COVER_HAMILTONIANS:
            print_kv("Valid path cover", layout.metadata.get("valid_path_cover"))
            print_kv("Path count", layout.metadata.get("path_count"))
            print_kv("Source / sink count", f"{layout.metadata.get('selected_source_count')} / {layout.metadata.get('selected_sink_count')}")
            print_kv("Source constraint violations", layout.metadata.get("source_constraint_violations"))
            print_kv("Sink constraint violations", layout.metadata.get("sink_constraint_violations"))
            print_kv("Isolated nodes", layout.metadata.get("isolated_node_count"))
        if layout.metadata.get("hamiltonian") == "pdf_assembly":
            print_kv("Length target", layout.metadata.get("length_target"))
            print_kv("GC target", fmt_float(layout.metadata.get("gc_target_fraction"), 4))
            print_kv("Total read len / GC", (
                f"{fmt_float(layout.metadata.get('total_read_len'), 1)} / "
                f"{fmt_float(layout.metadata.get('total_read_gc'), 1)}"
            ))
            print_kv("MI reward scale", layout.metadata.get("mi_reward_scale"))
        print_kv("Selected adjacent correct", selected_edge_report["adjacent_correct"])
        print_kv("Selected jump correct", selected_edge_report["jump_correct"])
        print_kv("Selected wrong edges", selected_edge_report["wrong"])
    else:
        print_kv("Read assignment violations", layout.metadata.get("read_assignment_violations"))
        print_kv("Position assignment violations", layout.metadata.get("position_assignment_violations"))
    order_label = "Adjacent correct" if layout.metadata.get("valid_binary_layout") else "Decoded order adjacent correct"
    wrong_label = "Wrong adjacencies" if layout.metadata.get("valid_binary_layout") else "Decoded order wrong adjacencies"
    print_kv(order_label, report.adjacent_correct_in_layout)
    print_kv(wrong_label, report.wrong_adjacencies_in_layout)
    print_kv("Inversion count", report.inversion_count)
    if layout.metadata.get("hamiltonian") not in EDGE_PATH_HAMILTONIANS or layout.metadata.get("valid_edge_path"):
        print_kv("Order", " -> ".join(layout.order))
    elif layout.metadata.get("valid_path_cover"):
        components = layout.metadata.get("path_components", [])
        formatted = " | ".join(" -> ".join(path) for path in components)
        print_kv("Path cover", formatted)
    else:
        print_kv("Order", "not available (selected edges do not form one path)")


def selected_edge_truth_counts(reads, selected_edges) -> dict[str, int]:
    rank = {
        read.rid: index
        for index, read in enumerate(sorted(reads, key=lambda item: item.true_start))
    }
    counts = {
        "adjacent_correct": 0,
        "jump_correct": 0,
        "wrong": 0,
    }
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


def print_qubo_config(genome=None) -> None:
    print_kv("Hamiltonian", QUBO_HAMILTONIAN)
    print_kv("Score mode", QUBO_SCORE_MODE)
    if QUBO_SCORE_MODE == "overlap_len_power":
        print_kv("Score gamma", QUBO_SCORE_GAMMA)
    if QUBO_HAMILTONIAN == "edge_path_dag":
        print_kv("Edge count penalty", QUBO_EDGE_COUNT_PENALTY)
        print_kv("Edge degree penalty", QUBO_EDGE_DEGREE_PENALTY)
    elif QUBO_HAMILTONIAN == "edge_path_cover_dag":
        print_kv("In/out degree penalty", QUBO_EDGE_DEGREE_PENALTY)
        print_kv("Isolate penalty", QUBO_EDGE_ISOLATE_PENALTY)
        print_kv("Two-path penalty", f"{QUBO_PATH_BREAK_PENALTY} (unused for center-square)")
        print_kv("Max path count", QUBO_MAX_PATH_COUNT)
        print_kv("Path cap coefficient", QUBO_PATH_COUNT_CAP_PENALTY)
    elif QUBO_HAMILTONIAN == "edge_cycle_cover_dag":
        print_kv("In/out degree penalty", QUBO_EDGE_DEGREE_PENALTY)
        print_kv("Void node", "__void__")
        print_kv("Cycle constraint", "all reads + void have exactly one input and output")
    elif QUBO_HAMILTONIAN == "pdf_assembly":
        print_kv("A path penalty", QUBO_EDGE_DEGREE_PENALTY)
        print_kv("A edge count target", "N - 1")
        print_kv("Length target", qubo_length_target(genome))
        print_kv("B length penalty", QUBO_ASSEMBLY_LENGTH_PENALTY)
        print_kv("GC target", fmt_float(qubo_gc_target_fraction(genome), 4))
        print_kv("C GC penalty", QUBO_GC_PENALTY)
        print_kv("Overlap GC method", QUBO_OVERLAP_GC_METHOD)
        print_kv("D MI reward scale", QUBO_MI_REWARD_SCALE)
        print_kv("MI score mode", QUBO_MI_SCORE_MODE)
    else:
        print_kv("Read once penalty", QUBO_READ_ONCE_PENALTY)
        print_kv("Position once penalty", QUBO_POSITION_ONCE_PENALTY)
        print_kv("Missing edge penalty", QUBO_MISSING_EDGE_PENALTY)
    print_kv("Edge reward scale", QUBO_EDGE_REWARD_SCALE)
    print_kv("Normalize rewards", QUBO_NORMALIZE_REWARDS)
    print_kv("Annealer backend", QUBO_ANNEALER_BACKEND)
    if QUBO_ANNEALER_BACKEND == "dwave_sa":
        print_kv("D-Wave reads / sweeps", f"{QUBO_DWAVE_NUM_READS} / {QUBO_DWAVE_NUM_SWEEPS}")
    elif QUBO_ANNEALER_BACKEND == "openjij_sqa":
        print_kv("OpenJij reads / sweeps", f"{QUBO_OPENJIJ_NUM_READS} / {QUBO_OPENJIJ_NUM_SWEEPS}")
        print_kv("OpenJij beta / trotter", f"{QUBO_OPENJIJ_BETA} / {QUBO_OPENJIJ_TROTTER}")
    elif QUBO_ANNEALER_BACKEND == "dwave_qpu":
        print_kv("D-Wave QPU reads", QUBO_DWAVE_QPU_NUM_READS)
        print_kv("D-Wave QPU chain strength", QUBO_DWAVE_QPU_CHAIN_STRENGTH)
        print_kv("D-Wave QPU annealing time", QUBO_DWAVE_QPU_ANNEALING_TIME)
    if QUBO_HAMILTONIAN not in EDGE_PATH_HAMILTONIANS:
        print_kv("Missing-edge effect", f"+{QUBO_MISSING_EDGE_PENALTY:.3f} per adjacent pair not in E")
    if QUBO_HAMILTONIAN in PATH_COVER_HAMILTONIANS:
        print_kv("Path count effect", (
            f"{QUBO_PATH_COUNT_CAP_PENALTY:.3f}*(m-1)^2"
        ))
    if QUBO_HAMILTONIAN != "pdf_assembly":
        print_kv("Edge reward effect", f"-{QUBO_EDGE_REWARD_SCALE:.3f} * normalized_reward per adjacent edge")
    if QUBO_HAMILTONIAN == "pdf_assembly":
        print_kv("MI reward effect", f"-{QUBO_MI_REWARD_SCALE:.3f} * raw_MI per selected edge")


def true_order_energy(reads, model) -> float | None:
    order = [
        read.rid
        for read in sorted(reads, key=lambda item: item.true_start)
    ]
    try:
        return model.energy(qubo_sample_for_order(model, order))
    except ValueError:
        return None


def qubo_effective_score_mode() -> str:
    if QUBO_SCORE_MODE == "overlap_len_power":
        return f"overlap_len_power:{QUBO_SCORE_GAMMA}"
    return QUBO_SCORE_MODE


def qubo_length_target(genome=None) -> float | None:
    if QUBO_ASSEMBLY_LENGTH_TARGET is not None:
        return float(QUBO_ASSEMBLY_LENGTH_TARGET)
    if genome is not None:
        return float(len(genome))
    return None


def qubo_gc_target_fraction(genome=None) -> float | None:
    if QUBO_GC_TARGET_FRACTION is not None:
        return float(QUBO_GC_TARGET_FRACTION)
    if genome:
        return gc_fraction(genome)
    return None


def build_qubo_hamiltonian(genome=None):
    effective_score_mode = qubo_effective_score_mode()
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
            score_mode=effective_score_mode,
            normalize_rewards=QUBO_NORMALIZE_REWARDS,
        ))
    if QUBO_HAMILTONIAN == "edge_path_dag":
        return EdgePathDAGQUBOHamiltonian(EdgePathDAGHamiltonianConfig(
            edge_count_penalty=QUBO_EDGE_COUNT_PENALTY,
            degree_penalty=QUBO_EDGE_DEGREE_PENALTY,
            edge_reward_scale=QUBO_EDGE_REWARD_SCALE,
            score_mode=effective_score_mode,
            normalize_rewards=QUBO_NORMALIZE_REWARDS,
            require_hamiltonian_path=True,
        ))
    if QUBO_HAMILTONIAN == "edge_path_cover_dag":
        return EdgePathCoverDAGQUBOHamiltonian(EdgePathCoverDAGHamiltonianConfig(
            degree_penalty=QUBO_EDGE_DEGREE_PENALTY,
            isolate_penalty=QUBO_EDGE_ISOLATE_PENALTY,
            path_break_penalty=QUBO_PATH_BREAK_PENALTY,
            max_path_count=QUBO_MAX_PATH_COUNT,
            path_count_cap_penalty=QUBO_PATH_COUNT_CAP_PENALTY,
            edge_reward_scale=QUBO_EDGE_REWARD_SCALE,
            score_mode=effective_score_mode,
            normalize_rewards=QUBO_NORMALIZE_REWARDS,
        ))
    if QUBO_HAMILTONIAN == "edge_cycle_cover_dag":
        return EdgeCycleCoverDAGQUBOHamiltonian(EdgeCycleCoverDAGHamiltonianConfig(
            degree_penalty=QUBO_EDGE_DEGREE_PENALTY,
            edge_reward_scale=QUBO_EDGE_REWARD_SCALE,
            score_mode=effective_score_mode,
            normalize_rewards=QUBO_NORMALIZE_REWARDS,
        ))
    if QUBO_HAMILTONIAN == "pdf_assembly":
        return PDFAssemblyQUBOHamiltonian(PDFAssemblyHamiltonianConfig(
            degree_penalty=QUBO_EDGE_DEGREE_PENALTY,
            length_target=qubo_length_target(genome),
            length_penalty=QUBO_ASSEMBLY_LENGTH_PENALTY,
            gc_target_fraction=qubo_gc_target_fraction(genome),
            gc_penalty=QUBO_GC_PENALTY,
            overlap_gc_method=QUBO_OVERLAP_GC_METHOD,
            mi_reward_scale=QUBO_MI_REWARD_SCALE,
            mi_score_mode=QUBO_MI_SCORE_MODE,
        ))
    raise ValueError(f"Unknown QUBO_HAMILTONIAN: {QUBO_HAMILTONIAN!r}")


def build_qubo_annealer():
    if QUBO_ANNEALER_BACKEND == "dwave_sa":
        return DWaveSimulatedAnnealer(DWaveAnnealingConfig(
            num_reads=QUBO_DWAVE_NUM_READS,
            num_sweeps=QUBO_DWAVE_NUM_SWEEPS,
            seed=QUBO_SEED,
        ))
    if QUBO_ANNEALER_BACKEND == "openjij_sqa":
        return OpenJijSimulatedQuantumAnnealer(OpenJijSQAConfig(
            num_reads=QUBO_OPENJIJ_NUM_READS,
            num_sweeps=QUBO_OPENJIJ_NUM_SWEEPS,
            seed=QUBO_SEED,
            beta=QUBO_OPENJIJ_BETA,
            trotter=QUBO_OPENJIJ_TROTTER,
        ))
    if QUBO_ANNEALER_BACKEND == "dwave_qpu":
        return DWaveQPUAnnealer(DWaveQPUConfig(
            num_reads=QUBO_DWAVE_QPU_NUM_READS,
            chain_strength=QUBO_DWAVE_QPU_CHAIN_STRENGTH,
            annealing_time=QUBO_DWAVE_QPU_ANNEALING_TIME,
        ))
    raise ValueError(f"Unknown QUBO_ANNEALER_BACKEND: {QUBO_ANNEALER_BACKEND!r}")


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
