"""Non-strict cut-cycle test for the existing edge-cycle DAG QUBO.

This demo deliberately uses the existing graph-only Hamilton-cycle witness to
choose a topological order.  It is therefore a feasibility/scale test for the
Hamiltonian, not an end-to-end assembly algorithm.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from time import perf_counter

from demo_phi174_common import DATA_DIR, PROJECT_DIR, run_reads_only_pipeline
from olc_pipeline.graph_builder import build_read_only_graph
from olc_pipeline.io_utils import read_fastq
from olc_pipeline.layout_solver import (
    AnnealingResult,
    DWaveAnnealingConfig,
    DWaveSimulatedAnnealer,
    EdgeCycleCoverDAGHamiltonianConfig,
    EdgeCycleCoverDAGQUBOModel,
    EdgeCycleCoverDAGQUBOHamiltonian,
    OpenJijSQAConfig,
    OpenJijSimulatedQuantumAnnealer,
    QUBOModel,
    QUBOLayoutSolver,
    qubo_sample_for_order,
)


# Experiment parameters are kept here for direct editing.
CUT_SEED = 20260904
ANNEALER_SEED = 20260904
MIN_OVERLAP = 80
DEGREE_PENALTY = 100.0
EDGE_REWARD_SCALE = 20.0
EDGE_SELECTION_PENALTY = 0.0
SCORE_MODE = "overlap_len_power2"
SQA_NUM_READS = 1
SQA_NUM_SWEEPS = 100
SQA_TROTTER = 4
SQA_BETA = None
SQA_BETA_START = None
SQA_GAMMA = None
SQA_SCHEDULE = "quartic"
SQA_START_S = 0.8
SQA_PAUSE_S = 0.5
SQA_PAUSE_FRACTION = 0.2
SQA_CLASSICAL_TAIL_FRACTION = 0.5
SELECT_BEST_TROTTER_SLICE = False
STEEPEST_POSTPROCESS = False
ANNEALER = "dwave-tabu"
FIX_ENDPOINTS = True
TABU_TIMEOUT_MS = 2000

DATASET = DATA_DIR / "SRR27862880_phiX174_OLC_cycle742_pilot"
GRAPH_DIR = PROJECT_DIR / "debug" / "phi174_cycle742_graph"
OUTPUT_DIR = PROJECT_DIR / "debug" / "qubo" / "cyclic_hamiltonian"
REPORT_PATH = OUTPUT_DIR / "cycle742_cut_dag_edge_cycle_audit.txt"
SOLVED_ORDER_PATH = OUTPUT_DIR / "cycle742_cut_dag_solved_order.tsv"
LOCAL_MINIMUM_PATH = OUTPUT_DIR / "cycle742_fixed_endpoint_sqa_local_minimum.txt"


def _load_cycle(path: Path) -> tuple[list[str], dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty Hamilton-cycle certificate: {path}")

    positions = [int(row["position"]) for row in rows]
    if positions != list(range(len(rows))):
        raise ValueError("certificate positions are not contiguous from zero")

    cycle = [row["read_id"] for row in rows]
    orientations = {
        row["read_id"]: int(row["orientation"])
        for row in rows
    }
    if len(set(cycle)) != len(cycle):
        raise ValueError("certificate contains a duplicate physical read")
    for index, row in enumerate(rows):
        expected = cycle[(index + 1) % len(cycle)]
        if row["next_read_id"] != expected:
            raise ValueError(
                f"certificate successor mismatch at position {index}: "
                f"{row['next_read_id']} != {expected}"
            )
    return cycle, orientations


def _build_accepted_graph(minimap2_bin: str):
    reads = read_fastq(DATASET / "SRR27862880.stride500.unique.fastq.gz")
    finder, result = run_reads_only_pipeline(
        reads,
        minimap2_bin=minimap2_bin,
        min_overlap=MIN_OVERLAP,
        min_paf_identity=0.990,
        max_error_rate_hint=0.01,
        max_error_rate=0.01,
        overhang_tolerance=20,
        extra_args=(
            "-k", "15", "-w", "5", "-m", "40", "-n", "2", "-X",
            "--secondary=yes", "-N", "1000", "-c", "--eqx",
        ),
        full_coverage_min_identity=0.990,
        edge_mode="paf",
    )
    graph = build_read_only_graph(
        reads,
        result.edges,
        finder.last_containment_evidence,
        min_in_support=1,
        min_out_support=1,
        score_mode=SCORE_MODE,
        apply_low_quality_filter=True,
        full_coverage_evidence=finder.last_full_coverage_evidence,
        apply_full_coverage_filter=True,
    )
    return graph


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _write_report(lines: list[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _reduce_fixed_variables(
    model: QUBOModel,
    fixed_values: dict[int, int],
) -> tuple[QUBOModel, list[int]]:
    """Substitute fixed binary variables and return the exact reduced QUBO."""
    if any(value not in (0, 1) for value in fixed_values.values()):
        raise ValueError("fixed QUBO values must be binary")
    if any(index < 0 or index >= model.num_variables for index in fixed_values):
        raise ValueError("fixed QUBO variable index is out of range")

    free_indices = [
        index
        for index in range(model.num_variables)
        if index not in fixed_values
    ]
    reduced_index = {
        original_index: index
        for index, original_index in enumerate(free_indices)
    }
    reduced = QUBOModel(
        read_ids=model.read_ids,
        linear=[0.0] * len(free_indices),
        constant=model.constant,
    )

    for original_index, coefficient in enumerate(model.linear):
        if original_index in fixed_values:
            reduced.constant += coefficient * fixed_values[original_index]
        else:
            reduced.linear[reduced_index[original_index]] += coefficient

    for (left, right), coefficient in model.quadratic.items():
        if coefficient == 0.0:
            continue
        left_fixed = left in fixed_values
        right_fixed = right in fixed_values
        if left_fixed and right_fixed:
            reduced.constant += (
                coefficient * fixed_values[left] * fixed_values[right]
            )
        elif left_fixed:
            reduced.linear[reduced_index[right]] += (
                coefficient * fixed_values[left]
            )
        elif right_fixed:
            reduced.linear[reduced_index[left]] += (
                coefficient * fixed_values[right]
            )
        else:
            reduced.add_quadratic(
                reduced_index[left], reduced_index[right], coefficient
            )
    return reduced, free_indices


def _fixed_endpoint_reduction(
    model: EdgeCycleCoverDAGQUBOModel,
    path_start: str,
    path_end: str,
) -> tuple[QUBOModel, list[int], dict[int, int]]:
    """Fix the known void edges for this witness-guided cut experiment."""
    fixed_values: dict[int, int] = {}
    for read_id in model.read_ids:
        fixed_values[model.source_variable_index(read_id)] = int(
            read_id == path_start
        )
        fixed_values[model.sink_variable_index(read_id)] = int(
            read_id == path_end
        )
    reduced, free_indices = _reduce_fixed_variables(model, fixed_values)
    return reduced, free_indices, fixed_values


def _expand_sample(
    reduced_sample: list[int],
    original_size: int,
    free_indices: list[int],
    fixed_values: dict[int, int],
) -> list[int]:
    if len(reduced_sample) != len(free_indices):
        raise ValueError("reduced sample size does not match free-variable map")
    sample = [0] * original_size
    for index, value in fixed_values.items():
        sample[index] = value
    for reduced_index, original_index in enumerate(free_indices):
        sample[original_index] = reduced_sample[reduced_index]
    return sample


def _make_annealer(args, initial_state: tuple[int, ...] | None = None):
    if args.annealer == "openjij-sqa":
        return OpenJijSimulatedQuantumAnnealer(OpenJijSQAConfig(
            num_reads=args.sqa_num_reads,
            num_sweeps=args.sqa_num_sweeps,
            seed=args.annealer_seed,
            beta=args.sqa_beta,
            gamma=args.sqa_gamma,
            trotter=args.sqa_trotter,
            schedule=_build_sqa_schedule(args),
            initial_state=initial_state,
        ))
    if args.annealer == "dwave-sa":
        return DWaveSimulatedAnnealer(DWaveAnnealingConfig(
            num_reads=args.sqa_num_reads,
            num_sweeps=args.sqa_num_sweeps,
            seed=args.annealer_seed,
        ))
    if args.annealer == "dwave-tabu":
        return _DWaveTabuAnnealer(
            num_reads=args.sqa_num_reads,
            seed=args.annealer_seed,
            timeout_ms=args.tabu_timeout_ms,
        )
    raise ValueError(f"unsupported annealer: {args.annealer}")


def _build_sqa_schedule(args) -> tuple[tuple[float, ...], ...] | None:
    """Build an OpenJij custom schedule with the requested total MC steps."""
    if args.sqa_schedule == "quartic" and args.sqa_beta_start is None:
        return None
    if args.sqa_num_sweeps < 2:
        raise ValueError("custom SQA schedules require at least two sweeps")
    if args.sqa_schedule == "classical-tail":
        if args.sqa_beta_start is None or args.sqa_beta is None:
            raise ValueError(
                "classical-tail requires --sqa-beta-start and --sqa-beta"
            )
        if args.sqa_beta_start <= 0.0:
            raise ValueError("SQA beta start must be positive")
        if not 0.0 < args.sqa_classical_tail_fraction < 1.0:
            raise ValueError("SQA classical-tail fraction must be between 0 and 1")
        tail_steps = max(
            2, round(args.sqa_num_sweeps * args.sqa_classical_tail_fraction)
        )
        quantum_steps = args.sqa_num_sweeps - tail_steps
        if quantum_steps < 2:
            raise ValueError("SQA classical-tail needs more total sweeps")
        quantum_denominator = quantum_steps - 1
        schedule = []
        for index in range(quantum_steps):
            raw_s = index / quantum_denominator
            anneal_s = raw_s ** 4 * (
                35 - 84 * raw_s + 70 * raw_s ** 2 - 20 * raw_s ** 3
            )
            schedule.append((anneal_s, args.sqa_beta_start, 1))
        tail_denominator = tail_steps - 1
        schedule.extend(
            (
                1.0,
                args.sqa_beta_start
                + (args.sqa_beta - args.sqa_beta_start)
                * index / tail_denominator,
                1,
            )
            for index in range(tail_steps)
        )
        return tuple(schedule)
    if args.sqa_schedule == "quartic":
        denominator = args.sqa_num_sweeps - 1
        schedule = []
        for index in range(args.sqa_num_sweeps):
            raw_s = index / denominator
            anneal_s = raw_s ** 4 * (
                35 - 84 * raw_s + 70 * raw_s ** 2 - 20 * raw_s ** 3
            )
            schedule.append((anneal_s, 1))
    elif args.sqa_schedule == "linear":
        denominator = args.sqa_num_sweeps - 1
        schedule = [
            (index / denominator, 1)
            for index in range(args.sqa_num_sweeps)
        ]
    elif args.sqa_schedule == "late-linear":
        if not 0.0 <= args.sqa_start_s < 1.0:
            raise ValueError("SQA start position must be in [0, 1)")
        denominator = args.sqa_num_sweeps - 1
        schedule = [
            (
                args.sqa_start_s
                + (1.0 - args.sqa_start_s) * index / denominator,
                1,
            )
            for index in range(args.sqa_num_sweeps)
        ]
    elif args.sqa_schedule == "pause":
        if not 0.0 < args.sqa_pause_s < 1.0:
            raise ValueError("SQA pause position must be between 0 and 1")
        if not 0.0 < args.sqa_pause_fraction < 1.0:
            raise ValueError("SQA pause fraction must be between 0 and 1")
        pause_steps = max(
            1, round(args.sqa_num_sweeps * args.sqa_pause_fraction)
        )
        ramp_steps = args.sqa_num_sweeps - pause_steps
        before_steps = max(2, ramp_steps // 2)
        after_steps = ramp_steps - before_steps
        if after_steps < 1:
            raise ValueError("SQA pause schedule needs more total sweeps")
        schedule = [
            (
                args.sqa_pause_s * index / (before_steps - 1),
                1,
            )
            for index in range(before_steps)
        ]
        schedule.extend(
            (args.sqa_pause_s, 1)
            for _ in range(pause_steps)
        )
        schedule.extend(
            (
                args.sqa_pause_s
                + (1.0 - args.sqa_pause_s) * (index + 1) / after_steps,
                1,
            )
            for index in range(after_steps)
        )
    else:
        raise ValueError(f"unsupported SQA schedule: {args.sqa_schedule}")

    if args.sqa_beta_start is None:
        return tuple(schedule)
    if args.sqa_beta is None:
        raise ValueError("SQA beta ramp requires --sqa-beta as its final beta")
    if args.sqa_beta_start <= 0.0:
        raise ValueError("SQA beta start must be positive")
    denominator = len(schedule) - 1
    return tuple(
        (
            anneal_s,
            args.sqa_beta_start
            + (args.sqa_beta - args.sqa_beta_start) * index / denominator,
            one_mc_step,
        )
        for index, (anneal_s, one_mc_step) in enumerate(schedule)
    )


def _single_flip_diagnostics(
    model: QUBOModel,
    sample: list[int],
    *,
    tolerance: float = 1e-9,
) -> dict[str, float | int]:
    """Summarize exact QUBO energy changes for all one-bit flips."""
    if len(sample) != model.num_variables:
        raise ValueError("sample size does not match QUBO model")
    adjacency = model.adjacency()
    deltas: list[float] = []
    for variable, value in enumerate(sample):
        local_field = model.linear[variable] + sum(
            coefficient * sample[neighbor]
            for neighbor, coefficient in adjacency[variable].items()
        )
        deltas.append(local_field if value == 0 else -local_field)
    improving = [delta for delta in deltas if delta < -tolerance]
    neutral = [delta for delta in deltas if abs(delta) <= tolerance]
    return {
        "improving_flip_count": len(improving),
        "neutral_flip_count": len(neutral),
        "minimum_flip_delta": min(deltas),
        "maximum_improvement": -min(improving) if improving else 0.0,
    }


def _steepest_descent(
    model: QUBOModel,
    initial_sample: list[int],
) -> tuple[list[int], float, float]:
    """Run deterministic single-bit descent from one supplied sample."""
    from dwave.samplers import SteepestDescentSolver

    start = perf_counter()
    sampleset = SteepestDescentSolver().sample(
        model.to_dimod_bqm(),
        initial_states=[initial_sample],
        initial_states_generator="none",
        num_reads=1,
    )
    elapsed = perf_counter() - start
    first = sampleset.first
    sample = [
        int(first.sample.get(index, 0))
        for index in range(model.num_variables)
    ]
    return sample, float(first.energy), elapsed


def _best_openjij_trotter_slice(
    annealer: OpenJijSimulatedQuantumAnnealer,
    model: QUBOModel,
) -> tuple[list[int], float, int] | None:
    response = annealer.last_response
    if response is None:
        return None
    candidates: list[tuple[float, list[int]]] = []
    for system_info in response.info.get("system", []):
        for raw_state in system_info.get("trotter_state", []):
            values = [int(value) for value in raw_state]
            if set(values).issubset({-1, 1}):
                sample = [(value + 1) // 2 for value in values]
            elif set(values).issubset({0, 1}):
                sample = values
            else:
                raise ValueError("unexpected OpenJij Trotter-state values")
            candidates.append((model.energy(sample), sample))
    if not candidates:
        return None
    best_energy, best_sample = min(candidates, key=lambda item: item[0])
    return best_sample, best_energy, len(candidates)


def _edge_difference_counts(
    model: EdgeCycleCoverDAGQUBOModel,
    sample: list[int],
    witness_sample: list[int],
) -> tuple[int, int]:
    edge_count = len(model.edge_pairs)
    false_positive = sum(
        sample[index] == 1 and witness_sample[index] == 0
        for index in range(edge_count)
    )
    false_negative = sum(
        sample[index] == 0 and witness_sample[index] == 1
        for index in range(edge_count)
    )
    return false_positive, false_negative


def _write_degree_conflict_report(
    model: EdgeCycleCoverDAGQUBOModel,
    sample: list[int],
) -> None:
    selected_edges = [
        pair
        for variable, pair in enumerate(model.edge_pairs)
        if sample[variable]
    ]
    incoming: dict[str, list[str]] = {
        read_id: [] for read_id in model.read_ids
    }
    outgoing: dict[str, list[str]] = {
        read_id: [] for read_id in model.read_ids
    }
    for left_id, right_id in selected_edges:
        outgoing[left_id].append(right_id)
        incoming[right_id].append(left_id)
    in_conflicts = {
        read_id: sorted(neighbors)
        for read_id, neighbors in incoming.items()
        if len(neighbors) > 1
    }
    out_conflicts = {
        read_id: sorted(neighbors)
        for read_id, neighbors in outgoing.items()
        if len(neighbors) > 1
    }
    edge_classes = {
        "both_conflict_endpoints": 0,
        "out_conflict_only": 0,
        "in_conflict_only": 0,
        "no_conflict_endpoint": 0,
    }
    for left_id, right_id in selected_edges:
        left_conflict = left_id in out_conflicts
        right_conflict = right_id in in_conflicts
        if left_conflict and right_conflict:
            edge_classes["both_conflict_endpoints"] += 1
        elif left_conflict:
            edge_classes["out_conflict_only"] += 1
        elif right_conflict:
            edge_classes["in_conflict_only"] += 1
        else:
            edge_classes["no_conflict_endpoint"] += 1

    lines = [
        "== fixed-endpoint SQA degree-conflict structure ==",
        f"selected_edges: {len(selected_edges)}",
        f"in_conflict_nodes: {len(in_conflicts)}",
        f"out_conflict_nodes: {len(out_conflicts)}",
        *(
            f"selected_edge_class_{name}: {count}"
            for name, count in edge_classes.items()
        ),
        "",
        "[in-degree conflicts]",
        *(
            f"{read_id}\t<-\t{','.join(neighbors)}"
            for read_id, neighbors in sorted(in_conflicts.items())
        ),
        "",
        "[out-degree conflicts]",
        *(
            f"{read_id}\t->\t{','.join(neighbors)}"
            for read_id, neighbors in sorted(out_conflicts.items())
        ),
    ]
    LOCAL_MINIMUM_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _DWaveTabuAnnealer:
    """Small demo-only adapter around dwave.samplers.TabuSampler."""

    def __init__(self, *, num_reads: int, seed: int, timeout_ms: int):
        if num_reads <= 0:
            raise ValueError("num_reads must be positive")
        if timeout_ms <= 0:
            raise ValueError("tabu timeout must be positive")
        self.num_reads = num_reads
        self.seed = seed
        self.timeout_ms = timeout_ms

    def solve(self, model: QUBOModel) -> AnnealingResult:
        from dwave.samplers import TabuSampler

        sampleset = TabuSampler().sample(
            model.to_dimod_bqm(),
            num_reads=self.num_reads,
            seed=self.seed,
            timeout=self.timeout_ms,
        )
        first = sampleset.first
        sample = [
            int(first.sample.get(index, 0))
            for index in range(model.num_variables)
        ]
        return AnnealingResult(
            sample=sample,
            energy=float(first.energy),
            iterations=-1,
            accepted_moves=-1,
            backend="dwave-tabu",
        )


def _write_solved_order(order: list[str], orientations: dict[str, int]) -> None:
    rows = ["position\tread_id\torientation\tnext_read_id"]
    for position, read_id in enumerate(order):
        next_read_id = order[position + 1] if position + 1 < len(order) else "__void__"
        rows.append(
            f"{position}\t{read_id}\t{orientations[read_id]:+d}\t{next_read_id}"
        )
    SOLVED_ORDER_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimap2-bin", default="minimap2")
    parser.add_argument("--cut-seed", type=int, default=CUT_SEED)
    parser.add_argument("--annealer-seed", type=int, default=ANNEALER_SEED)
    parser.add_argument(
        "--solve",
        action="store_true",
        help="also run a deliberately small OpenJij SQA trial",
    )
    parser.add_argument("--sqa-num-reads", type=int, default=SQA_NUM_READS)
    parser.add_argument("--sqa-num-sweeps", type=int, default=SQA_NUM_SWEEPS)
    parser.add_argument("--sqa-trotter", type=int, default=SQA_TROTTER)
    parser.add_argument("--sqa-beta", type=float, default=SQA_BETA)
    parser.add_argument("--sqa-beta-start", type=float, default=SQA_BETA_START)
    parser.add_argument("--sqa-gamma", type=float, default=SQA_GAMMA)
    parser.add_argument(
        "--sqa-schedule",
        choices=("quartic", "linear", "late-linear", "pause", "classical-tail"),
        default=SQA_SCHEDULE,
    )
    parser.add_argument("--sqa-start-s", type=float, default=SQA_START_S)
    parser.add_argument("--sqa-pause-s", type=float, default=SQA_PAUSE_S)
    parser.add_argument(
        "--sqa-pause-fraction", type=float, default=SQA_PAUSE_FRACTION
    )
    parser.add_argument(
        "--sqa-classical-tail-fraction",
        type=float,
        default=SQA_CLASSICAL_TAIL_FRACTION,
    )
    parser.add_argument(
        "--steepest-postprocess",
        action=argparse.BooleanOptionalAction,
        default=STEEPEST_POSTPROCESS,
        help="run deterministic single-bit descent from the annealer sample",
    )
    parser.add_argument(
        "--select-best-trotter-slice",
        action=argparse.BooleanOptionalAction,
        default=SELECT_BEST_TROTTER_SLICE,
        help="select the lowest-energy recorded OpenJij Trotter slice",
    )
    parser.add_argument(
        "--sqa-initial-state",
        choices=("random", "certificate"),
        default="random",
        help="diagnostic SQA initial state; certificate is non-strict",
    )
    parser.add_argument("--degree-penalty", type=float, default=DEGREE_PENALTY)
    parser.add_argument("--edge-reward-scale", type=float, default=EDGE_REWARD_SCALE)
    parser.add_argument(
        "--edge-selection-penalty",
        type=float,
        default=EDGE_SELECTION_PENALTY,
        help="linear penalty per selected real edge; zero preserves the original model",
    )
    parser.add_argument(
        "--annealer",
        choices=("openjij-sqa", "dwave-sa", "dwave-tabu"),
        default=ANNEALER,
    )
    parser.add_argument("--tabu-timeout-ms", type=int, default=TABU_TIMEOUT_MS)
    parser.add_argument(
        "--fix-endpoints",
        action=argparse.BooleanOptionalAction,
        default=FIX_ENDPOINTS,
        help="fix all void variables using the known cut endpoints",
    )
    args = parser.parse_args()

    cycle, certificate_orientations = _load_cycle(
        GRAPH_DIR / "cycle742_hamilton_cycle.tsv"
    )
    graph = _build_accepted_graph(args.minimap2_bin)
    graph_ids = {read.rid for read in graph.reads}
    if graph_ids != set(cycle):
        missing = sorted(set(cycle) - graph_ids)
        extra = sorted(graph_ids - set(cycle))
        raise ValueError(
            f"graph/certificate read mismatch: missing={missing}, extra={extra}"
        )
    if graph.orientation_by_read != certificate_orientations:
        mismatches = sorted(
            read_id
            for read_id in graph_ids
            if graph.orientation_by_read[read_id]
            != certificate_orientations[read_id]
        )
        raise ValueError(f"graph/certificate orientation mismatch: {mismatches}")

    graph_pairs = {(edge.left_id, edge.right_id) for edge in graph.edges}
    cycle_pairs = list(zip(cycle, cycle[1:] + cycle[:1]))
    missing_cycle_pairs = [pair for pair in cycle_pairs if pair not in graph_pairs]
    if missing_cycle_pairs:
        raise ValueError(
            f"certificate contains {len(missing_cycle_pairs)} absent graph edges"
        )

    cut_position = random.Random(args.cut_seed).randrange(len(cycle))
    cut_left = cycle[cut_position]
    cut_right = cycle[(cut_position + 1) % len(cycle)]
    rotated_order = cycle[cut_position + 1:] + cycle[:cut_position + 1]
    rank = {read_id: index for index, read_id in enumerate(rotated_order)}
    dag_edges = [
        edge
        for edge in graph.edges
        if rank[edge.left_id] < rank[edge.right_id]
    ]
    dag_pairs = {(edge.left_id, edge.right_id) for edge in dag_edges}
    path_pairs = list(zip(rotated_order, rotated_order[1:]))
    missing_path_pairs = [pair for pair in path_pairs if pair not in dag_pairs]
    if missing_path_pairs:
        raise ValueError(
            f"cut DAG lost {len(missing_path_pairs)} certificate path edges"
        )

    hamiltonian = EdgeCycleCoverDAGQUBOHamiltonian(
        EdgeCycleCoverDAGHamiltonianConfig(
            degree_penalty=args.degree_penalty,
            edge_reward_scale=args.edge_reward_scale,
            edge_selection_penalty=args.edge_selection_penalty,
            score_mode=SCORE_MODE,
            normalize_rewards=True,
        )
    )
    build_start = perf_counter()
    model = hamiltonian.build(graph.reads, dag_edges, weight_mode=SCORE_MODE)
    build_seconds = perf_counter() - build_start

    witness_sample = qubo_sample_for_order(model, rotated_order)
    witness_energy = model.energy(witness_sample)
    decoded_order, witness_metadata = QUBOLayoutSolver._decode_edge_cycle_cover(
        model, witness_sample
    )
    if decoded_order != rotated_order:
        raise ValueError("decoded witness order differs from the rotated certificate")
    if not witness_metadata["valid_edge_cycle"]:
        raise ValueError("rotated certificate is not valid under the QUBO model")

    reduced_model, free_indices, fixed_values = _fixed_endpoint_reduction(
        model, rotated_order[0], rotated_order[-1]
    )
    reduced_witness = [witness_sample[index] for index in free_indices]
    reduced_witness_energy = reduced_model.energy(reduced_witness)
    if abs(reduced_witness_energy - witness_energy) > 1e-8:
        raise ValueError(
            "fixed-variable reduction changed the witness QUBO energy"
        )

    lines = [
        "== phi174 cycle742 cut-DAG edge-cycle QUBO audit ==",
        "test_class: non_strict_known_witness_guided",
        "reference_used: no",
        "known_graph_hamilton_cycle_used_to_define_rank: yes",
        f"cut_seed: {args.cut_seed}",
        f"cut_position: {cut_position}",
        f"cut_edge: {cut_left} -> {cut_right}",
        f"cut_edge_orientations: {graph.orientation_by_read[cut_left]:+d} -> {graph.orientation_by_read[cut_right]:+d}",
        f"path_start: {rotated_order[0]}",
        f"path_end: {rotated_order[-1]}",
        f"graph_nodes: {len(graph.reads)}",
        f"graph_unique_directed_edges: {len(graph_pairs)}",
        f"dag_unique_directed_edges: {len(dag_pairs)}",
        f"backward_or_cut_edges_removed: {len(graph_pairs) - len(dag_pairs)}",
        f"certificate_path_edges_preserved: {len(path_pairs) - len(missing_path_pairs)}/{len(path_pairs)}",
        "dag_rule: keep edge u->v iff rank(u) < rank(v)",
        "dag_verified_by_hamiltonian_builder: yes",
        f"degree_penalty: {args.degree_penalty}",
        f"edge_reward_scale: {args.edge_reward_scale}",
        f"edge_selection_penalty: {args.edge_selection_penalty}",
        f"score_mode: {SCORE_MODE}",
        "normalize_rewards: True",
        f"qubo_candidate_edges: {hamiltonian.last_edge_count}",
        f"qubo_variables: {model.num_variables}",
        f"qubo_quadratic_terms: {len(model.quadratic)}",
        f"qubo_build_sec: {build_seconds:.6f}",
        f"reward_min: {_format_value(hamiltonian.last_reward_min)}",
        f"reward_max: {_format_value(hamiltonian.last_reward_max)}",
        f"witness_selected_variables: {sum(witness_sample)}",
        f"witness_energy: {witness_energy:.12g}",
        f"witness_valid_edge_cycle: {witness_metadata['valid_edge_cycle']}",
        f"witness_selected_edge_count: {witness_metadata['selected_edge_count']}",
        f"witness_selected_source_count: {witness_metadata['selected_source_count']}",
        f"witness_selected_sink_count: {witness_metadata['selected_sink_count']}",
        f"witness_read_in_constraint_violations: {witness_metadata['read_in_constraint_violations']}",
        f"witness_read_out_constraint_violations: {witness_metadata['read_out_constraint_violations']}",
        f"witness_void_in_constraint_violation: {witness_metadata['void_in_constraint_violation']}",
        f"witness_void_out_constraint_violation: {witness_metadata['void_out_constraint_violation']}",
        f"fixed_endpoint_variables: {len(fixed_values)}",
        f"reduced_qubo_variables: {reduced_model.num_variables}",
        f"reduced_qubo_quadratic_terms: {len(reduced_model.quadratic)}",
        f"reduced_witness_energy: {reduced_witness_energy:.12g}",
        f"reduction_energy_invariant: {abs(reduced_witness_energy - witness_energy) <= 1e-8}",
    ]
    _write_report(lines)
    print("\n".join(lines))
    print(f"audit_report: {REPORT_PATH}")

    if args.solve:
        solve_model = reduced_model if args.fix_endpoints else model
        initial_state = None
        if args.sqa_initial_state == "certificate":
            initial_state = tuple(
                reduced_witness if args.fix_endpoints else witness_sample
            )
        annealer = _make_annealer(args, initial_state=initial_state)
        solve_start = perf_counter()
        annealing_result = annealer.solve(solve_model)
        solve_seconds = perf_counter() - solve_start
        raw_solver_sample = list(annealing_result.sample)
        trotter_candidate = None
        if args.annealer == "openjij-sqa":
            trotter_candidate = _best_openjij_trotter_slice(
                annealer, solve_model
            )
            if args.select_best_trotter_slice and trotter_candidate is not None:
                raw_solver_sample = trotter_candidate[0]
        raw_flip_diagnostics = _single_flip_diagnostics(
            solve_model, raw_solver_sample
        )
        if args.fix_endpoints:
            solver_sample = _expand_sample(
                raw_solver_sample,
                model.num_variables,
                free_indices,
                fixed_values,
            )
        else:
            solver_sample = raw_solver_sample
        solver_energy = model.energy(solver_sample)
        solver_order, solver_metadata = QUBOLayoutSolver._decode_edge_cycle_cover(
            model, solver_sample
        )
        false_positive_edges, false_negative_edges = _edge_difference_counts(
            model, solver_sample, witness_sample
        )
        if args.annealer == "openjij-sqa" and args.fix_endpoints:
            _write_degree_conflict_report(model, solver_sample)
        solver_lines = [
            "",
            "== low-budget annealing trial ==",
            f"solver_annealer: {args.annealer}",
            f"solver_annealer_seed: {args.annealer_seed}",
            f"solver_fixed_endpoints: {args.fix_endpoints}",
            f"solver_sqa_initial_state: {args.sqa_initial_state if args.annealer == 'openjij-sqa' else 'not_applicable'}",
            f"annealer_num_reads: {args.sqa_num_reads}",
            f"annealer_num_sweeps: {args.sqa_num_sweeps if args.annealer != 'dwave-tabu' else 'not_applicable'}",
            f"sqa_trotter: {args.sqa_trotter if args.annealer == 'openjij-sqa' else 'not_applicable'}",
            f"sqa_beta: {args.sqa_beta if args.annealer == 'openjij-sqa' else 'not_applicable'}",
            f"sqa_beta_start: {args.sqa_beta_start if args.annealer == 'openjij-sqa' else 'not_applicable'}",
            f"sqa_gamma: {args.sqa_gamma if args.annealer == 'openjij-sqa' else 'not_applicable'}",
            f"sqa_schedule: {args.sqa_schedule if args.annealer == 'openjij-sqa' else 'not_applicable'}",
            f"sqa_start_s: {args.sqa_start_s if args.annealer == 'openjij-sqa' and args.sqa_schedule == 'late-linear' else 'not_applicable'}",
            f"sqa_pause_s: {args.sqa_pause_s if args.annealer == 'openjij-sqa' and args.sqa_schedule == 'pause' else 'not_applicable'}",
            f"sqa_pause_fraction: {args.sqa_pause_fraction if args.annealer == 'openjij-sqa' and args.sqa_schedule == 'pause' else 'not_applicable'}",
            f"sqa_classical_tail_fraction: {args.sqa_classical_tail_fraction if args.annealer == 'openjij-sqa' and args.sqa_schedule == 'classical-tail' else 'not_applicable'}",
            f"tabu_timeout_ms: {args.tabu_timeout_ms if args.annealer == 'dwave-tabu' else 'not_applicable'}",
            f"solver_variables: {solve_model.num_variables}",
            f"solver_quadratic_terms: {len(solve_model.quadratic)}",
            f"solver_energy: {_format_value(solver_energy)}",
            f"solver_backend_reported_energy: {_format_value(annealing_result.energy)}",
            f"solver_select_best_trotter_slice: {args.select_best_trotter_slice if args.annealer == 'openjij-sqa' else 'not_applicable'}",
            f"solver_trotter_slice_candidates: {trotter_candidate[2] if trotter_candidate is not None else 'not_available'}",
            f"solver_best_trotter_slice_energy: {_format_value(trotter_candidate[1]) if trotter_candidate is not None else 'not_available'}",
            f"solver_valid_edge_cycle: {solver_metadata['valid_edge_cycle']}",
            f"solver_matches_rotated_witness: {solver_order == rotated_order}",
            f"solver_selected_edge_count: {solver_metadata['selected_edge_count']}",
            f"solver_selected_source_count: {solver_metadata['selected_source_count']}",
            f"solver_selected_sink_count: {solver_metadata['selected_sink_count']}",
            f"solver_read_in_constraint_violations: {solver_metadata['read_in_constraint_violations']}",
            f"solver_read_out_constraint_violations: {solver_metadata['read_out_constraint_violations']}",
            f"solver_void_in_constraint_violation: {solver_metadata['void_in_constraint_violation']}",
            f"solver_void_out_constraint_violation: {solver_metadata['void_out_constraint_violation']}",
            f"solver_false_positive_edges_vs_witness: {false_positive_edges}",
            f"solver_false_negative_edges_vs_witness: {false_negative_edges}",
            f"solver_improving_single_flips: {raw_flip_diagnostics['improving_flip_count']}",
            f"solver_neutral_single_flips: {raw_flip_diagnostics['neutral_flip_count']}",
            f"solver_minimum_single_flip_delta: {_format_value(raw_flip_diagnostics['minimum_flip_delta'])}",
            f"solver_maximum_single_flip_improvement: {_format_value(raw_flip_diagnostics['maximum_improvement'])}",
            f"solver_anneal_sec: {_format_value(solve_seconds)}",
            f"solver_degree_conflict_report: {LOCAL_MINIMUM_PATH if args.annealer == 'openjij-sqa' and args.fix_endpoints else 'not_written'}",
        ]
        if solver_metadata["valid_edge_cycle"]:
            _write_solved_order(solver_order, graph.orientation_by_read)
            solver_lines.append(f"solver_order_tsv: {SOLVED_ORDER_PATH}")

        if args.steepest_postprocess:
            post_sample, post_backend_energy, post_seconds = _steepest_descent(
                solve_model, raw_solver_sample
            )
            post_flip_diagnostics = _single_flip_diagnostics(
                solve_model, post_sample
            )
            if args.fix_endpoints:
                post_full_sample = _expand_sample(
                    post_sample,
                    model.num_variables,
                    free_indices,
                    fixed_values,
                )
            else:
                post_full_sample = post_sample
            post_energy = model.energy(post_full_sample)
            post_order, post_metadata = QUBOLayoutSolver._decode_edge_cycle_cover(
                model, post_full_sample
            )
            post_false_positive, post_false_negative = _edge_difference_counts(
                model, post_full_sample, witness_sample
            )
            post_lines = [
                "",
                "== deterministic single-bit postprocess ==",
                "postprocess_backend: dwave-steepest-descent",
                f"postprocess_initial_hamming_flips: {sum(left != right for left, right in zip(raw_solver_sample, post_sample))}",
                f"postprocess_energy: {_format_value(post_energy)}",
                f"postprocess_backend_reported_energy: {_format_value(post_backend_energy)}",
                f"postprocess_energy_improvement: {_format_value(solver_energy - post_energy)}",
                f"postprocess_valid_edge_cycle: {post_metadata['valid_edge_cycle']}",
                f"postprocess_matches_rotated_witness: {post_order == rotated_order}",
                f"postprocess_selected_edge_count: {post_metadata['selected_edge_count']}",
                f"postprocess_read_in_constraint_violations: {post_metadata['read_in_constraint_violations']}",
                f"postprocess_read_out_constraint_violations: {post_metadata['read_out_constraint_violations']}",
                f"postprocess_false_positive_edges_vs_witness: {post_false_positive}",
                f"postprocess_false_negative_edges_vs_witness: {post_false_negative}",
                f"postprocess_improving_single_flips: {post_flip_diagnostics['improving_flip_count']}",
                f"postprocess_neutral_single_flips: {post_flip_diagnostics['neutral_flip_count']}",
                f"postprocess_minimum_single_flip_delta: {_format_value(post_flip_diagnostics['minimum_flip_delta'])}",
                f"postprocess_sec: {_format_value(post_seconds)}",
            ]
            if post_metadata["valid_edge_cycle"]:
                _write_solved_order(post_order, graph.orientation_by_read)
                post_lines.append(f"postprocess_order_tsv: {SOLVED_ORDER_PATH}")
            solver_lines.extend(post_lines)
        lines.extend(solver_lines)
        _write_report(lines)
        print("\n".join(solver_lines))


if __name__ == "__main__":
    main()
