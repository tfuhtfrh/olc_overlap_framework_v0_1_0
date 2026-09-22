"""Reads-only OLC graph and reference-mapping audit for the 12-read dataset."""

from __future__ import annotations

import argparse
import statistics
import subprocess
from pathlib import Path

from demo_phi174_common import DATA_DIR, PROJECT_DIR, run_reads_only_pipeline
from olc_pipeline.graph_builder import build_read_only_graph, check_hamilton_cycle
from olc_pipeline.graph_viz import write_oriented_overlap_graph_dot
from olc_pipeline.io_utils import read_fastq
from olc_pipeline.reference_evaluator import CircularReferenceEvaluator, read_fasta_sequence


DATASET = DATA_DIR / "phiX174_OLC_cycle12_SRR37543650_v2"
OUTPUT_DIR = PROJECT_DIR / "debug" / "phi174_cycle12_graph"


def render(dot_path: Path, output_type: str) -> Path:
    output_path = dot_path.with_suffix(f".{output_type}")
    subprocess.run(
        ["dot", f"-T{output_type}", str(dot_path), "-o", str(output_path)],
        check=True,
    )
    return output_path


def write_reference_mapping_tsv(
    reads,
    placements,
    path: Path,
) -> tuple[float, float, float, float]:
    rows: list[tuple[str, int, int, int, float, float, int]] = []
    for read in reads:
        placement = placements.get(read.rid)
        if placement is None:
            continue
        error_rate = 1.0 - placement.identity
        rows.append((
            read.rid,
            placement.orientation,
            placement.reference_start,
            placement.reference_end,
            placement.identity,
            error_rate,
            placement.mapq,
        ))
    rows.sort(key=lambda row: (row[2], row[0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("read_id\torientation\tref_start\tref_end\tidentity\terror_rate\tmapq\n")
        for row in rows:
            handle.write(
                f"{row[0]}\t{row[1]:+d}\t{row[2]}\t{row[3]}\t"
                f"{row[4]:.8f}\t{row[5]:.8f}\t{row[6]}\n"
            )
    if not rows:
        return (0.0, 0.0, 0.0, 0.0)
    errors = [row[5] for row in rows]
    weighted_identity = sum(row[4] * (row[3] - row[2]) for row in rows)
    total_span = sum(row[3] - row[2] for row in rows)
    return (
        statistics.mean(errors),
        statistics.median(errors),
        max(errors),
        1.0 - weighted_identity / total_span if total_span else 0.0,
    )


def orientation_agreement(selection, placements) -> tuple[float, float]:
    """Compare graph-only strand choices with mapping, up to global RC symmetry."""
    direct: list[bool] = []
    flipped: list[bool] = []
    for read_id, orientation in selection.orientation_by_read.items():
        if read_id not in placements:
            continue
        reference_orientation = placements[read_id].orientation
        direct.append(orientation == reference_orientation)
        flipped.append(-orientation == reference_orientation)
    if not direct:
        return (0.0, 0.0)
    return (
        sum(direct) / len(direct),
        sum(flipped) / len(flipped),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimap2-bin", default="minimap2")
    args = parser.parse_args()

    reads = read_fastq(
        DATASET / "SRR37543650.cycle12.scrubbed.original_orientation.fastq.gz"
    )
    finder, result = run_reads_only_pipeline(
        reads,
        minimap2_bin=args.minimap2_bin,
        min_overlap=200,
        min_paf_identity=0.96,
        max_error_rate_hint=0.04,
        max_error_rate=0.10,
        overhang_tolerance=20,
        extra_args=("-k", "9", "-w", "5", "-m", "20", "-n", "2", "-s", "40", "-c", "--eqx", "-X"),
        # The published 12-read benchmark contains genuine overlap edges
        # below 0.990; 0.990 is reserved for full-coverage duplicate removal.
        full_coverage_min_identity=0.990,
        edge_mode="paf",
    )

    # This preprocessing is graph-only.  The reference mapping below is an
    # audit and is not passed into candidate finding or graph filtering.
    graph = build_read_only_graph(
        reads,
        result.edges,
        finder.last_containment_evidence,
        min_in_support=1,
        min_out_support=1,
        score_mode="overlap_len_power2",
        apply_low_quality_filter=False,
        full_coverage_evidence=finder.last_full_coverage_evidence,
        apply_full_coverage_filter=True,
    )
    hamilton = check_hamilton_cycle(
        [
            (read_id, orientation)
            for read_id, orientation in graph.orientation_by_read.items()
        ],
        graph.edges,
        time_limit_sec=30.0,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    graph_dot = OUTPUT_DIR / "cycle12_single_orientation_graph.dot"
    write_oriented_overlap_graph_dot(
        graph.reads,
        graph.edges,
        graph_dot,
        score_mode="overlap_len_power2",
        orientation_by_read=graph.orientation_by_read,
    )
    render(graph_dot, "svg")
    render(graph_dot, "png")

    compact_dot = OUTPUT_DIR / "cycle12_single_orientation_graph_compact.dot"
    write_oriented_overlap_graph_dot(
        graph.reads,
        graph.edges,
        compact_dot,
        score_mode="overlap_len_power2",
        rankdir="TB",
        show_edge_labels=False,
        orientation_by_read=graph.orientation_by_read,
    )
    render(compact_dot, "svg")
    render(compact_dot, "png")

    evaluator = CircularReferenceEvaluator(
        read_fasta_sequence(DATASET / "NC_001422.1.fasta"),
        minimap2_bin=args.minimap2_bin,
        preset="map-ont",
    )
    placements = evaluator.map_reads(reads)
    mean_error, median_error, max_error, weighted_error = write_reference_mapping_tsv(
        reads,
        placements,
        OUTPUT_DIR / "cycle12_reference_mapping.tsv",
    )
    direct_agreement, flipped_agreement = orientation_agreement(graph, placements)

    print("== phi174 cycle12 graph audit ==")
    print("graph_input: reads only")
    print("edge_preprocessing: minimap2_paf_direct_no_dp")
    print(f"reads: {len(reads)}")
    print(f"candidate_records: {len(result.candidates)}")
    print(f"paf_edges_before_orientation_filter: {len(result.edges)}")
    print(f"full_coverage_evidence: {len(finder.last_full_coverage_evidence)}")
    print(f"full_coverage_reads_removed: {len(graph.full_coverage_read_ids)}")
    print(f"contained_reads_removed: {len(graph.contained_read_ids)}")
    print(f"low_quality_reads_removed: {len(graph.low_quality_read_ids)}")
    print(f"retained_physical_reads: {len(graph.reads)}")
    print(f"single_orientation_edges: {len(graph.edges)}")
    print(f"orientation_selection_rounds: {graph.orientation_rounds}")
    print(f"orientation_constraints: {graph.orientation_constraint_count}")
    print(f"orientation_constraint_conflicts: {graph.orientation_constraint_conflict_count}")
    print(f"orientation_constraint_components: {graph.orientation_constraint_component_count}")
    print(f"orientation_score: {graph.retained_score:.3f}")
    print(f"hamilton_cycle: {hamilton.status}")
    print(f"hamilton_cycle_nodes: {len(hamilton.cycle) - 1 if hamilton.cycle else 0}")
    print(f"minimap2_filter_counts: {dict(finder.last_filter_counts)}")
    print("== reference mapping audit (not graph input) ==")
    print(f"mapped_reads: {len(placements)}/{len(reads)}")
    print(f"mean_error_rate: {mean_error:.8f}")
    print(f"median_error_rate: {median_error:.8f}")
    print(f"max_error_rate: {max_error:.8f}")
    print(f"length_weighted_error_rate: {weighted_error:.8f}")
    print(f"orientation_agreement_direct: {direct_agreement:.4f}")
    print(f"orientation_agreement_global_reverse: {flipped_agreement:.4f}")
    print(f"mapping_report: {OUTPUT_DIR / 'cycle12_reference_mapping.tsv'}")
    print(f"graph_dot: {graph_dot}")


if __name__ == "__main__":
    main()
