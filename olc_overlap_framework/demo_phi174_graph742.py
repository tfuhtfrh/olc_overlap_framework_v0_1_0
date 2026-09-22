"""Reads-only OLC graph audit for the phi174 742-cycle pilot dataset."""

from __future__ import annotations

import argparse
import statistics
import subprocess
from collections import deque
from pathlib import Path

from demo_phi174_common import DATA_DIR, PROJECT_DIR, run_reads_only_pipeline
from olc_pipeline.data import OverlapEdge
from olc_pipeline.graph_builder import build_read_only_graph, check_hamilton_cycle
from olc_pipeline.graph_viz import write_oriented_overlap_graph_dot
from olc_pipeline.io_utils import read_fastq
from olc_pipeline.layout_solver import OverlapRewardScorer
from olc_pipeline.reference_evaluator import CircularReferenceEvaluator, read_fasta_sequence


DATASET = DATA_DIR / "SRR27862880_phiX174_OLC_cycle742_pilot"
OUTPUT_DIR = PROJECT_DIR / "debug" / "phi174_cycle742_graph"


Node = tuple[str, int]


def _oriented_node(read_id: str, orientation: int) -> Node:
    return read_id, orientation


def _selected_graph(reads, edges: list[OverlapEdge], orientations: dict[str, int]):
    nodes = [_oriented_node(read.rid, orientations[read.rid]) for read in reads]
    node_set = set(nodes)
    adjacency: dict[Node, set[Node]] = {node: set() for node in nodes}
    reverse: dict[Node, set[Node]] = {node: set() for node in nodes}
    for edge in edges:
        left = _oriented_node(edge.left_id, edge.left_orientation)
        right = _oriented_node(edge.right_id, edge.right_orientation)
        if left in node_set and right in node_set and left != right:
            adjacency[left].add(right)
            reverse[right].add(left)
    return nodes, adjacency, reverse


def _weak_components(nodes, adjacency, reverse):
    unseen = set(nodes)
    components = []
    while unseen:
        start = min(unseen, key=str)
        component = {start}
        queue = deque([start])
        unseen.remove(start)
        while queue:
            node = queue.popleft()
            neighbors = adjacency[node] | reverse[node]
            for neighbor in neighbors:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _strong_components(nodes, adjacency, reverse):
    visited = set()
    order = []
    for start in nodes:
        if start in visited:
            continue
        stack = [(start, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            stack.extend((neighbor, False) for neighbor in adjacency[node] if neighbor not in visited)

    components = []
    assigned = set()
    for start in reversed(order):
        if start in assigned:
            continue
        component = set()
        stack = [start]
        assigned.add(start)
        while stack:
            node = stack.pop()
            component.add(node)
            for neighbor in reverse[node]:
                if neighbor not in assigned:
                    assigned.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hamilton_cycle(path: Path, cycle, edges: list[OverlapEdge]) -> list[OverlapEdge]:
    """Write the graph-valid Hamilton witness with one row per selected edge."""
    best_edge = {}
    for edge in edges:
        key = (
            (edge.left_id, edge.left_orientation),
            (edge.right_id, edge.right_orientation),
        )
        old = best_edge.get(key)
        if old is None or (edge.overlap_len, edge.identity) > (old.overlap_len, old.identity):
            best_edge[key] = edge

    selected = []
    rows = ["position\tread_id\torientation\tnext_read_id\tnext_orientation\toverlap_len\tshift\tidentity\tmapq"]
    for position, (left, right) in enumerate(zip(cycle, cycle[1:])):
        edge = best_edge[(left, right)]
        selected.append(edge)
        rows.append(
            f"{position}\t{left[0]}\t{left[1]:+d}\t{right[0]}\t{right[1]:+d}\t"
            f"{edge.overlap_len}\t{edge.shift}\t{edge.identity:.8f}\t{edge.mapq}"
        )
    write_report(path, rows)
    return selected


def _strongest_incident_edges(edges: list[OverlapEdge], orientations: dict[str, int]):
    """Keep strongest incoming/outgoing edge per selected oriented node."""
    scorer = OverlapRewardScorer()
    node_set = set(orientations.items())
    best_out = {}
    best_in = {}
    for edge in edges:
        left = (edge.left_id, edge.left_orientation)
        right = (edge.right_id, edge.right_orientation)
        if left not in node_set or right not in node_set:
            continue
        score = scorer.score(edge, "overlap_len_power2")
        edge_key = (edge.left_id, edge.right_id, edge.left_orientation, edge.right_orientation)
        current = best_out.get(left)
        if current is None or (score, edge.overlap_len, edge_key) > current[0]:
            best_out[left] = ((score, edge.overlap_len, edge_key), edge)
        current = best_in.get(right)
        if current is None or (score, edge.overlap_len, edge_key) > current[0]:
            best_in[right] = ((score, edge.overlap_len, edge_key), edge)

    selected = {}
    for _, edge in [*best_out.values(), *best_in.values()]:
        key = (edge.left_id, edge.right_id, edge.left_orientation, edge.right_orientation)
        selected[key] = edge
    return list(selected.values())


def render_graph(reads, edges, orientations) -> tuple[Path, Path, Path, Path]:
    """Write the full graph and a renderable strongest-incident-edge view."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_dot_path = OUTPUT_DIR / "cycle742_filtered_graph_full.dot"
    dot_path = OUTPUT_DIR / "cycle742_filtered_graph_skeleton.dot"
    svg_path = dot_path.with_suffix(".svg")
    png_path = dot_path.with_suffix(".png")
    write_oriented_overlap_graph_dot(
        reads,
        edges,
        full_dot_path,
        score_mode="overlap_len_power2",
        show_edge_labels=False,
        orientation_by_read=orientations,
    )
    skeleton_edges = _strongest_incident_edges(edges, orientations)
    write_oriented_overlap_graph_dot(
        reads,
        skeleton_edges,
        dot_path,
        score_mode="overlap_len_power2",
        show_edge_labels=False,
        orientation_by_read=orientations,
    )
    subprocess.run(
        ["dot", "-Goverlap=false", "-Gsplines=true", "-Tsvg", str(dot_path), "-o", str(svg_path)],
        check=True,
    )
    subprocess.run(
        ["dot", "-Goverlap=false", "-Gsplines=true", "-Tpng", str(dot_path), "-o", str(png_path)],
        check=True,
    )
    return full_dot_path, dot_path, svg_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimap2-bin", default="minimap2")
    parser.add_argument("--min-overlap", type=int, default=80)
    parser.add_argument("--hamilton-time-limit", type=float, default=120.0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    reads = read_fastq(DATASET / "SRR27862880.stride500.unique.fastq.gz")
    finder, result = run_reads_only_pipeline(
        reads,
        minimap2_bin=args.minimap2_bin,
        min_overlap=args.min_overlap,
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
        score_mode="overlap_len_power2",
        # One-pass removal of reads that lack either incoming or outgoing
        # support in the selected 80 bp graph.  This removes the audited 34
        # reads without recursively peeling newly exposed graph boundaries.
        apply_low_quality_filter=True,
        full_coverage_evidence=finder.last_full_coverage_evidence,
        apply_full_coverage_filter=True,
    )
    orientation_by_read = graph.orientation_by_read
    nodes, adjacency, reverse = _selected_graph(
        graph.reads, graph.edges, orientation_by_read
    )
    weak = _weak_components(nodes, adjacency, reverse)
    strong = _strong_components(nodes, adjacency, reverse)
    hamilton = check_hamilton_cycle(
        nodes, graph.edges, time_limit_sec=args.hamilton_time_limit
    )

    cycle_path = OUTPUT_DIR / "cycle742_hamilton_cycle.tsv"
    cycle_edges = []
    reference_report = None
    truth_graph_edges_present = 0
    truth_graph_edge_count = 0
    truth_missing_path = OUTPUT_DIR / "cycle742_reference_truth_missing_edges.tsv"
    if hamilton.cycle:
        cycle_edges = write_hamilton_cycle(cycle_path, hamilton.cycle, graph.edges)
        evaluator = CircularReferenceEvaluator(
            read_fasta_sequence(DATASET / "NC_001422.1.fasta"),
            minimap2_bin=args.minimap2_bin,
            preset="sr",
        )
        placements = evaluator.map_reads(graph.reads)
        reference_report = evaluator.evaluate(hamilton.cycle[:-1], placements)
        truth_nodes = [
            (read_id, placements[read_id].orientation)
            for read_id in reference_report.truth_order
        ]
        graph_edge_keys = {
            (
                (edge.left_id, edge.left_orientation),
                (edge.right_id, edge.right_orientation),
            )
            for edge in graph.edges
        }
        truth_edges = list(zip(truth_nodes, truth_nodes[1:] + truth_nodes[:1]))
        truth_graph_edge_count = len(truth_edges)
        truth_graph_edges_present = sum(edge in graph_edge_keys for edge in truth_edges)
        missing_rows = ["left_read_id\tleft_orientation\tright_read_id\tright_orientation"]
        for left, right in truth_edges:
            if (left, right) not in graph_edge_keys:
                missing_rows.append(
                    f"{left[0]}\t{left[1]:+d}\t{right[0]}\t{right[1]:+d}"
                )
        write_report(truth_missing_path, missing_rows)

    isolated = [node for node in nodes if not adjacency[node] or not reverse[node]]
    zero_out = [node for node in nodes if not adjacency[node]]
    zero_in = [node for node in nodes if not reverse[node]]
    out_degrees = [len(adjacency[node]) for node in nodes]
    in_degrees = [len(reverse[node]) for node in nodes]
    lines = [
        "== phi174 cycle742 graph audit ==",
        "graph_input: reads only",
        "edge_preprocessing: minimap2_paf_direct_no_dp",
        f"reads: {len(reads)}",
        f"read_length_mean: {statistics.mean(len(read.seq) for read in reads):.6f}",
        f"read_length_median: {statistics.median(len(read.seq) for read in reads):.3f}",
        f"read_length_min: {min(len(read.seq) for read in reads)}",
        f"read_length_max: {max(len(read.seq) for read in reads)}",
        f"min_overlap: {args.min_overlap}",
        f"candidate_records: {len(result.candidates)}",
        f"paf_edges_before_orientation_filter: {len(result.edges)}",
        f"full_coverage_evidence: {len(finder.last_full_coverage_evidence)}",
        f"full_coverage_reads_removed: {len(graph.full_coverage_read_ids)}",
        "full_coverage_removed_read_ids: " + ", ".join(
            sorted(graph.full_coverage_read_ids)
        ),
        f"contained_reads_removed: {len(graph.contained_read_ids)}",
        f"low_quality_reads_removed: {len(graph.low_quality_read_ids)}",
        "low_quality_removed_read_ids: " + ", ".join(
            sorted(graph.low_quality_read_ids)
        ),
        f"retained_physical_reads: {len(graph.reads)}",
        f"excluded_input_reads: {len(reads) - len(graph.reads)}",
        f"single_orientation_edges: {len(graph.edges)}",
        f"orientation_constraints: {graph.orientation_constraint_count}",
        f"orientation_constraint_conflicts: {graph.orientation_constraint_conflict_count}",
        f"orientation_constraint_components: {graph.orientation_constraint_component_count}",
        f"minimap2_filter_counts: {dict(finder.last_filter_counts)}",
        f"graph_nodes: {len(nodes)}",
        f"unique_directed_edges: {sum(len(neighbors) for neighbors in adjacency.values())}",
        f"out_degree_min_mean_median_max: {min(out_degrees)}/{statistics.mean(out_degrees):.6f}/{statistics.median(out_degrees):.3f}/{max(out_degrees)}",
        f"in_degree_min_mean_median_max: {min(in_degrees)}/{statistics.mean(in_degrees):.6f}/{statistics.median(in_degrees):.3f}/{max(in_degrees)}",
        f"weak_components: {len(weak)}",
        f"largest_weak_component: {max((len(component) for component in weak), default=0)}",
        f"strong_components: {len(strong)}",
        f"largest_strong_component: {max((len(component) for component in strong), default=0)}",
        f"all_points_weakly_connected: {len(weak) == 1}",
        f"all_points_strongly_connected: {len(strong) == 1}",
        f"zero_out_degree_nodes: {len(zero_out)}",
        f"zero_in_degree_nodes: {len(zero_in)}",
        f"nodes_missing_in_or_out_edge: {len(isolated)}",
        f"hamilton_cycle: {hamilton.status}",
    ]
    if hamilton.cycle:
        lines.append(f"hamilton_cycle_nodes: {len(hamilton.cycle) - 1}")
        lines.extend([
            f"hamilton_cycle_edge_overlap_mean: {statistics.mean(edge.overlap_len for edge in cycle_edges):.6f}",
            f"hamilton_cycle_edge_overlap_min: {min(edge.overlap_len for edge in cycle_edges)}",
            f"hamilton_cycle_edge_overlap_max: {max(edge.overlap_len for edge in cycle_edges)}",
            f"hamilton_cycle_edge_identity_mean: {statistics.mean(edge.identity for edge in cycle_edges):.8f}",
            f"hamilton_cycle_tsv: {cycle_path}",
        ])
    if reference_report is not None:
        lines.extend([
            "reference_used_for_graph: no",
            "reference_used_for_hamilton_search: no",
            f"hamilton_reference_mapped_reads: {reference_report.mapped_predicted_reads}/{reference_report.total_predicted_reads}",
            f"hamilton_reference_coverage_bp: {reference_report.reference_coverage_bp}/{evaluator.reference_length}",
            f"hamilton_orientation_accuracy: {reference_report.orientation_accuracy}",
            f"hamilton_forward_edge_accuracy: {reference_report.forward_edge_accuracy:.8f}",
            f"hamilton_reverse_edge_accuracy: {reference_report.reverse_edge_accuracy:.8f}",
            f"reference_truth_edges_present_in_graph: {truth_graph_edges_present}/{truth_graph_edge_count}",
            f"reference_truth_edge_recall: {truth_graph_edges_present / truth_graph_edge_count:.8f}",
            f"reference_truth_missing_edges_tsv: {truth_missing_path}",
            f"hamilton_exact_forward_cycle: {reference_report.exact_forward_cycle}",
            f"hamilton_exact_reverse_equivalent_cycle: {reference_report.exact_reverse_equivalent_cycle}",
        ])
    if zero_out:
        lines.append("zero_out_degree_ids: " + ", ".join(f"{rid}{'+' if ori > 0 else '-'}" for rid, ori in zero_out))
    if zero_in:
        lines.append("zero_in_degree_ids: " + ", ".join(f"{rid}{'+' if ori > 0 else '-'}" for rid, ori in zero_in))
    report_path = OUTPUT_DIR / "cycle742_graph_audit.txt"
    write_report(report_path, lines)
    print("\n".join(lines))
    print(f"audit_report: {report_path}")
    if args.render:
        full_dot_path, dot_path, svg_path, png_path = render_graph(
            graph.reads,
            graph.edges,
            orientation_by_read,
        )
        print(f"graph_full_dot: {full_dot_path}")
        print(f"graph_dot: {dot_path}")
        print(f"graph_svg: {svg_path}")
        print(f"graph_png: {png_path}")


if __name__ == "__main__":
    main()
