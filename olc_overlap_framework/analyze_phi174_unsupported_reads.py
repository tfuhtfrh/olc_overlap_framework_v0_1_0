"""Reference-only audit of reads lacking one-side graph support."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from demo_phi174_common import DATA_DIR, run_reads_only_pipeline
from olc_pipeline.graph_builder import build_read_only_graph, select_one_orientation_per_read
from olc_pipeline.io_utils import read_fastq
from olc_pipeline.reference_evaluator import CircularReferenceEvaluator, read_fasta_sequence


DATASET = DATA_DIR / "SRR27862880_phiX174_OLC_cycle742_pilot"
OUTPUT_DIR = Path(__file__).resolve().parent / "debug" / "phi174_cycle742_graph"
Node = tuple[str, int]
PairKey = tuple[str, str]


@dataclass
class PafPairAudit:
    record_count: int = 0
    max_alignment_block: int = 0
    max_identity: float = 0.0
    best_alignment_block: int = 0
    best_strand: str = ""
    best_coordinates: str = ""
    has_min_overlap: bool = False
    has_min_identity: bool = False
    has_error_hint: bool = False
    has_dovetail_geometry: bool = False


def _pair_key(first_id: str, second_id: str) -> PairKey:
    return tuple(sorted((first_id, second_id)))


def _edge_key(edge) -> tuple[Node, Node]:
    return (
        (edge.left_id, edge.left_orientation),
        (edge.right_id, edge.right_orientation),
    )


def _load_paf_pair_audits(path: Path, finder) -> dict[PairKey, PafPairAudit]:
    audits: dict[PairKey, PafPairAudit] = {}
    config = finder.config
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 12 or fields[0] == fields[5]:
                continue
            q_id = fields[0]
            q_len = int(fields[1])
            q_st = int(fields[2])
            q_en = int(fields[3])
            strand = fields[4]
            t_id = fields[5]
            t_len = int(fields[6])
            t_st = int(fields[7])
            t_en = int(fields[8])
            n_match = int(fields[9])
            aln_block_len = int(fields[10])
            mapq = int(fields[11])
            if aln_block_len <= 0:
                continue
            identity = n_match / aln_block_len
            audit = audits.setdefault(_pair_key(q_id, t_id), PafPairAudit())
            audit.record_count += 1
            audit.max_alignment_block = max(audit.max_alignment_block, aln_block_len)
            rank = (identity, aln_block_len)
            if rank > (audit.max_identity, audit.best_alignment_block):
                audit.max_identity = identity
                audit.best_alignment_block = aln_block_len
                audit.best_strand = strand
                audit.best_coordinates = (
                    f"q:{q_st}-{q_en}/{q_len};t:{t_st}-{t_en}/{t_len}"
                )

            passes_overlap = aln_block_len >= config.min_overlap
            audit.has_min_overlap |= passes_overlap
            if not passes_overlap or mapq < config.min_mapq:
                continue
            passes_identity = (
                config.min_paf_identity is None
                or identity >= config.min_paf_identity
            )
            audit.has_min_identity |= passes_identity
            if not passes_identity:
                continue
            tags = finder._parse_optional_tags(fields[12:])
            passes_error_hint = (
                finder._error_rate_hint(n_match, aln_block_len, tags)
                <= config.max_error_rate_hint
            )
            audit.has_error_hint |= passes_error_hint
            if not passes_error_hint:
                continue
            candidate = finder._paf_to_directed_candidate(
                q_id=q_id,
                q_len=q_len,
                q_st=q_st,
                q_en=q_en,
                strand=strand,
                t_id=t_id,
                t_len=t_len,
                t_st=t_st,
                t_en=t_en,
                n_match=n_match,
                aln_block_len=aln_block_len,
                mapq=mapq,
            )
            audit.has_dovetail_geometry |= candidate is not None
    return audits


def _paf_stage_status(audit: PafPairAudit | None) -> str:
    if audit is None:
        return "no_paf_record"
    if not audit.has_min_overlap:
        return "below_min_overlap"
    if not audit.has_min_identity:
        return "below_identity_0.990"
    if not audit.has_error_hint:
        return "above_error_hint_0.010"
    if not audit.has_dovetail_geometry:
        return "no_positive_shift_dovetail"
    return "directed_candidate_geometry"


def _node_label(node: Node) -> str:
    return f"{node[0]}{'+' if node[1] > 0 else '-'}"


def _edge_label(edge) -> str:
    if edge is None:
        return ""
    return f"overlap={edge.overlap_len};identity={edge.identity:.6f};error={edge.error_rate:.6f}"


def _build_support(selection):
    incoming: dict[str, set[Node]] = {read.rid: set() for read in selection.reads}
    outgoing: dict[str, set[Node]] = {read.rid: set() for read in selection.reads}
    edge_by_nodes = {}
    for edge in selection.edges:
        left = (edge.left_id, edge.left_orientation)
        right = (edge.right_id, edge.right_orientation)
        if left[0] not in incoming or right[0] not in incoming:
            continue
        outgoing[left[0]].add(right)
        incoming[right[0]].add(left)
        edge_by_nodes.setdefault((left, right), edge)
    return incoming, outgoing, edge_by_nodes


def _build_physical_support(edges, read_ids):
    """Count directed physical-read support before orientation projection."""
    read_id_set = set(read_ids)
    incoming: dict[str, set[str]] = {read_id: set() for read_id in read_ids}
    outgoing: dict[str, set[str]] = {read_id: set() for read_id in read_ids}
    for edge in edges:
        if edge.left_id not in read_id_set or edge.right_id not in read_id_set:
            continue
        outgoing[edge.left_id].add(edge.right_id)
        incoming[edge.right_id].add(edge.left_id)
    return incoming, outgoing


def _support_status(read_id: str, incoming, outgoing) -> str:
    has_in = bool(incoming[read_id])
    has_out = bool(outgoing[read_id])
    if has_in and has_out:
        return "supported"
    if not has_in and not has_out:
        return "no_in_or_out_support"
    if not has_in:
        return "no_in_support"
    return "no_out_support"


def _write_reference_placements(path: Path, placements) -> None:
    rows = sorted(placements.values(), key=lambda item: (item.reference_start, item.read_id))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "read_id", "reference_orientation", "reference_start", "reference_end",
            "alignment_length", "identity", "error_rate", "mapq",
        ])
        for placement in rows:
            writer.writerow([
                placement.read_id,
                f"{placement.orientation:+d}",
                placement.reference_start,
                placement.reference_end,
                placement.alignment_length,
                f"{placement.identity:.8f}",
                f"{1.0 - placement.identity:.8f}",
                placement.mapq,
            ])


def _write_unsupported_reads(
    path: Path,
    flagged,
    placements,
    incoming,
    outgoing,
    raw_incoming,
    raw_outgoing,
    pre_filter_incoming,
    pre_filter_outgoing,
    orientations,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "read_id", "read_length", "support_status", "selected_orientation", "reference_orientation",
            "reference_start", "reference_end", "alignment_length", "identity", "mapq",
            "incoming_support_count", "outgoing_support_count", "incoming_ids", "outgoing_ids",
            "raw_physical_incoming_count", "raw_physical_outgoing_count",
            "pre_full_filter_incoming_count", "pre_full_filter_outgoing_count",
        ])
        for read in sorted(flagged, key=lambda item: item.rid):
            placement = placements.get(read.rid)
            writer.writerow([
                read.rid,
                len(read.seq),
                _support_status(read.rid, incoming, outgoing),
                f"{orientations[read.rid]:+d}",
                f"{placement.orientation:+d}" if placement else "",
                placement.reference_start if placement else "",
                placement.reference_end if placement else "",
                placement.alignment_length if placement else "",
                f"{placement.identity:.8f}" if placement else "",
                placement.mapq if placement else "",
                len(incoming[read.rid]),
                len(outgoing[read.rid]),
                "|".join(sorted(_node_label(node) for node in incoming[read.rid])),
                "|".join(sorted(_node_label(node) for node in outgoing[read.rid])),
                len(raw_incoming[read.rid]),
                len(raw_outgoing[read.rid]),
                len(pre_filter_incoming[read.rid]),
                len(pre_filter_outgoing[read.rid]),
            ])


def _write_neighbors(
    path: Path,
    flagged,
    all_placements,
    retained_placements,
    incoming,
    outgoing,
    orientations,
    edge_by_nodes,
    removal_reason_by_read,
    paf_audits,
    reference_length: int,
    neighbor_count: int,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "unsupported_read_id", "support_status", "unsupported_reference_start", "neighbor_scope",
            "neighbor_side", "neighbor_rank", "start_distance_bp", "neighbor_id",
            "neighbor_reference_orientation", "neighbor_reference_start", "neighbor_reference_end",
            "neighbor_identity", "neighbor_graph_status", "neighbor_removal_reason",
            "graph_edge_flag_to_neighbor", "graph_edge_neighbor_to_flag",
            "paf_stage", "paf_record_count", "paf_max_identity", "paf_max_alignment_block",
            "paf_best_alignment_block", "paf_best_strand", "paf_best_coordinates",
        ])
        scopes = (
            ("all_mapped", all_placements),
            ("retained_graph", retained_placements),
        )
        for scope_name, placements in scopes:
            ordered = sorted(
                placements.values(),
                key=lambda item: (item.reference_start, item.read_id),
            )
            index_by_id = {
                placement.read_id: index for index, placement in enumerate(ordered)
            }
            for read in sorted(flagged, key=lambda item: item.rid):
                placement = placements.get(read.rid)
                if placement is None or read.rid not in index_by_id:
                    continue
                flag_node = (read.rid, orientations[read.rid])
                index = index_by_id[read.rid]
                seen: set[str] = set()
                neighbors = []
                for side, sign in (("previous", -1), ("next", +1)):
                    for rank in range(1, neighbor_count + 1):
                        neighbor_placement = ordered[(index + sign * rank) % len(ordered)]
                        if neighbor_placement.read_id in seen:
                            continue
                        seen.add(neighbor_placement.read_id)
                        if side == "previous":
                            distance = (
                                placement.reference_start
                                - neighbor_placement.reference_start
                            ) % reference_length
                        else:
                            distance = (
                                neighbor_placement.reference_start
                                - placement.reference_start
                            ) % reference_length
                        neighbors.append((side, rank, distance, neighbor_placement))
                for side, rank, distance, neighbor_placement in neighbors:
                    neighbor_orientation = orientations.get(neighbor_placement.read_id)
                    neighbor_node = (
                        (neighbor_placement.read_id, neighbor_orientation)
                        if neighbor_orientation is not None
                        else None
                    )
                    audit = paf_audits.get(
                        _pair_key(read.rid, neighbor_placement.read_id)
                    )
                    writer.writerow([
                        read.rid,
                        _support_status(read.rid, incoming, outgoing),
                        placement.reference_start,
                        scope_name,
                        side,
                        rank,
                        distance,
                        neighbor_placement.read_id,
                        f"{neighbor_placement.orientation:+d}",
                        neighbor_placement.reference_start,
                        neighbor_placement.reference_end,
                        f"{neighbor_placement.identity:.8f}",
                        "retained" if neighbor_orientation is not None else "removed",
                        removal_reason_by_read.get(neighbor_placement.read_id, ""),
                        _edge_label(
                            edge_by_nodes.get((flag_node, neighbor_node))
                            if neighbor_node is not None else None
                        ),
                        _edge_label(
                            edge_by_nodes.get((neighbor_node, flag_node))
                            if neighbor_node is not None else None
                        ),
                        _paf_stage_status(audit),
                        audit.record_count if audit else 0,
                        f"{audit.max_identity:.8f}" if audit else "",
                        audit.max_alignment_block if audit else "",
                        audit.best_alignment_block if audit else "",
                        audit.best_strand if audit else "",
                        audit.best_coordinates if audit else "",
                    ])


def _neighbor_edge_summary(flagged, placements, orientations, edge_by_nodes, neighbor_count):
    ordered = sorted(placements.values(), key=lambda item: (item.reference_start, item.read_id))
    index_by_id = {placement.read_id: index for index, placement in enumerate(ordered)}
    summary = {}
    for read in flagged:
        placement = placements.get(read.rid)
        if placement is None or read.rid not in index_by_id:
            continue
        flag_node = (read.rid, orientations[read.rid])
        index = index_by_id[read.rid]
        found = False
        observations = 0
        for sign in (-1, +1):
            for rank in range(1, neighbor_count + 1):
                neighbor = ordered[(index + sign * rank) % len(ordered)]
                if neighbor.read_id == read.rid:
                    continue
                neighbor_node = (neighbor.read_id, orientations[neighbor.read_id])
                observations += 1
                if (flag_node, neighbor_node) in edge_by_nodes or (neighbor_node, flag_node) in edge_by_nodes:
                    found = True
        summary[read.rid] = (found, observations)
    return summary


def _desired_edge_status(
    flag_node: Node,
    neighbor_node: Node,
    missing_side: str,
    paf_audit: PafPairAudit | None,
    candidate_edge_keys,
    preprocessed_edge_keys,
    final_edge_keys,
) -> str:
    desired = (
        (neighbor_node, flag_node)
        if missing_side == "incoming"
        else (flag_node, neighbor_node)
    )
    if desired in final_edge_keys:
        return "unexpected_final_edge"
    if desired in preprocessed_edge_keys:
        return "paf_edge_removed_after_projection"
    if desired in candidate_edge_keys:
        return "candidate_deduplicated"

    pair = _pair_key(flag_node[0], neighbor_node[0])
    if any(_pair_key(left[0], right[0]) == pair for left, right in preprocessed_edge_keys):
        return "paf_only_opposite_or_other_orientation"
    if any(_pair_key(left[0], right[0]) == pair for left, right in candidate_edge_keys):
        return "candidate_only_opposite_or_other_orientation"
    return _paf_stage_status(paf_audit)


def _primary_missing_side_cause(statuses: list[str]) -> str:
    priority = (
        "unexpected_final_edge",
        "paf_edge_removed_after_projection",
        "candidate_deduplicated",
        "paf_only_opposite_or_other_orientation",
        "candidate_only_opposite_or_other_orientation",
        "directed_candidate_geometry",
        "no_positive_shift_dovetail",
        "above_error_hint_0.010",
        "below_identity_0.990",
        "below_min_overlap",
        "no_paf_record",
    )
    for status in priority:
        if status in statuses:
            return status
    return "no_nearby_retained_read"


def _write_missing_side_analysis(
    path: Path,
    flagged,
    placements,
    orientations,
    incoming,
    outgoing,
    paf_audits,
    candidate_edge_keys,
    preprocessed_edge_keys,
    final_edge_keys,
    reference_length: int,
    neighbor_count: int,
) -> Counter[str]:
    ordered = sorted(placements.values(), key=lambda item: (item.reference_start, item.read_id))
    index_by_id = {placement.read_id: index for index, placement in enumerate(ordered)}
    cause_counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "read_id", "support_status", "missing_side", "selected_reference_phase",
            "reference_start", "reference_end", "reference_identity", "expected_neighbor_side",
            "primary_cause", "nearby_retained_ids", "nearby_stage_statuses",
            "nearby_paf_max_identities", "nearby_start_distances_bp",
        ])
        for read in sorted(flagged, key=lambda item: item.rid):
            status = _support_status(read.rid, incoming, outgoing)
            missing_sides = []
            if not incoming[read.rid]:
                missing_sides.append("incoming")
            if not outgoing[read.rid]:
                missing_sides.append("outgoing")
            placement = placements.get(read.rid)
            if placement is None or read.rid not in index_by_id:
                for missing_side in missing_sides:
                    cause_counts["reference_unmapped"] += 1
                    writer.writerow([
                        read.rid, status, missing_side, "", "", "", "", "",
                        "reference_unmapped", "", "", "", "",
                    ])
                continue

            phase = orientations[read.rid] * placement.orientation
            index = index_by_id[read.rid]
            flag_node = (read.rid, orientations[read.rid])
            for missing_side in missing_sides:
                if missing_side == "incoming":
                    expected_side = "previous" if phase > 0 else "next"
                else:
                    expected_side = "next" if phase > 0 else "previous"
                sign = -1 if expected_side == "previous" else +1
                neighbor_rows = []
                for rank in range(1, neighbor_count + 1):
                    neighbor = ordered[(index + sign * rank) % len(ordered)]
                    if neighbor.read_id == read.rid:
                        continue
                    distance = (
                        (placement.reference_start - neighbor.reference_start)
                        if expected_side == "previous"
                        else (neighbor.reference_start - placement.reference_start)
                    ) % reference_length
                    neighbor_node = (neighbor.read_id, orientations[neighbor.read_id])
                    audit = paf_audits.get(_pair_key(read.rid, neighbor.read_id))
                    edge_status = _desired_edge_status(
                        flag_node,
                        neighbor_node,
                        missing_side,
                        audit,
                        candidate_edge_keys,
                        preprocessed_edge_keys,
                        final_edge_keys,
                    )
                    neighbor_rows.append((neighbor, distance, edge_status, audit))
                stages = [row[2] for row in neighbor_rows]
                primary_cause = _primary_missing_side_cause(stages)
                cause_counts[primary_cause] += 1
                writer.writerow([
                    read.rid,
                    status,
                    missing_side,
                    f"{phase:+d}",
                    placement.reference_start,
                    placement.reference_end,
                    f"{placement.identity:.8f}",
                    expected_side,
                    primary_cause,
                    "|".join(row[0].read_id for row in neighbor_rows),
                    "|".join(row[2] for row in neighbor_rows),
                    "|".join(
                        f"{row[3].max_identity:.8f}" if row[3] else ""
                        for row in neighbor_rows
                    ),
                    "|".join(str(row[1]) for row in neighbor_rows),
                ])
    return cause_counts


def _write_unmapped_audit(
    path: Path,
    flagged,
    placements,
    paf_audits,
    candidate_edge_keys,
    preprocessed_edge_keys,
    incoming,
    outgoing,
) -> int:
    unmapped = [read for read in flagged if read.rid not in placements]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "read_id", "read_length", "support_status", "incident_paf_pairs",
            "accepted_candidate_edges", "preprocessed_paf_edges", "best_partner_ids",
            "best_pair_stages", "best_pair_identities", "best_pair_alignment_blocks",
        ])
        for read in sorted(unmapped, key=lambda item: item.rid):
            incident = []
            for pair, audit in paf_audits.items():
                if read.rid not in pair:
                    continue
                partner = pair[1] if pair[0] == read.rid else pair[0]
                incident.append((partner, audit))
            incident.sort(
                key=lambda item: (
                    item[1].has_dovetail_geometry,
                    item[1].has_min_identity,
                    item[1].max_alignment_block,
                    item[1].max_identity,
                ),
                reverse=True,
            )
            best = incident[:10]
            candidate_count = sum(
                read.rid in (left[0], right[0])
                for left, right in candidate_edge_keys
            )
            preprocessed_count = sum(
                read.rid in (left[0], right[0])
                for left, right in preprocessed_edge_keys
            )
            writer.writerow([
                read.rid,
                len(read.seq),
                _support_status(read.rid, incoming, outgoing),
                len(incident),
                candidate_count,
                preprocessed_count,
                "|".join(item[0] for item in best),
                "|".join(_paf_stage_status(item[1]) for item in best),
                "|".join(f"{item[1].max_identity:.8f}" for item in best),
                "|".join(str(item[1].max_alignment_block) for item in best),
            ])
    return len(unmapped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimap2-bin", default="minimap2")
    parser.add_argument("--reference-preset", default="sr")
    parser.add_argument("--neighbor-count", type=int, default=3)
    parser.add_argument("--min-overlap", type=int, default=80)
    args = parser.parse_args()
    if args.neighbor_count < 1:
        raise ValueError("neighbor-count must be positive")

    reads = read_fastq(DATASET / "SRR27862880.stride500.unique.fastq.gz")
    finder, result = run_reads_only_pipeline(
        reads,
        minimap2_bin=args.minimap2_bin,
        min_overlap=args.min_overlap,
        min_paf_identity=0.990,
        max_error_rate_hint=0.01,
        max_error_rate=0.01,
        overhang_tolerance=20,
        extra_args=("-k", "15", "-w", "5", "-m", "40", "-n", "2", "-X", "--secondary=yes", "-N", "1000", "-c", "--eqx"),
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
        apply_low_quality_filter=False,
        full_coverage_evidence=finder.last_full_coverage_evidence,
        apply_full_coverage_filter=True,
    )
    pre_filter_selection = select_one_orientation_per_read(
        reads,
        result.edges,
        score_mode="overlap_len_power2",
    )
    pre_filter_incoming, pre_filter_outgoing, _ = _build_support(
        pre_filter_selection
    )
    raw_incoming, raw_outgoing = _build_physical_support(
        result.edges,
        [read.rid for read in graph.reads],
    )
    orientations = graph.orientation_by_read
    incoming, outgoing, edge_by_nodes = _build_support(graph)
    flagged = [
        read for read in graph.reads
        if _support_status(read.rid, incoming, outgoing) != "supported"
    ]

    reference = CircularReferenceEvaluator(
        read_fasta_sequence(DATASET / "NC_001422.1.fasta"),
        minimap2_bin=args.minimap2_bin,
        preset=args.reference_preset,
    )
    placements = reference.map_reads(reads)
    retained_placements = {
        read_id: placements[read_id]
        for read_id in orientations
        if read_id in placements
    }
    paf_audits = _load_paf_pair_audits(DATASET / "all_vs_all.exact.paf", finder)
    candidate_edge_keys = {_edge_key(edge) for edge in result.candidates}
    preprocessed_edge_keys = {_edge_key(edge) for edge in result.edges}
    final_edge_keys = {_edge_key(edge) for edge in graph.edges}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    placements_path = OUTPUT_DIR / "cycle742_reference_placements.tsv"
    unsupported_path = OUTPUT_DIR / "cycle742_unsupported_reads.tsv"
    neighbors_path = OUTPUT_DIR / "cycle742_unsupported_read_neighbors.tsv"
    missing_side_path = OUTPUT_DIR / "cycle742_unsupported_missing_sides.tsv"
    unmapped_path = OUTPUT_DIR / "cycle742_unsupported_unmapped_reads.tsv"
    summary_path = OUTPUT_DIR / "cycle742_unsupported_read_analysis.txt"
    _write_reference_placements(placements_path, placements)
    _write_unsupported_reads(
        unsupported_path,
        flagged,
        placements,
        incoming,
        outgoing,
        raw_incoming,
        raw_outgoing,
        pre_filter_incoming,
        pre_filter_outgoing,
        orientations,
    )
    _write_neighbors(
        neighbors_path,
        flagged,
        placements,
        retained_placements,
        incoming,
        outgoing,
        orientations,
        edge_by_nodes,
        graph.removal_reason_by_read,
        paf_audits,
        reference.reference_length,
        args.neighbor_count,
    )
    missing_side_cause_counts = _write_missing_side_analysis(
        missing_side_path,
        flagged,
        retained_placements,
        orientations,
        incoming,
        outgoing,
        paf_audits,
        candidate_edge_keys,
        preprocessed_edge_keys,
        final_edge_keys,
        reference.reference_length,
        args.neighbor_count,
    )
    unsupported_unmapped_count = _write_unmapped_audit(
        unmapped_path,
        flagged,
        placements,
        paf_audits,
        candidate_edge_keys,
        preprocessed_edge_keys,
        incoming,
        outgoing,
    )

    status_counts = {}
    for read in flagged:
        status = _support_status(read.rid, incoming, outgoing)
        status_counts[status] = status_counts.get(status, 0) + 1
    flagged_placements = [placements[read.rid] for read in flagged if read.rid in placements]
    orientation_direct = [
        orientations[read.rid] == placements[read.rid].orientation
        for read in flagged
        if read.rid in placements
    ]
    neighbor_summary = _neighbor_edge_summary(
        flagged,
        retained_placements,
        orientations,
        edge_by_nodes,
        args.neighbor_count,
    )
    flagged_with_nearby_edge = sum(found for found, _ in neighbor_summary.values())
    flagged_mapped_by_status = {}
    raw_status_counts = {}
    lost_in_support = 0
    lost_out_support = 0
    full_filter_lost_in_support = 0
    full_filter_lost_out_support = 0
    for read in flagged:
        if read.rid in placements:
            status = _support_status(read.rid, incoming, outgoing)
            flagged_mapped_by_status[status] = flagged_mapped_by_status.get(status, 0) + 1
        raw_status = _support_status(read.rid, raw_incoming, raw_outgoing)
        raw_status_counts[raw_status] = raw_status_counts.get(raw_status, 0) + 1
        if not incoming[read.rid] and raw_incoming[read.rid]:
            lost_in_support += 1
        if not outgoing[read.rid] and raw_outgoing[read.rid]:
            lost_out_support += 1
        if not incoming[read.rid] and pre_filter_incoming[read.rid]:
            full_filter_lost_in_support += 1
        if not outgoing[read.rid] and pre_filter_outgoing[read.rid]:
            full_filter_lost_out_support += 1
    raw_no_support_reads = [
        read for read in flagged
        if _support_status(read.rid, raw_incoming, raw_outgoing) == "no_in_or_out_support"
    ]
    raw_no_support_placements = [
        placements[read.rid]
        for read in raw_no_support_reads
        if read.rid in placements
    ]
    flagged_ids = {read.rid for read in flagged}
    retained_graph_placements = [
        placements[read.rid]
        for read in graph.reads
        if read.rid in placements
    ]
    supported_only_placements = [
        placements[read.rid]
        for read in graph.reads
        if read.rid not in flagged_ids and read.rid in placements
    ]
    read_lengths = {read.rid: len(read.seq) for read in reads}
    lines = [
        "== phi174 cycle742 unsupported-read reference audit ==",
        "audit_scope: before_approved_one_pass_low_support_deletion",
        "current_action_for_flagged_reads: removed_by_cycle742_demo",
        "reference_used_for_graph: no",
        "edge_preprocessing: minimap2_paf_direct_no_dp",
        f"reads: {len(reads)}",
        f"read_length_mean: {mean(read_lengths.values()):.6f}",
        f"read_length_min: {min(read_lengths.values())}",
        f"read_length_max: {max(read_lengths.values())}",
        f"mapped_reads: {len(placements)}/{len(reads)}",
        f"min_overlap: {args.min_overlap}",
        "candidate_identity_threshold: 0.990",
        "full_coverage_identity_threshold: 0.990",
        "full_coverage_terminal_tolerance: 0",
        f"selected_one_side_edges: {len(graph.edges)}",
        f"orientation_constraints: {graph.orientation_constraint_count}",
        f"orientation_constraint_conflicts: {graph.orientation_constraint_conflict_count}",
        f"full_coverage_evidence: {len(finder.last_full_coverage_evidence)}",
        f"full_coverage_reads_removed: {len(graph.full_coverage_read_ids)}",
        f"containment_evidence: {len(finder.last_containment_evidence)}",
        f"contained_reads_removed: {len(graph.contained_read_ids)}",
        f"unsupported_reads: {len(flagged)}",
        f"unsupported_status_counts: {dict(sorted(status_counts.items()))}",
        f"unsupported_mapped_reads: {len(flagged_placements)}/{len(flagged)}",
        f"unsupported_mapped_status_counts: {dict(sorted(flagged_mapped_by_status.items()))}",
        f"unsupported_raw_physical_status_counts: {dict(sorted(raw_status_counts.items()))}",
        f"unsupported_selected_zero_in_but_raw_in_exists: {lost_in_support}",
        f"unsupported_selected_zero_out_but_raw_out_exists: {lost_out_support}",
        f"unsupported_final_zero_in_but_pre_full_filter_in_exists: {full_filter_lost_in_support}",
        f"unsupported_final_zero_out_but_pre_full_filter_out_exists: {full_filter_lost_out_support}",
        f"raw_no_support_mapped_reads: {len(raw_no_support_placements)}/{len(raw_no_support_reads)}",
        f"raw_no_support_unmapped_ids: {','.join(read.rid for read in raw_no_support_reads if read.rid not in placements)}",
        f"raw_no_support_read_length_mean: {mean(read_lengths[read.rid] for read in raw_no_support_reads):.3f}",
        (
            f"raw_no_support_mapped_identity_mean: {mean(item.identity for item in raw_no_support_placements):.8f}"
            f";min={min(item.identity for item in raw_no_support_placements):.8f}"
            f";max={max(item.identity for item in raw_no_support_placements):.8f}"
        ) if raw_no_support_placements else "raw_no_support_mapped_identity_mean: n/a",
        f"unsupported_read_length_mean: {mean(read_lengths[read.rid] for read in flagged):.3f}",
        f"unsupported_mapped_identity_mean: {mean(item.identity for item in flagged_placements):.8f}" if flagged_placements else "unsupported_mapped_identity_mean: n/a",
        f"unsupported_orientation_agreement_direct: {mean(orientation_direct):.6f}" if orientation_direct else "unsupported_orientation_agreement_direct: n/a",
        f"unsupported_with_edge_to_one_of_{args.neighbor_count * 2}_reference_neighbors: {flagged_with_nearby_edge}/{len(neighbor_summary)}",
        f"missing_side_count: {sum(missing_side_cause_counts.values())}",
        f"missing_side_primary_cause_counts: {dict(sorted(missing_side_cause_counts.items()))}",
        f"unsupported_reference_unmapped_reads: {unsupported_unmapped_count}",
        f"retained_graph_mapped_reads: {len(retained_graph_placements)}/{len(graph.reads)}",
        f"retained_graph_reference_coverage_bp: {reference._coverage_bp(retained_graph_placements)}/{reference.reference_length}",
        f"supported_only_reads: {len(graph.reads) - len(flagged)}",
        f"supported_only_mapped_reads: {len(supported_only_placements)}/{len(graph.reads) - len(flagged)}",
        f"supported_only_reference_coverage_bp: {reference._coverage_bp(supported_only_placements)}/{reference.reference_length}",
        f"reference_coverage_bp: {reference._coverage_bp(placements.values())}/{reference.reference_length}",
        f"reference_preset: {args.reference_preset}",
        f"placements_tsv: {placements_path}",
        f"unsupported_reads_tsv: {unsupported_path}",
        f"neighbors_tsv: {neighbors_path}",
        f"missing_sides_tsv: {missing_side_path}",
        f"unmapped_reads_tsv: {unmapped_path}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
