"""Find an A->B, C->B, D->A orientation-projection example on phi174."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from demo_phi174_common import DATA_DIR, run_reads_only_pipeline
from olc_pipeline.graph_builder import select_one_orientation_per_read
from olc_pipeline.io_utils import read_fastq
from olc_pipeline.reference_evaluator import CircularReferenceEvaluator, read_fasta_sequence


DATASET = DATA_DIR / "SRR27862880_phiX174_OLC_cycle742_pilot"
OUTPUT_PATH = Path(__file__).resolve().parent / "debug" / "phi174_cycle742_graph" / "cycle742_orientation_projection_abcd.txt"


def _edge_key(edge):
    return (edge.left_id, edge.left_orientation, edge.right_id, edge.right_orientation)


def _edge_score(edge):
    return (edge.overlap_len, edge.identity, -edge.error_rate)


def _support(edges, read_ids):
    incoming = {read_id: [] for read_id in read_ids}
    outgoing = {read_id: [] for read_id in read_ids}
    for edge in edges:
        if edge.left_id not in outgoing or edge.right_id not in incoming:
            continue
        outgoing[edge.left_id].append(edge)
        incoming[edge.right_id].append(edge)
    return incoming, outgoing


def _cyclic_span(starts, reference_length):
    values = sorted(starts)
    if len(values) < 2:
        return 0
    gaps = [
        (values[(index + 1) % len(values)] - values[index]) % reference_length
        for index in range(len(values))
    ]
    return reference_length - max(gaps)


def _placement_line(read_id, placements):
    placement = placements.get(read_id)
    if placement is None:
        return f"{read_id}: UNMAPPED"
    return (
        f"{read_id}: ref={placement.reference_start}-{placement.reference_end}"
        f" orient={placement.orientation:+d}"
        f" aln={placement.alignment_length}"
        f" identity={placement.identity:.8f}"
    )


def _edge_line(label, edge, selected_orientations=None):
    if edge is None:
        return f"{label}: NONE"
    selected_text = ""
    if selected_orientations is not None:
        selected_text = (
            f" selected_endpoints=({selected_orientations[edge.left_id]:+d},"
            f"{selected_orientations[edge.right_id]:+d})"
        )
    return (
        f"{label}: {edge.left_id}{edge.left_orientation:+d}"
        f" -> {edge.right_id}{edge.right_orientation:+d}"
        f" overlap={edge.overlap_len} identity={edge.identity:.8f}"
        f" error={edge.error_rate:.8f}{selected_text}"
    )


def _reverse_complement(sequence: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return sequence.translate(table)[::-1]


def _sequence_identity(first: str, second: str) -> float:
    if len(first) != len(second) or not first:
        return 0.0
    return sum(left == right for left, right in zip(first, second)) / len(first)


def _candidate_line(candidate) -> str:
    return (
        f"{candidate.left_id}{candidate.left_orientation:+d}"
        f" -> {candidate.right_id}{candidate.right_orientation:+d}"
        f" overlap={candidate.rough_overlap_len}"
        f" shift={candidate.rough_shift}"
    )


def main() -> None:
    reads = read_fastq(DATASET / "SRR27862880.stride500.unique.fastq.gz")
    finder, result = run_reads_only_pipeline(
        reads,
        minimap2_bin="minimap2",
        min_overlap=80,
        min_paf_identity=0.995,
        max_error_rate_hint=0.01,
        max_error_rate=0.01,
        overhang_tolerance=20,
        extra_args=("-k", "15", "-w", "5", "-m", "40", "-n", "2", "-X", "--secondary=yes", "-N", "1000", "-c", "--eqx"),
    )
    read_ids = [read.rid for read in reads]
    read_by_id = {read.rid: read for read in reads}
    selected = select_one_orientation_per_read(reads, result.edges)
    selected_in, selected_out = _support(selected.edges, read_ids)
    raw_in, raw_out = _support(result.edges, read_ids)
    raw_in_by_read = defaultdict(list)
    for edge in result.edges:
        if edge.right_id in read_ids and edge.left_id != edge.right_id:
            raw_in_by_read[edge.right_id].append(edge)

    reference = CircularReferenceEvaluator(
        read_fasta_sequence(DATASET / "NC_001422.1.fasta"),
        minimap2_bin="minimap2",
        preset="sr",
    )
    placements = reference.map_reads(reads)

    candidates = []
    for a_id in read_ids:
        if selected_in[a_id] or not selected_out[a_id] or not raw_in[a_id]:
            continue
        for ab in sorted(selected_out[a_id], key=_edge_score, reverse=True):
            b_id = ab.right_id
            for cb in selected_in[b_id]:
                c_id = cb.left_id
                if c_id in {a_id, b_id}:
                    continue
                for da in sorted(raw_in_by_read[a_id], key=_edge_score, reverse=True):
                    d_id = da.left_id
                    if d_id in {a_id, b_id, c_id}:
                        continue
                    ids = (a_id, b_id, c_id, d_id)
                    mapped = [placements.get(read_id) for read_id in ids]
                    span = _cyclic_span(
                        [placement.reference_start for placement in mapped if placement],
                        reference.reference_length,
                    )
                    missing = sum(placement is None for placement in mapped)
                    candidates.append((
                        missing,
                        span,
                        -min(ab.overlap_len, cb.overlap_len, da.overlap_len),
                        a_id,
                        b_id,
                        c_id,
                        d_id,
                        ab,
                        cb,
                        da,
                    ))

    candidates.sort(key=lambda item: item[:3])
    if not candidates:
        raise RuntimeError("No A->B, C->B, D->A example found")

    best = candidates[0]
    _, span, _, a_id, b_id, c_id, d_id, ab, cb, da = best
    reverse_da = next(
        (
            edge for edge in result.edges
            if edge.left_id == a_id
            and edge.right_id == d_id
            and edge.left_orientation == -da.right_orientation
            and edge.right_orientation == -da.left_orientation
        ),
        None,
    )
    selected_da = next(
        (
            edge for edge in selected.edges
            if edge.left_id == d_id and edge.right_id == a_id
        ),
        None,
    )
    selected_ad = next(
        (
            edge for edge in selected.edges
            if edge.left_id == a_id and edge.right_id == d_id
        ),
        None,
    )
    ac_candidates = [
        candidate for candidate in result.candidates
        if {candidate.left_id, candidate.right_id} == {a_id, c_id}
    ]
    ac_raw_edges = [
        edge for edge in result.edges
        if {edge.left_id, edge.right_id} == {a_id, c_id}
    ]
    ac_selected_edges = [
        edge for edge in selected.edges
        if {edge.left_id, edge.right_id} == {a_id, c_id}
    ]
    a_selected_seq = (
        read_by_id[a_id].seq
        if selected.orientation_by_read[a_id] > 0
        else _reverse_complement(read_by_id[a_id].seq)
    )
    c_selected_seq = (
        read_by_id[c_id].seq
        if selected.orientation_by_read[c_id] > 0
        else _reverse_complement(read_by_id[c_id].seq)
    )

    lines = [
        "== phi174 cycle742 orientation-projection A-B-C-D audit ==",
        "graph_input: reads only",
        f"candidate_records: {len(result.candidates)}",
        f"refined_edges_before_orientation_projection: {len(result.edges)}",
        f"selected_one_side_edges: {len(selected.edges)}",
        f"orientation_constraint_conflicts: {selected.constraint_conflict_count}",
        f"candidate_examples_found: {len(candidates)}",
        f"selected_example_reference_span_bp: {span}",
        "",
        "Selected-side relation:",
        _edge_line("A->B", ab, selected.orientation_by_read),
        _edge_line("C->B", cb, selected.orientation_by_read),
        "",
        "Pre-projection incoming relation:",
        _edge_line("D->A (raw)", da, selected.orientation_by_read),
        _edge_line("reverse-complement counterpart A->D (raw)", reverse_da, selected.orientation_by_read),
        _edge_line("D->A (selected)", selected_da, selected.orientation_by_read),
        _edge_line("A->D (selected)", selected_ad, selected.orientation_by_read),
        "",
        "Reference coordinates:",
        _placement_line(a_id, placements),
        _placement_line(b_id, placements),
        _placement_line(c_id, placements),
        _placement_line(d_id, placements),
        "",
        "A-C same-locus check:",
        f"A-C candidates_after_PAF_filter: {len(ac_candidates)}",
        f"A-C refined_edges_before_projection: {len(ac_raw_edges)}",
        f"A-C selected_edges: {len(ac_selected_edges)}",
        f"A-C selected_oriented_full_length_identity: {_sequence_identity(a_selected_seq, c_selected_seq):.8f}",
        f"A-C selected_oriented_lengths: {len(a_selected_seq)},{len(c_selected_seq)}",
        *[_candidate_line(candidate) for candidate in ac_candidates],
        "",
        "Selected orientation and support counts:",
    ]
    for read_id in (a_id, b_id, c_id, d_id):
        lines.append(
            f"{read_id}: selected={selected.orientation_by_read[read_id]:+d}"
            f" selected_in={len(selected_in[read_id])}"
            f" selected_out={len(selected_out[read_id])}"
            f" raw_in={len(raw_in[read_id])}"
            f" raw_out={len(raw_out[read_id])}"
        )
    lines.extend([
        "",
        "Interpretation:",
        "The raw D->A edge is paired with its reverse-complement A->D edge.",
        "If the selected phase matches D->A, D->A is retained; if it matches the",
        "opposite phase, A->D is retained.  A selected-side zero-in A can therefore",
        "be a direction reversal of raw incoming information, not an information loss.",
        f"report: {OUTPUT_PATH}",
    ])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
