"""
Overlap/read feature helpers for layout Hamiltonians.

The functions here keep sequence-derived quantities out of the QUBO builders so
different feature definitions can be tested without changing solver plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import OverlapEdge, Read


GC_BASES = frozenset({"G", "C", "g", "c"})


@dataclass(frozen=True)
class OverlapComposition:
    """GC composition estimate for one directed overlap edge."""

    overlap_len: float
    gc_count: float
    gc_fraction: float
    source: str


def gc_count(sequence: str) -> int:
    """Count G/C bases in a sequence, ignoring gaps and ambiguous bases."""
    return sum(1 for base in sequence if base in GC_BASES)


def gc_fraction(sequence: str) -> float:
    """Return G/C fraction over non-gap bases."""
    bases = [base for base in sequence if base != "-"]
    if not bases:
        return 0.0
    return gc_count("".join(bases)) / len(bases)


def read_gc_count(read: Read) -> int:
    return gc_count(read.seq)


def read_gc_fraction(read: Read) -> float:
    return gc_fraction(read.seq)


def overlap_composition(
    edge: OverlapEdge,
    reads_by_id: dict[str, Read],
    method: str = "average",
) -> OverlapComposition:
    """
    Estimate the GC content of the overlap represented by an edge.

    `method` controls which side of a noisy overlap is trusted:
        - "average": average left and right overlap windows when both exist.
        - "left": use the left-read aligned/sliced overlap.
        - "right": use the right-read aligned/sliced overlap.

    Refined alignments are preferred because they respect gaps and trimmed
    overlap boundaries. Coordinate slices are used as a fallback.
    """

    if method not in {"average", "left", "right"}:
        raise ValueError(f"Unsupported overlap GC method: {method!r}")

    segments = _alignment_segments(edge)
    source = "alignment"
    if not segments:
        segments = _coordinate_segments(edge, reads_by_id)
        source = "coordinates"
    if not segments:
        fallback = _fallback_overlap_composition(edge, reads_by_id)
        return fallback

    selected = _select_segments(segments, method)
    if not selected:
        fallback = _fallback_overlap_composition(edge, reads_by_id)
        return fallback

    lengths = [_non_gap_len(segment) for segment in selected]
    gc_values = [gc_count(segment) for segment in selected]
    overlap_len = sum(lengths) / len(lengths)
    gc_value = sum(gc_values) / len(gc_values)
    fraction = 0.0 if overlap_len <= 0.0 else gc_value / overlap_len
    return OverlapComposition(
        overlap_len=overlap_len,
        gc_count=gc_value,
        gc_fraction=fraction,
        source=source,
    )


def overlap_gc_count(
    edge: OverlapEdge,
    reads_by_id: dict[str, Read],
    method: str = "average",
) -> float:
    return overlap_composition(edge, reads_by_id, method=method).gc_count


def overlap_gc_fraction(
    edge: OverlapEdge,
    reads_by_id: dict[str, Read],
    method: str = "average",
) -> float:
    return overlap_composition(edge, reads_by_id, method=method).gc_fraction


def _alignment_segments(edge: OverlapEdge) -> dict[str, str]:
    if edge.alignment is None:
        return {}
    segments: dict[str, str] = {}
    left = edge.alignment.aligned_left.replace("-", "")
    right = edge.alignment.aligned_right.replace("-", "")
    if left:
        segments["left"] = left
    if right:
        segments["right"] = right
    return segments


def _coordinate_segments(
    edge: OverlapEdge,
    reads_by_id: dict[str, Read],
) -> dict[str, str]:
    segments: dict[str, str] = {}
    left = reads_by_id.get(edge.left_id)
    if left is not None:
        segment = _safe_slice(left.seq, edge.left_start, edge.left_end)
        if segment:
            segments["left"] = segment
    right = reads_by_id.get(edge.right_id)
    if right is not None:
        segment = _safe_slice(right.seq, edge.right_start, edge.right_end)
        if segment:
            segments["right"] = segment
    return segments


def _fallback_overlap_composition(
    edge: OverlapEdge,
    reads_by_id: dict[str, Read],
) -> OverlapComposition:
    left = reads_by_id.get(edge.left_id)
    right = reads_by_id.get(edge.right_id)
    fractions = [
        read_gc_fraction(read)
        for read in (left, right)
        if read is not None and read.seq
    ]
    fraction = sum(fractions) / len(fractions) if fractions else 0.0
    overlap_len = max(0.0, float(edge.overlap_len))
    return OverlapComposition(
        overlap_len=overlap_len,
        gc_count=fraction * overlap_len,
        gc_fraction=fraction,
        source="read_average_fallback",
    )


def _select_segments(segments: dict[str, str], method: str) -> list[str]:
    if method == "average":
        return list(segments.values())
    segment = segments.get(method)
    return [segment] if segment else []


def _safe_slice(sequence: str, start: int, end: int) -> str:
    start = max(0, min(len(sequence), start))
    end = max(start, min(len(sequence), end))
    return sequence[start:end]


def _non_gap_len(sequence: str) -> int:
    return sum(1 for base in sequence if base != "-")
