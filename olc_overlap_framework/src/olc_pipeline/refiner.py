"""
olc_pipeline.refiner
Version: 0.1.0

Overlap refinement modules. ParasailOverlapRefiner is the main intended
refiner. It aligns only minimap2/original candidates, not all read pairs.

This module uses parasail's semi-global traceback alignment to refine overlap
boundaries and build aligned strings for MI scoring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Optional

import parasail

from .data import Read, OverlapCandidate, PairwiseAlignment, OverlapEdge

MODULE_VERSION = "0.1.0"


class OverlapRefiner(ABC):
    """
    Interface for candidate-edge refinement.

    Input:
        OverlapCandidate + reads_by_id
    Output:
        OverlapEdge containing refined geometry, edit statistics, and optional
        alignment strings.
    """

    @abstractmethod
    def refine(self, candidate: OverlapCandidate, reads_by_id: dict[str, Read]) -> OverlapEdge:
        raise NotImplementedError


@dataclass(frozen=True)
class RefinerConfig:
    """Parameters for overlap refinement and edge acceptance."""

    min_overlap: int = 500
    max_error_rate: float = 0.15
    margin: int = 80
    match_score: int = 2
    mismatch_penalty: int = 3
    gap_open: int = 5
    gap_extend: int = 1


class ParasailOverlapRefiner(OverlapRefiner):
    """
    Refine candidate overlaps with parasail-style semi-global alignment.

    Current implementation notes:
        - It first extracts a suffix region from left and a prefix region from right
          using PAF/candidate hints and a configurable margin.
        - It uses parasail's query-begin / database-end free semi-global mode,
          which matches suffix(left_region) vs prefix(right_region).
        - It decodes the parasail CIGAR into aligned strings for MI scoring.

    The interface is stable even if the internal alignment engine is replaced by
    edlib/WFA/another parasail function later.
    """

    def __init__(self, config: Optional[RefinerConfig] = None):
        self.config = config or RefinerConfig()

    def refine(self, candidate: OverlapCandidate, reads_by_id: dict[str, Read]) -> OverlapEdge:
        left = reads_by_id[candidate.left_id]
        right = reads_by_id[candidate.right_id]

        left_region, right_region, left_offset, right_offset = self._extract_regions(candidate, left, right)

        alignment = self._align_overlap_parasail(left_region, right_region, left_offset, right_offset)

        overlap_len = alignment.alignment_length
        error_rate = alignment.edit_distance / overlap_len if overlap_len > 0 else 1.0
        identity = alignment.matches / overlap_len if overlap_len > 0 else 0.0

        shift = alignment.left_start - alignment.right_start
        accepted = overlap_len >= self.config.min_overlap and error_rate <= self.config.max_error_rate
        weight_dp = overlap_len * identity

        return OverlapEdge(
            left_id=candidate.left_id,
            right_id=candidate.right_id,
            left_start=alignment.left_start,
            left_end=alignment.left_end,
            right_start=alignment.right_start,
            right_end=alignment.right_end,
            overlap_len=overlap_len,
            shift=shift,
            matches=alignment.matches,
            mismatches=alignment.mismatches,
            insertions=alignment.insertions,
            deletions=alignment.deletions,
            gaps=alignment.gaps,
            edit_distance=alignment.edit_distance,
            error_rate=error_rate,
            identity=identity,
            dp_score=alignment.score,
            weight_dp=weight_dp,
            candidate_source=candidate.source,
            mapq=candidate.mapq,
            accepted=accepted,
            alignment=alignment,
        )

    def _extract_regions(
        self,
        candidate: OverlapCandidate,
        left: Read,
        right: Read,
    ) -> tuple[str, str, int, int]:
        """
        Extract left suffix and right prefix regions with a small margin.

        Returns:
            left_region, right_region, left_offset, right_offset
        where offsets are coordinates of region starts in the original reads.
        """
        margin = self.config.margin
        left_start = max(0, candidate.left_start_hint - margin)
        left_end = min(len(left.seq), candidate.left_end_hint)

        # For suffix-prefix overlap, right_start is usually 0. We keep it fixed at
        # 0 for now to avoid accidentally aligning non-overlap prefixes.
        right_start = max(0, candidate.right_start_hint)
        right_end = min(len(right.seq), candidate.right_end_hint + margin)

        if left_start >= left_end:
            left_start, left_end = 0, len(left.seq)
        if right_start >= right_end:
            right_start, right_end = 0, len(right.seq)

        return left.seq[left_start:left_end], right.seq[right_start:right_end], left_start, right_start

    def _align_overlap_parasail(
        self,
        a: str,
        b: str,
        a_offset: int,
        b_offset: int,
    ) -> PairwiseAlignment:
        """
        Parasail suffix-prefix overlap alignment.

        `a` is the left-read suffix region and `b` is the right-read prefix
        region. sg_qb_de makes the query beginning free and the database end
        free, so leading bases in `a` and trailing bases in `b` are outside the
        overlap.
        """
        matrix = parasail.matrix_create("ACGT", self.config.match_score, -self.config.mismatch_penalty)
        result = parasail.sg_qb_de_trace_striped_sat(
            a,
            b,
            self.config.gap_open,
            self.config.gap_extend,
            matrix,
        )
        if result.saturated:
            result = parasail.sg_qb_de_trace_striped_64(
                a,
                b,
                self.config.gap_open,
                self.config.gap_extend,
                matrix,
            )

        cigar = result.cigar.decode
        if isinstance(cigar, bytes):
            cigar = cigar.decode("ascii")

        aligned_left, aligned_right, start_i, end_i, start_j, end_j = self._alignment_from_cigar(
            cigar,
            a,
            b,
        )

        stats = self._count_alignment_stats(aligned_left, aligned_right)

        return PairwiseAlignment(
            aligned_left=aligned_left,
            aligned_right=aligned_right,
            left_start=a_offset + start_i,
            left_end=a_offset + end_i,
            right_start=b_offset + start_j,
            right_end=b_offset + end_j,
            score=result.score,
            matches=stats["matches"],
            mismatches=stats["mismatches"],
            insertions=stats["insertions"],
            deletions=stats["deletions"],
            gaps=stats["gaps"],
            edit_distance=stats["edit_distance"],
            alignment_length=len(aligned_left),
        )

    @staticmethod
    def _alignment_from_cigar(
        cigar: str,
        query: str,
        ref: str,
    ) -> tuple[str, str, int, int, int, int]:
        """
        Decode parasail CIGAR and remove free semi-global flanks.

        For sg_qb_de, parasail reports the free query prefix as leading `I`
        operations and the free reference suffix as trailing `D` operations.
        Those bases are outside the overlap and should not contribute to
        alignment strings, edit distance, or refined boundaries.
        """
        ops = [(int(length), op) for length, op in re.findall(r"(\d+)([=XIDM])", cigar)]

        free_query_prefix = 0
        while ops and ops[0][1] == "I":
            free_query_prefix += ops[0][0]
            ops.pop(0)

        while ops and ops[-1][1] == "D":
            ops.pop()

        q_pos = free_query_prefix
        r_pos = 0
        q_start = q_pos
        r_start = r_pos
        aligned_query: list[str] = []
        aligned_ref: list[str] = []

        for length, op in ops:
            if op in {"=", "X", "M"}:
                q_chunk = query[q_pos:q_pos + length]
                r_chunk = ref[r_pos:r_pos + length]
                aligned_query.append(q_chunk)
                aligned_ref.append(r_chunk)
                q_pos += length
                r_pos += length
            elif op == "I":
                aligned_query.append(query[q_pos:q_pos + length])
                aligned_ref.append("-" * length)
                q_pos += length
            elif op == "D":
                aligned_query.append("-" * length)
                aligned_ref.append(ref[r_pos:r_pos + length])
                r_pos += length

        return (
            "".join(aligned_query),
            "".join(aligned_ref),
            q_start,
            q_pos,
            r_start,
            r_pos,
        )

    @staticmethod
    def _count_alignment_stats(aligned_left: str, aligned_right: str) -> dict[str, int]:
        matches = mismatches = insertions = deletions = 0
        for x, y in zip(aligned_left, aligned_right):
            if x == "-" and y == "-":
                continue
            if x == "-":
                insertions += 1
            elif y == "-":
                deletions += 1
            elif x == y:
                matches += 1
            else:
                mismatches += 1
        gaps = insertions + deletions
        edit_distance = mismatches + gaps
        return {
            "matches": matches,
            "mismatches": mismatches,
            "insertions": insertions,
            "deletions": deletions,
            "gaps": gaps,
            "edit_distance": edit_distance,
        }
