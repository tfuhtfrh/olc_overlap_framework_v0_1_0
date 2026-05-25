"""
olc_pipeline.mi_scorer
Version: 0.1.0

Mutual-information edge scoring for refined alignments.
This module implements a simple Shannon MI/NMI scorer over aligned columns with
alphabet {A,C,G,T,-}. It is designed to compare MI-derived edge weights against
DP-derived weights.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable

from .data import PairwiseAlignment, OverlapEdge

MODULE_VERSION = "0.1.0"


@dataclass(frozen=True)
class MIScore:
    """Mutual-information statistics for one pairwise alignment."""

    mi: float
    nmi: float
    mi_distance: float
    h_left: float
    h_right: float
    alignment_length: int


class AlignmentMIScorer:
    """
    Compute Shannon MI and NMI from aligned read columns.

    Public interface:
        score_alignment(alignment) -> MIScore
        attach_to_edge(edge) -> OverlapEdge
    """

    def __init__(self, log_base: float = 2.0, include_gap: bool = True):
        self.log_base = log_base
        self.include_gap = include_gap

    def score_alignment(self, alignment: PairwiseAlignment) -> MIScore:
        left = alignment.aligned_left
        right = alignment.aligned_right
        if len(left) != len(right):
            raise ValueError("aligned_left and aligned_right must have equal length")

        pairs: list[tuple[str, str]] = []
        for x, y in zip(left, right):
            if not self.include_gap and (x == "-" or y == "-"):
                continue
            pairs.append((x, y))

        n = len(pairs)
        if n == 0:
            return MIScore(mi=0.0, nmi=0.0, mi_distance=1.0, h_left=0.0, h_right=0.0, alignment_length=0)

        joint = Counter(pairs)
        left_counts = Counter(x for x, _ in pairs)
        right_counts = Counter(y for _, y in pairs)

        mi = 0.0
        for (x, y), c_xy in joint.items():
            p_xy = c_xy / n
            p_x = left_counts[x] / n
            p_y = right_counts[y] / n
            mi += p_xy * self._log(p_xy / (p_x * p_y))

        h_left = self._entropy(left_counts.values(), n)
        h_right = self._entropy(right_counts.values(), n)

        denom = h_left + h_right
        nmi = (2.0 * mi / denom) if denom > 0 else 0.0
        nmi = max(0.0, min(1.0, nmi))
        mi_distance = 1.0 - nmi

        return MIScore(
            mi=mi,
            nmi=nmi,
            mi_distance=mi_distance,
            h_left=h_left,
            h_right=h_right,
            alignment_length=n,
        )

    def attach_to_edge(self, edge: OverlapEdge) -> OverlapEdge:
        """Compute MI statistics and store them in the edge object in-place."""
        if edge.alignment is None:
            raise ValueError("edge.alignment is required for MI scoring")
        score = self.score_alignment(edge.alignment)
        edge.mi = score.mi
        edge.nmi = score.nmi
        edge.mi_distance = score.mi_distance
        edge.weight_mi = edge.overlap_len * score.nmi
        return edge

    def _entropy(self, counts: Iterable[int], total: int) -> float:
        h = 0.0
        for c in counts:
            p = c / total
            if p > 0:
                h -= p * self._log(p)
        return h

    def _log(self, x: float) -> float:
        return math.log(x, self.log_base)
