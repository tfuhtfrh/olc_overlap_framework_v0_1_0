"""
olc_pipeline.data
Version: 0.1.0

Shared dataclasses for the OLC overlap/layout reproduction framework.
This file intentionally contains no algorithmic logic. Other modules exchange
only these structures so candidate finders, refiners, MI scorers, and layout
solvers can be replaced independently without changing the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal

MODULE_VERSION = "0.1.0"


@dataclass(frozen=True)
class Read:
    """
    A simulated or imported sequencing read.

    true_start / true_end are ground-truth coordinates on the simulated genome.
    For real data, these can be set to -1 and evaluation functions that require
    ground truth should be disabled.
    """

    rid: str
    seq: str
    true_start: int = -1
    true_end: int = -1
    strand: int = +1
    ref_coords: Optional[tuple[int, ...]] = None


@dataclass(frozen=True)
class OverlapCandidate:
    """
    Coarse directed overlap candidate from minimap2 or another finder.

    Convention:
        left_id -> right_id means suffix(left) is expected to overlap prefix(right).

    PAF fields are preserved so downstream refiners can use exact query/target
    coordinates rather than only the directed edge interpretation.
    """

    left_id: str
    right_id: str

    # Source and orientation information.
    source: str
    query_id: str
    target_id: str
    strand: str

    # PAF geometry.
    q_len: int
    q_st: int
    q_en: int
    t_len: int
    t_st: int
    t_en: int

    # PAF statistics.
    n_match: int
    aln_block_len: int
    mapq: int

    # Directed-edge geometry hints, expressed in left/right coordinates.
    left_start_hint: int
    left_end_hint: int
    right_start_hint: int
    right_end_hint: int
    rough_overlap_len: int
    rough_shift: int


@dataclass
class PairwiseAlignment:
    """
    Refined pairwise overlap alignment.

    aligned_left and aligned_right must have the same length and may include '-'.
    The coordinates are half-open intervals on the original read sequences:
        left[left_start:left_end]
        right[right_start:right_end]
    """

    aligned_left: str
    aligned_right: str
    left_start: int
    left_end: int
    right_start: int
    right_end: int
    score: float
    matches: int
    mismatches: int
    insertions: int
    deletions: int
    gaps: int
    edit_distance: int
    alignment_length: int


@dataclass
class OverlapEdge:
    """
    Final accepted or rejected refined overlap edge.

    Geometry:
        shift is the predicted start(right) - start(left) in layout coordinates.
        overlap_len is the refined alignment length or effective overlap length.

    Weights:
        weight_dp and weight_mi are optional so experiments can compare direct
        DP-derived weights with mutual-information-derived weights.
    """

    left_id: str
    right_id: str

    left_start: int
    left_end: int
    right_start: int
    right_end: int
    overlap_len: int
    shift: int

    matches: int
    mismatches: int
    insertions: int
    deletions: int
    gaps: int
    edit_distance: int
    error_rate: float
    identity: float
    dp_score: float

    mi: Optional[float] = None
    nmi: Optional[float] = None
    mi_distance: Optional[float] = None
    weight_dp: Optional[float] = None
    weight_mi: Optional[float] = None

    candidate_source: str = "unknown"
    mapq: int = 0
    accepted: bool = False
    alignment: Optional[PairwiseAlignment] = None


@dataclass(frozen=True)
class EdgeTruthLabel:
    """Ground-truth label of a directed overlap edge."""

    label: Literal["adjacent_correct", "jump_correct", "wrong_edge"]
    true_overlap_len: int
    true_shift: int


@dataclass
class EdgeEvaluationReport:
    """Aggregate edge-level evaluation against simulated ground truth."""

    total_edges: int
    adjacent_correct: int
    jump_correct: int
    wrong_edges: int
    missing_adjacent_edges: int
    edge_precision: float
    adjacent_recall: float
    mean_shift_error: Optional[float] = None
    median_shift_error: Optional[float] = None
    max_shift_error: Optional[int] = None
    notes: list[str] = field(default_factory=list)


@dataclass
class RefinementEvaluationReport:
    """Geometry and edit-quality summary for refined accepted edges."""

    total_edges: int
    mean_error_rate: Optional[float] = None
    median_error_rate: Optional[float] = None
    max_error_rate: Optional[float] = None
    mean_identity: Optional[float] = None
    mean_overlap_len_error: Optional[float] = None
    median_overlap_len_error: Optional[float] = None
    max_overlap_len_error: Optional[int] = None
    total_mismatches: int = 0
    total_insertions: int = 0
    total_deletions: int = 0
    total_gaps: int = 0
    total_edit_distance: int = 0
    total_alignment_length: int = 0
    aggregate_error_rate: Optional[float] = None
    truth_evaluable_edges: int = 0
    gap_columns: int = 0
    truth_supported_gap_columns: int = 0
    unsupported_gap_columns: int = 0
    gap_truth_precision: Optional[float] = None
    homologous_aligned_columns: int = 0
    misaligned_non_gap_columns: int = 0
    non_gap_coord_error_rate: Optional[float] = None
    notes: list[str] = field(default_factory=list)


@dataclass
class LayoutResult:
    """Output of a layout solver. Currently only the order is required."""

    order: list[str]
    objective_value: Optional[float] = None
    solver_name: str = "unknown"
    metadata: dict = field(default_factory=dict)


@dataclass
class LayoutEvaluationReport:
    """Optional layout-level evaluation. Used when a solver is implemented."""

    order_length: int
    adjacent_correct_in_layout: int
    wrong_adjacencies_in_layout: int
    inversion_count: Optional[int] = None
    notes: list[str] = field(default_factory=list)
