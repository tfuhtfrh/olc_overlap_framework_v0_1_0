"""
olc_pipeline.evaluator
Version: 0.1.0

Evaluation functions for simulated data. Edge-level evaluation labels each
predicted directed edge as adjacent-correct, jump-correct, or wrong using the
stored true_start/true_end coordinates. Layout-level evaluation is a placeholder
for later OR-Tools/SA/QUBO solver integration.
"""

from __future__ import annotations

from statistics import mean, median

from .data import (
    Read,
    OverlapCandidate,
    OverlapEdge,
    EdgeEvaluationReport,
    EdgeTruthLabel,
    LayoutEvaluationReport,
    RefinementEvaluationReport,
)

MODULE_VERSION = "0.1.0"


class EdgeEvaluator:
    """
    Evaluate refined overlap edges against simulated ground truth.

    Public interface:
        label_edge(edge, reads_by_id, rank) -> EdgeTruthLabel
        evaluate(reads, edges) -> EdgeEvaluationReport
    """

    def evaluate(self, reads: list[Read], edges: list[OverlapEdge]) -> EdgeEvaluationReport:
        reads_sorted = sorted(reads, key=lambda r: r.true_start)
        reads_by_id = {r.rid: r for r in reads}
        rank = {r.rid: i for i, r in enumerate(reads_sorted)}

        true_adjacent = self._true_adjacent_edges(reads_sorted)
        recovered = {(e.left_id, e.right_id) for e in edges}

        adjacent_correct = 0
        jump_correct = 0
        wrong_edges = 0
        shift_errors: list[int] = []

        for edge in edges:
            label = self.label_edge(edge, reads_by_id, rank)
            if label.label == "adjacent_correct":
                adjacent_correct += 1
                shift_errors.append(abs(edge.shift - label.true_shift))
            elif label.label == "jump_correct":
                jump_correct += 1
                shift_errors.append(abs(edge.shift - label.true_shift))
            else:
                wrong_edges += 1

        correct_edges = adjacent_correct + jump_correct
        edge_precision = correct_edges / len(edges) if edges else 0.0
        missing_adjacent_edges = len(true_adjacent - recovered)
        adjacent_recall = adjacent_correct / len(true_adjacent) if true_adjacent else 0.0

        return EdgeEvaluationReport(
            total_edges=len(edges),
            adjacent_correct=adjacent_correct,
            jump_correct=jump_correct,
            wrong_edges=wrong_edges,
            missing_adjacent_edges=missing_adjacent_edges,
            edge_precision=edge_precision,
            adjacent_recall=adjacent_recall,
            mean_shift_error=mean(shift_errors) if shift_errors else None,
            median_shift_error=median(shift_errors) if shift_errors else None,
            max_shift_error=max(shift_errors) if shift_errors else None,
        )

    def evaluate_candidates(
        self,
        reads: list[Read],
        candidates: list[OverlapCandidate],
    ) -> EdgeEvaluationReport:
        """Evaluate raw directed candidates before DP refinement."""
        candidate_edges = [
            OverlapEdge(
                left_id=c.left_id,
                right_id=c.right_id,
                left_start=c.left_start_hint,
                left_end=c.left_end_hint,
                right_start=c.right_start_hint,
                right_end=c.right_end_hint,
                overlap_len=c.rough_overlap_len,
                shift=c.rough_shift,
                matches=c.n_match,
                mismatches=0,
                insertions=0,
                deletions=0,
                gaps=0,
                edit_distance=0,
                error_rate=0.0,
                identity=0.0,
                dp_score=0.0,
                candidate_source=c.source,
                mapq=c.mapq,
                accepted=True,
                alignment=None,
            )
            for c in candidates
        ]
        report = self.evaluate(reads, candidate_edges)
        report.notes.append("Evaluated raw directed candidates before DP refinement.")
        return report

    def evaluate_refinement(
        self,
        reads: list[Read],
        edges: list[OverlapEdge],
    ) -> RefinementEvaluationReport:
        """Summarize DP-refined edit rates and overlap boundary errors."""
        reads_by_id = {r.rid: r for r in reads}

        error_rates: list[float] = []
        identities: list[float] = []
        overlap_len_errors: list[int] = []

        total_mismatches = 0
        total_insertions = 0
        total_deletions = 0
        total_gaps = 0
        total_edit_distance = 0
        total_alignment_length = 0
        truth_evaluable_edges = 0
        gap_columns = 0
        truth_supported_gap_columns = 0
        unsupported_gap_columns = 0
        homologous_aligned_columns = 0
        misaligned_non_gap_columns = 0

        for edge in edges:
            left = reads_by_id[edge.left_id]
            right = reads_by_id[edge.right_id]
            true_overlap_len = max(0, min(left.true_end, right.true_end) - max(left.true_start, right.true_start))

            error_rates.append(edge.error_rate)
            identities.append(edge.identity)
            overlap_len_errors.append(abs(edge.overlap_len - true_overlap_len))
            total_mismatches += edge.mismatches
            total_insertions += edge.insertions
            total_deletions += edge.deletions
            total_gaps += edge.gaps
            total_edit_distance += edge.edit_distance
            total_alignment_length += edge.overlap_len
            gap_stats = self._evaluate_alignment_gap_truth(left, right, edge)
            if gap_stats is not None:
                truth_evaluable_edges += 1
                gap_columns += gap_stats["gap_columns"]
                truth_supported_gap_columns += gap_stats["truth_supported_gap_columns"]
                unsupported_gap_columns += gap_stats["unsupported_gap_columns"]
                homologous_aligned_columns += gap_stats["homologous_aligned_columns"]
                misaligned_non_gap_columns += gap_stats["misaligned_non_gap_columns"]

        return RefinementEvaluationReport(
            total_edges=len(edges),
            mean_error_rate=mean(error_rates) if error_rates else None,
            median_error_rate=median(error_rates) if error_rates else None,
            max_error_rate=max(error_rates) if error_rates else None,
            mean_identity=mean(identities) if identities else None,
            mean_overlap_len_error=mean(overlap_len_errors) if overlap_len_errors else None,
            median_overlap_len_error=median(overlap_len_errors) if overlap_len_errors else None,
            max_overlap_len_error=max(overlap_len_errors) if overlap_len_errors else None,
            total_mismatches=total_mismatches,
            total_insertions=total_insertions,
            total_deletions=total_deletions,
            total_gaps=total_gaps,
            total_edit_distance=total_edit_distance,
            total_alignment_length=total_alignment_length,
            aggregate_error_rate=(
                total_edit_distance / total_alignment_length
                if total_alignment_length > 0
                else None
            ),
            truth_evaluable_edges=truth_evaluable_edges,
            gap_columns=gap_columns,
            truth_supported_gap_columns=truth_supported_gap_columns,
            unsupported_gap_columns=unsupported_gap_columns,
            gap_truth_precision=(
                truth_supported_gap_columns / gap_columns
                if gap_columns > 0
                else None
            ),
            homologous_aligned_columns=homologous_aligned_columns,
            misaligned_non_gap_columns=misaligned_non_gap_columns,
            non_gap_coord_error_rate=(
                misaligned_non_gap_columns
                / (homologous_aligned_columns + misaligned_non_gap_columns)
                if homologous_aligned_columns + misaligned_non_gap_columns > 0
                else None
            ),
        )

    def _evaluate_alignment_gap_truth(
        self,
        left: Read,
        right: Read,
        edge: OverlapEdge,
    ) -> dict[str, int] | None:
        if edge.alignment is None or left.ref_coords is None or right.ref_coords is None:
            return None

        left_ref_coords = left.ref_coords
        right_ref_coords = right.ref_coords
        left_ref_set = {coord for coord in left_ref_coords if coord >= 0}
        right_ref_set = {coord for coord in right_ref_coords if coord >= 0}

        left_idx = edge.left_start
        right_idx = edge.right_start
        stats = {
            "gap_columns": 0,
            "truth_supported_gap_columns": 0,
            "unsupported_gap_columns": 0,
            "homologous_aligned_columns": 0,
            "misaligned_non_gap_columns": 0,
        }

        for left_base, right_base in zip(edge.alignment.aligned_left, edge.alignment.aligned_right):
            left_coord = None
            right_coord = None
            if left_base != "-":
                left_coord = left_ref_coords[left_idx]
                left_idx += 1
            if right_base != "-":
                right_coord = right_ref_coords[right_idx]
                right_idx += 1

            if left_base == "-" or right_base == "-":
                stats["gap_columns"] += 1
                if left_base == "-":
                    supported = self._gap_supported_by_truth(
                        non_gap_coord=right_coord,
                        gapped_read=left,
                        gapped_ref_set=left_ref_set,
                    )
                else:
                    supported = self._gap_supported_by_truth(
                        non_gap_coord=left_coord,
                        gapped_read=right,
                        gapped_ref_set=right_ref_set,
                    )

                if supported:
                    stats["truth_supported_gap_columns"] += 1
                else:
                    stats["unsupported_gap_columns"] += 1
                continue

            if left_coord is not None and right_coord is not None and left_coord >= 0 and right_coord >= 0:
                if left_coord == right_coord:
                    stats["homologous_aligned_columns"] += 1
                else:
                    stats["misaligned_non_gap_columns"] += 1
            else:
                stats["misaligned_non_gap_columns"] += 1

        return stats

    @staticmethod
    def _gap_supported_by_truth(
        non_gap_coord: int | None,
        gapped_read: Read,
        gapped_ref_set: set[int],
    ) -> bool:
        if non_gap_coord is None:
            return False
        if non_gap_coord < 0:
            return True
        return (
            gapped_read.true_start <= non_gap_coord < gapped_read.true_end
            and non_gap_coord not in gapped_ref_set
        )

    def label_edge(
        self,
        edge: OverlapEdge,
        reads_by_id: dict[str, Read],
        rank: dict[str, int],
    ) -> EdgeTruthLabel:
        left = reads_by_id[edge.left_id]
        right = reads_by_id[edge.right_id]
        true_shift = right.true_start - left.true_start
        true_overlap_len = max(0, min(left.true_end, right.true_end) - max(left.true_start, right.true_start))

        forward_order = rank[left.rid] < rank[right.rid]
        has_true_overlap = left.true_end > right.true_start and right.true_end > left.true_start

        if forward_order and has_true_overlap:
            if rank[right.rid] == rank[left.rid] + 1:
                return EdgeTruthLabel("adjacent_correct", true_overlap_len, true_shift)
            return EdgeTruthLabel("jump_correct", true_overlap_len, true_shift)

        return EdgeTruthLabel("wrong_edge", true_overlap_len, true_shift)

    @staticmethod
    def _true_adjacent_edges(reads_sorted: list[Read]) -> set[tuple[str, str]]:
        true_edges: set[tuple[str, str]] = set()
        for i in range(len(reads_sorted) - 1):
            a = reads_sorted[i]
            b = reads_sorted[i + 1]
            if a.true_end > b.true_start:
                true_edges.add((a.rid, b.rid))
        return true_edges


class LayoutEvaluator:
    """
    Placeholder for layout-order evaluation.

    This module will become useful after OR-Tools/SA/QUBO solvers are connected.
    """

    def evaluate_order(self, reads: list[Read], predicted_order: list[str]) -> LayoutEvaluationReport:
        reads_sorted = sorted(reads, key=lambda r: r.true_start)
        rank = {r.rid: i for i, r in enumerate(reads_sorted)}

        adjacent_correct = 0
        wrong_adjacencies = 0
        for a, b in zip(predicted_order, predicted_order[1:]):
            if rank.get(b, -10**9) == rank.get(a, 10**9) + 1:
                adjacent_correct += 1
            else:
                wrong_adjacencies += 1

        return LayoutEvaluationReport(
            order_length=len(predicted_order),
            adjacent_correct_in_layout=adjacent_correct,
            wrong_adjacencies_in_layout=wrong_adjacencies,
            inversion_count=self._inversion_count([rank[rid] for rid in predicted_order if rid in rank]),
        )

    @staticmethod
    def _inversion_count(values: list[int]) -> int:
        inv = 0
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if values[i] > values[j]:
                    inv += 1
        return inv
