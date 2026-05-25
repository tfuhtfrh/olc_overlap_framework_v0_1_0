import unittest

from olc_pipeline.data import OverlapEdge, PairwiseAlignment, Read
from olc_pipeline.evaluator import EdgeEvaluator


class RefinementGapTruthTests(unittest.TestCase):
    def test_gap_truth_precision_counts_supported_insertion_gap(self):
        left = Read("left", "ACGT", true_start=0, true_end=4, ref_coords=(0, 1, 2, 3))
        right = Read("right", "ACTGT", true_start=0, true_end=4, ref_coords=(0, 1, -1, 2, 3))
        edge = OverlapEdge(
            left_id="left",
            right_id="right",
            left_start=0,
            left_end=4,
            right_start=0,
            right_end=5,
            overlap_len=5,
            shift=0,
            matches=4,
            mismatches=0,
            insertions=1,
            deletions=0,
            gaps=1,
            edit_distance=1,
            error_rate=0.2,
            identity=0.8,
            dp_score=0.0,
            accepted=True,
            alignment=PairwiseAlignment(
                aligned_left="AC-GT",
                aligned_right="ACTGT",
                left_start=0,
                left_end=4,
                right_start=0,
                right_end=5,
                score=0.0,
                matches=4,
                mismatches=0,
                insertions=1,
                deletions=0,
                gaps=1,
                edit_distance=1,
                alignment_length=5,
            ),
        )

        report = EdgeEvaluator().evaluate_refinement([left, right], [edge])

        self.assertEqual(report.truth_evaluable_edges, 1)
        self.assertEqual(report.gap_columns, 1)
        self.assertEqual(report.truth_supported_gap_columns, 1)
        self.assertEqual(report.unsupported_gap_columns, 0)
        self.assertEqual(report.gap_truth_precision, 1.0)
        self.assertEqual(report.non_gap_coord_error_rate, 0.0)

    def test_gap_truth_precision_counts_unsupported_gap(self):
        left = Read("left", "ACGT", true_start=0, true_end=4, ref_coords=(0, 1, 2, 3))
        right = Read("right", "ACGT", true_start=0, true_end=4, ref_coords=(0, 1, 2, 3))
        edge = OverlapEdge(
            left_id="left",
            right_id="right",
            left_start=0,
            left_end=4,
            right_start=0,
            right_end=4,
            overlap_len=5,
            shift=0,
            matches=3,
            mismatches=0,
            insertions=1,
            deletions=0,
            gaps=1,
            edit_distance=1,
            error_rate=0.2,
            identity=0.6,
            dp_score=0.0,
            accepted=True,
            alignment=PairwiseAlignment(
                aligned_left="AC-GT",
                aligned_right="ACG-T",
                left_start=0,
                left_end=4,
                right_start=0,
                right_end=4,
                score=0.0,
                matches=3,
                mismatches=0,
                insertions=1,
                deletions=0,
                gaps=1,
                edit_distance=1,
                alignment_length=5,
            ),
        )

        report = EdgeEvaluator().evaluate_refinement([left, right], [edge])

        self.assertEqual(report.gap_columns, 2)
        self.assertEqual(report.truth_supported_gap_columns, 0)
        self.assertEqual(report.unsupported_gap_columns, 2)
        self.assertEqual(report.gap_truth_precision, 0.0)


if __name__ == "__main__":
    unittest.main()
