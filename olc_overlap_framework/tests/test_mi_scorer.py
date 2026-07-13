import unittest

from olc_pipeline.data import PairwiseAlignment
from olc_pipeline.mi_scorer import AlignmentMIScorer


def make_alignment(left: str, right: str) -> PairwiseAlignment:
    matches = sum(a == b for a, b in zip(left, right))
    return PairwiseAlignment(
        aligned_left=left,
        aligned_right=right,
        left_start=0,
        left_end=len(left.replace("-", "")),
        right_start=0,
        right_end=len(right.replace("-", "")),
        score=float(matches),
        matches=matches,
        mismatches=len(left) - matches,
        insertions=0,
        deletions=0,
        gaps=left.count("-") + right.count("-"),
        edit_distance=len(left) - matches,
        alignment_length=len(left),
    )


class AlignmentMIScorerTests(unittest.TestCase):
    def test_exact_balanced_alignment_has_two_bits_and_unit_nmi(self):
        score = AlignmentMIScorer().score_alignment(
            make_alignment("ACGTACGT", "ACGTACGT")
        )

        self.assertAlmostEqual(score.mi, 2.0)
        self.assertAlmostEqual(score.nmi, 1.0)
        self.assertAlmostEqual(score.mi_distance, 0.0)

    def test_exact_constant_alignment_has_zero_entropy_and_zero_nmi(self):
        score = AlignmentMIScorer().score_alignment(
            make_alignment("AAAAAAAA", "AAAAAAAA")
        )

        self.assertAlmostEqual(score.mi, 0.0)
        self.assertAlmostEqual(score.nmi, 0.0)
        self.assertAlmostEqual(score.mi_distance, 1.0)


if __name__ == "__main__":
    unittest.main()
