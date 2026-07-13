import unittest

from olc_pipeline.data import OverlapEdge, Read
from olc_pipeline.overlap_features import (
    gc_count,
    gc_fraction,
    overlap_composition,
    overlap_gc_count,
    overlap_gc_fraction,
)


def make_edge(left_id: str, right_id: str) -> OverlapEdge:
    return OverlapEdge(
        left_id=left_id,
        right_id=right_id,
        left_start=2,
        left_end=6,
        right_start=0,
        right_end=4,
        overlap_len=4,
        shift=4,
        matches=4,
        mismatches=0,
        insertions=0,
        deletions=0,
        gaps=0,
        edit_distance=0,
        error_rate=0.0,
        identity=1.0,
        dp_score=1.0,
        accepted=True,
    )


class OverlapFeatureTests(unittest.TestCase):
    def test_gc_count_and_fraction_ignore_gaps(self):
        self.assertEqual(gc_count("A-CGgnN"), 3)
        self.assertAlmostEqual(gc_fraction("A-CG"), 2 / 3)

    def test_overlap_composition_averages_left_and_right_windows(self):
        reads = {
            "left": Read("left", "TTGGAA"),
            "right": Read("right", "CCCCAA"),
        }
        edge = make_edge("left", "right")

        composition = overlap_composition(edge, reads)

        self.assertEqual(composition.source, "coordinates")
        self.assertEqual(composition.overlap_len, 4.0)
        self.assertEqual(composition.gc_count, 3.0)
        self.assertEqual(composition.gc_fraction, 0.75)
        self.assertEqual(overlap_gc_count(edge, reads), 3.0)
        self.assertEqual(overlap_gc_fraction(edge, reads), 0.75)


if __name__ == "__main__":
    unittest.main()
