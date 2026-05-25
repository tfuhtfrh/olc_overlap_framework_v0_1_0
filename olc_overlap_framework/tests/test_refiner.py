import unittest

from olc_pipeline.data import OverlapCandidate, Read
from olc_pipeline.refiner import ParasailOverlapRefiner, RefinerConfig


class ParasailOverlapRefinerTests(unittest.TestCase):
    def test_parasail_alignment_trims_free_semiglobal_flanks(self):
        refiner = ParasailOverlapRefiner(RefinerConfig(min_overlap=1))

        alignment = refiner._align_overlap_parasail("GGGGACGT", "ACGTTTTT", 0, 0)

        self.assertEqual(alignment.aligned_left, "ACGT")
        self.assertEqual(alignment.aligned_right, "ACGT")
        self.assertEqual((alignment.left_start, alignment.left_end), (4, 8))
        self.assertEqual((alignment.right_start, alignment.right_end), (0, 4))
        self.assertEqual(alignment.matches, 4)
        self.assertEqual(alignment.edit_distance, 0)

    def test_refine_builds_overlap_edge_from_parasail_alignment(self):
        reads = {
            "left": Read("left", "GGGGACGT", true_start=0, true_end=8),
            "right": Read("right", "ACGTTTTT", true_start=4, true_end=12),
        }
        candidate = OverlapCandidate(
            left_id="left",
            right_id="right",
            source="test",
            query_id="right",
            target_id="left",
            strand="+",
            q_len=8,
            q_st=0,
            q_en=4,
            t_len=8,
            t_st=4,
            t_en=8,
            n_match=4,
            aln_block_len=4,
            mapq=60,
            left_start_hint=0,
            left_end_hint=8,
            right_start_hint=0,
            right_end_hint=8,
            rough_overlap_len=4,
            rough_shift=4,
        )
        refiner = ParasailOverlapRefiner(RefinerConfig(min_overlap=1, margin=0))

        edge = refiner.refine(candidate, reads)

        self.assertTrue(edge.accepted)
        self.assertEqual((edge.left_start, edge.left_end), (4, 8))
        self.assertEqual((edge.right_start, edge.right_end), (0, 4))
        self.assertEqual(edge.overlap_len, 4)
        self.assertEqual(edge.shift, 4)
        self.assertEqual(edge.error_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
