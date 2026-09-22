import unittest

from olc_pipeline.candidate_finder import Minimap2CandidateFinder
from olc_pipeline.data import OverlapCandidate, Read
from olc_pipeline.refiner import PafCandidateEdgeAdapter, ParasailOverlapRefiner, RefinerConfig


class PafCandidateEdgeAdapterTests(unittest.TestCase):
    def test_direct_adapter_preserves_paf_geometry_without_alignment(self):
        candidate = OverlapCandidate(
            left_id="left",
            right_id="right",
            source="minimap2",
            query_id="right",
            target_id="left",
            strand="+",
            q_len=150,
            q_st=0,
            q_en=90,
            t_len=151,
            t_st=61,
            t_en=151,
            n_match=89,
            aln_block_len=90,
            mapq=60,
            left_start_hint=61,
            left_end_hint=151,
            right_start_hint=0,
            right_end_hint=90,
            rough_overlap_len=90,
            rough_shift=61,
            left_orientation=-1,
            right_orientation=-1,
        )

        edge = PafCandidateEdgeAdapter().refine(candidate, {})

        self.assertTrue(edge.accepted)
        self.assertIsNone(edge.alignment)
        self.assertEqual((edge.left_start, edge.left_end), (61, 151))
        self.assertEqual((edge.right_start, edge.right_end), (0, 90))
        self.assertEqual((edge.overlap_len, edge.shift), (90, 61))
        self.assertEqual(edge.matches, 89)
        self.assertEqual(edge.edit_distance, 1)
        self.assertAlmostEqual(edge.identity, 89 / 90)
        self.assertEqual((edge.left_orientation, edge.right_orientation), (-1, -1))
        self.assertEqual(edge.candidate_source, "minimap2_paf_direct")


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

    def test_refine_is_symmetric_for_generated_reverse_candidate(self):
        reads = {
            "left": Read("left", "TTTTACGT"),
            "right": Read("right", "ACGTGGGGGG"),
        }
        candidate = OverlapCandidate(
            left_id="left",
            right_id="right",
            source="test",
            query_id="left",
            target_id="right",
            strand="+",
            q_len=8,
            q_st=4,
            q_en=8,
            t_len=10,
            t_st=0,
            t_en=4,
            n_match=4,
            aln_block_len=4,
            mapq=60,
            left_start_hint=4,
            left_end_hint=8,
            right_start_hint=0,
            right_end_hint=4,
            rough_overlap_len=4,
            rough_shift=4,
        )
        reciprocal = Minimap2CandidateFinder._reverse_complement_candidate(candidate)
        refiner = ParasailOverlapRefiner(RefinerConfig(min_overlap=4, margin=0))

        forward = refiner.refine(candidate, reads)
        reverse = refiner.refine(reciprocal, reads)

        self.assertTrue(forward.accepted)
        self.assertTrue(reverse.accepted)
        self.assertEqual(forward.overlap_len, reverse.overlap_len)
        self.assertEqual(forward.identity, reverse.identity)
        self.assertEqual(forward.error_rate, reverse.error_rate)


if __name__ == "__main__":
    unittest.main()
