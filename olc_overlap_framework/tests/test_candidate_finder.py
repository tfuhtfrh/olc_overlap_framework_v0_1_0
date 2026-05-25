import unittest

from olc_pipeline.candidate_finder import Minimap2CandidateFinder, Minimap2Config


class Minimap2CandidateFinderTests(unittest.TestCase):
    def test_parse_target_suffix_query_prefix_as_target_to_query(self):
        finder = Minimap2CandidateFinder(Minimap2Config(min_overlap=100, overhang_tolerance=10))

        candidates = self._parse_lines(
            finder,
            [
                # query read_1 prefix aligns target read_0 suffix.
                "read_1\t1000\t0\t600\t+\tread_0\t1000\t400\t1000\t580\t600\t60\n",
            ],
        )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual((candidate.left_id, candidate.right_id), ("read_0", "read_1"))
        self.assertEqual((candidate.left_start_hint, candidate.left_end_hint), (400, 1000))
        self.assertEqual((candidate.right_start_hint, candidate.right_end_hint), (0, 600))
        self.assertEqual(candidate.rough_shift, 400)

    def test_parse_query_suffix_target_prefix_as_query_to_target(self):
        finder = Minimap2CandidateFinder(Minimap2Config(min_overlap=100, overhang_tolerance=10))

        candidates = self._parse_lines(
            finder,
            [
                # query read_0 suffix aligns target read_1 prefix.
                "read_0\t1000\t400\t1000\t+\tread_1\t1000\t0\t600\t580\t600\t60\n",
            ],
        )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual((candidate.left_id, candidate.right_id), ("read_0", "read_1"))
        self.assertEqual((candidate.left_start_hint, candidate.left_end_hint), (400, 1000))
        self.assertEqual((candidate.right_start_hint, candidate.right_end_hint), (0, 600))
        self.assertEqual(candidate.rough_shift, 400)

    def test_parse_skips_reverse_strand_for_version_0_1_0(self):
        finder = Minimap2CandidateFinder(Minimap2Config(min_overlap=100, overhang_tolerance=10))

        candidates = self._parse_lines(
            finder,
            [
                "read_0\t1000\t400\t1000\t-\tread_1\t1000\t0\t600\t580\t600\t60\n",
            ],
        )

        self.assertEqual(candidates, [])

    def test_parse_uses_minimap2_divergence_tag_for_error_hint(self):
        finder = Minimap2CandidateFinder(Minimap2Config(min_overlap=500, max_error_rate_hint=0.30))

        candidates = self._parse_lines(
            finder,
            [
                # ava overlap PAF can have low n_match / block_len but a good
                # minimap2 divergence tag.
                "read_0\t3000\t500\t2950\t+\tread_1\t3000\t20\t2470\t1300\t2500\t0"
                "\ttp:A:S\tcm:i:250\ts1:i:1290\tdv:f:0.075\n",
            ],
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual((candidates[0].left_id, candidates[0].right_id), ("read_0", "read_1"))

    def test_parse_falls_back_to_mandatory_paf_identity_without_divergence_tag(self):
        finder = Minimap2CandidateFinder(Minimap2Config(min_overlap=500, max_error_rate_hint=0.30))

        candidates = self._parse_lines(
            finder,
            [
                "read_0\t3000\t500\t2950\t+\tread_1\t3000\t20\t2470\t1300\t2500\t0\n",
            ],
        )

        self.assertEqual(candidates, [])

    @staticmethod
    def _parse_lines(finder: Minimap2CandidateFinder, lines: list[str]):
        return list(finder._parse_paf_lines(lines))


if __name__ == "__main__":
    unittest.main()
