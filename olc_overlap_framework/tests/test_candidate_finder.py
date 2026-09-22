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

    def test_parse_supports_reverse_strand_geometry_when_enabled(self):
        finder = Minimap2CandidateFinder(Minimap2Config(
            min_overlap=100,
            overhang_tolerance=10,
            support_reverse_strand=True,
        ))

        candidates = self._parse_lines(
            finder,
            [
                # The target is reverse-complemented and the query suffix
                # overlaps the target prefix with a positive 5 bp shift.
                "read_0\t1000\t5\t1000\t-\tread_1\t1000\t5\t1000\t980\t995\t60\n",
            ],
        )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual((candidate.left_id, candidate.right_id), ("read_0", "read_1"))
        self.assertEqual((candidate.left_orientation, candidate.right_orientation), (+1, -1))
        self.assertEqual(candidate.rough_shift, 5)

    def test_parse_can_add_reciprocal_reverse_complement_edge(self):
        finder = Minimap2CandidateFinder(Minimap2Config(
            min_overlap=100,
            overhang_tolerance=10,
            support_reverse_strand=True,
            include_reverse_complement_edges=True,
        ))

        candidates = self._parse_lines(
            finder,
            [
                "read_0\t1000\t5\t1000\t-\tread_1\t1000\t5\t1000\t980\t995\t60\n",
            ],
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            {(c.left_id, c.left_orientation, c.right_id, c.right_orientation) for c in candidates},
            {
                ("read_0", +1, "read_1", -1),
                ("read_1", +1, "read_0", -1),
            },
        )

    def test_reciprocal_hint_coordinates_use_swapped_read_lengths(self):
        finder = Minimap2CandidateFinder(Minimap2Config(
            min_overlap=80,
            overhang_tolerance=20,
            support_reverse_strand=True,
            include_reverse_complement_edges=False,
        ))

        candidates = self._parse_lines(
            finder,
            [
                "left\t148\t67\t148\t-\tright\t151\t70\t151\t81\t81\t60\n",
            ],
        )
        self.assertEqual(len(candidates), 1)
        reciprocal = Minimap2CandidateFinder._reverse_complement_candidate(candidates[0])

        self.assertEqual((reciprocal.left_id, reciprocal.right_id), ("right", "left"))
        self.assertEqual(
            (reciprocal.left_start_hint, reciprocal.left_end_hint),
            (70, 151),
        )
        self.assertEqual(
            (reciprocal.right_start_hint, reciprocal.right_end_hint),
            (0, 81),
        )

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

    def test_parse_records_full_coverage_before_directed_identity_filter(self):
        finder = Minimap2CandidateFinder(Minimap2Config(
            min_overlap=100,
            min_paf_identity=0.995,
            full_coverage_min_identity=0.990,
            support_reverse_strand=True,
        ))

        # The pair is coextensive on the reverse strand.  It has no positive
        # suffix-prefix shift, so it is evidence for node reduction rather
        # than a directed OLC edge.
        candidates = self._parse_lines(
            finder,
            [
                "read_a\t150\t0\t150\t-\tread_b\t150\t0\t150\t149\t150\t60\n",
            ],
        )

        self.assertEqual(candidates, [])
        self.assertEqual(len(finder.last_full_coverage_evidence), 1)
        evidence = finder.last_full_coverage_evidence[0]
        self.assertEqual({evidence.first_id, evidence.second_id}, {"read_a", "read_b"})
        self.assertEqual((evidence.first_orientation, evidence.second_orientation), (+1, -1))
        self.assertGreaterEqual(evidence.identity, 0.990)

    def test_full_coverage_requires_exact_endpoints_without_tolerance(self):
        finder = Minimap2CandidateFinder(Minimap2Config(
            min_overlap=100,
            min_paf_identity=0.990,
            support_reverse_strand=True,
        ))

        self._parse_lines(
            finder,
            [
                # One-base terminal overhang must not be treated as full coverage.
                "read_a\t150\t1\t150\t+\tread_b\t150\t0\t149\t149\t149\t60\n",
            ],
        )

        self.assertEqual(finder.last_full_coverage_evidence, [])

    @staticmethod
    def _parse_lines(finder: Minimap2CandidateFinder, lines: list[str]):
        return list(finder._parse_paf_lines(lines))


if __name__ == "__main__":
    unittest.main()
