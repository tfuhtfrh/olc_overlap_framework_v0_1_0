import unittest

from olc_pipeline.overlap_features import gc_fraction
from olc_pipeline.simulator import RandomReadSimulator, SimulationConfig


class RandomReadSimulatorTests(unittest.TestCase):
    def test_fixed_layout_preserves_read_count_and_coordinates(self):
        _, reads = RandomReadSimulator().simulate(SimulationConfig(
            genome_len=10_000,
            read_len=2_000,
            step=1_000,
            mismatch_rate=0.0,
            ins_rate=0.0,
            del_rate=0.0,
            seed=1,
            shuffle_reads=False,
        ))

        self.assertEqual(len(reads), 9)
        self.assertEqual(reads[0].true_start, 0)
        self.assertEqual(reads[1].true_start, 1_000)
        self.assertEqual(reads[-1].true_end, 10_000)

    def test_random_layout_uses_configured_read_and_overlap_ranges(self):
        _, reads = RandomReadSimulator().simulate(SimulationConfig(
            genome_len=100_000,
            read_len_min=2_000,
            read_len_max=8_000,
            overlap_fraction_min=0.10,
            overlap_fraction_max=0.80,
            adjacent_overlap_min=1_000,
            mismatch_rate=0.0,
            ins_rate=0.0,
            del_rate=0.0,
            seed=42,
            shuffle_reads=False,
        ))

        self.assertGreater(len(reads), 2)

        for read in reads:
            read_len = read.true_end - read.true_start
            self.assertGreaterEqual(read_len, 2_000)
            self.assertLessEqual(read_len, 8_000)
            self.assertEqual(len(read.seq), read_len)

        for left, right in zip(reads, reads[1:]):
            self.assertGreater(right.true_start, left.true_start)
            overlap_len = min(left.true_end, right.true_end) - max(left.true_start, right.true_start)
            shorter_len = min(left.true_end - left.true_start, right.true_end - right.true_start)
            self.assertGreater(overlap_len, 0)
            self.assertGreaterEqual(overlap_len, 1_000)
            self.assertGreaterEqual(overlap_len, 0.10 * shorter_len - 1)
            self.assertLessEqual(overlap_len, 0.80 * shorter_len + 1)

    def test_random_layout_rejects_unsatisfiable_minimum_overlap(self):
        config = SimulationConfig(
            genome_len=100_000,
            read_len_min=2_000,
            read_len_max=8_000,
            overlap_fraction_min=0.10,
            overlap_fraction_max=0.80,
            adjacent_overlap_min=1_700,
            mismatch_rate=0.0,
            ins_rate=0.0,
            del_rate=0.0,
        )

        with self.assertRaisesRegex(ValueError, "adjacent_overlap_min"):
            RandomReadSimulator().simulate(config)

    def test_fixed_layout_uses_configured_gc_fraction(self):
        genome, reads = RandomReadSimulator().simulate(SimulationConfig(
            genome_len=50_000,
            read_len=2_000,
            step=2_000,
            mismatch_rate=0.0,
            ins_rate=0.0,
            del_rate=0.0,
            gc_fraction=0.70,
            seed=7,
            shuffle_reads=False,
        ))

        self.assertGreater(gc_fraction(genome), 0.67)
        self.assertLess(gc_fraction(genome), 0.73)
        self.assertGreater(gc_fraction(reads[0].seq), 0.64)
        self.assertLess(gc_fraction(reads[0].seq), 0.76)

    def test_simulation_rejects_invalid_gc_fraction(self):
        with self.assertRaisesRegex(ValueError, "gc_fraction"):
            RandomReadSimulator().simulate(SimulationConfig(gc_fraction=1.1))


if __name__ == "__main__":
    unittest.main()
