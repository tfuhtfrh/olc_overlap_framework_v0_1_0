import unittest

from olc_pipeline.data import ContainmentEvidence, FullCoverageEvidence, OverlapEdge, Read
from olc_pipeline.graph_builder import (
    build_read_only_graph,
    check_hamilton_cycle,
    remove_contained_reads,
    remove_full_coverage_reads,
    remove_low_quality_reads,
    select_largest_oriented_scc,
    select_one_orientation_per_read,
)
from olc_pipeline.reference_evaluator import (
    CircularReferenceEvaluator,
    ReferencePlacement,
)


def edge(left: str, right: str) -> OverlapEdge:
    return OverlapEdge(
        left_id=left,
        right_id=right,
        left_start=2,
        left_end=4,
        right_start=0,
        right_end=2,
        overlap_len=2,
        shift=2,
        matches=2,
        mismatches=0,
        insertions=0,
        deletions=0,
        gaps=0,
        edit_distance=0,
        error_rate=0.0,
        identity=1.0,
        dp_score=2.0,
        weight_dp=2.0,
        accepted=True,
    )


class RealDataHelperTests(unittest.TestCase):
    def test_largest_oriented_scc_uses_no_reference(self):
        reads = [Read(f"r{index}", "AAAA") for index in range(4)]
        edges = [
            edge("r0", "r1"),
            edge("r1", "r2"),
            edge("r2", "r0"),
            edge("r2", "r3"),
        ]

        result = select_largest_oriented_scc(reads, edges)

        self.assertEqual([read.rid for read in result.reads], ["r0", "r1", "r2"])
        self.assertEqual(len(result.edges), 3)

    def test_circular_reference_evaluation_ignores_start_rotation(self):
        evaluator = CircularReferenceEvaluator("ACGTACGT")
        placements = {
            "r0": ReferencePlacement("r0", +1, 0, 4, 4, 1.0, 60),
            "r1": ReferencePlacement("r1", +1, 2, 6, 4, 1.0, 60),
            "r2": ReferencePlacement("r2", +1, 4, 8, 4, 1.0, 60),
        }

        report = evaluator.evaluate(
            [("r1", +1), ("r2", +1), ("r0", +1)],
            placements,
        )

        self.assertTrue(report.exact_forward_cycle)
        self.assertEqual(report.forward_edge_accuracy, 1.0)
        self.assertEqual(report.orientation_accuracy, 1.0)
        self.assertEqual(report.reference_coverage_bp, 8)

    def test_single_orientation_selection_keeps_one_copy_per_read(self):
        reads = [Read(f"r{index}", "AAAA") for index in range(3)]
        edges = [
            edge("r0", "r1"),
            edge("r1", "r2"),
            edge("r2", "r0"),
        ]
        for item in edges:
            item.left_orientation = +1
            item.right_orientation = +1

        result = select_one_orientation_per_read(reads, edges)

        self.assertEqual(result.orientation_by_read, {"r0": +1, "r1": +1, "r2": +1})
        self.assertEqual(len(result.edges), 3)
        self.assertEqual(
            {(edge.left_id, edge.left_orientation) for edge in result.edges},
            {("r0", +1), ("r1", +1), ("r2", +1)},
        )

    def test_single_orientation_selection_propagates_relative_direction(self):
        reads = [Read("r0", "AAAA"), Read("r1", "AAAA")]
        oriented_edge = edge("r0", "r1")
        oriented_edge.left_orientation = +1
        oriented_edge.right_orientation = -1

        result = select_one_orientation_per_read(reads, [oriented_edge])

        self.assertEqual(result.orientation_by_read["r0"], +1)
        self.assertEqual(result.orientation_by_read["r1"], -1)
        self.assertEqual(result.constraint_conflict_count, 0)
        self.assertEqual(len(result.edges), 1)

    def test_containment_filter_removes_only_contained_reads(self):
        reads = [Read("short", "AAAA"), Read("long", "AAAAAAAA")]
        evidence = [ContainmentEvidence("short", "long", +1, +1, 4, 1.0, 60)]

        result = remove_contained_reads(reads, [], evidence)

        self.assertEqual([read.rid for read in result.reads], ["long"])
        self.assertEqual(result.removal_reason_by_read["short"], "contained_in:long")

    def test_full_coverage_filter_keeps_longest_read_and_preserves_other_edges(self):
        reads = [
            Read("short", "AAAA"),
            Read("long", "AAAAAA"),
            Read("middle", "AAAAA"),
            Read("outside", "CCCC"),
        ]
        evidence = [
            FullCoverageEvidence("short", "middle", +1, +1, 4, 0.995, 60),
            FullCoverageEvidence("middle", "long", +1, -1, 5, 0.995, 60),
        ]
        edges = [edge("long", "outside"), edge("outside", "long")]

        result = remove_full_coverage_reads(reads, edges, evidence)

        self.assertEqual(
            [read.rid for read in result.reads], ["long", "outside"],
        )
        self.assertEqual(result.removed_read_ids, frozenset({"short", "middle"}))
        self.assertEqual(len(result.edges), 2)
        self.assertEqual(result.removal_reason_by_read["short"], "full_coverage_with:long")
        self.assertEqual(result.removal_reason_by_read["middle"], "full_coverage_with:long")

    def test_full_coverage_filter_uses_deterministic_id_tie_break(self):
        reads = [Read("b", "AAAA"), Read("a", "TTTT")]
        evidence = [FullCoverageEvidence("a", "b", +1, +1, 4, 1.0, 60)]

        result = remove_full_coverage_reads(reads, [], evidence)

        self.assertEqual([read.rid for read in result.reads], ["b"])
        self.assertEqual(result.removed_read_ids, frozenset({"a"}))

    def test_low_quality_filter_only_removes_zero_support_nodes(self):
        reads = [Read("r0", "AAAA"), Read("r1", "AAAA"), Read("r2", "AAAA")]
        edges = [edge("r0", "r1"), edge("r1", "r0")]

        result = remove_low_quality_reads(reads, edges)

        self.assertEqual([read.rid for read in result.reads], ["r0", "r1"])
        self.assertEqual(len(result.edges), 2)
        self.assertIn("r2", result.removed_read_ids)

    def test_low_quality_filter_removes_missing_side_once_without_cascade(self):
        reads = [Read("left", "AAAA"), Read("middle", "AAAA"), Read("right", "AAAA")]
        edges = [edge("left", "middle"), edge("middle", "right")]

        result = remove_low_quality_reads(reads, edges)

        self.assertEqual(result.removed_read_ids, frozenset({"left", "right"}))
        self.assertEqual([read.rid for read in result.reads], ["middle"])
        self.assertEqual(result.edges, [])

    def test_hamilton_check_allows_branching_graph(self):
        reads = [Read(f"r{index}", "AAAA") for index in range(4)]
        cycle_edges = [
            edge("r0", "r1"),
            edge("r1", "r2"),
            edge("r2", "r3"),
            edge("r3", "r0"),
            edge("r0", "r2"),
        ]
        nodes = [(read.rid, +1) for read in reads]

        result = check_hamilton_cycle(nodes, cycle_edges, time_limit_sec=1.0)

        self.assertEqual(result.status, "yes")
        self.assertEqual(len(result.cycle), 5)

    def test_hamilton_check_merges_cycle_cover_without_dfs_time(self):
        nodes = [(f"r{index}", +1) for index in range(4)]
        edges = [
            edge("r0", "r1"), edge("r1", "r0"),
            edge("r2", "r3"), edge("r3", "r2"),
            edge("r0", "r3"), edge("r2", "r1"),
        ]

        result = check_hamilton_cycle(nodes, edges, time_limit_sec=0.0)

        self.assertEqual(result.status, "yes")
        self.assertEqual(len(result.cycle), 5)

    def test_read_only_graph_does_not_reduce_transitive_edges(self):
        reads = [Read(f"r{index}", "AAAA") for index in range(3)]
        edges = [
            edge("r0", "r1"),
            edge("r1", "r2"),
            edge("r2", "r0"),
            edge("r0", "r2"),
        ]

        result = build_read_only_graph(reads, edges)

        self.assertEqual(len(result.edges), 4)

    def test_read_only_graph_keeps_low_support_reads_by_default(self):
        reads = [Read("r0", "AAAA"), Read("r1", "AAAA"), Read("r2", "AAAA")]
        edges = [edge("r0", "r1"), edge("r1", "r0")]

        result = build_read_only_graph(reads, edges)

        self.assertEqual([read.rid for read in result.reads], ["r0", "r1", "r2"])
        self.assertEqual(result.low_quality_read_ids, frozenset())
        self.assertEqual(len(result.edges), 2)


if __name__ == "__main__":
    unittest.main()
