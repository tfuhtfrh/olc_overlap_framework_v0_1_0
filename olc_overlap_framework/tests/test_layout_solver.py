import unittest

from olc_pipeline.data import OverlapEdge, Read
from olc_pipeline.layout_solver import (
    AnnealingResult,
    BinaryAnnealingConfig,
    BinarySimulatedAnnealer,
    EdgePathDAGHamiltonianConfig,
    EdgePathDAGQUBOHamiltonian,
    EdgePathDAGQUBOModel,
    EdgeCycleCoverDAGHamiltonianConfig,
    EdgeCycleCoverDAGQUBOHamiltonian,
    EdgeCycleCoverDAGQUBOModel,
    EdgePathCoverDAGHamiltonianConfig,
    EdgePathCoverDAGQUBOHamiltonian,
    EdgePathCoverDAGQUBOModel,
    MissingEdgeHamiltonianConfig,
    MissingEdgeQUBOHamiltonian,
    OverlapRewardScorer,
    PDFAssemblyHamiltonianConfig,
    PDFAssemblyQUBOHamiltonian,
    PDFAssemblyQUBOModel,
    PermutationEnergyEvaluator,
    PermutationLocalSearchPolisher,
    QUBOHamiltonianBuilder,
    QUBOLayoutSolver,
    QUBOModel,
    WeightedOverlapHamiltonianConfig,
    WeightedOverlapQUBOHamiltonian,
    qubo_sample_for_order,
)


def make_edge(
    left_id: str,
    right_id: str,
    weight: float = 1.0,
    overlap_len: int = 10,
    mi: float | None = None,
) -> OverlapEdge:
    return OverlapEdge(
        left_id=left_id,
        right_id=right_id,
        left_start=0,
        left_end=overlap_len,
        right_start=0,
        right_end=overlap_len,
        overlap_len=overlap_len,
        shift=overlap_len,
        matches=overlap_len,
        mismatches=0,
        insertions=0,
        deletions=0,
        gaps=0,
        edit_distance=0,
        error_rate=0.0,
        identity=1.0,
        dp_score=weight,
        mi=mi,
        weight_mi=mi,
        weight_dp=weight,
        accepted=True,
    )


class QUBOLayoutSolverTests(unittest.TestCase):
    def test_missing_edge_hamiltonian_matches_valid_path_energy(self):
        reads = [Read("r0", "A"), Read("r1", "C"), Read("r2", "G")]
        hamiltonian = MissingEdgeQUBOHamiltonian(MissingEdgeHamiltonianConfig(
            read_once_penalty=100.0,
            position_once_penalty=100.0,
            missing_edge_penalty=3.0,
        ))
        model = hamiltonian.build(reads, [make_edge("r0", "r1"), make_edge("r1", "r2")])

        sample = [0] * model.num_variables
        sample[model.variable_index(0, 0)] = 1
        sample[model.variable_index(1, 1)] = 1
        sample[model.variable_index(2, 2)] = 1

        self.assertEqual(model.energy(sample), 0.0)

        sample_missing_edge = [0] * model.num_variables
        sample_missing_edge[model.variable_index(0, 0)] = 1
        sample_missing_edge[model.variable_index(2, 1)] = 1
        sample_missing_edge[model.variable_index(1, 2)] = 1

        self.assertEqual(model.energy(sample_missing_edge), 6.0)

    def test_qubo_solver_recovers_directed_overlap_chain(self):
        reads = [Read("r0", "A"), Read("r1", "C"), Read("r2", "G")]
        solver = QUBOLayoutSolver(
            hamiltonian=MissingEdgeQUBOHamiltonian(MissingEdgeHamiltonianConfig(
                read_once_penalty=100.0,
                position_once_penalty=100.0,
                missing_edge_penalty=5.0,
            )),
            annealer=BinarySimulatedAnnealer(BinaryAnnealingConfig(
                initial_temperature=20.0,
                final_temperature=0.01,
                cooling_rate=0.90,
                sweeps_per_temperature=10,
                seed=7,
                random_restarts=8,
                swap_move_probability=0.8,
            )),
        )

        result = solver.solve(reads, [make_edge("r0", "r1"), make_edge("r1", "r2")])

        self.assertEqual(result.order, ["r0", "r1", "r2"])
        self.assertTrue(result.metadata["valid_binary_layout"])
        self.assertEqual(result.objective_value, 0.0)

    def test_custom_hamiltonian_builder_can_be_injected(self):
        class PutR2FirstHamiltonian(QUBOHamiltonianBuilder):
            def build(self, reads, edges, weight_mode="dp"):
                del edges, weight_mode
                model = QUBOModel.for_reads(reads)
                for read_index, read in enumerate(reads):
                    variable = model.variable_index(read_index, 0)
                    model.add_linear(variable, -10.0 if read.rid == "r2" else 10.0)
                return model

        reads = [Read("r0", "A"), Read("r1", "C"), Read("r2", "G")]
        solver = QUBOLayoutSolver(
            hamiltonian=PutR2FirstHamiltonian(),
            annealer=BinarySimulatedAnnealer(BinaryAnnealingConfig(
                initial_temperature=5.0,
                final_temperature=0.01,
                cooling_rate=0.90,
                sweeps_per_temperature=5,
                seed=3,
                random_restarts=4,
            )),
        )

        result = solver.solve(reads, [])

        self.assertEqual(result.order[0], "r2")

    def test_overlap_reward_scorer_supports_candidate_metrics(self):
        edge = make_edge("r0", "r1", weight=7.0)
        edge.weight_mi = 5.0
        edge.mi = 1.5
        edge.nmi = 0.5

        scorer = OverlapRewardScorer()

        self.assertEqual(scorer.score(edge, "overlap_len"), 10.0)
        self.assertEqual(scorer.score(edge, "overlap_len_power:1.5"), 10.0 ** 1.5)
        self.assertEqual(scorer.score(edge, "overlap_len_power2"), 100.0)
        self.assertEqual(scorer.score(edge, "overlap_len_power3"), 1000.0)
        self.assertEqual(scorer.score(edge, "dp"), 7.0)
        self.assertEqual(scorer.score(edge, "mi"), 5.0)
        self.assertEqual(scorer.score(edge, "raw_mi"), 1.5)
        self.assertEqual(scorer.score(edge, "nmi"), 0.5)
        self.assertEqual(scorer.score(edge, "weighted_nmi"), 5.0)

    def test_weighted_overlap_hamiltonian_rewards_stronger_edges(self):
        reads = [Read("r0", "A"), Read("r1", "C"), Read("r2", "G")]
        hamiltonian = WeightedOverlapQUBOHamiltonian(WeightedOverlapHamiltonianConfig(
            read_once_penalty=100.0,
            position_once_penalty=100.0,
            missing_edge_penalty=3.0,
            edge_reward_scale=2.0,
            score_mode="dp",
            normalize_rewards=True,
        ))
        model = hamiltonian.build(
            reads,
            [
                make_edge("r0", "r1", weight=10.0),
                make_edge("r1", "r2", weight=10.0),
                make_edge("r0", "r2", weight=1.0),
            ],
        )

        strong_path = [0] * model.num_variables
        strong_path[model.variable_index(0, 0)] = 1
        strong_path[model.variable_index(1, 1)] = 1
        strong_path[model.variable_index(2, 2)] = 1

        weak_then_missing_path = [0] * model.num_variables
        weak_then_missing_path[model.variable_index(0, 0)] = 1
        weak_then_missing_path[model.variable_index(2, 1)] = 1
        weak_then_missing_path[model.variable_index(1, 2)] = 1

        self.assertLess(model.energy(strong_path), model.energy(weak_then_missing_path))
        self.assertEqual(hamiltonian.last_edge_count, 3)

    def test_permutation_energy_matches_full_qubo_energy_for_valid_order(self):
        reads = [Read("r0", "A"), Read("r1", "C"), Read("r2", "G")]
        hamiltonian = WeightedOverlapQUBOHamiltonian(WeightedOverlapHamiltonianConfig(
            read_once_penalty=100.0,
            position_once_penalty=100.0,
            missing_edge_penalty=3.0,
            edge_reward_scale=2.0,
            score_mode="dp",
            normalize_rewards=True,
        ))
        model = hamiltonian.build(reads, [make_edge("r0", "r1", weight=10.0)])
        order = ["r0", "r1", "r2"]

        sample = PermutationLocalSearchPolisher._order_to_sample(model, order)
        fast_energy = PermutationEnergyEvaluator(model).energy(order)

        self.assertEqual(fast_energy, model.energy(sample))

    def test_edge_path_dag_hamiltonian_uses_one_variable_per_unique_edge(self):
        reads = [Read(f"r{index}", "A") for index in range(4)]
        edges = [
            make_edge("r0", "r1"),
            make_edge("r1", "r2"),
            make_edge("r2", "r3"),
            make_edge("r0", "r2"),
            make_edge("r0", "r2", weight=2.0),
        ]
        hamiltonian = EdgePathDAGQUBOHamiltonian(EdgePathDAGHamiltonianConfig(
            edge_count_penalty=10.0,
            degree_penalty=10.0,
            edge_reward_scale=0.0,
            score_mode="dp",
        ))

        model = hamiltonian.build(reads, edges)

        self.assertIsInstance(model, EdgePathDAGQUBOModel)
        self.assertEqual(model.num_variables, 4)
        self.assertEqual(hamiltonian.last_edge_count, 4)

        valid_path = qubo_sample_for_order(model, ["r0", "r1", "r2", "r3"])
        self.assertEqual(model.energy(valid_path), 0.0)

        too_few_edges = valid_path.copy()
        too_few_edges[model.edge_variable_index("r2", "r3")] = 0
        self.assertEqual(model.energy(too_few_edges), 10.0)

        branching = [0] * model.num_variables
        for edge_pair in [("r0", "r1"), ("r0", "r2"), ("r2", "r3")]:
            branching[model.edge_variable_index(*edge_pair)] = 1
        self.assertEqual(model.energy(branching), 10.0)

    def test_edge_path_dag_hamiltonian_rejects_cycles(self):
        reads = [Read("r0", "A"), Read("r1", "C"), Read("r2", "G")]
        edges = [
            make_edge("r0", "r1"),
            make_edge("r1", "r2"),
            make_edge("r2", "r0"),
        ]

        with self.assertRaisesRegex(ValueError, "acyclic"):
            EdgePathDAGQUBOHamiltonian().build(reads, edges)

    def test_edge_path_dag_solver_decodes_selected_path(self):
        class FixedEdgeAnnealer:
            def solve(self, model):
                sample = qubo_sample_for_order(model, ["r0", "r1", "r2", "r3"])
                return AnnealingResult(
                    sample=sample,
                    energy=model.energy(sample),
                    iterations=1,
                    accepted_moves=0,
                    backend="fixed",
                )

        reads = [Read(f"r{index}", "A") for index in range(4)]
        edges = [
            make_edge("r0", "r1"),
            make_edge("r1", "r2"),
            make_edge("r2", "r3"),
            make_edge("r0", "r2"),
        ]
        solver = QUBOLayoutSolver(
            hamiltonian=EdgePathDAGQUBOHamiltonian(EdgePathDAGHamiltonianConfig(
                edge_count_penalty=10.0,
                degree_penalty=10.0,
                edge_reward_scale=1.0,
                score_mode="dp",
            )),
            annealer=FixedEdgeAnnealer(),
        )

        result = solver.solve(reads, edges)

        self.assertEqual(result.order, ["r0", "r1", "r2", "r3"])
        self.assertTrue(result.metadata["valid_edge_path"])
        self.assertEqual(result.metadata["selected_edge_count"], 3)
        self.assertEqual(result.metadata["in_degree_violations"], 0)
        self.assertEqual(result.metadata["out_degree_violations"], 0)
        self.assertEqual(result.metadata["in_degree_conflicts"], {})
        self.assertEqual(result.metadata["out_degree_conflicts"], {})

    def test_edge_path_cover_dag_uses_edge_source_and_sink_variables(self):
        reads = [Read(f"r{index}", "A") for index in range(4)]
        edges = [
            make_edge("r0", "r1"),
            make_edge("r1", "r2"),
            make_edge("r2", "r3"),
            make_edge("r0", "r2"),
        ]
        hamiltonian = EdgePathCoverDAGQUBOHamiltonian(EdgePathCoverDAGHamiltonianConfig(
            degree_penalty=10.0,
            isolate_penalty=10.0,
            path_break_penalty=5.0,
            edge_reward_scale=0.0,
            score_mode="dp",
        ))

        model = hamiltonian.build(reads, edges)

        self.assertIsInstance(model, EdgePathCoverDAGQUBOModel)
        self.assertEqual(model.num_variables, 4 + 2 * len(reads))

        valid_path = qubo_sample_for_order(model, ["r0", "r1", "r2", "r3"])
        self.assertEqual(model.energy(valid_path), 0.0)

    def test_edge_path_cover_dag_decodes_disconnected_path_cover(self):
        reads = [Read(f"r{index}", "A") for index in range(4)]
        edges = [
            make_edge("r0", "r1"),
            make_edge("r1", "r2"),
            make_edge("r2", "r3"),
        ]
        hamiltonian = EdgePathCoverDAGQUBOHamiltonian(EdgePathCoverDAGHamiltonianConfig(
            degree_penalty=10.0,
            isolate_penalty=10.0,
            path_break_penalty=5.0,
            edge_reward_scale=0.0,
            score_mode="dp",
        ))
        model = hamiltonian.build(reads, edges)
        sample = [0] * model.num_variables
        sample[model.edge_variable_index("r0", "r1")] = 1
        sample[model.edge_variable_index("r2", "r3")] = 1
        for source_id in ["r0", "r2"]:
            sample[model.source_variable_index(source_id)] = 1
        for sink_id in ["r1", "r3"]:
            sample[model.sink_variable_index(sink_id)] = 1

        order, metadata = QUBOLayoutSolver._decode_edge_path_cover(model, sample)

        self.assertEqual(order, ["r0", "r1", "r2", "r3"])
        self.assertTrue(metadata["valid_path_cover"])
        self.assertFalse(metadata["single_path_layout"])
        self.assertEqual(metadata["path_count"], 2)
        self.assertEqual(metadata["source_constraint_violations"], 0)
        self.assertEqual(metadata["sink_constraint_violations"], 0)
        self.assertEqual(metadata["isolated_node_count"], 0)
        self.assertEqual(model.energy(sample), 100.0)

    def test_edge_path_cover_dag_penalizes_more_than_two_paths_strongly(self):
        reads = [Read(f"r{index}", "A") for index in range(6)]
        edges = [
            make_edge("r0", "r1"),
            make_edge("r2", "r3"),
            make_edge("r4", "r5"),
        ]
        hamiltonian = EdgePathCoverDAGQUBOHamiltonian(EdgePathCoverDAGHamiltonianConfig(
            degree_penalty=10.0,
            isolate_penalty=10.0,
            path_break_penalty=5.0,
            path_count_cap_penalty=100.0,
            edge_reward_scale=0.0,
            score_mode="dp",
        ))
        model = hamiltonian.build(reads, edges)
        sample = [0] * model.num_variables
        for edge in edges:
            sample[model.edge_variable_index(edge.left_id, edge.right_id)] = 1
        for source_id in ["r0", "r2", "r4"]:
            sample[model.source_variable_index(source_id)] = 1
        for sink_id in ["r1", "r3", "r5"]:
            sample[model.sink_variable_index(sink_id)] = 1

        _, metadata = QUBOLayoutSolver._decode_edge_path_cover(model, sample)

        self.assertTrue(metadata["valid_path_cover"])
        self.assertEqual(metadata["path_count"], 3)
        self.assertEqual(model.energy(sample), 400.0)

    def test_edge_path_cover_dag_penalizes_isolated_nodes(self):
        reads = [Read(f"r{index}", "A") for index in range(3)]
        hamiltonian = EdgePathCoverDAGQUBOHamiltonian(EdgePathCoverDAGHamiltonianConfig(
            degree_penalty=10.0,
            isolate_penalty=7.0,
            path_break_penalty=0.0,
            edge_reward_scale=0.0,
            score_mode="dp",
        ))
        model = hamiltonian.build(
            reads,
            [make_edge("r0", "r1"), make_edge("r1", "r2")],
        )
        sample = qubo_sample_for_order(model, ["r0", "r1", "r2"])
        sample[model.edge_variable_index("r1", "r2")] = 0
        sample[model.sink_variable_index("r1")] = 1
        sample[model.source_variable_index("r2")] = 1
        sample[model.sink_variable_index("r2")] = 1

        _, metadata = QUBOLayoutSolver._decode_edge_path_cover(model, sample)

        self.assertFalse(metadata["valid_path_cover"])
        self.assertEqual(metadata["isolated_nodes"], ["r2"])
        self.assertEqual(model.energy(sample), 107.0)

    def test_edge_cycle_cover_dag_uses_void_edges(self):
        reads = [Read(f"r{index}", "A") for index in range(4)]
        edges = [
            make_edge("r0", "r1"),
            make_edge("r1", "r2"),
            make_edge("r2", "r3"),
            make_edge("r0", "r2"),
        ]
        hamiltonian = EdgeCycleCoverDAGQUBOHamiltonian(EdgeCycleCoverDAGHamiltonianConfig(
            degree_penalty=10.0,
            edge_reward_scale=0.0,
            score_mode="dp",
        ))

        model = hamiltonian.build(reads, edges)

        self.assertIsInstance(model, EdgeCycleCoverDAGQUBOModel)
        self.assertEqual(model.num_variables, 4 + 2 * len(reads))

        valid_cycle = qubo_sample_for_order(model, ["r0", "r1", "r2", "r3"])
        self.assertEqual(model.energy(valid_cycle), 0.0)

        two_void_out_edges = valid_cycle.copy()
        two_void_out_edges[model.source_variable_index("r2")] = 1
        self.assertGreater(model.energy(two_void_out_edges), model.energy(valid_cycle))

    def test_edge_cycle_cover_dag_decodes_single_void_cycle(self):
        reads = [Read(f"r{index}", "A") for index in range(4)]
        edges = [
            make_edge("r0", "r1"),
            make_edge("r1", "r2"),
            make_edge("r2", "r3"),
            make_edge("r0", "r2"),
        ]
        hamiltonian = EdgeCycleCoverDAGQUBOHamiltonian(EdgeCycleCoverDAGHamiltonianConfig(
            degree_penalty=10.0,
            edge_reward_scale=0.0,
            score_mode="dp",
        ))
        model = hamiltonian.build(reads, edges)
        sample = qubo_sample_for_order(model, ["r0", "r1", "r2", "r3"])

        order, metadata = QUBOLayoutSolver._decode_edge_cycle_cover(model, sample)

        self.assertEqual(order, ["r0", "r1", "r2", "r3"])
        self.assertTrue(metadata["valid_edge_cycle"])
        self.assertTrue(metadata["single_cycle_cover"])
        self.assertEqual(metadata["selected_sources"], ["r0"])
        self.assertEqual(metadata["selected_sinks"], ["r3"])
        self.assertEqual(
            metadata["cycle_components"],
            [["__void__", "r0", "r1", "r2", "r3", "__void__"]],
        )

    def test_pdf_assembly_hamiltonian_uses_path_a_with_pdf_terms(self):
        reads = [Read(f"r{index}", "AAAA") for index in range(3)]
        edges = [
            make_edge("r0", "r1", overlap_len=2),
            make_edge("r1", "r2", overlap_len=2),
            make_edge("r0", "r2", overlap_len=1),
        ]
        hamiltonian = PDFAssemblyQUBOHamiltonian(
            PDFAssemblyHamiltonianConfig(
                degree_penalty=10.0,
                length_target=8.0,
                length_penalty=1.0,
            )
        )
        model = hamiltonian.build(reads, edges)

        self.assertIsInstance(model, PDFAssemblyQUBOModel)
        self.assertEqual(model.num_variables, len(edges))

        target_path = qubo_sample_for_order(model, ["r0", "r1", "r2"])
        _, metadata = QUBOLayoutSolver._decode_pdf_assembly(model, target_path)

        self.assertTrue(metadata["valid_edge_path"])
        self.assertEqual(metadata["selected_edge_count"], 2)
        self.assertEqual(metadata["target_edge_count"], 2)
        self.assertEqual(model.energy(target_path), 0.0)

        conflict_sample = [0] * model.num_variables
        conflict_sample[model.edge_variable_index("r0", "r1")] = 1
        conflict_sample[model.edge_variable_index("r0", "r2")] = 1
        self.assertEqual(model.energy(conflict_sample), 11.0)


if __name__ == "__main__":
    unittest.main()
