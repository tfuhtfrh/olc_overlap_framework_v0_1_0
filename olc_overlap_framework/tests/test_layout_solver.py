import unittest

from olc_pipeline.data import OverlapEdge, Read
from olc_pipeline.layout_solver import (
    BinaryAnnealingConfig,
    BinarySimulatedAnnealer,
    MissingEdgeHamiltonianConfig,
    MissingEdgeQUBOHamiltonian,
    OverlapRewardScorer,
    PermutationEnergyEvaluator,
    PermutationLocalSearchPolisher,
    QUBOHamiltonianBuilder,
    QUBOLayoutSolver,
    QUBOModel,
    WeightedOverlapHamiltonianConfig,
    WeightedOverlapQUBOHamiltonian,
)


def make_edge(left_id: str, right_id: str, weight: float = 1.0) -> OverlapEdge:
    return OverlapEdge(
        left_id=left_id,
        right_id=right_id,
        left_start=0,
        left_end=10,
        right_start=0,
        right_end=10,
        overlap_len=10,
        shift=10,
        matches=10,
        mismatches=0,
        insertions=0,
        deletions=0,
        gaps=0,
        edit_distance=0,
        error_rate=0.0,
        identity=1.0,
        dp_score=weight,
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
        edge.nmi = 0.5

        scorer = OverlapRewardScorer()

        self.assertEqual(scorer.score(edge, "overlap_len"), 10.0)
        self.assertEqual(scorer.score(edge, "dp"), 7.0)
        self.assertEqual(scorer.score(edge, "mi"), 5.0)
        self.assertEqual(scorer.score(edge, "nmi"), 0.5)

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


if __name__ == "__main__":
    unittest.main()
