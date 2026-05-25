"""
olc_pipeline.layout_solver
Version: 0.1.0

Layout solver interfaces. OR-Tools remains a reserved implementation slot.
The QUBO solver builds a binary x[v, j] layout model and optimizes it with a
small built-in simulated annealer so experiments can run without extra solver
dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
import random
from time import perf_counter
from typing import Callable, Optional

from .data import Read, OverlapEdge, LayoutResult

MODULE_VERSION = "0.1.0"
SUPPORTED_OVERLAP_SCORE_MODES = (
    "overlap_len",
    "identity",
    "dp",
    "mi",
    "nmi",
    "mapq",
    "matches",
    "quality",
)


class LayoutSolver(ABC):
    """
    Interface for all layout solvers.

    weight_mode should be one of:
        - "dp": use edge.weight_dp / edge.dp_score
        - "mi": use edge.weight_mi / edge.nmi
        - custom modes can be added later
    """

    @abstractmethod
    def solve(self, reads: list[Read], edges: list[OverlapEdge], weight_mode: str = "dp") -> LayoutResult:
        raise NotImplementedError


class DummyLayoutSolver(LayoutSolver):
    """No-op solver used while layout is not the current focus."""

    def solve(self, reads: list[Read], edges: list[OverlapEdge], weight_mode: str = "dp") -> LayoutResult:
        return LayoutResult(order=[], objective_value=None, solver_name="dummy")


class ORToolsLayoutSolver(LayoutSolver):
    """
    Reserved implementation slot for Google's OR-Tools.

    Expected problem type:
        directed maximum-weight Hamiltonian path / ATSP-like layout.
    """

    def solve(self, reads: list[Read], edges: list[OverlapEdge], weight_mode: str = "dp") -> LayoutResult:
        raise NotImplementedError("ORToolsLayoutSolver is not implemented in version 0.1.0")


@dataclass
class QUBOModel:
    """
    Upper-triangular QUBO model.

    Energy convention:
        E(x) = constant + sum_i linear[i] * x_i
             + sum_{i < j} quadratic[(i, j)] * x_i * x_j
    """

    read_ids: list[str]
    linear: list[float]
    quadratic: dict[tuple[int, int], float] = field(default_factory=dict)
    constant: float = 0.0

    @classmethod
    def for_reads(cls, reads: list[Read]) -> "QUBOModel":
        n = len(reads)
        return cls(read_ids=[read.rid for read in reads], linear=[0.0] * (n * n))

    @property
    def num_reads(self) -> int:
        return len(self.read_ids)

    @property
    def num_variables(self) -> int:
        return len(self.linear)

    def variable_index(self, read_index: int, position_index: int) -> int:
        return read_index * self.num_reads + position_index

    def variable_label(self, index: int) -> str:
        read_index, position_index = divmod(index, self.num_reads)
        return f"x[{self.read_ids[read_index]},{position_index}]"

    def add_constant(self, value: float) -> None:
        self.constant += value

    def add_linear(self, index: int, value: float) -> None:
        self.linear[index] += value

    def add_quadratic(self, left: int, right: int, value: float) -> None:
        if left == right:
            self.add_linear(left, value)
            return
        key = (left, right) if left < right else (right, left)
        self.quadratic[key] = self.quadratic.get(key, 0.0) + value

    def energy(self, sample: list[int]) -> float:
        value = self.constant
        for index, bit in enumerate(sample):
            if bit:
                value += self.linear[index]
        for (left, right), coeff in self.quadratic.items():
            if sample[left] and sample[right]:
                value += coeff
        return value

    def to_dimod_bqm(self):
        """
        Convert to dimod.BinaryQuadraticModel.

        dimod is imported lazily so the built-in solver and unit tests do not
        require D-Wave packages unless this backend is selected.
        """
        try:
            import dimod
        except ImportError as exc:
            raise RuntimeError(
                "dimod is required for the D-Wave sampler backend. "
                "Install it with: pip install dwave-samplers dimod"
            ) from exc

        bqm = dimod.BinaryQuadraticModel({}, {}, self.constant, dimod.BINARY)
        for index, bias in enumerate(self.linear):
            bqm.add_variable(index, bias)
        for (left, right), bias in self.quadratic.items():
            bqm.add_interaction(left, right, bias)
        return bqm

    def adjacency(self) -> list[dict[int, float]]:
        adj: list[dict[int, float]] = [dict() for _ in range(self.num_variables)]
        for (left, right), coeff in self.quadratic.items():
            if coeff == 0.0:
                continue
            adj[left][right] = adj[left].get(right, 0.0) + coeff
            adj[right][left] = adj[right].get(left, 0.0) + coeff
        return adj


class QUBOHamiltonianBuilder(ABC):
    """Build a QUBO model from overlap edges. Replace this for new Hamiltonians."""

    @abstractmethod
    def build(self, reads: list[Read], edges: list[OverlapEdge], weight_mode: str = "dp") -> QUBOModel:
        raise NotImplementedError


@dataclass(frozen=True)
class MissingEdgeHamiltonianConfig:
    """
    Coefficients for the initial Hamiltonian:

        A1 * sum_v (1 - sum_j x[v,j])^2
      + A2 * sum_j (1 - sum_v x[v,j])^2
      + B  * sum_(u,v not in E) sum_j x[u,j] x[v,j+1]

    The first two coefficients should be high because they penalize invalid
    layouts. The third coefficient is the editable edge-missing penalty.
    """

    read_once_penalty: float = 100.0
    position_once_penalty: float = 100.0
    missing_edge_penalty: float = 1.0


class MissingEdgeQUBOHamiltonian(QUBOHamiltonianBuilder):
    """Initial QUBO Hamiltonian matching the image supplied by the user."""

    def __init__(self, config: Optional[MissingEdgeHamiltonianConfig] = None):
        self.config = config or MissingEdgeHamiltonianConfig()
        self.last_edge_count = 0

    def build(self, reads: list[Read], edges: list[OverlapEdge], weight_mode: str = "dp") -> QUBOModel:
        del weight_mode  # This Hamiltonian uses E as a directed edge set.
        model = QUBOModel.for_reads(reads)
        n = model.num_reads
        read_ids = set(model.read_ids)
        edge_set = {
            (edge.left_id, edge.right_id)
            for edge in edges
            if edge.left_id in read_ids and edge.right_id in read_ids
        }
        self.last_edge_count = len(edge_set)

        self._add_read_once_terms(model, self.config.read_once_penalty)
        self._add_position_once_terms(model, self.config.position_once_penalty)
        self._add_missing_edge_terms(model, edge_set, self.config.missing_edge_penalty)
        return model

    @staticmethod
    def _add_read_once_terms(model: QUBOModel, penalty: float) -> None:
        n = model.num_reads
        for read_index in range(n):
            model.add_constant(penalty)
            variables = [model.variable_index(read_index, pos) for pos in range(n)]
            for variable in variables:
                model.add_linear(variable, -penalty)
            for left_offset, left in enumerate(variables):
                for right in variables[left_offset + 1:]:
                    model.add_quadratic(left, right, 2.0 * penalty)

    @staticmethod
    def _add_position_once_terms(model: QUBOModel, penalty: float) -> None:
        n = model.num_reads
        for position_index in range(n):
            model.add_constant(penalty)
            variables = [model.variable_index(read_index, position_index) for read_index in range(n)]
            for variable in variables:
                model.add_linear(variable, -penalty)
            for left_offset, left in enumerate(variables):
                for right in variables[left_offset + 1:]:
                    model.add_quadratic(left, right, 2.0 * penalty)

    @staticmethod
    def _add_missing_edge_terms(
        model: QUBOModel,
        edge_set: set[tuple[str, str]],
        penalty: float,
    ) -> None:
        n = model.num_reads
        for position_index in range(n - 1):
            for left_read_index, left_id in enumerate(model.read_ids):
                left_variable = model.variable_index(left_read_index, position_index)
                for right_read_index, right_id in enumerate(model.read_ids):
                    if (left_id, right_id) in edge_set:
                        continue
                    right_variable = model.variable_index(right_read_index, position_index + 1)
                    model.add_quadratic(left_variable, right_variable, penalty)


@dataclass(frozen=True)
class OverlapRewardScorerConfig:
    """
    Select how an OverlapEdge becomes a scalar reward.

    Supported modes:
        - "overlap_len": refined overlap length
        - "identity": refined identity
        - "dp": weight_dp, then dp_score, then overlap_len * identity
        - "mi": weight_mi, then overlap_len * nmi, then mi
        - "nmi": normalized mutual information
        - "mapq": minimap2 mapq
        - "matches": refined match count
        - "quality": overlap_len * identity * (1 - error_rate)
    """

    score_mode: str = "dp"


class OverlapRewardScorer:
    """Reusable overlap-edge scorer for weighted QUBO Hamiltonians."""

    def __init__(
        self,
        config: Optional[OverlapRewardScorerConfig] = None,
        custom_score: Optional[Callable[[OverlapEdge], float]] = None,
    ):
        self.config = config or OverlapRewardScorerConfig()
        self.custom_score = custom_score

    def score(self, edge: OverlapEdge, weight_mode: Optional[str] = None) -> float:
        if self.custom_score is not None:
            return float(self.custom_score(edge))

        mode = weight_mode or self.config.score_mode
        if mode == "overlap_len":
            return float(edge.overlap_len)
        if mode == "identity":
            return float(edge.identity)
        if mode == "dp":
            if edge.weight_dp is not None:
                return float(edge.weight_dp)
            if edge.dp_score is not None:
                return float(edge.dp_score)
            return float(edge.overlap_len * edge.identity)
        if mode == "mi":
            if edge.weight_mi is not None:
                return float(edge.weight_mi)
            if edge.nmi is not None:
                return float(edge.overlap_len * edge.nmi)
            if edge.mi is not None:
                return float(edge.mi)
            return 0.0
        if mode == "nmi":
            return float(edge.nmi or 0.0)
        if mode == "mapq":
            return float(edge.mapq)
        if mode == "matches":
            return float(edge.matches)
        if mode == "quality":
            return float(edge.overlap_len * edge.identity * max(0.0, 1.0 - edge.error_rate))
        raise ValueError(
            f"Unsupported overlap reward score mode: {mode!r}. "
            f"Supported modes: {', '.join(SUPPORTED_OVERLAP_SCORE_MODES)}"
        )


@dataclass(frozen=True)
class WeightedOverlapHamiltonianConfig:
    """
    Coefficients for rewarding strong overlap edges:

        A_read * read-once constraints
      + A_pos  * position-once constraints
      + C_miss * adjacent pairs not in E
      - B_edge * reward(u, v) for adjacent pairs in E

    Keep A_read/A_pos high enough that invalid binary layouts remain expensive
    even when edge rewards are large.
    """

    read_once_penalty: float = 100.0
    position_once_penalty: float = 100.0
    missing_edge_penalty: float = 1.0
    edge_reward_scale: float = 1.0
    score_mode: Optional[str] = None
    normalize_rewards: bool = True
    max_reward_score: Optional[float] = None
    min_reward_score: float = 0.0


class WeightedOverlapQUBOHamiltonian(QUBOHamiltonianBuilder):
    """
    QUBO Hamiltonian that rewards stronger overlap edges.

    This is the general testing frame for overlap scoring candidates. Use
    `score_mode` or a custom OverlapRewardScorer to compare overlap length, DP
    weights, MI weights, NMI, or future quality metrics.
    """

    def __init__(
        self,
        config: Optional[WeightedOverlapHamiltonianConfig] = None,
        scorer: Optional[OverlapRewardScorer] = None,
    ):
        self.config = config or WeightedOverlapHamiltonianConfig()
        self.scorer = scorer or OverlapRewardScorer()
        self.last_edge_count = 0
        self.last_reward_min: Optional[float] = None
        self.last_reward_max: Optional[float] = None

    def build(self, reads: list[Read], edges: list[OverlapEdge], weight_mode: str = "dp") -> QUBOModel:
        model = QUBOModel.for_reads(reads)
        score_mode = self.config.score_mode or weight_mode
        reward_by_pair = self._best_rewards_by_pair(model.read_ids, edges, score_mode)

        self.last_edge_count = len(reward_by_pair)
        self.last_reward_min = min(reward_by_pair.values()) if reward_by_pair else None
        self.last_reward_max = max(reward_by_pair.values()) if reward_by_pair else None

        MissingEdgeQUBOHamiltonian._add_read_once_terms(model, self.config.read_once_penalty)
        MissingEdgeQUBOHamiltonian._add_position_once_terms(model, self.config.position_once_penalty)
        self._add_weighted_adjacency_terms(model, reward_by_pair)
        return model

    def _best_rewards_by_pair(
        self,
        read_ids: list[str],
        edges: list[OverlapEdge],
        score_mode: str,
    ) -> dict[tuple[str, str], float]:
        read_id_set = set(read_ids)
        raw_scores: dict[tuple[str, str], float] = {}
        for edge in edges:
            if edge.left_id not in read_id_set or edge.right_id not in read_id_set:
                continue
            if edge.left_id == edge.right_id:
                continue
            score = self.scorer.score(edge, score_mode)
            if score < self.config.min_reward_score:
                continue
            key = (edge.left_id, edge.right_id)
            old_score = raw_scores.get(key)
            if old_score is None or score > old_score:
                raw_scores[key] = score

        if not self.config.normalize_rewards:
            return raw_scores

        denominator = self.config.max_reward_score
        if denominator is None:
            denominator = max(raw_scores.values(), default=1.0)
        if denominator <= 0.0:
            return {key: 0.0 for key in raw_scores}
        return {key: value / denominator for key, value in raw_scores.items()}

    def _add_weighted_adjacency_terms(
        self,
        model: QUBOModel,
        reward_by_pair: dict[tuple[str, str], float],
    ) -> None:
        n = model.num_reads
        for position_index in range(n - 1):
            for left_read_index, left_id in enumerate(model.read_ids):
                left_variable = model.variable_index(left_read_index, position_index)
                for right_read_index, right_id in enumerate(model.read_ids):
                    right_variable = model.variable_index(right_read_index, position_index + 1)
                    reward = reward_by_pair.get((left_id, right_id))
                    if reward is None:
                        model.add_quadratic(
                            left_variable,
                            right_variable,
                            self.config.missing_edge_penalty,
                        )
                    else:
                        model.add_quadratic(
                            left_variable,
                            right_variable,
                            -self.config.edge_reward_scale * reward,
                        )


@dataclass(frozen=True)
class BinaryAnnealingConfig:
    """Settings for the built-in binary simulated annealer."""

    initial_temperature: float = 10.0
    final_temperature: float = 0.01
    cooling_rate: float = 0.95
    sweeps_per_temperature: int = 5
    seed: int = 42
    random_restarts: int = 4
    start_from_valid_permutation: bool = True
    swap_move_probability: float = 0.5

    def validate(self) -> None:
        if self.initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be positive")
        if self.final_temperature <= 0.0:
            raise ValueError("final_temperature must be positive")
        if not 0.0 < self.cooling_rate < 1.0:
            raise ValueError("cooling_rate must be in (0, 1)")
        if self.sweeps_per_temperature <= 0:
            raise ValueError("sweeps_per_temperature must be positive")
        if self.random_restarts <= 0:
            raise ValueError("random_restarts must be positive")
        if not 0.0 <= self.swap_move_probability <= 1.0:
            raise ValueError("swap_move_probability must be in [0, 1]")


@dataclass(frozen=True)
class AnnealingResult:
    sample: list[int]
    energy: float
    iterations: int
    accepted_moves: int
    backend: str = "builtin"


@dataclass(frozen=True)
class PermutationPolishConfig:
    """Local search over valid read permutations after binary annealing."""

    enabled: bool = True
    max_passes: int = 20
    use_swap_moves: bool = True
    use_insert_moves: bool = True
    use_segment_insert_moves: bool = True
    max_segment_len: int = 12

    def validate(self) -> None:
        if self.max_passes <= 0:
            raise ValueError("max_passes must be positive")
        if self.max_segment_len <= 0:
            raise ValueError("max_segment_len must be positive")


@dataclass(frozen=True)
class PermutationPolishResult:
    order: list[str]
    energy: float
    passes: int
    improvements: int
    seconds: float


class PermutationLocalSearchPolisher:
    """
    Improve a decoded layout with permutation-aware moves.

    D-Wave's simulated annealer works with individual binary flips. For one-hot
    layout QUBOs, moving between valid permutations often requires coordinated
    bit changes. This polisher keeps every move in valid permutation space and
    evaluates the same QUBO energy as the annealer.
    """

    def __init__(self, config: Optional[PermutationPolishConfig] = None):
        self.config = config or PermutationPolishConfig()
        self.config.validate()

    def polish(self, model: QUBOModel, order: list[str]) -> PermutationPolishResult:
        start = perf_counter()
        current_order = order.copy()
        evaluator = PermutationEnergyEvaluator(model)
        current_energy = evaluator.energy(current_order)
        improvements = 0
        passes_completed = 0

        for pass_index in range(self.config.max_passes):
            passes_completed = pass_index + 1
            move = self._best_improving_move(evaluator, current_order, current_energy)
            if move is None:
                break
            current_order, current_energy = move
            improvements += 1

        return PermutationPolishResult(
            order=current_order,
            energy=current_energy,
            passes=passes_completed,
            improvements=improvements,
            seconds=perf_counter() - start,
        )

    def _best_improving_move(
        self,
        evaluator: "PermutationEnergyEvaluator",
        order: list[str],
        current_energy: float,
    ) -> Optional[tuple[list[str], float]]:
        best_order: Optional[list[str]] = None
        best_energy = current_energy
        n = len(order)

        if self.config.use_swap_moves:
            for left in range(n - 1):
                for right in range(left + 1, n):
                    candidate = order.copy()
                    candidate[left], candidate[right] = candidate[right], candidate[left]
                    energy = evaluator.energy(candidate)
                    if energy < best_energy:
                        best_order = candidate
                        best_energy = energy

        if self.config.use_insert_moves:
            for src in range(n):
                for dst in range(n):
                    if src == dst:
                        continue
                    candidate = order.copy()
                    item = candidate.pop(src)
                    candidate.insert(dst, item)
                    energy = evaluator.energy(candidate)
                    if energy < best_energy:
                        best_order = candidate
                        best_energy = energy

        if self.config.use_segment_insert_moves:
            max_len = min(self.config.max_segment_len, n)
            for start in range(n):
                for length in range(2, max_len + 1):
                    end = start + length
                    if end > n:
                        break
                    segment = order[start:end]
                    remainder = order[:start] + order[end:]
                    for dst in range(len(remainder) + 1):
                        if dst == start:
                            continue
                        candidate = remainder[:dst] + segment + remainder[dst:]
                        energy = evaluator.energy(candidate)
                        if energy < best_energy:
                            best_order = candidate
                            best_energy = energy

        if best_order is None:
            return None
        return best_order, best_energy

    @staticmethod
    def _order_to_sample(model: QUBOModel, order: list[str]) -> list[int]:
        read_index_by_id = {read_id: idx for idx, read_id in enumerate(model.read_ids)}
        sample = [0] * model.num_variables
        for position_index, read_id in enumerate(order):
            sample[model.variable_index(read_index_by_id[read_id], position_index)] = 1
        return sample


class PermutationEnergyEvaluator:
    """
    Fast energy evaluation for valid permutation samples.

    For a valid one-hot layout, read-once and position-once quadratic penalties
    are inactive. Energy is the model constant, the selected one-hot linear
    terms, and adjacent-position quadratic terms.
    """

    def __init__(self, model: QUBOModel):
        self.model = model
        self.read_index_by_id = {read_id: idx for idx, read_id in enumerate(model.read_ids)}
        self.constant_terms = model.constant
        self.linear_by_read_position: dict[tuple[str, int], float] = {}
        self.adjacent_pair_bias: dict[tuple[str, str, int], float] = {}
        n = model.num_reads

        for read_index, read_id in enumerate(model.read_ids):
            for position_index in range(n):
                variable = model.variable_index(read_index, position_index)
                self.linear_by_read_position[(read_id, position_index)] = model.linear[variable]

        for (left_var, right_var), coeff in model.quadratic.items():
            left_read, left_pos = divmod(left_var, n)
            right_read, right_pos = divmod(right_var, n)
            if right_pos == left_pos + 1:
                key = (model.read_ids[left_read], model.read_ids[right_read], left_pos)
                self.adjacent_pair_bias[key] = self.adjacent_pair_bias.get(
                    key,
                    0.0,
                ) + coeff
            elif left_pos == right_pos + 1:
                key = (model.read_ids[right_read], model.read_ids[left_read], right_pos)
                self.adjacent_pair_bias[key] = self.adjacent_pair_bias.get(
                    key,
                    0.0,
                ) + coeff

    def energy(self, order: list[str]) -> float:
        value = self.constant_terms
        for position_index, read_id in enumerate(order):
            value += self.linear_by_read_position.get((read_id, position_index), 0.0)
        for position_index, (left_id, right_id) in enumerate(zip(order, order[1:])):
            value += self.adjacent_pair_bias.get((left_id, right_id, position_index), 0.0)
        return value


class BinarySimulatedAnnealer:
    """Small dependency-free simulated annealer for QUBO smoke tests and demos."""

    def __init__(self, config: Optional[BinaryAnnealingConfig] = None):
        self.config = config or BinaryAnnealingConfig()
        self.config.validate()

    def solve(self, model: QUBOModel) -> AnnealingResult:
        rng = random.Random(self.config.seed)
        adjacency = model.adjacency()
        best_sample: list[int] | None = None
        best_energy = float("inf")
        total_iterations = 0
        total_accepted = 0

        for restart in range(self.config.random_restarts):
            sample = self._initial_sample(model, rng, restart)
            energy = model.energy(sample)
            if energy < best_energy:
                best_sample = sample.copy()
                best_energy = energy

            temperature = self.config.initial_temperature
            while temperature > self.config.final_temperature:
                for _ in range(self.config.sweeps_per_temperature * model.num_variables):
                    proposal = None
                    if (
                        self.config.start_from_valid_permutation
                        and model.num_reads > 1
                        and rng.random() < self.config.swap_move_probability
                    ):
                        proposal = self._swap_position_proposal(model, sample, rng)

                    if proposal is not None:
                        proposal_energy = model.energy(proposal)
                        delta = proposal_energy - energy
                    else:
                        index = rng.randrange(model.num_variables)
                        delta = self._flip_delta(model, adjacency, sample, index)

                    total_iterations += 1
                    if delta <= 0.0 or rng.random() < math.exp(-delta / temperature):
                        if proposal is not None:
                            sample = proposal
                        else:
                            sample[index] = 1 - sample[index]
                        energy += delta
                        total_accepted += 1
                        if energy < best_energy:
                            best_sample = sample.copy()
                            best_energy = energy
                temperature *= self.config.cooling_rate

        assert best_sample is not None
        return AnnealingResult(
            sample=best_sample,
            energy=best_energy,
            iterations=total_iterations,
            accepted_moves=total_accepted,
            backend="builtin",
        )

    def _initial_sample(self, model: QUBOModel, rng: random.Random, restart: int) -> list[int]:
        n = model.num_reads
        sample = [0] * model.num_variables
        if self.config.start_from_valid_permutation or restart == 0:
            order = list(range(n))
            rng.shuffle(order)
            for position_index, read_index in enumerate(order):
                sample[model.variable_index(read_index, position_index)] = 1
            return sample

        for index in range(model.num_variables):
            sample[index] = rng.randrange(2)
        return sample

    @staticmethod
    def _swap_position_proposal(
        model: QUBOModel,
        sample: list[int],
        rng: random.Random,
    ) -> list[int] | None:
        n = model.num_reads
        selected: list[int | None] = [None] * n
        for read_index in range(n):
            for position_index in range(n):
                variable = model.variable_index(read_index, position_index)
                if not sample[variable]:
                    continue
                if selected[position_index] is not None:
                    return None
                selected[position_index] = read_index
        if any(value is None for value in selected):
            return None

        left_pos, right_pos = rng.sample(range(n), 2)
        proposal = sample.copy()
        left_read = selected[left_pos]
        right_read = selected[right_pos]
        assert left_read is not None and right_read is not None
        proposal[model.variable_index(left_read, left_pos)] = 0
        proposal[model.variable_index(right_read, right_pos)] = 0
        proposal[model.variable_index(left_read, right_pos)] = 1
        proposal[model.variable_index(right_read, left_pos)] = 1
        return proposal

    @staticmethod
    def _flip_delta(
        model: QUBOModel,
        adjacency: list[dict[int, float]],
        sample: list[int],
        index: int,
    ) -> float:
        local = model.linear[index]
        for other, coeff in adjacency[index].items():
            if sample[other]:
                local += coeff
        return local if sample[index] == 0 else -local


@dataclass(frozen=True)
class DWaveAnnealingConfig:
    """Settings forwarded to dwave.samplers.SimulatedAnnealingSampler."""

    num_reads: int = 20
    num_sweeps: int = 1000
    seed: int = 42
    beta_range: Optional[tuple[float, float]] = None
    beta_schedule_type: str = "geometric"
    randomize_order: bool = False

    def validate(self) -> None:
        if self.num_reads <= 0:
            raise ValueError("num_reads must be positive")
        if self.num_sweeps <= 0:
            raise ValueError("num_sweeps must be positive")
        if self.beta_range is not None:
            if len(self.beta_range) != 2:
                raise ValueError("beta_range must contain two values")
            if self.beta_range[0] < 0.0 or self.beta_range[1] < 0.0:
                raise ValueError("beta_range values must be non-negative")


class DWaveSimulatedAnnealer:
    """D-Wave Ocean simulated annealing backend for QUBOModel."""

    def __init__(self, config: Optional[DWaveAnnealingConfig] = None):
        self.config = config or DWaveAnnealingConfig()
        self.config.validate()
        self.last_sampleset = None

    def solve(self, model: QUBOModel) -> AnnealingResult:
        try:
            from dwave.samplers import SimulatedAnnealingSampler
        except ImportError as exc:
            raise RuntimeError(
                "dwave-samplers is required for DWaveSimulatedAnnealer. "
                "Install it with: pip install dwave-samplers dimod"
            ) from exc

        bqm = model.to_dimod_bqm()
        sampler = SimulatedAnnealingSampler()
        kwargs: dict[str, object] = {
            "num_reads": self.config.num_reads,
            "num_sweeps": self.config.num_sweeps,
            "seed": self.config.seed,
            "beta_schedule_type": self.config.beta_schedule_type,
            "randomize_order": self.config.randomize_order,
        }
        if self.config.beta_range is not None:
            kwargs["beta_range"] = self.config.beta_range

        sampleset = sampler.sample(bqm, **kwargs)
        self.last_sampleset = sampleset
        first = sampleset.first
        sample = [int(first.sample.get(index, 0)) for index in range(model.num_variables)]
        return AnnealingResult(
            sample=sample,
            energy=float(first.energy),
            iterations=self.config.num_reads * self.config.num_sweeps,
            accepted_moves=-1,
            backend="dwave-samplers",
        )


class QUBOLayoutSolver(LayoutSolver):
    """
    QUBO-based layout optimization with pluggable Hamiltonian builders.

    The default Hamiltonian uses binary variables x[v, j], where read v appears
    at layout position j. It enforces one read per position, one position per
    read, and penalizes adjacent layout pairs that are not present in the
    directed overlap edge set.
    """

    def __init__(
        self,
        hamiltonian: Optional[QUBOHamiltonianBuilder] = None,
        annealer: Optional[BinarySimulatedAnnealer] = None,
        polisher: Optional[PermutationLocalSearchPolisher] = None,
    ):
        self.hamiltonian = hamiltonian or MissingEdgeQUBOHamiltonian()
        self.annealer = annealer or BinarySimulatedAnnealer()
        self.polisher = polisher
        self.last_model: Optional[QUBOModel] = None
        self.last_sample: Optional[list[int]] = None

    def solve(self, reads: list[Read], edges: list[OverlapEdge], weight_mode: str = "dp") -> LayoutResult:
        if not reads:
            return LayoutResult(order=[], objective_value=0.0, solver_name="qubo_simulated_annealing")

        total_start = perf_counter()
        build_start = perf_counter()
        model = self.hamiltonian.build(reads, edges, weight_mode=weight_mode)
        build_seconds = perf_counter() - build_start

        anneal_start = perf_counter()
        annealing_result = self.annealer.solve(model)
        anneal_seconds = perf_counter() - anneal_start

        decode_start = perf_counter()
        order, decode_metadata = self._decode_order(model, annealing_result.sample)
        decode_seconds = perf_counter() - decode_start
        polish_metadata: dict[str, object] = {
            "polish_enabled": False,
            "polish_sec": 0.0,
            "polish_improvements": 0,
            "polish_passes": 0,
        }
        if self.polisher is not None:
            polish_result = self.polisher.polish(model, order)
            order = polish_result.order
            annealing_result = AnnealingResult(
                sample=PermutationLocalSearchPolisher._order_to_sample(model, order),
                energy=polish_result.energy,
                iterations=annealing_result.iterations,
                accepted_moves=annealing_result.accepted_moves,
                backend=annealing_result.backend,
            )
            polish_metadata = {
                "polish_enabled": True,
                "polish_sec": polish_result.seconds,
                "polish_improvements": polish_result.improvements,
                "polish_passes": polish_result.passes,
            }
        total_seconds = perf_counter() - total_start

        self.last_model = model
        self.last_sample = annealing_result.sample

        metadata = {
            "variables": model.num_variables,
            "quadratic_terms": len(model.quadratic),
            "constant": model.constant,
            "energy": annealing_result.energy,
            "iterations": annealing_result.iterations,
            "accepted_moves": annealing_result.accepted_moves,
            "annealer_backend": annealing_result.backend,
            "weight_mode": weight_mode,
            "build_sec": build_seconds,
            "anneal_sec": anneal_seconds,
            "decode_sec": decode_seconds,
            "total_sec": total_seconds,
            **polish_metadata,
            **decode_metadata,
        }
        if isinstance(self.hamiltonian, MissingEdgeQUBOHamiltonian):
            metadata.update({
                "hamiltonian": "missing_edge",
                "read_once_penalty": self.hamiltonian.config.read_once_penalty,
                "position_once_penalty": self.hamiltonian.config.position_once_penalty,
                "missing_edge_penalty": self.hamiltonian.config.missing_edge_penalty,
                "edge_count": self.hamiltonian.last_edge_count,
            })
        elif isinstance(self.hamiltonian, WeightedOverlapQUBOHamiltonian):
            metadata.update({
                "hamiltonian": "weighted_overlap",
                "read_once_penalty": self.hamiltonian.config.read_once_penalty,
                "position_once_penalty": self.hamiltonian.config.position_once_penalty,
                "missing_edge_penalty": self.hamiltonian.config.missing_edge_penalty,
                "edge_reward_scale": self.hamiltonian.config.edge_reward_scale,
                "score_mode": self.hamiltonian.config.score_mode or weight_mode,
                "normalize_rewards": self.hamiltonian.config.normalize_rewards,
                "edge_count": self.hamiltonian.last_edge_count,
                "reward_min": self.hamiltonian.last_reward_min,
                "reward_max": self.hamiltonian.last_reward_max,
            })

        return LayoutResult(
            order=order,
            objective_value=annealing_result.energy,
            solver_name="qubo_simulated_annealing",
            metadata=metadata,
        )

    @staticmethod
    def _decode_order(model: QUBOModel, sample: list[int]) -> tuple[list[str], dict[str, object]]:
        n = model.num_reads
        selected_by_position: list[list[int]] = [[] for _ in range(n)]
        read_counts = [0] * n
        position_counts = [0] * n

        for read_index in range(n):
            for position_index in range(n):
                variable = model.variable_index(read_index, position_index)
                if sample[variable]:
                    selected_by_position[position_index].append(read_index)
                    read_counts[read_index] += 1
                    position_counts[position_index] += 1

        order_indices: list[int | None] = []
        used: set[int] = set()
        for candidates in selected_by_position:
            chosen = next((idx for idx in candidates if idx not in used), None)
            order_indices.append(chosen)
            if chosen is not None:
                used.add(chosen)

        unused = [idx for idx in range(n) if idx not in used]
        for position_index, value in enumerate(order_indices):
            if value is None:
                order_indices[position_index] = unused.pop(0)

        order = [model.read_ids[index] for index in order_indices if index is not None]
        read_violations = sum(abs(1 - count) for count in read_counts)
        position_violations = sum(abs(1 - count) for count in position_counts)
        valid_sample = read_violations == 0 and position_violations == 0

        return order, {
            "valid_binary_layout": valid_sample,
            "read_assignment_violations": read_violations,
            "position_assignment_violations": position_violations,
        }


class SimulatedAnnealingLayoutSolver(QUBOLayoutSolver):
    """Backward-compatible alias for the built-in QUBO simulated annealer."""
