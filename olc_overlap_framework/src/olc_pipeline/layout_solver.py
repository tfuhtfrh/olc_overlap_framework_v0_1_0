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
    "overlap_len_power",
    "overlap_len_power2",
    "overlap_len_power3",
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


class EdgePathDAGQUBOModel(QUBOModel):
    """QUBO model whose binary variables select directed overlap edges."""

    def __init__(self, read_ids: list[str], edge_pairs: list[tuple[str, str]]):
        super().__init__(read_ids=read_ids, linear=[0.0] * len(edge_pairs))
        self.edge_pairs = edge_pairs
        self.edge_index_by_pair = {
            edge_pair: index
            for index, edge_pair in enumerate(edge_pairs)
        }

    def variable_label(self, index: int) -> str:
        left_id, right_id = self.edge_pairs[index]
        return f"y[{left_id},{right_id}]"

    def edge_variable_index(self, left_id: str, right_id: str) -> int:
        return self.edge_index_by_pair[(left_id, right_id)]


class EdgePathCoverDAGQUBOModel(EdgePathDAGQUBOModel):
    """Edge-path-cover QUBO model with source and sink node variables."""

    def __init__(self, read_ids: list[str], edge_pairs: list[tuple[str, str]]):
        QUBOModel.__init__(
            self,
            read_ids=read_ids,
            linear=[0.0] * (len(edge_pairs) + 2 * len(read_ids)),
        )
        self.edge_pairs = edge_pairs
        self.edge_index_by_pair = {
            edge_pair: index
            for index, edge_pair in enumerate(edge_pairs)
        }
        self._source_offset = len(edge_pairs)
        self._sink_offset = len(edge_pairs) + len(read_ids)

    def source_variable_index(self, read_id: str) -> int:
        return self._source_offset + self.read_ids.index(read_id)

    def sink_variable_index(self, read_id: str) -> int:
        return self._sink_offset + self.read_ids.index(read_id)

    def variable_label(self, index: int) -> str:
        if index < len(self.edge_pairs):
            return super().variable_label(index)
        if index < self._sink_offset:
            return f"s[{self.read_ids[index - self._source_offset]}]"
        return f"t[{self.read_ids[index - self._sink_offset]}]"


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
        - "overlap_len_power": refined overlap length raised to a gamma supplied as "overlap_len_power:<gamma>"
        - "overlap_len_power2": squared refined overlap length
        - "overlap_len_power3": cubed refined overlap length
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
        if mode == "overlap_len_power":
            return float(edge.overlap_len ** 2)
        if mode.startswith("overlap_len_power:"):
            try:
                gamma = float(mode.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid overlap_len_power gamma in score mode: {mode!r}") from exc
            if gamma <= 0.0:
                raise ValueError(f"overlap_len_power gamma must be positive: {mode!r}")
            return float(edge.overlap_len ** gamma)
        if mode == "overlap_len_power2":
            return float(edge.overlap_len ** 2)
        if mode == "overlap_len_power3":
            return float(edge.overlap_len ** 3)
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
class EdgePathDAGHamiltonianConfig:
    """
    Coefficients for a DAG edge-selection Hamiltonian.

    Each valid directed overlap edge has one binary variable y[u,v]. The model
    selects N-1 edges, limits every read to at most one predecessor and one
    successor, and rewards strong overlaps. On a DAG these constraints define
    one Hamiltonian path when they are all satisfied.
    """

    edge_count_penalty: float = 100.0
    degree_penalty: float = 100.0
    edge_reward_scale: float = 1.0
    score_mode: Optional[str] = None
    normalize_rewards: bool = True
    max_reward_score: Optional[float] = None
    min_reward_score: float = 0.0
    require_hamiltonian_path: bool = True

    def validate(self) -> None:
        if self.edge_count_penalty <= 0.0:
            raise ValueError("edge_count_penalty must be positive")
        if self.degree_penalty <= 0.0:
            raise ValueError("degree_penalty must be positive")
        if self.edge_reward_scale < 0.0:
            raise ValueError("edge_reward_scale must be non-negative")


class EdgePathDAGQUBOHamiltonian(QUBOHamiltonianBuilder):
    """
    Select a maximum-reward Hamiltonian path from a directed acyclic graph.

    The variable count is the number of unique valid directed overlap edges,
    instead of N squared read-position variables.
    """

    def __init__(
        self,
        config: Optional[EdgePathDAGHamiltonianConfig] = None,
        scorer: Optional[OverlapRewardScorer] = None,
    ):
        self.config = config or EdgePathDAGHamiltonianConfig()
        self.config.validate()
        self.scorer = scorer or OverlapRewardScorer()
        self.last_edge_count = 0
        self.last_reward_min: Optional[float] = None
        self.last_reward_max: Optional[float] = None
        self.last_longest_path_edges = 0

    def build(self, reads: list[Read], edges: list[OverlapEdge], weight_mode: str = "dp") -> QUBOModel:
        read_ids = [read.rid for read in reads]
        score_mode = self.config.score_mode or weight_mode
        reward_by_pair = self._best_rewards_by_pair(read_ids, edges, score_mode)
        edge_pairs = sorted(
            reward_by_pair,
            key=lambda pair: (read_ids.index(pair[0]), read_ids.index(pair[1])),
        )

        topological_order = self._topological_order(read_ids, edge_pairs)
        self.last_longest_path_edges = self._longest_path_edge_count(
            topological_order,
            edge_pairs,
        )
        if (
            self.config.require_hamiltonian_path
            and read_ids
            and self.last_longest_path_edges != len(read_ids) - 1
        ):
            raise ValueError(
                "Candidate overlap DAG does not contain a Hamiltonian path "
                f"covering all {len(read_ids)} reads; longest path contains "
                f"{self.last_longest_path_edges} edges."
            )

        model = EdgePathDAGQUBOModel(read_ids, edge_pairs)
        self.last_edge_count = len(edge_pairs)
        self.last_reward_min = min(reward_by_pair.values()) if reward_by_pair else None
        self.last_reward_max = max(reward_by_pair.values()) if reward_by_pair else None

        self._add_edge_count_constraint(
            model,
            target=max(0, len(read_ids) - 1),
            penalty=self.config.edge_count_penalty,
        )
        self._add_degree_constraints(model, self.config.degree_penalty)
        for edge_pair, variable in model.edge_index_by_pair.items():
            model.add_linear(
                variable,
                -self.config.edge_reward_scale * reward_by_pair[edge_pair],
            )
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
            edge_pair = (edge.left_id, edge.right_id)
            old_score = raw_scores.get(edge_pair)
            if old_score is None or score > old_score:
                raw_scores[edge_pair] = score

        if not self.config.normalize_rewards:
            return raw_scores

        denominator = self.config.max_reward_score
        if denominator is None:
            denominator = max(raw_scores.values(), default=1.0)
        if denominator <= 0.0:
            return {edge_pair: 0.0 for edge_pair in raw_scores}
        return {
            edge_pair: score / denominator
            for edge_pair, score in raw_scores.items()
        }

    @staticmethod
    def _add_edge_count_constraint(
        model: EdgePathDAGQUBOModel,
        target: int,
        penalty: float,
    ) -> None:
        model.add_constant(penalty * target * target)
        linear_bias = penalty * (1.0 - 2.0 * target)
        for variable in range(model.num_variables):
            model.add_linear(variable, linear_bias)
        for left in range(model.num_variables - 1):
            for right in range(left + 1, model.num_variables):
                model.add_quadratic(left, right, 2.0 * penalty)

    @staticmethod
    def _add_degree_constraints(
        model: EdgePathDAGQUBOModel,
        penalty: float,
    ) -> None:
        incoming: dict[str, list[int]] = {read_id: [] for read_id in model.read_ids}
        outgoing: dict[str, list[int]] = {read_id: [] for read_id in model.read_ids}
        for variable, (left_id, right_id) in enumerate(model.edge_pairs):
            outgoing[left_id].append(variable)
            incoming[right_id].append(variable)

        for variables in [*incoming.values(), *outgoing.values()]:
            for left_offset, left in enumerate(variables):
                for right in variables[left_offset + 1:]:
                    model.add_quadratic(left, right, penalty)

    @staticmethod
    def _topological_order(
        read_ids: list[str],
        edge_pairs: list[tuple[str, str]],
    ) -> list[str]:
        rank = {read_id: index for index, read_id in enumerate(read_ids)}
        incoming_count = {read_id: 0 for read_id in read_ids}
        outgoing = {read_id: [] for read_id in read_ids}
        for left_id, right_id in edge_pairs:
            outgoing[left_id].append(right_id)
            incoming_count[right_id] += 1

        ready = sorted(
            (read_id for read_id, count in incoming_count.items() if count == 0),
            key=rank.get,
        )
        order: list[str] = []
        while ready:
            read_id = ready.pop(0)
            order.append(read_id)
            for right_id in sorted(outgoing[read_id], key=rank.get):
                incoming_count[right_id] -= 1
                if incoming_count[right_id] == 0:
                    ready.append(right_id)
                    ready.sort(key=rank.get)

        if len(order) != len(read_ids):
            raise ValueError("EdgePathDAGQUBOHamiltonian requires an acyclic candidate graph.")
        return order

    @staticmethod
    def _longest_path_edge_count(
        topological_order: list[str],
        edge_pairs: list[tuple[str, str]],
    ) -> int:
        outgoing = {read_id: [] for read_id in topological_order}
        for left_id, right_id in edge_pairs:
            outgoing[left_id].append(right_id)

        distance = {read_id: 0 for read_id in topological_order}
        for left_id in topological_order:
            for right_id in outgoing[left_id]:
                distance[right_id] = max(distance[right_id], distance[left_id] + 1)
        return max(distance.values(), default=0)


@dataclass(frozen=True)
class EdgePathCoverDAGHamiltonianConfig:
    """
    Coefficients for a DAG path-cover Hamiltonian.

    Edge variables select a disjoint path cover. Source/sink variables allow
    multiple path components while penalizing isolated reads and extra breaks.
    """

    degree_penalty: float = 100.0
    isolate_penalty: float = 100.0
    path_break_penalty: float = 10.0
    max_path_count: int = 2
    path_count_cap_penalty: float = 100.0
    edge_reward_scale: float = 1.0
    score_mode: Optional[str] = None
    normalize_rewards: bool = True
    max_reward_score: Optional[float] = None
    min_reward_score: float = 0.0

    def validate(self) -> None:
        if self.degree_penalty <= 0.0:
            raise ValueError("degree_penalty must be positive")
        if self.isolate_penalty <= 0.0:
            raise ValueError("isolate_penalty must be positive")
        if self.path_break_penalty < 0.0:
            raise ValueError("path_break_penalty must be non-negative")
        if self.max_path_count < 1:
            raise ValueError("max_path_count must be at least 1")
        if self.max_path_count != 2:
            raise ValueError("EdgePathCoverDAGQUBOHamiltonian currently supports max_path_count=2")
        if self.path_count_cap_penalty < 0.0:
            raise ValueError("path_count_cap_penalty must be non-negative")
        if self.edge_reward_scale < 0.0:
            raise ValueError("edge_reward_scale must be non-negative")


class EdgePathCoverDAGQUBOHamiltonian(QUBOHamiltonianBuilder):
    """
    Select a rewarded disjoint path cover on a directed acyclic graph.

    Every read must have either one incoming edge or a source marker, and either
    one outgoing edge or a sink marker. A source+sink pair on the same read is
    penalized to discourage isolated nodes. Extra source markers represent path
    breaks and receive a soft linear penalty.
    """

    def __init__(
        self,
        config: Optional[EdgePathCoverDAGHamiltonianConfig] = None,
        scorer: Optional[OverlapRewardScorer] = None,
    ):
        self.config = config or EdgePathCoverDAGHamiltonianConfig()
        self.config.validate()
        self.scorer = scorer or OverlapRewardScorer()
        self.last_edge_count = 0
        self.last_reward_min: Optional[float] = None
        self.last_reward_max: Optional[float] = None

    def build(self, reads: list[Read], edges: list[OverlapEdge], weight_mode: str = "dp") -> QUBOModel:
        read_ids = [read.rid for read in reads]
        score_mode = self.config.score_mode or weight_mode
        reward_by_pair = self._best_rewards_by_pair(read_ids, edges, score_mode)
        rank = {read_id: index for index, read_id in enumerate(read_ids)}
        edge_pairs = sorted(
            reward_by_pair,
            key=lambda pair: (rank[pair[0]], rank[pair[1]]),
        )

        EdgePathDAGQUBOHamiltonian._topological_order(read_ids, edge_pairs)
        model = EdgePathCoverDAGQUBOModel(read_ids, edge_pairs)
        self.last_edge_count = len(edge_pairs)
        self.last_reward_min = min(reward_by_pair.values()) if reward_by_pair else None
        self.last_reward_max = max(reward_by_pair.values()) if reward_by_pair else None

        incoming: dict[str, list[int]] = {read_id: [] for read_id in read_ids}
        outgoing: dict[str, list[int]] = {read_id: [] for read_id in read_ids}
        for variable, (left_id, right_id) in enumerate(edge_pairs):
            outgoing[left_id].append(variable)
            incoming[right_id].append(variable)
            model.add_linear(variable, -self.config.edge_reward_scale * reward_by_pair[(left_id, right_id)])

        for read_id in read_ids:
            self._add_exactly_one_constraint(
                model,
                [model.source_variable_index(read_id), *incoming[read_id]],
                self.config.degree_penalty,
            )
            self._add_exactly_one_constraint(
                model,
                [model.sink_variable_index(read_id), *outgoing[read_id]],
                self.config.degree_penalty,
            )
            model.add_quadratic(
                model.source_variable_index(read_id),
                model.sink_variable_index(read_id),
                self.config.isolate_penalty,
            )
        source_variables = [
            model.source_variable_index(read_id)
            for read_id in read_ids
        ]
        self._add_path_count_shaping(
            model,
            source_variables,
            self.config.path_break_penalty,
            self.config.path_count_cap_penalty,
        )

        return model

    def _best_rewards_by_pair(
        self,
        read_ids: list[str],
        edges: list[OverlapEdge],
        score_mode: str,
    ) -> dict[tuple[str, str], float]:
        helper = EdgePathDAGQUBOHamiltonian(EdgePathDAGHamiltonianConfig(
            edge_reward_scale=self.config.edge_reward_scale,
            score_mode=self.config.score_mode,
            normalize_rewards=self.config.normalize_rewards,
            max_reward_score=self.config.max_reward_score,
            min_reward_score=self.config.min_reward_score,
        ), scorer=self.scorer)
        return helper._best_rewards_by_pair(read_ids, edges, score_mode)

    @staticmethod
    def _add_exactly_one_constraint(
        model: QUBOModel,
        variables: list[int],
        penalty: float,
    ) -> None:
        model.add_constant(penalty)
        for variable in variables:
            model.add_linear(variable, -penalty)
        for left_offset, left in enumerate(variables):
            for right in variables[left_offset + 1:]:
                model.add_quadratic(left, right, 2.0 * penalty)

    @staticmethod
    def _add_path_count_shaping(
        model: QUBOModel,
        source_variables: list[int],
        two_path_penalty: float,
        over_two_penalty: float,
    ) -> None:
        """
        Shape path-count energy for m=sum(source_variables):

            over_two_penalty * (m - 1)^2

        This makes one path the reference, penalizes zero or two paths equally,
        and penalizes more than two paths quadratically:
            m=0 -> over_two_penalty
            m=1 -> 0
            m=2 -> over_two_penalty
            m=3 -> 4*over_two_penalty
            m=4 -> 9*over_two_penalty
        """
        del two_path_penalty  # Kept in the signature for config/API compatibility.
        if not source_variables:
            return
        model.add_constant(over_two_penalty)
        linear_bias = -over_two_penalty
        for variable in source_variables:
            model.add_linear(variable, linear_bias)
        for left_offset, left in enumerate(source_variables):
            for right in source_variables[left_offset + 1:]:
                model.add_quadratic(left, right, 2.0 * over_two_penalty)


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
                        and model.num_variables == model.num_reads * model.num_reads
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
        is_position_model = model.num_variables == n * n
        if is_position_model and (self.config.start_from_valid_permutation or restart == 0):
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


@dataclass(frozen=True)
class OpenJijSQAConfig:
    """Settings forwarded to openjij.SQASampler."""

    num_reads: int = 20
    num_sweeps: int = 1000
    seed: Optional[int] = 42
    beta: Optional[float] = None
    trotter: Optional[int] = None

    def validate(self) -> None:
        if self.num_reads <= 0:
            raise ValueError("num_reads must be positive")
        if self.num_sweeps <= 0:
            raise ValueError("num_sweeps must be positive")
        if self.beta is not None and self.beta <= 0.0:
            raise ValueError("beta must be positive")
        if self.trotter is not None and self.trotter <= 0:
            raise ValueError("trotter must be positive")


class OpenJijSimulatedQuantumAnnealer:
    """OpenJij simulated quantum annealing backend for QUBOModel."""

    def __init__(self, config: Optional[OpenJijSQAConfig] = None):
        self.config = config or OpenJijSQAConfig()
        self.config.validate()
        self.last_response = None

    def solve(self, model: QUBOModel) -> AnnealingResult:
        try:
            import openjij as oj
        except ImportError as exc:
            raise RuntimeError(
                "openjij is required for OpenJijSimulatedQuantumAnnealer. "
                "Install it with: pip install openjij"
            ) from exc

        qubo = self._qubo_dict(model)
        sampler = oj.SQASampler()
        kwargs: dict[str, object] = {
            "num_reads": self.config.num_reads,
            "num_sweeps": self.config.num_sweeps,
        }
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed
        if self.config.beta is not None:
            kwargs["beta"] = self.config.beta
        if self.config.trotter is not None:
            kwargs["trotter"] = self.config.trotter

        response = sampler.sample_qubo(qubo, **kwargs)
        self.last_response = response
        first = response.first
        sample_map = first.sample
        sample = [int(sample_map.get(index, 0)) for index in range(model.num_variables)]
        return AnnealingResult(
            sample=sample,
            energy=float(first.energy + model.constant),
            iterations=self.config.num_reads * self.config.num_sweeps,
            accepted_moves=-1,
            backend="openjij-sqa",
        )

    @staticmethod
    def _qubo_dict(model: QUBOModel) -> dict[tuple[int, int], float]:
        qubo: dict[tuple[int, int], float] = {}
        for index, bias in enumerate(model.linear):
            if bias != 0.0:
                qubo[(index, index)] = qubo.get((index, index), 0.0) + bias
        for key, bias in model.quadratic.items():
            if bias != 0.0:
                qubo[key] = qubo.get(key, 0.0) + bias
        return qubo


@dataclass(frozen=True)
class DWaveQPUConfig:
    """Settings for a future real D-Wave quantum annealing backend."""

    num_reads: int = 100
    chain_strength: Optional[float] = None
    annealing_time: Optional[float] = None
    solver: Optional[str] = None
    token: Optional[str] = None
    endpoint: Optional[str] = None

    def validate(self) -> None:
        if self.num_reads <= 0:
            raise ValueError("num_reads must be positive")
        if self.chain_strength is not None and self.chain_strength <= 0.0:
            raise ValueError("chain_strength must be positive")
        if self.annealing_time is not None and self.annealing_time <= 0.0:
            raise ValueError("annealing_time must be positive")


class DWaveQPUAnnealer:
    """
    Real D-Wave QPU backend placeholder using Ocean's DWaveSampler.

    This backend is intentionally isolated from the demo defaults. It shares the
    same QUBOModel input as local SA/SQA backends, so switching to a QPU later
    should not require changing Hamiltonian builders.
    """

    def __init__(self, config: Optional[DWaveQPUConfig] = None):
        self.config = config or DWaveQPUConfig()
        self.config.validate()
        self.last_sampleset = None

    def solve(self, model: QUBOModel) -> AnnealingResult:
        try:
            from dwave.system import DWaveSampler, EmbeddingComposite
        except ImportError as exc:
            raise RuntimeError(
                "dwave-system is required for DWaveQPUAnnealer. "
                "Install it with: pip install dwave-system"
            ) from exc

        sampler_kwargs: dict[str, object] = {}
        if self.config.solver is not None:
            sampler_kwargs["solver"] = self.config.solver
        if self.config.token is not None:
            sampler_kwargs["token"] = self.config.token
        if self.config.endpoint is not None:
            sampler_kwargs["endpoint"] = self.config.endpoint

        sample_kwargs: dict[str, object] = {"num_reads": self.config.num_reads}
        if self.config.chain_strength is not None:
            sample_kwargs["chain_strength"] = self.config.chain_strength
        if self.config.annealing_time is not None:
            sample_kwargs["annealing_time"] = self.config.annealing_time

        bqm = model.to_dimod_bqm()
        sampler = EmbeddingComposite(DWaveSampler(**sampler_kwargs))
        sampleset = sampler.sample(bqm, **sample_kwargs)
        self.last_sampleset = sampleset
        first = sampleset.first
        sample = [int(first.sample.get(index, 0)) for index in range(model.num_variables)]
        return AnnealingResult(
            sample=sample,
            energy=float(first.energy),
            iterations=self.config.num_reads,
            accepted_moves=-1,
            backend="dwave-qpu",
        )


def qubo_sample_for_order(model: QUBOModel, order: list[str]) -> list[int]:
    """Encode a complete read order for either position or edge-path QUBOs."""
    sample = [0] * model.num_variables
    if isinstance(model, EdgePathCoverDAGQUBOModel):
        for edge_pair in zip(order, order[1:]):
            variable = model.edge_index_by_pair.get(edge_pair)
            if variable is None:
                raise ValueError(
                    f"Order uses edge {edge_pair[0]} -> {edge_pair[1]} "
                    "that is absent from the candidate DAG."
                )
            sample[variable] = 1
        if order:
            sample[model.source_variable_index(order[0])] = 1
            sample[model.sink_variable_index(order[-1])] = 1
        return sample

    if isinstance(model, EdgePathDAGQUBOModel):
        for edge_pair in zip(order, order[1:]):
            variable = model.edge_index_by_pair.get(edge_pair)
            if variable is None:
                raise ValueError(
                    f"Order uses edge {edge_pair[0]} -> {edge_pair[1]} "
                    "that is absent from the candidate DAG."
                )
            sample[variable] = 1
        return sample

    read_index_by_id = {
        read_id: read_index
        for read_index, read_id in enumerate(model.read_ids)
    }
    for position_index, read_id in enumerate(order):
        sample[model.variable_index(read_index_by_id[read_id], position_index)] = 1
    return sample


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
        if model.num_variables == 0:
            annealing_result = AnnealingResult(
                sample=[],
                energy=model.constant,
                iterations=0,
                accepted_moves=0,
                backend="trivial",
            )
        else:
            annealing_result = self.annealer.solve(model)
        anneal_seconds = perf_counter() - anneal_start

        decode_start = perf_counter()
        if isinstance(model, EdgePathCoverDAGQUBOModel):
            order, decode_metadata = self._decode_edge_path_cover(model, annealing_result.sample)
        elif isinstance(model, EdgePathDAGQUBOModel):
            order, decode_metadata = self._decode_edge_path(model, annealing_result.sample)
        else:
            order, decode_metadata = self._decode_order(model, annealing_result.sample)
        decode_seconds = perf_counter() - decode_start
        polish_metadata: dict[str, object] = {
            "polish_enabled": False,
            "polish_sec": 0.0,
            "polish_improvements": 0,
            "polish_passes": 0,
        }
        if self.polisher is not None and isinstance(model, EdgePathDAGQUBOModel):
            raise ValueError("Permutation polish is not supported for edge-path QUBO models.")
        if self.polisher is not None:
            polish_result = self.polisher.polish(model, order)
            order = polish_result.order
            polished_sample = PermutationLocalSearchPolisher._order_to_sample(model, order)
            order, decode_metadata = self._decode_order(model, polished_sample)
            annealing_result = AnnealingResult(
                sample=polished_sample,
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
        elif isinstance(self.hamiltonian, EdgePathDAGQUBOHamiltonian):
            metadata.update({
                "hamiltonian": "edge_path_dag",
                "edge_count_penalty": self.hamiltonian.config.edge_count_penalty,
                "degree_penalty": self.hamiltonian.config.degree_penalty,
                "edge_reward_scale": self.hamiltonian.config.edge_reward_scale,
                "score_mode": self.hamiltonian.config.score_mode or weight_mode,
                "normalize_rewards": self.hamiltonian.config.normalize_rewards,
                "edge_count": self.hamiltonian.last_edge_count,
                "reward_min": self.hamiltonian.last_reward_min,
                "reward_max": self.hamiltonian.last_reward_max,
                "candidate_dag": True,
                "candidate_longest_path_edges": self.hamiltonian.last_longest_path_edges,
            })
        elif isinstance(self.hamiltonian, EdgePathCoverDAGQUBOHamiltonian):
            metadata.update({
                "hamiltonian": "edge_path_cover_dag",
                "degree_penalty": self.hamiltonian.config.degree_penalty,
                "isolate_penalty": self.hamiltonian.config.isolate_penalty,
                "path_break_penalty": self.hamiltonian.config.path_break_penalty,
                "max_path_count": self.hamiltonian.config.max_path_count,
                "path_count_cap_penalty": self.hamiltonian.config.path_count_cap_penalty,
                "edge_reward_scale": self.hamiltonian.config.edge_reward_scale,
                "score_mode": self.hamiltonian.config.score_mode or weight_mode,
                "normalize_rewards": self.hamiltonian.config.normalize_rewards,
                "edge_count": self.hamiltonian.last_edge_count,
                "reward_min": self.hamiltonian.last_reward_min,
                "reward_max": self.hamiltonian.last_reward_max,
                "candidate_dag": True,
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

    @staticmethod
    def _decode_edge_path(
        model: EdgePathDAGQUBOModel,
        sample: list[int],
    ) -> tuple[list[str], dict[str, object]]:
        rank = {read_id: index for index, read_id in enumerate(model.read_ids)}
        selected_edges = [
            edge_pair
            for variable, edge_pair in enumerate(model.edge_pairs)
            if sample[variable]
        ]
        incoming: dict[str, list[str]] = {read_id: [] for read_id in model.read_ids}
        outgoing: dict[str, list[str]] = {read_id: [] for read_id in model.read_ids}
        for left_id, right_id in selected_edges:
            outgoing[left_id].append(right_id)
            incoming[right_id].append(left_id)

        in_degree_violations = sum(max(0, len(values) - 1) for values in incoming.values())
        out_degree_violations = sum(max(0, len(values) - 1) for values in outgoing.values())
        in_degree_conflicts = {
            read_id: values
            for read_id, values in incoming.items()
            if len(values) > 1
        }
        out_degree_conflicts = {
            read_id: values
            for read_id, values in outgoing.items()
            if len(values) > 1
        }
        target_edge_count = max(0, len(model.read_ids) - 1)
        edge_count_violation = abs(len(selected_edges) - target_edge_count)

        incoming_count = {
            read_id: len(values)
            for read_id, values in incoming.items()
        }
        ready = sorted(
            (read_id for read_id, count in incoming_count.items() if count == 0),
            key=rank.get,
        )
        topological_order: list[str] = []
        while ready:
            read_id = ready.pop(0)
            topological_order.append(read_id)
            for right_id in sorted(outgoing[read_id], key=rank.get):
                incoming_count[right_id] -= 1
                if incoming_count[right_id] == 0:
                    ready.append(right_id)
                    ready.sort(key=rank.get)

        selected_graph_acyclic = len(topological_order) == len(model.read_ids)
        if not selected_graph_acyclic:
            used = set(topological_order)
            topological_order.extend(
                read_id
                for read_id in model.read_ids
                if read_id not in used
            )

        single_path_layout = False
        if (
            edge_count_violation == 0
            and in_degree_violations == 0
            and out_degree_violations == 0
            and selected_graph_acyclic
        ):
            sources = [
                read_id
                for read_id in model.read_ids
                if not incoming[read_id]
            ]
            if len(sources) == 1:
                path = [sources[0]]
                seen = {sources[0]}
                while outgoing[path[-1]]:
                    next_id = outgoing[path[-1]][0]
                    if next_id in seen:
                        break
                    path.append(next_id)
                    seen.add(next_id)
                if len(path) == len(model.read_ids):
                    topological_order = path
                    single_path_layout = True

        return topological_order, {
            "valid_binary_layout": single_path_layout,
            "valid_edge_path": single_path_layout,
            "selected_edge_count": len(selected_edges),
            "target_edge_count": target_edge_count,
            "edge_count_violation": edge_count_violation,
            "in_degree_violations": in_degree_violations,
            "out_degree_violations": out_degree_violations,
            "in_degree_conflicts": in_degree_conflicts,
            "out_degree_conflicts": out_degree_conflicts,
            "selected_graph_acyclic": selected_graph_acyclic,
            "single_path_layout": single_path_layout,
            "selected_edges": selected_edges,
        }

    @staticmethod
    def _decode_edge_path_cover(
        model: EdgePathCoverDAGQUBOModel,
        sample: list[int],
    ) -> tuple[list[str], dict[str, object]]:
        order, base = QUBOLayoutSolver._decode_edge_path(model, sample)
        selected_sources = [
            read_id
            for read_id in model.read_ids
            if sample[model.source_variable_index(read_id)]
        ]
        selected_sinks = [
            read_id
            for read_id in model.read_ids
            if sample[model.sink_variable_index(read_id)]
        ]
        selected_edges = base["selected_edges"]
        incoming = {read_id: [] for read_id in model.read_ids}
        outgoing = {read_id: [] for read_id in model.read_ids}
        for left_id, right_id in selected_edges:
            outgoing[left_id].append(right_id)
            incoming[right_id].append(left_id)

        source_constraint_violations = 0
        sink_constraint_violations = 0
        isolated_nodes: list[str] = []
        for read_id in model.read_ids:
            source_constraint_violations += abs(
                1 - len(incoming[read_id]) - int(read_id in selected_sources)
            )
            sink_constraint_violations += abs(
                1 - len(outgoing[read_id]) - int(read_id in selected_sinks)
            )
            if read_id in selected_sources and read_id in selected_sinks:
                isolated_nodes.append(read_id)

        paths: list[list[str]] = []
        for source_id in selected_sources:
            path = [source_id]
            seen = {source_id}
            while outgoing[path[-1]]:
                next_id = outgoing[path[-1]][0]
                if next_id in seen:
                    break
                path.append(next_id)
                seen.add(next_id)
            paths.append(path)

        covered = {
            read_id
            for path in paths
            for read_id in path
        }
        uncovered_nodes = [
            read_id
            for read_id in model.read_ids
            if read_id not in covered
        ]
        path_cover_valid = (
            source_constraint_violations == 0
            and sink_constraint_violations == 0
            and base["in_degree_violations"] == 0
            and base["out_degree_violations"] == 0
            and not isolated_nodes
            and not uncovered_nodes
            and base["selected_graph_acyclic"]
        )
        single_path_layout = path_cover_valid and len(paths) == 1
        if single_path_layout:
            order = paths[0]
        else:
            paths.sort(key=lambda path: model.read_ids.index(path[0]))
            order = [
                read_id
                for path in paths
                for read_id in path
            ] + uncovered_nodes

        metadata = {
            **base,
            "valid_binary_layout": single_path_layout,
            "valid_edge_path": single_path_layout,
            "valid_path_cover": path_cover_valid,
            "single_path_layout": single_path_layout,
            "path_count": len(paths),
            "selected_source_count": len(selected_sources),
            "selected_sink_count": len(selected_sinks),
            "selected_sources": selected_sources,
            "selected_sinks": selected_sinks,
            "source_constraint_violations": source_constraint_violations,
            "sink_constraint_violations": sink_constraint_violations,
            "isolated_nodes": isolated_nodes,
            "isolated_node_count": len(isolated_nodes),
            "uncovered_nodes": uncovered_nodes,
            "path_components": paths,
        }
        return order, metadata


class SimulatedAnnealingLayoutSolver(QUBOLayoutSolver):
    """Backward-compatible alias for the built-in QUBO simulated annealer."""
