from __future__ import annotations

"""
Complete OLC Plan-B pipeline (single-file integration).

This file intentionally keeps the whole research pipeline in one place:
  1. conservative hifiasm-inspired edge/read cleanup
  2. multi-entry / multi-exit Plan-B compression
  3. recursive QA handling for oversized non-DAG blocks
  4. exact classical handling for DAG blocks
  5. explicit compression-priority evaluation
  6. cycle-closed boundary protection against exit -> ... -> entry return paths
  7. DFS-style whole-graph recursion and final solution reconstruction

Main rule for a selected block B:

    f(B):
        if B is DAG:
            solve B classically and pack the solved alternatives into macros
        elif B fits the QA limit:
            solve B by QA and pack the solved alternatives into macros
        else:
            find a child C inside B, call f(C), then re-evaluate B

Whole-graph rule:

    if the ORIGINAL top graph is DAG:
        solve it classically
    elif the ORIGINAL top graph is non-DAG but directly fits QA:
        solve it directly by QA (Plan B is unnecessary)
    else:
        enter Plan B and keep eliminating non-DAG top-level structure.
        Once Plan B has started, do NOT stop merely because the top graph later
        becomes QA-sized.  Continue until the visible top graph is a DAG, then
        solve that final DAG once classically.

A solved macro edge stores expansion_edge_ids, so after the final top solution
is obtained the selected macros are recursively expanded back to layer-0 edges.
The wrapper also restores safe-unitig contractions to return the raw OLC edge
and node path when possible.

Dependency: networkx
"""

from dataclasses import dataclass, field, asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Mapping, Protocol, Sequence
import copy
import math
import uuid

import networkx as nx


Node = Hashable


# ============================================================================
# COMMON DATA MODEL
# ============================================================================

@dataclass
class EdgeRecord:
    """
    Original edges and macro edges are stored in one registry.

    Compression is structural only:
      - original edge weight may already be known
      - macro edge weight remains None until a later solver determines it
    """

    edge_id: str
    u: Node
    v: Node
    kind: str                      # "original" or "macro"
    layer: int                     # original = 0
    weight: float | None
    block_id: str | None = None
    input_edge_id: str | None = None
    output_edge_id: str | None = None
    expansion_edge_ids: list[str] | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionBlock:
    """One structurally compressed region."""

    block_id: str
    layer: int
    region_nodes: frozenset[Node]

    internal_edge_ids: tuple[str, ...]
    input_edge_ids: tuple[str, ...]
    output_edge_ids: tuple[str, ...]
    macro_edge_ids: tuple[str, ...]

    # Snapshot of the evaluation at the instant this region was compressed.
    # This is intentionally plain data so later changes to the scoring model
    # do not break old hierarchy files.
    evaluation_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphHierarchy:
    """
    Hierarchically compressed OLC graph.

    graphs_by_layer[0] : normalized original graph
    graphs_by_layer[n] : graph after compression step n

    In the priority-aware version, one compression is performed per step by
    default. This is deliberate: after each compression the priorities are
    recomputed because another candidate may become much cheaper.
    """

    original_graph: nx.MultiDiGraph
    current_graph: nx.MultiDiGraph
    edge_records: dict[str, EdgeRecord]
    blocks_by_layer: dict[int, list[CompressionBlock]] = field(default_factory=dict)
    graphs_by_layer: dict[int, nx.MultiDiGraph] = field(default_factory=dict)

    # Human-readable history of why each compression was selected.
    compression_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def max_layer(self) -> int:
        return max(self.blocks_by_layer, default=0)

    def blocks(self) -> list[CompressionBlock]:
        return [
            block
            for layer in sorted(self.blocks_by_layer)
            for block in self.blocks_by_layer[layer]
        ]

    def block_map(self) -> dict[str, CompressionBlock]:
        return {block.block_id: block for block in self.blocks()}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def normalize_olc_graph(
    graph: nx.DiGraph | nx.MultiDiGraph,
    *,
    weight_key: str = "weight",
) -> tuple[nx.MultiDiGraph, dict[str, EdgeRecord]]:
    """Convert the input to MultiDiGraph and assign unique edge IDs."""

    normalized = nx.MultiDiGraph()
    normalized.add_nodes_from(
        (node, copy.deepcopy(data))
        for node, data in graph.nodes(data=True)
    )

    records: dict[str, EdgeRecord] = {}

    if graph.is_multigraph():
        edge_iter = graph.edges(keys=True, data=True)
        for u, v, _key, raw_data in edge_iter:
            data = copy.deepcopy(dict(raw_data))
            edge_id = str(data.get("edge_id", _new_id("E")))
            while edge_id in records:
                edge_id = _new_id("E")

            raw_weight = data.get(weight_key)
            weight = None if raw_weight is None else float(raw_weight)

            records[edge_id] = EdgeRecord(
                edge_id=edge_id,
                u=u,
                v=v,
                kind="original",
                layer=0,
                weight=weight,
                data=data,
            )
            normalized.add_edge(
                u,
                v,
                key=edge_id,
                edge_id=edge_id,
                kind="original",
                layer=0,
                block_id=None,
            )
    else:
        for u, v, raw_data in graph.edges(data=True):
            data = copy.deepcopy(dict(raw_data))
            edge_id = str(data.get("edge_id", _new_id("E")))
            while edge_id in records:
                edge_id = _new_id("E")

            raw_weight = data.get(weight_key)
            weight = None if raw_weight is None else float(raw_weight)

            records[edge_id] = EdgeRecord(
                edge_id=edge_id,
                u=u,
                v=v,
                kind="original",
                layer=0,
                weight=weight,
                data=data,
            )
            normalized.add_edge(
                u,
                v,
                key=edge_id,
                edge_id=edge_id,
                kind="original",
                layer=0,
                block_id=None,
            )

    return normalized, records


def graph_edge_ids(graph: nx.MultiDiGraph) -> list[str]:
    return [
        data["edge_id"]
        for _, _, _, data in graph.edges(keys=True, data=True)
    ]


def boundary_edge_ids(
    graph: nx.MultiDiGraph,
    region_nodes: set[Node],
) -> tuple[list[str], list[str], list[str]]:
    """
    Return:
        internal : region -> region
        incoming : outside -> region
        outgoing : region -> outside
    """

    internal: list[str] = []
    incoming: list[str] = []
    outgoing: list[str] = []

    for u, v, _key, data in graph.edges(keys=True, data=True):
        edge_id = data["edge_id"]
        u_inside = u in region_nodes
        v_inside = v in region_nodes

        if u_inside and v_inside:
            internal.append(edge_id)
        elif (not u_inside) and v_inside:
            incoming.append(edge_id)
        elif u_inside and (not v_inside):
            outgoing.append(edge_id)

    return internal, incoming, outgoing


@dataclass(frozen=True)
class CycleClosureResult:
    """Result of expanding a candidate until no outside exit->entry return path remains."""

    region_nodes: frozenset[Node]
    is_cycle_closed: bool
    rounds: int
    added_nodes: frozenset[Node]
    return_pairs: tuple[tuple[Node, Node], ...]
    exceeded_node_limit: bool = False
    message: str = ""


def _external_boundary_nodes(
    graph: nx.MultiDiGraph,
    region_nodes: set[Node],
) -> tuple[set[Node], set[Node]]:
    """
    Return (external_entry_nodes, external_exit_nodes).

    external_entry_nodes p:
        p -> inside is an incoming boundary edge.

    external_exit_nodes q:
        inside -> q is an outgoing boundary edge.

    A macro edge created by Plan B has the form p -> q. Therefore an outside
    path q ~> p would immediately close a directed cycle after compression.
    """

    external_entries: set[Node] = set()
    external_exits: set[Node] = set()

    for u, v in graph.edges():
        u_inside = u in region_nodes
        v_inside = v in region_nodes
        if (not u_inside) and v_inside:
            external_entries.add(u)
        elif u_inside and (not v_inside):
            external_exits.add(v)

    return external_entries, external_exits


def external_return_path_nodes(
    graph: nx.MultiDiGraph,
    region_nodes: set[Node],
) -> tuple[set[Node], list[tuple[Node, Node]]]:
    """
    Find all outside vertices that lie on at least one path

        q ~> p

    where q is an outside endpoint of an outgoing boundary edge and p is an
    outside endpoint of an incoming boundary edge.

    Such a path is exactly what would combine with a future macro edge p->q to
    create a new cycle after compression.

    The search is performed only in G[V \\ region].
    """

    outside_nodes = set(graph.nodes) - set(region_nodes)
    if not outside_nodes:
        return set(), []

    outside = nx.DiGraph(graph.subgraph(outside_nodes))
    entries, exits = _external_boundary_nodes(graph, region_nodes)
    entries &= outside_nodes
    exits &= outside_nodes

    if not entries or not exits:
        return set(), []

    return_nodes: set[Node] = set()
    violating_pairs: list[tuple[Node, Node]] = []

    # Ports are expected to be small in Plan B, so pairwise reachability is
    # acceptable and keeps the implementation transparent.
    for q in exits:
        if q not in outside:
            continue
        reachable_from_q = {q}
        reachable_from_q.update(nx.descendants(outside, q))

        for p in entries:
            if p not in reachable_from_q or p not in outside:
                continue

            violating_pairs.append((q, p))

            # Vertices lying on at least one q~>p path are exactly those that
            # are reachable from q and can reach p.
            can_reach_p = {p}
            can_reach_p.update(nx.ancestors(outside, p))
            return_nodes.update(reachable_from_q & can_reach_p)

    return return_nodes, violating_pairs


def is_cycle_closed_region(
    graph: nx.MultiDiGraph,
    region_nodes: set[Node],
) -> bool:
    """
    A region is cycle-closed iff no outside path q~>p remains, where p is an
    external entry endpoint and q is an external exit endpoint.

    If this condition holds, replacing the region by any p->q macro edge
    cannot create a directed cycle solely by combining that macro edge with
    the outside graph.
    """

    nodes, pairs = external_return_path_nodes(graph, region_nodes)
    return not nodes and not pairs


def cycle_close_region(
    graph: nx.MultiDiGraph,
    region_nodes: set[Node],
    *,
    max_region_nodes: int | None = None,
    max_rounds: int = 32,
) -> CycleClosureResult:
    """
    Expand `region_nodes` until the cycle-closed fixed point is reached.

    At each round, every outside vertex that lies on an exit->entry return path
    is absorbed into the region. Boundary ports are then recomputed, so the
    next round naturally pushes the cut farther outward when necessary.

    This operation does NOT choose an assembly path and does NOT delete edges.
    It only changes where a candidate region is cut.
    """

    region = set(region_nodes)
    original = set(region_nodes)
    last_pairs: list[tuple[Node, Node]] = []

    if not region <= set(graph.nodes):
        return CycleClosureResult(
            region_nodes=frozenset(region),
            is_cycle_closed=False,
            rounds=0,
            added_nodes=frozenset(),
            return_pairs=(),
            message="region contains nodes not present in graph",
        )

    for round_index in range(max(1, max_rounds)):
        return_nodes, pairs = external_return_path_nodes(graph, region)
        last_pairs = pairs

        new_nodes = return_nodes - region
        if not pairs:
            return CycleClosureResult(
                region_nodes=frozenset(region),
                is_cycle_closed=True,
                rounds=round_index,
                added_nodes=frozenset(region - original),
                return_pairs=(),
                message="cycle-closed fixed point reached",
            )

        if not new_nodes:
            return CycleClosureResult(
                region_nodes=frozenset(region),
                is_cycle_closed=False,
                rounds=round_index,
                added_nodes=frozenset(region - original),
                return_pairs=tuple(pairs),
                message="return path exists but no new outside node can be absorbed",
            )

        candidate = region | new_nodes
        if max_region_nodes is not None and len(candidate) > max_region_nodes:
            return CycleClosureResult(
                region_nodes=frozenset(candidate),
                is_cycle_closed=False,
                rounds=round_index + 1,
                added_nodes=frozenset(candidate - original),
                return_pairs=tuple(pairs),
                exceeded_node_limit=True,
                message=(
                    f"cycle closure requires {len(candidate)} nodes, exceeding "
                    f"max_region_nodes={max_region_nodes}"
                ),
            )

        region = candidate

    return CycleClosureResult(
        region_nodes=frozenset(region),
        is_cycle_closed=is_cycle_closed_region(graph, region),
        rounds=max_rounds,
        added_nodes=frozenset(region - original),
        return_pairs=tuple(last_pairs),
        message="maximum cycle-closure rounds reached",
    )


def _macro_edges_grouped_by_block(
    graph: nx.MultiDiGraph,
    edge_records: Mapping[str, EdgeRecord],
) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}

    for edge_id in graph_edge_ids(graph):
        record = edge_records[edge_id]
        if record.kind == "macro" and record.block_id is not None:
            grouped.setdefault(record.block_id, set()).add(edge_id)

    return grouped


def respects_lower_block_atomicity(
    graph: nx.MultiDiGraph,
    region_nodes: set[Node],
    edge_records: Mapping[str, EdgeRecord],
) -> bool:
    """
    A higher-level region must not cut a visible lower block in half.

    All currently visible macro edges of one child block must either be
    completely inside the new region or completely untouched.
    """

    grouped = _macro_edges_grouped_by_block(graph, edge_records)

    for macro_edge_ids in grouped.values():
        touching = 0
        internal = 0

        for edge_id in macro_edge_ids:
            record = edge_records[edge_id]
            u_inside = record.u in region_nodes
            v_inside = record.v in region_nodes

            if u_inside or v_inside:
                touching += 1
            if u_inside and v_inside:
                internal += 1

        if touching == 0:
            continue
        if internal == len(macro_edge_ids):
            continue

        # After a lower block has been solved and pruned, it may have exactly
        # one feasible visible macro edge left.  That single resolved edge is
        # now an atomic edge in its own right and may safely serve as an input
        # or output boundary edge of a higher recursive block.  Without this
        # exception, recursive Plan B can get stuck immediately after packing
        # a child block.
        if len(macro_edge_ids) == 1 and internal == 0 and touching == 1:
            only_edge_id = next(iter(macro_edge_ids))
            only_record = edge_records[only_edge_id]
            status = str(
                only_record.data.get("solve_status", "")
            ).strip().lower()
            if only_record.weight is not None and status in {"", "feasible"}:
                continue

        return False

    return True


def region_has_basic_directional_support(
    graph: nx.MultiDiGraph,
    region_nodes: set[Node],
    input_edge_ids: Sequence[str],
    output_edge_ids: Sequence[str],
    edge_records: Mapping[str, EdgeRecord],
) -> bool:
    """
    Cheap necessary condition before accepting a candidate.

    Every internal node must be reachable from at least one input-side node,
    and must be able to reach at least one output-side node.
    """

    induced = graph.subgraph(region_nodes).copy()

    entry_inside_nodes = {
        edge_records[edge_id].v for edge_id in input_edge_ids
    }
    exit_inside_nodes = {
        edge_records[edge_id].u for edge_id in output_edge_ids
    }

    if not entry_inside_nodes or not exit_inside_nodes:
        return False

    reachable_from_input: set[Node] = set()
    for source in entry_inside_nodes:
        reachable_from_input.add(source)
        reachable_from_input.update(nx.descendants(induced, source))

    can_reach_output: set[Node] = set()
    reverse_graph = induced.reverse(copy=False)
    for target in exit_inside_nodes:
        can_reach_output.add(target)
        can_reach_output.update(nx.descendants(reverse_graph, target))

    return (
        region_nodes <= reachable_from_input
        and region_nodes <= can_reach_output
    )


# ============================================================================
# PART A — COMPRESSION EVALUATION SYSTEM
# ============================================================================
#
# IMPORTANT DESIGN RULE:
#   The evaluation system is intentionally isolated from the actual compressor.
#
# If the priority formula changes later, normally only:
#       EvaluationConfig
#       DefaultPriorityModel
# need to be edited.
#
# The structural compression functions below do not know the score formula.
# ============================================================================


@dataclass
class EvaluationConfig:
    """
    Tunable parameters for candidate evaluation.

    There are three kinds of thresholds:

    1) final target
       When the remaining graph reaches this estimated top-level QUBO size,
       compression may stop.

    2) preferred local limits
       Soft limits. Candidates exceeding them are NOT rejected.
       They are merely moved to a lower priority tier and will be used later
       if the final graph is still too large.

    3) absolute local limits
       Optional hard limits. Set to None by default because the exact
       hardware-feasible size may change with embedding and hardware.
    """

    # ---- final stopping target ----
    target_top_qubo_variables: int | None = 180
    target_top_qubo_couplings: int | None = None
    target_top_qubo_density: float | None = None
    stop_if_top_graph_is_dag: bool = True

    # ---- soft QA preferences for a non-DAG region ----
    preferred_local_qubo_variables: int = 180
    preferred_local_qubo_avg_degree: float | None = None
    preferred_local_qubo_density: float | None = None
    preferred_port_pair_count: int = 4

    # ---- optional hard limits ----
    absolute_local_qubo_variables: int | None = None
    absolute_port_pair_count: int | None = None

    # ---- benefit weights ----
    # These affect only ordering inside a priority tier.
    node_reduction_weight: float = 1.0
    edge_reduction_weight: float = 1.0
    future_qubo_variable_gain_weight: float = 1.0

    # ---- cost weights ----
    # DAG cost is based on O(a*b*(K+1)*(V+E)).
    dag_child_block_factor: float = 1.0

    # non-DAG structural cost:
    #   pair_count * Q * (1 + degree_weight * avg_degree)
    qa_avg_degree_cost_weight: float = 0.15

    # Soft penalty when preferred local limits are exceeded.
    qa_soft_oversize_penalty: float = 2.0
    qa_soft_port_penalty: float = 1.0

    # Numerical stabilizer.
    epsilon: float = 1.0e-12


@dataclass(frozen=True)
class QUBOStructureEstimate:
    """Structural estimate for one fixed input-output QUBO."""

    logical_variables: int
    quadratic_couplings: int
    density: float
    average_degree: float
    max_degree: int


@dataclass(frozen=True)
class TopProblemEstimate:
    """
    Estimate of the current final problem.

    The coupling count is a structural proxy because the final source/target
    auxiliary cycle is not known here. Variable count is the main stopping
    quantity.
    """

    node_count: int
    edge_count: int
    is_dag: bool

    estimated_qubo_variables: int
    estimated_qubo_couplings: int
    estimated_qubo_density: float
    estimated_qubo_average_degree: float
    estimated_qubo_max_degree: int

    qa_target_reached: bool


@dataclass
class RegionEvaluation:
    """Complete evaluation record for one candidate region."""

    region_nodes: frozenset[Node]

    # Basic graph size
    node_count: int
    internal_edge_count: int
    input_count: int
    output_count: int
    port_pair_count: int

    # Sparsity
    edge_to_node_ratio: float
    simple_directed_density: float

    # Structural class
    is_dag: bool
    visible_child_block_count: int
    visible_child_port_variable_count: int

    # Compression gain
    node_reduction: int
    edge_reduction: int
    future_qubo_variables_before_proxy: int
    future_qubo_variables_after_proxy: int
    future_qubo_variable_gain_proxy: int

    # Local non-DAG QUBO estimates.
    # None for DAG candidates because QA is not needed there.
    local_qubo_min_variables: int | None
    local_qubo_max_variables: int | None
    local_qubo_mean_variables: float | None

    local_qubo_min_couplings: int | None
    local_qubo_max_couplings: int | None
    local_qubo_mean_couplings: float | None

    local_qubo_max_density: float | None
    local_qubo_max_average_degree: float | None
    local_qubo_max_degree: int | None

    # Ranking result
    priority_tier: int
    priority_label: str
    priority_score: float
    eligible: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["region_nodes"] = [str(x) for x in sorted(self.region_nodes, key=str)]
        return data


class PriorityModel(Protocol):
    """
    Pluggable scoring interface.

    You can replace DefaultPriorityModel with another model later without
    touching the compression engine.
    """

    def rank(
        self,
        metrics: Mapping[str, Any],
        config: EvaluationConfig,
    ) -> tuple[int, str, float, bool, str]:
        ...


class DefaultPriorityModel:
    """
    Default priority rule.

    Priority tiers:
        0 : DAG
            Classical. No D-Wave size limit is needed.

        1 : non-DAG, preferred QA size
            Good QA candidate.

        2 : non-DAG, relaxed QA size
            Not ideal, but still kept. It will be used later if higher-priority
            compression is insufficient.

        99: hard-ineligible
            Only used when an explicitly configured absolute hard limit is
            exceeded.

    Inside each tier, a larger score is better.

    This model is intentionally simple and easy to replace.
    """

    def rank(
        self,
        metrics: Mapping[str, Any],
        config: EvaluationConfig,
    ) -> tuple[int, str, float, bool, str]:

        n = int(metrics["node_count"])
        m = int(metrics["internal_edge_count"])
        pair_count = int(metrics["port_pair_count"])
        child_count = int(metrics["visible_child_block_count"])
        is_dag = bool(metrics["is_dag"])

        edge_reduction = int(metrics["edge_reduction"])
        future_q_gain = int(metrics["future_qubo_variable_gain_proxy"])

        benefit = (
            config.node_reduction_weight * n
            + config.edge_reduction_weight * edge_reduction
            + config.future_qubo_variable_gain_weight * future_q_gain
        )

        # Do not force benefit to be positive conceptually.
        # We only stabilize the denominator/ratio numerically.
        benefit_for_ratio = max(config.epsilon, benefit)

        if is_dag:
            classical_cost = (
                max(1, pair_count)
                * max(1.0, 1.0 + config.dag_child_block_factor * child_count)
                * max(1, n + m)
            )
            score = benefit_for_ratio / classical_cost

            return (
                0,
                "DAG_CLASSICAL",
                float(score),
                True,
                (
                    "DAG: classical dynamic programming is preferred. "
                    "Region size is not restricted by the QA target."
                ),
            )

        q_max = int(metrics["local_qubo_max_variables"])
        avg_degree = float(metrics["local_qubo_max_average_degree"])
        density = float(metrics["local_qubo_max_density"])

        # Optional absolute limits.
        if (
            config.absolute_local_qubo_variables is not None
            and q_max > config.absolute_local_qubo_variables
        ):
            return (
                99,
                "HARD_LIMIT",
                0.0,
                False,
                (
                    f"Estimated local QUBO variables {q_max} exceed absolute "
                    f"limit {config.absolute_local_qubo_variables}."
                ),
            )

        if (
            config.absolute_port_pair_count is not None
            and pair_count > config.absolute_port_pair_count
        ):
            return (
                99,
                "HARD_LIMIT",
                0.0,
                False,
                (
                    f"Port-pair count {pair_count} exceeds absolute limit "
                    f"{config.absolute_port_pair_count}."
                ),
            )

        preferred = True
        soft_reasons: list[str] = []

        if q_max > config.preferred_local_qubo_variables:
            preferred = False
            soft_reasons.append(
                f"Q={q_max}>{config.preferred_local_qubo_variables}"
            )

        if pair_count > config.preferred_port_pair_count:
            preferred = False
            soft_reasons.append(
                f"a*b={pair_count}>{config.preferred_port_pair_count}"
            )

        if (
            config.preferred_local_qubo_avg_degree is not None
            and avg_degree > config.preferred_local_qubo_avg_degree
        ):
            preferred = False
            soft_reasons.append(
                f"avg_degree={avg_degree:.3g}>"
                f"{config.preferred_local_qubo_avg_degree}"
            )

        if (
            config.preferred_local_qubo_density is not None
            and density > config.preferred_local_qubo_density
        ):
            preferred = False
            soft_reasons.append(
                f"density={density:.3g}>"
                f"{config.preferred_local_qubo_density}"
            )

        # Structural cost proxy for a non-DAG block.
        qa_cost = (
            max(1, pair_count)
            * max(1, q_max)
            * (
                1.0
                + config.qa_avg_degree_cost_weight * max(0.0, avg_degree)
            )
        )

        # Soft oversize penalties.
        q_ratio = max(
            1.0,
            q_max / max(1, config.preferred_local_qubo_variables),
        )
        port_ratio = max(
            1.0,
            pair_count / max(1, config.preferred_port_pair_count),
        )

        soft_penalty = (
            q_ratio ** config.qa_soft_oversize_penalty
            * port_ratio ** config.qa_soft_port_penalty
        )

        score = benefit_for_ratio / max(
            config.epsilon,
            qa_cost * soft_penalty,
        )

        if preferred:
            return (
                1,
                "QA_PREFERRED",
                float(score),
                True,
                "non-DAG but within preferred local QUBO / port limits.",
            )

        return (
            2,
            "QA_RELAXED",
            float(score),
            True,
            (
                "non-DAG and outside preferred limits; kept as a lower-priority "
                "candidate because it may be required later. "
                + ("; ".join(soft_reasons) if soft_reasons else "")
            ),
        )


def _simple_directed_density(
    graph: nx.MultiDiGraph,
    region_nodes: set[Node],
) -> float:
    """
    Density after collapsing parallel edges.

    For n <= 1 return 0.
    """
    n = len(region_nodes)
    if n <= 1:
        return 0.0

    simple_edges = {
        (u, v)
        for u, v in graph.subgraph(region_nodes).edges()
        if u != v
    }
    return len(simple_edges) / (n * (n - 1))


def _visible_child_blocks_in_region(
    hierarchy: GraphHierarchy,
    internal_edge_ids: Sequence[str],
) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}

    for edge_id in internal_edge_ids:
        record = hierarchy.edge_records[edge_id]
        if record.kind == "macro" and record.block_id is not None:
            grouped.setdefault(record.block_id, set()).add(edge_id)

    return grouped


def _child_port_counts_from_visible_edges(
    hierarchy: GraphHierarchy,
    grouped: Mapping[str, set[str]],
) -> tuple[int, dict[str, tuple[int, int]]]:
    """
    Return:
        total child port-variable count
        child_block_id -> (visible_input_ports, visible_output_ports)
    """

    total = 0
    counts: dict[str, tuple[int, int]] = {}

    for block_id, edge_ids in grouped.items():
        inputs: set[str] = set()
        outputs: set[str] = set()

        for edge_id in edge_ids:
            record = hierarchy.edge_records[edge_id]
            if record.input_edge_id is not None:
                inputs.add(record.input_edge_id)
            if record.output_edge_id is not None:
                outputs.add(record.output_edge_id)

        a = len(inputs)
        b = len(outputs)
        counts[block_id] = (a, b)
        total += a + b

    return total, counts


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _estimate_local_qubo_structure_for_pair(
    hierarchy: GraphHierarchy,
    region_nodes: set[Node],
    internal_edge_ids: Sequence[str],
    input_edge_id: str,
    output_edge_id: str,
) -> QUBOStructureEstimate:
    """
    Estimate the structural QUBO graph for one fixed input-output pair.

    This mirrors the current Plan-B QUBO structure without solving it:
      - one selected incoming boundary edge
      - all current internal edges
      - one selected outgoing boundary edge
      - q -> auxiliary -> p
      - A constraints
      - visible child block port/link constraints

    The estimate counts UNIQUE quadratic variable pairs.
    """

    records = hierarchy.edge_records
    input_record = records[input_edge_id]
    output_record = records[output_edge_id]

    source = "__EVAL_SOURCE__"
    target = "__EVAL_TARGET__"
    auxiliary = "__EVAL_AUX__"

    local_graph = nx.MultiDiGraph()
    local_graph.add_nodes_from(region_nodes)
    local_graph.add_nodes_from([source, target, auxiliary])

    edge_var_by_id: dict[str, str] = {}

    def add_edge_var(
        edge_id: str,
        u: Node,
        v: Node,
    ) -> str:
        var = f"x::{edge_id}"
        edge_var_by_id[edge_id] = var
        local_graph.add_edge(u, v, key=var, variable=var)
        return var

    # Fixed incoming boundary.
    add_edge_var(
        input_edge_id,
        source,
        input_record.v,
    )

    # All internal edges.
    for edge_id in internal_edge_ids:
        record = records[edge_id]
        add_edge_var(edge_id, record.u, record.v)

    # Fixed outgoing boundary.
    add_edge_var(
        output_edge_id,
        output_record.u,
        target,
    )

    # Two QA auxiliary edges.
    aux1 = "aux::eval::q_to_r"
    aux2 = "aux::eval::r_to_p"
    local_graph.add_edge(target, auxiliary, key=aux1, variable=aux1)
    local_graph.add_edge(auxiliary, source, key=aux2, variable=aux2)

    variables: set[str] = {
        data["variable"]
        for _, _, _, data in local_graph.edges(keys=True, data=True)
    }
    couplings: set[tuple[str, str]] = set()

    # A-condition quadratic interactions.
    for node in local_graph.nodes:
        incoming = [
            data["variable"]
            for _, _, _, data in local_graph.in_edges(
                node, keys=True, data=True
            )
        ]
        outgoing = [
            data["variable"]
            for _, _, _, data in local_graph.out_edges(
                node, keys=True, data=True
            )
        ]

        for x, y in combinations(sorted(set(incoming)), 2):
            couplings.add(_pair_key(x, y))

        for x, y in combinations(sorted(set(outgoing)), 2):
            couplings.add(_pair_key(x, y))

    # Visible child block macro edges.
    child_edges: dict[str, list[str]] = {}
    for edge_id in internal_edge_ids:
        record = records[edge_id]
        if record.kind == "macro" and record.block_id is not None:
            child_edges.setdefault(record.block_id, []).append(edge_id)

    for child_block_id, macro_ids in child_edges.items():
        rows: dict[str, list[str]] = {}
        columns: dict[str, list[str]] = {}

        for edge_id in macro_ids:
            record = records[edge_id]
            if record.input_edge_id is None or record.output_edge_id is None:
                continue

            rows.setdefault(record.input_edge_id, []).append(
                edge_var_by_id[edge_id]
            )
            columns.setdefault(record.output_edge_id, []).append(
                edge_var_by_id[edge_id]
            )

        p_vars: list[str] = []
        q_vars: list[str] = []

        # p_i = sum_j z_ij
        for input_port_id, z_vars in rows.items():
            p = f"p::{child_block_id}::{input_port_id}"
            variables.add(p)
            p_vars.append(p)

            unique_z = sorted(set(z_vars))
            for z in unique_z:
                couplings.add(_pair_key(p, z))
            for z1, z2 in combinations(unique_z, 2):
                couplings.add(_pair_key(z1, z2))

        # q_j = sum_i z_ij
        for output_port_id, z_vars in columns.items():
            q = f"q::{child_block_id}::{output_port_id}"
            variables.add(q)
            q_vars.append(q)

            unique_z = sorted(set(z_vars))
            for z in unique_z:
                couplings.add(_pair_key(q, z))
            for z1, z2 in combinations(unique_z, 2):
                couplings.add(_pair_key(z1, z2))

        # one-hot p and q.
        for p1, p2 in combinations(sorted(set(p_vars)), 2):
            couplings.add(_pair_key(p1, p2))
        for q1, q2 in combinations(sorted(set(q_vars)), 2):
            couplings.add(_pair_key(q1, q2))

    q = len(variables)
    j = len(couplings)

    degree: dict[str, int] = {var: 0 for var in variables}
    for x, y in couplings:
        degree[x] = degree.get(x, 0) + 1
        degree[y] = degree.get(y, 0) + 1

    if q <= 1:
        density = 0.0
        avg_degree = 0.0
    else:
        density = 2.0 * j / (q * (q - 1))
        avg_degree = 2.0 * j / q

    max_degree = max(degree.values(), default=0)

    return QUBOStructureEstimate(
        logical_variables=q,
        quadratic_couplings=j,
        density=density,
        average_degree=avg_degree,
        max_degree=max_degree,
    )


def estimate_top_problem(
    hierarchy: GraphHierarchy,
    config: EvaluationConfig,
) -> TopProblemEstimate:
    """
    Estimate whether the current top graph is already easy enough.

    If it is a DAG, the classical branch can solve it and compression may stop.

    For non-DAG, estimate the top-level QUBO size:
        edge variables
        + 2 future auxiliary variables
        + visible child port variables

    Couplings are a proxy because the final source/target auxiliary cycle is not
    specified in this compression-only file.
    """

    graph = hierarchy.current_graph
    records = hierarchy.edge_records

    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    is_dag = nx.is_directed_acyclic_graph(graph)

    visible_groups = _macro_edges_grouped_by_block(graph, records)
    port_var_count, _ = _child_port_counts_from_visible_edges(
        hierarchy,
        visible_groups,
    )

    q_variables = m + 2 + port_var_count

    # QUBO coupling proxy: A-condition interactions on current visible edges
    # + port/link interactions. No source/target auxiliary interactions here.
    variables: set[str] = set()
    couplings: set[tuple[str, str]] = set()

    for edge_id in graph_edge_ids(graph):
        variables.add(f"x::{edge_id}")

    for node in graph.nodes:
        incoming = [
            f"x::{data['edge_id']}"
            for _, _, _, data in graph.in_edges(node, keys=True, data=True)
        ]
        outgoing = [
            f"x::{data['edge_id']}"
            for _, _, _, data in graph.out_edges(node, keys=True, data=True)
        ]

        for x, y in combinations(sorted(set(incoming)), 2):
            couplings.add(_pair_key(x, y))
        for x, y in combinations(sorted(set(outgoing)), 2):
            couplings.add(_pair_key(x, y))

    for block_id, edge_ids in visible_groups.items():
        rows: dict[str, list[str]] = {}
        cols: dict[str, list[str]] = {}

        for edge_id in edge_ids:
            record = records[edge_id]
            if record.input_edge_id is None or record.output_edge_id is None:
                continue
            z = f"x::{edge_id}"
            rows.setdefault(record.input_edge_id, []).append(z)
            cols.setdefault(record.output_edge_id, []).append(z)

        p_vars: list[str] = []
        q_vars: list[str] = []

        for port_id, z_vars in rows.items():
            p = f"p::{block_id}::{port_id}"
            variables.add(p)
            p_vars.append(p)
            for z in set(z_vars):
                couplings.add(_pair_key(p, z))
            for z1, z2 in combinations(sorted(set(z_vars)), 2):
                couplings.add(_pair_key(z1, z2))

        for port_id, z_vars in cols.items():
            qv = f"q::{block_id}::{port_id}"
            variables.add(qv)
            q_vars.append(qv)
            for z in set(z_vars):
                couplings.add(_pair_key(qv, z))
            for z1, z2 in combinations(sorted(set(z_vars)), 2):
                couplings.add(_pair_key(z1, z2))

        for p1, p2 in combinations(sorted(set(p_vars)), 2):
            couplings.add(_pair_key(p1, p2))
        for q1, q2 in combinations(sorted(set(q_vars)), 2):
            couplings.add(_pair_key(q1, q2))

    # +2 future auxiliary variables.
    q_for_density = max(q_variables, len(variables) + 2)
    j = len(couplings)

    if q_for_density <= 1:
        density = 0.0
        avg_degree = 0.0
    else:
        density = 2.0 * j / (q_for_density * (q_for_density - 1))
        avg_degree = 2.0 * j / q_for_density

    degree: dict[str, int] = {var: 0 for var in variables}
    for x, y in couplings:
        degree[x] = degree.get(x, 0) + 1
        degree[y] = degree.get(y, 0) + 1
    max_degree = max(degree.values(), default=0)

    target_configured = any(
        value is not None
        for value in (
            config.target_top_qubo_variables,
            config.target_top_qubo_couplings,
            config.target_top_qubo_density,
        )
    )
    target_ok = target_configured

    if config.target_top_qubo_variables is not None:
        target_ok = target_ok and (
            q_variables <= config.target_top_qubo_variables
        )

    if config.target_top_qubo_couplings is not None:
        target_ok = target_ok and (
            j <= config.target_top_qubo_couplings
        )

    if config.target_top_qubo_density is not None:
        target_ok = target_ok and (
            density <= config.target_top_qubo_density
        )

    return TopProblemEstimate(
        node_count=n,
        edge_count=m,
        is_dag=is_dag,
        estimated_qubo_variables=q_variables,
        estimated_qubo_couplings=j,
        estimated_qubo_density=density,
        estimated_qubo_average_degree=avg_degree,
        estimated_qubo_max_degree=max_degree,
        qa_target_reached=target_ok,
    )


def evaluate_region_candidate(
    hierarchy: GraphHierarchy,
    region_nodes: set[Node],
    *,
    config: EvaluationConfig | None = None,
    priority_model: PriorityModel | None = None,
) -> RegionEvaluation:
    """
    Evaluate one candidate region.

    This function does NOT modify the graph.

    For DAG:
        calculate classical cost-oriented metrics.

    For non-DAG:
        estimate every input-output local QUBO structurally and aggregate
        min / max / mean values.
    """

    if config is None:
        config = EvaluationConfig()
    if priority_model is None:
        priority_model = DefaultPriorityModel()

    graph = hierarchy.current_graph
    records = hierarchy.edge_records

    if not region_nodes:
        raise ValueError("region_nodes is empty.")
    if not region_nodes <= set(graph.nodes):
        raise ValueError("region contains nodes not present in current graph.")

    internal, incoming, outgoing = boundary_edge_ids(graph, region_nodes)

    n = len(region_nodes)
    e = len(internal)
    a = len(incoming)
    b = len(outgoing)
    pair_count = a * b

    induced = graph.subgraph(region_nodes).copy()
    is_dag = nx.is_directed_acyclic_graph(induced)

    edge_to_node_ratio = e / max(1, n)
    simple_density = _simple_directed_density(graph, region_nodes)

    grouped_children = _visible_child_blocks_in_region(
        hierarchy,
        internal,
    )
    child_port_vars, _child_port_counts = _child_port_counts_from_visible_edges(
        hierarchy,
        grouped_children,
    )

    # Exact graph-size effect of this structural compression.
    node_reduction = n
    edge_reduction = e + a + b - pair_count

    # Future upper-QUBO footprint proxy.
    #
    # Before compression:
    #   internal edge variables
    #   + boundary edges
    #   + currently visible child port variables
    #
    # After compression:
    #   a*b macro-edge variables
    #   + a+b port variables when this block appears as a child
    #
    # Hence:
    #   gain proxy = e + child_port_vars - a*b
    before_proxy = e + a + b + child_port_vars
    after_proxy = pair_count + a + b
    future_q_gain = before_proxy - after_proxy

    q_estimates: list[QUBOStructureEstimate] = []

    if not is_dag and a > 0 and b > 0:
        for input_edge_id in incoming:
            for output_edge_id in outgoing:
                q_estimates.append(
                    _estimate_local_qubo_structure_for_pair(
                        hierarchy,
                        region_nodes,
                        internal,
                        input_edge_id,
                        output_edge_id,
                    )
                )

    if q_estimates:
        q_values = [x.logical_variables for x in q_estimates]
        j_values = [x.quadratic_couplings for x in q_estimates]

        q_min = min(q_values)
        q_max = max(q_values)
        q_mean = sum(q_values) / len(q_values)

        j_min = min(j_values)
        j_max = max(j_values)
        j_mean = sum(j_values) / len(j_values)

        density_max = max(x.density for x in q_estimates)
        avg_degree_max = max(x.average_degree for x in q_estimates)
        max_degree = max(x.max_degree for x in q_estimates)
    else:
        q_min = q_max = None
        q_mean = None
        j_min = j_max = None
        j_mean = None
        density_max = None
        avg_degree_max = None
        max_degree = None

    raw_metrics: dict[str, Any] = {
        "node_count": n,
        "internal_edge_count": e,
        "input_count": a,
        "output_count": b,
        "port_pair_count": pair_count,
        "edge_to_node_ratio": edge_to_node_ratio,
        "simple_directed_density": simple_density,
        "is_dag": is_dag,
        "visible_child_block_count": len(grouped_children),
        "visible_child_port_variable_count": child_port_vars,
        "node_reduction": node_reduction,
        "edge_reduction": edge_reduction,
        "future_qubo_variables_before_proxy": before_proxy,
        "future_qubo_variables_after_proxy": after_proxy,
        "future_qubo_variable_gain_proxy": future_q_gain,
        "local_qubo_min_variables": q_min,
        "local_qubo_max_variables": q_max,
        "local_qubo_mean_variables": q_mean,
        "local_qubo_min_couplings": j_min,
        "local_qubo_max_couplings": j_max,
        "local_qubo_mean_couplings": j_mean,
        "local_qubo_max_density": density_max,
        "local_qubo_max_average_degree": avg_degree_max,
        "local_qubo_max_degree": max_degree,
    }

    # Structural validity is checked before ranking.
    if a < 1 or b < 1:
        tier, label, score, eligible, reason = (
            99,
            "INVALID_BOUNDARY",
            0.0,
            False,
            "Region must have at least one input and one output.",
        )
    else:
        tier, label, score, eligible, reason = priority_model.rank(
            raw_metrics,
            config,
        )

    return RegionEvaluation(
        region_nodes=frozenset(region_nodes),
        **raw_metrics,
        priority_tier=tier,
        priority_label=label,
        priority_score=score,
        eligible=eligible,
        reason=reason,
    )


def evaluate_candidates(
    hierarchy: GraphHierarchy,
    candidates: Sequence[set[Node]],
    *,
    config: EvaluationConfig | None = None,
    priority_model: PriorityModel | None = None,
) -> list[RegionEvaluation]:
    """Evaluate and rank a list of candidate regions."""

    if config is None:
        config = EvaluationConfig()
    if priority_model is None:
        priority_model = DefaultPriorityModel()

    evaluations: list[RegionEvaluation] = []

    for region in candidates:
        try:
            ev = evaluate_region_candidate(
                hierarchy,
                set(region),
                config=config,
                priority_model=priority_model,
            )
        except (ValueError, nx.NetworkXError):
            continue
        evaluations.append(ev)

    evaluations.sort(
        key=lambda ev: (
            ev.priority_tier,
            -ev.priority_score,
            -ev.node_reduction,
            ev.port_pair_count,
        )
    )
    return evaluations


# ============================================================================
# PART B — STRUCTURAL COMPRESSION ENGINE
# ============================================================================


def _cheap_search_score(
    graph: nx.MultiDiGraph,
    region_nodes: set[Node],
    input_count: int,
    output_count: int,
) -> float:
    """
    Cheap score used ONLY inside beam search.

    This is NOT the final compression priority.
    Final selection always uses evaluate_region_candidate().

    Keeping this cheap avoids building QUBO estimates for every temporary
    beam-search state.
    """

    internal_edge_count = graph.subgraph(region_nodes).number_of_edges()
    macro_edge_count = input_count * output_count

    return (
        5.0 * len(region_nodes)
        + float(internal_edge_count - macro_edge_count)
        - float(input_count + output_count)
    )


def find_bounded_boundary_regions(
    graph: nx.MultiDiGraph,
    edge_records: Mapping[str, EdgeRecord],
    *,
    max_input_ports: int,
    max_output_ports: int,
    min_region_nodes: int = 2,
    max_region_nodes: int = 20,
    beam_width: int = 24,
    max_candidates: int = 256,
    max_seeds: int | None = None,
    require_cycle_closed: bool = True,
    auto_expand_cycle_closure: bool = True,
    max_cycle_closure_rounds: int = 32,
) -> list[set[Node]]:
    """
    Heuristic candidate discovery.

    It searches weakly connected regions satisfying bounded input/output ports.
    Final priority is NOT decided here.
    """

    if max_input_ports < 1 or max_output_ports < 1:
        raise ValueError("max_input_ports/max_output_ports must be >= 1.")

    undirected = graph.to_undirected(as_view=True)
    seeds = list(graph.nodes)
    seeds.sort(
        key=lambda node: graph.in_degree(node) + graph.out_degree(node),
        reverse=True,
    )

    if max_seeds is not None:
        seeds = seeds[:max_seeds]

    globally_seen: set[frozenset[Node]] = set()
    candidates: dict[frozenset[Node], float] = {}

    for seed in seeds:
        beam: list[frozenset[Node]] = [frozenset({seed})]

        for _ in range(max_region_nodes):
            next_beam_scores: dict[frozenset[Node], float] = {}

            for frozen_region in beam:
                if frozen_region in globally_seen:
                    continue
                globally_seen.add(frozen_region)

                region = set(frozen_region)
                _internal, incoming, outgoing = boundary_edge_ids(
                    graph,
                    region,
                )

                valid_boundary = (
                    1 <= len(incoming) <= max_input_ports
                    and 1 <= len(outgoing) <= max_output_ports
                )

                if (
                    len(region) >= min_region_nodes
                    and valid_boundary
                    and nx.is_weakly_connected(graph.subgraph(region))
                ):
                    candidate_region = set(region)

                    if require_cycle_closed:
                        if auto_expand_cycle_closure:
                            closure = cycle_close_region(
                                graph,
                                candidate_region,
                                max_region_nodes=max_region_nodes,
                                max_rounds=max_cycle_closure_rounds,
                            )
                            if not closure.is_cycle_closed or closure.exceeded_node_limit:
                                candidate_region = set()
                            else:
                                candidate_region = set(closure.region_nodes)
                        elif not is_cycle_closed_region(graph, candidate_region):
                            candidate_region = set()

                    if candidate_region:
                        _ci, c_in, c_out = boundary_edge_ids(
                            graph, candidate_region
                        )
                        candidate_valid_boundary = (
                            1 <= len(c_in) <= max_input_ports
                            and 1 <= len(c_out) <= max_output_ports
                        )

                        if (
                            len(candidate_region) >= min_region_nodes
                            and len(candidate_region) <= max_region_nodes
                            and candidate_valid_boundary
                            and nx.is_weakly_connected(graph.subgraph(candidate_region))
                            and respects_lower_block_atomicity(
                                graph, candidate_region, edge_records
                            )
                            and region_has_basic_directional_support(
                                graph,
                                candidate_region,
                                c_in,
                                c_out,
                                edge_records,
                            )
                        ):
                            frozen_candidate = frozenset(candidate_region)
                            candidates[frozen_candidate] = _cheap_search_score(
                                graph,
                                candidate_region,
                                len(c_in),
                                len(c_out),
                            )

                if len(region) >= max_region_nodes:
                    continue

                frontier: set[Node] = set()
                for node in region:
                    frontier.update(undirected.neighbors(node))
                frontier.difference_update(region)

                for new_node in frontier:
                    expanded = frozenset(region | {new_node})
                    if expanded in globally_seen:
                        continue

                    _i, expanded_in, expanded_out = boundary_edge_ids(
                        graph,
                        set(expanded),
                    )

                    # Allow temporary boundary overflow during beam expansion.
                    if (
                        len(expanded_in) > 2 * max_input_ports + 2
                        or len(expanded_out) > 2 * max_output_ports + 2
                    ):
                        continue

                    next_beam_scores[expanded] = _cheap_search_score(
                        graph,
                        set(expanded),
                        len(expanded_in),
                        len(expanded_out),
                    )

            if not next_beam_scores:
                break

            beam = [
                region
                for region, _score in sorted(
                    next_beam_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:beam_width]
            ]

    ranked = sorted(
        candidates.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        set(region)
        for region, _score in ranked[:max_candidates]
    ]


def _validate_region_for_compression(
    hierarchy: GraphHierarchy,
    region_nodes: set[Node],
    *,
    max_input_ports: int,
    max_output_ports: int,
    min_region_nodes: int,
    require_cycle_closed: bool = True,
) -> bool:
    graph = hierarchy.current_graph

    if not region_nodes:
        return False
    if not region_nodes <= set(graph.nodes):
        return False
    if len(region_nodes) < min_region_nodes:
        return False
    if not nx.is_weakly_connected(graph.subgraph(region_nodes)):
        return False

    _internal, incoming, outgoing = boundary_edge_ids(
        graph,
        region_nodes,
    )

    if not (
        1 <= len(incoming) <= max_input_ports
        and 1 <= len(outgoing) <= max_output_ports
    ):
        return False

    if not respects_lower_block_atomicity(
        graph,
        region_nodes,
        hierarchy.edge_records,
    ):
        return False

    if not region_has_basic_directional_support(
        graph,
        region_nodes,
        incoming,
        outgoing,
        hierarchy.edge_records,
    ):
        return False

    if require_cycle_closed and not is_cycle_closed_region(
        graph, region_nodes
    ):
        return False

    return True


def compress_one_region(
    hierarchy: GraphHierarchy,
    region_nodes: set[Node],
    *,
    layer: int,
    max_input_ports: int,
    max_output_ports: int,
    evaluation: RegionEvaluation | None = None,
    require_cycle_closed: bool = True,
) -> CompressionBlock:
    """
    Structurally replace one region by all input x output macro edges.

    No weight solving is performed.
    """

    graph = hierarchy.current_graph
    records = hierarchy.edge_records

    if not region_nodes <= set(graph.nodes):
        raise ValueError("region contains nodes not present in current graph.")

    if not nx.is_weakly_connected(graph.subgraph(region_nodes)):
        raise ValueError("region must be weakly connected.")

    internal, incoming, outgoing = boundary_edge_ids(
        graph,
        region_nodes,
    )

    if not (1 <= len(incoming) <= max_input_ports):
        raise ValueError(
            f"input count {len(incoming)} is outside allowed range."
        )

    if not (1 <= len(outgoing) <= max_output_ports):
        raise ValueError(
            f"output count {len(outgoing)} is outside allowed range."
        )

    if not respects_lower_block_atomicity(
        graph,
        region_nodes,
        records,
    ):
        raise ValueError("region cuts a lower block in half.")

    if require_cycle_closed and not is_cycle_closed_region(graph, region_nodes):
        raise ValueError(
            "region is not cycle-closed: an outside exit->entry return path "
            "would recreate a cycle after macro-edge replacement."
        )

    block_id = _new_id(f"B{layer}")
    macro_edge_ids: list[str] = []

    # Remove the region from the current visible graph.
    # All removed edges remain in edge_records for later reconstruction.
    graph.remove_nodes_from(region_nodes)

    for input_edge_id in incoming:
        input_record = records[input_edge_id]

        for output_edge_id in outgoing:
            output_record = records[output_edge_id]
            macro_edge_id = _new_id(f"M{layer}")

            record = EdgeRecord(
                edge_id=macro_edge_id,
                u=input_record.u,
                v=output_record.v,
                kind="macro",
                layer=layer,
                weight=None,
                block_id=block_id,
                input_edge_id=input_edge_id,
                output_edge_id=output_edge_id,
                expansion_edge_ids=None,
                data={
                    "entry_inside_node": input_record.v,
                    "exit_inside_node": output_record.u,
                    "solve_status": "unresolved",
                },
            )

            records[macro_edge_id] = record
            macro_edge_ids.append(macro_edge_id)

            graph.add_edge(
                record.u,
                record.v,
                key=macro_edge_id,
                edge_id=macro_edge_id,
                kind="macro",
                layer=layer,
                block_id=block_id,
            )

    snapshot = {} if evaluation is None else evaluation.to_dict()

    return CompressionBlock(
        block_id=block_id,
        layer=layer,
        region_nodes=frozenset(region_nodes),
        internal_edge_ids=tuple(internal),
        input_edge_ids=tuple(incoming),
        output_edge_ids=tuple(outgoing),
        macro_edge_ids=tuple(macro_edge_ids),
        evaluation_snapshot=snapshot,
    )


def _should_stop(
    top: TopProblemEstimate,
    config: EvaluationConfig,
) -> tuple[bool, str]:
    if config.stop_if_top_graph_is_dag and top.is_dag:
        return True, "TOP_GRAPH_IS_DAG"

    if top.qa_target_reached:
        return True, "TOP_QUBO_TARGET_REACHED"

    return False, ""


def compress_olc_hierarchy_prioritized(
    graph: nx.DiGraph | nx.MultiDiGraph,
    *,
    max_input_ports: int,
    max_output_ports: int,

    # Evaluation / stopping policy
    evaluation_config: EvaluationConfig | None = None,
    priority_model: PriorityModel | None = None,

    # Compression limits
    max_compression_steps: int = 100,
    target_node_count: int | None = None,

    # Candidate supply
    regions_by_step: Sequence[Sequence[Iterable[Node]]] | None = None,
    region_finder: Callable[
        [nx.MultiDiGraph, Mapping[str, EdgeRecord], int, int, int],
        Sequence[set[Node]],
    ] | None = None,

    # Built-in search settings
    min_region_nodes: int = 2,
    max_region_nodes: int = 20,
    beam_width: int = 24,
    max_candidates: int = 256,

    # Boundary closure policy
    # A cycle-closed region has no outside q~>p path from an exit endpoint
    # back to an entry endpoint. This prevents a new macro edge p->q from
    # immediately recreating a cycle through the outside graph.
    require_cycle_closed_regions: bool = True,
    auto_expand_cycle_closure: bool = True,
    max_cycle_closure_rounds: int = 32,

    # Greedy policy
    # Default = one region per step, then recompute all priorities.
    # This makes the priority system adaptive.
    regions_per_step: int = 1,
) -> GraphHierarchy:
    """
    Priority-aware structural compression.

    Core behavior:
        1. estimate current top problem
        2. if DAG -> stop
        3. if top QUBO target reached -> stop
        4. discover candidates
        5. evaluate ALL candidates
        6. choose highest priority
        7. compress
        8. recompute priorities from the new graph
        9. if still too large, lower-priority candidates are naturally used later

    Therefore:
        low priority != reject
        low priority == use later if needed

    Only candidates violating an explicitly configured absolute hard limit are
    excluded.
    """

    if evaluation_config is None:
        evaluation_config = EvaluationConfig()

    if priority_model is None:
        priority_model = DefaultPriorityModel()

    if regions_per_step < 1:
        raise ValueError("regions_per_step must be >= 1.")

    normalized, records = normalize_olc_graph(graph)

    hierarchy = GraphHierarchy(
        original_graph=normalized.copy(),
        current_graph=normalized.copy(),
        edge_records=records,
        graphs_by_layer={0: normalized.copy()},
    )

    for step in range(1, max_compression_steps + 1):

        # ------------------------------------------------------------
        # Global stopping decision
        # ------------------------------------------------------------
        top_before = estimate_top_problem(
            hierarchy,
            evaluation_config,
        )

        if (
            target_node_count is not None
            and hierarchy.current_graph.number_of_nodes() <= target_node_count
        ):
            hierarchy.compression_log.append({
                "step": step,
                "action": "stop",
                "reason": "TARGET_NODE_COUNT_REACHED",
                "top_before": asdict(top_before),
            })
            break

        stop, stop_reason = _should_stop(
            top_before,
            evaluation_config,
        )
        if stop:
            hierarchy.compression_log.append({
                "step": step,
                "action": "stop",
                "reason": stop_reason,
                "top_before": asdict(top_before),
            })
            break

        # ------------------------------------------------------------
        # Candidate discovery
        # ------------------------------------------------------------
        if regions_by_step is not None:
            if step > len(regions_by_step):
                hierarchy.compression_log.append({
                    "step": step,
                    "action": "stop",
                    "reason": "NO_MORE_MANUAL_CANDIDATES",
                    "top_before": asdict(top_before),
                })
                break

            raw_candidates = [
                set(region)
                for region in regions_by_step[step - 1]
            ]

        elif region_finder is not None:
            raw_candidates = list(
                region_finder(
                    hierarchy.current_graph,
                    hierarchy.edge_records,
                    max_input_ports,
                    max_output_ports,
                    step,
                )
            )

        else:
            raw_candidates = find_bounded_boundary_regions(
                hierarchy.current_graph,
                hierarchy.edge_records,
                max_input_ports=max_input_ports,
                max_output_ports=max_output_ports,
                min_region_nodes=min_region_nodes,
                max_region_nodes=max_region_nodes,
                beam_width=beam_width,
                max_candidates=max_candidates,
                require_cycle_closed=require_cycle_closed_regions,
                auto_expand_cycle_closure=auto_expand_cycle_closure,
                max_cycle_closure_rounds=max_cycle_closure_rounds,
            )

        # Manual/custom candidates are also normalized to the same cycle-closed
        # rule used by the built-in finder. This keeps the invariant independent
        # of how candidates are supplied.
        prepared_candidates: list[set[Node]] = []
        prepared_seen: set[frozenset[Node]] = set()

        for raw_region in raw_candidates:
            region = set(raw_region)
            if require_cycle_closed_regions:
                if auto_expand_cycle_closure:
                    closure = cycle_close_region(
                        hierarchy.current_graph,
                        region,
                        max_region_nodes=max_region_nodes,
                        max_rounds=max_cycle_closure_rounds,
                    )
                    if not closure.is_cycle_closed or closure.exceeded_node_limit:
                        continue
                    region = set(closure.region_nodes)
                elif not is_cycle_closed_region(
                    hierarchy.current_graph, region
                ):
                    continue

            frozen = frozenset(region)
            if frozen in prepared_seen:
                continue
            prepared_seen.add(frozen)
            prepared_candidates.append(region)

        valid_candidates = [
            region
            for region in prepared_candidates
            if _validate_region_for_compression(
                hierarchy,
                set(region),
                max_input_ports=max_input_ports,
                max_output_ports=max_output_ports,
                min_region_nodes=min_region_nodes,
                require_cycle_closed=require_cycle_closed_regions,
            )
        ]

        if not valid_candidates:
            hierarchy.compression_log.append({
                "step": step,
                "action": "stop",
                "reason": "NO_VALID_CANDIDATE",
                "top_before": asdict(top_before),
            })
            break

        # ------------------------------------------------------------
        # FULL evaluation + ranking
        # ------------------------------------------------------------
        evaluations = evaluate_candidates(
            hierarchy,
            valid_candidates,
            config=evaluation_config,
            priority_model=priority_model,
        )

        eligible = [
            ev
            for ev in evaluations
            if ev.eligible
        ]

        if not eligible:
            hierarchy.compression_log.append({
                "step": step,
                "action": "stop",
                "reason": "NO_ELIGIBLE_CANDIDATE",
                "top_before": asdict(top_before),
                "candidate_count": len(evaluations),
            })
            break

        # ------------------------------------------------------------
        # Greedy selection
        #
        # Default regions_per_step = 1.
        # If >1, choose non-overlapping candidates in priority order.
        # ------------------------------------------------------------
        selected: list[RegionEvaluation] = []
        selected_nodes: set[Node] = set()

        for ev in eligible:
            region = set(ev.region_nodes)

            if region & selected_nodes:
                continue

            selected.append(ev)
            selected_nodes.update(region)

            if len(selected) >= regions_per_step:
                break

        if not selected:
            hierarchy.compression_log.append({
                "step": step,
                "action": "stop",
                "reason": "SELECTION_FAILED",
                "top_before": asdict(top_before),
            })
            break

        # ------------------------------------------------------------
        # Structural compression
        # ------------------------------------------------------------
        blocks: list[CompressionBlock] = []

        # With regions_per_step=1 this loop runs once.
        # For >1, candidates are node-disjoint. They are revalidated because
        # the current graph changes after each compression.
        for ev in selected:
            region = set(ev.region_nodes)

            if not _validate_region_for_compression(
                hierarchy,
                region,
                max_input_ports=max_input_ports,
                max_output_ports=max_output_ports,
                min_region_nodes=min_region_nodes,
                require_cycle_closed=require_cycle_closed_regions,
            ):
                continue

            block = compress_one_region(
                hierarchy,
                region,
                layer=step,
                max_input_ports=max_input_ports,
                max_output_ports=max_output_ports,
                evaluation=ev,
            )
            blocks.append(block)

        if not blocks:
            hierarchy.compression_log.append({
                "step": step,
                "action": "stop",
                "reason": "ALL_SELECTED_CANDIDATES_INVALIDATED",
                "top_before": asdict(top_before),
            })
            break

        hierarchy.blocks_by_layer[step] = blocks
        hierarchy.graphs_by_layer[step] = hierarchy.current_graph.copy()

        top_after = estimate_top_problem(
            hierarchy,
            evaluation_config,
        )

        hierarchy.compression_log.append({
            "step": step,
            "action": "compress",
            "selected_blocks": [b.block_id for b in blocks],
            "selected_evaluations": [
                b.evaluation_snapshot for b in blocks
            ],
            "top_before": asdict(top_before),
            "top_after": asdict(top_after),
        })

    return hierarchy


# Backward-friendly alias.
compress_olc_hierarchy = compress_olc_hierarchy_prioritized


def hierarchy_summary(
    hierarchy: GraphHierarchy,
    *,
    evaluation_config: EvaluationConfig | None = None,
) -> dict[str, Any]:

    if evaluation_config is None:
        evaluation_config = EvaluationConfig()

    top = estimate_top_problem(
        hierarchy,
        evaluation_config,
    )

    return {
        "original_nodes": hierarchy.original_graph.number_of_nodes(),
        "original_edges": hierarchy.original_graph.number_of_edges(),
        "final_nodes": hierarchy.current_graph.number_of_nodes(),
        "final_edges": hierarchy.current_graph.number_of_edges(),
        "max_layer": hierarchy.max_layer,
        "top_problem": asdict(top),
        "layers": {
            layer: {
                "block_count": len(blocks),
                "macro_edge_count": sum(
                    len(block.macro_edge_ids)
                    for block in blocks
                ),
                "graph_nodes_after_layer": hierarchy.graphs_by_layer[
                    layer
                ].number_of_nodes(),
                "graph_edges_after_layer": hierarchy.graphs_by_layer[
                    layer
                ].number_of_edges(),
                "blocks": [
                    {
                        "block_id": block.block_id,
                        "evaluation": block.evaluation_snapshot,
                    }
                    for block in blocks
                ],
            }
            for layer, blocks in hierarchy.blocks_by_layer.items()
        },
        "compression_log": copy.deepcopy(hierarchy.compression_log),
    }


def export_current_graph_graphml(
    hierarchy: GraphHierarchy,
    path: str | Path,
) -> None:
    """Export the currently visible compressed graph."""

    output = nx.MultiDiGraph()

    for node, data in hierarchy.current_graph.nodes(data=True):
        output.add_node(
            node,
            **{
                str(key): str(value)
                for key, value in data.items()
            },
        )

    for u, v, key, data in hierarchy.current_graph.edges(
        keys=True,
        data=True,
    ):
        record = hierarchy.edge_records[data["edge_id"]]

        output.add_edge(
            u,
            v,
            key=str(key),
            edge_id=record.edge_id,
            kind=record.kind,
            layer=int(record.layer),
            block_id="" if record.block_id is None else record.block_id,
            weight="" if record.weight is None else float(record.weight),
        )

    nx.write_graphml(output, str(path))


def format_candidate_table(
    evaluations: Sequence[RegionEvaluation],
) -> str:
    """
    Plain-text candidate table for debugging / reports.

    Keeps the evaluator transparent: you can inspect WHY a candidate was chosen.
    """

    header = (
        "tier  label          score        DAG   V    E    E/V   "
        "a  b  ab   Qmax  Jmax  dQ_future  reason"
    )
    lines = [header, "-" * len(header)]

    for ev in evaluations:
        qmax = "-" if ev.local_qubo_max_variables is None else str(
            ev.local_qubo_max_variables
        )
        jmax = "-" if ev.local_qubo_max_couplings is None else str(
            ev.local_qubo_max_couplings
        )

        lines.append(
            f"{ev.priority_tier:>4}  "
            f"{ev.priority_label:<13} "
            f"{ev.priority_score:>10.4g}  "
            f"{str(ev.is_dag):<5} "
            f"{ev.node_count:>4} "
            f"{ev.internal_edge_count:>4} "
            f"{ev.edge_to_node_ratio:>5.2f} "
            f"{ev.input_count:>2} "
            f"{ev.output_count:>2} "
            f"{ev.port_pair_count:>3} "
            f"{qmax:>5} "
            f"{jmax:>5} "
            f"{ev.future_qubo_variable_gain_proxy:>9}  "
            f"{ev.reason}"
        )

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SafePreprocessConfig:
    """Conservative settings for pre-Plan-B graph cleanup."""

    # Strict containment: default 0 means the shorter read must be aligned
    # from coordinate 0 to its full length.
    full_contain_end_tolerance: int = 0

    # Optional rejection thresholds. Defaults deliberately do not reject
    # anything by score alone.
    min_overlap_length: int = 0
    min_identity: float = 0.0

    # If True, end-to-end overlap is required for an edge to survive when
    # coordinate fields are available. Internal alignments are removed.
    remove_internal_alignments: bool = True

    # hifiasm-like multi-arc cleanup: one directed u->v representative.
    deduplicate_parallel_edges: bool = True

    # Contract forced linear components, but never directed cycles.
    contract_linear_unitigs: bool = True

    # PAF-like edge attribute aliases. The first existing key is used.
    q_start_keys: tuple[str, ...] = ("q_start", "qstart", "qs")
    q_end_keys: tuple[str, ...] = ("q_end", "qend", "qe")
    q_len_keys: tuple[str, ...] = ("q_len", "qlen", "query_length")
    t_start_keys: tuple[str, ...] = ("t_start", "tstart", "ts")
    t_end_keys: tuple[str, ...] = ("t_end", "tend", "te")
    t_len_keys: tuple[str, ...] = ("t_len", "tlen", "target_length")
    match_keys: tuple[str, ...] = ("matches", "nmatch", "match")
    block_len_keys: tuple[str, ...] = ("block_len", "alen", "alignment_length", "overlap_len")
    mapq_keys: tuple[str, ...] = ("mapq", "mapping_quality")
    identity_keys: tuple[str, ...] = ("identity", "id")


@dataclass(frozen=True)
class OverlapClassification:
    kind: str  # DOVETAIL / Q_CONTAINED / T_CONTAINED / INTERNAL / TOO_SHORT / UNKNOWN
    overlap_length: int | None
    identity: float | None
    reason: str


@dataclass
class SafePreprocessResult:
    graph: nx.MultiDiGraph
    removed_contained_nodes: set[Node] = field(default_factory=set)
    removed_edge_records: list[dict[str, Any]] = field(default_factory=list)
    duplicate_groups: list[dict[str, Any]] = field(default_factory=list)
    unitigs: dict[Node, tuple[Node, ...]] = field(default_factory=dict)
    node_to_unitig: dict[Node, Node] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------------

def _first(data: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _as_int(value: Any | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _alignment_fields(
    data: Mapping[str, Any],
    config: SafePreprocessConfig,
) -> dict[str, int | float | None]:
    qs = _as_int(_first(data, config.q_start_keys))
    qe = _as_int(_first(data, config.q_end_keys))
    ql = _as_int(_first(data, config.q_len_keys))
    ts = _as_int(_first(data, config.t_start_keys))
    te = _as_int(_first(data, config.t_end_keys))
    tl = _as_int(_first(data, config.t_len_keys))

    block_len = _as_int(_first(data, config.block_len_keys))
    matches = _as_int(_first(data, config.match_keys))
    identity = _as_float(_first(data, config.identity_keys))

    if identity is None and matches is not None and block_len not in (None, 0):
        identity = matches / block_len

    if block_len is None and None not in (qs, qe):
        assert qs is not None and qe is not None
        block_len = max(0, qe - qs)

    return {
        "qs": qs,
        "qe": qe,
        "ql": ql,
        "ts": ts,
        "te": te,
        "tl": tl,
        "block_len": block_len,
        "identity": identity,
    }


# ---------------------------------------------------------------------------
# 1) hifiasm-like conservative overlap classification
# ---------------------------------------------------------------------------

def classify_overlap_safe(
    data: Mapping[str, Any],
    *,
    config: SafePreprocessConfig | None = None,
) -> OverlapClassification:
    """
    Conservative hifiasm-inspired overlap classification.

    Unlike hifiasm's permissive hang/fraction rules, this default version only
    calls a read "contained" when the shorter read is aligned end-to-end
    (tolerance configurable, default 0).

    This function classifies geometry only; it does not choose an assembly path.
    """

    if config is None:
        config = SafePreprocessConfig()

    f = _alignment_fields(data, config)
    qs, qe, ql = f["qs"], f["qe"], f["ql"]
    ts, te, tl = f["ts"], f["te"], f["tl"]
    block_len = f["block_len"]
    identity = f["identity"]

    if block_len is not None and block_len < config.min_overlap_length:
        return OverlapClassification(
            "TOO_SHORT", block_len, identity,
            f"overlap_length={block_len} < {config.min_overlap_length}",
        )

    if identity is not None and identity < config.min_identity:
        return OverlapClassification(
            "TOO_SHORT", block_len, identity,
            f"identity={identity:.6g} < {config.min_identity}",
        )

    # If geometry is absent, do not guess. Keep the edge as UNKNOWN.
    if None in (qs, qe, ql, ts, te, tl):
        return OverlapClassification(
            "UNKNOWN", block_len, identity,
            "PAF-like query/target coordinates are incomplete; edge is kept.",
        )

    assert isinstance(qs, int) and isinstance(qe, int) and isinstance(ql, int)
    assert isinstance(ts, int) and isinstance(te, int) and isinstance(tl, int)

    tol = max(0, int(config.full_contain_end_tolerance))

    q_left = qs <= tol
    q_right = (ql - qe) <= tol
    t_left = ts <= tol
    t_right = (tl - te) <= tol

    q_full = q_left and q_right
    t_full = t_left and t_right

    # Strict *proper* containment. Equal-length full overlaps are not deleted
    # here because neither read is strictly contained in the other.
    if q_full and ql < tl:
        return OverlapClassification(
            "Q_CONTAINED", block_len, identity,
            "query is strictly shorter and aligned end-to-end inside target",
        )

    if t_full and tl < ql:
        return OverlapClassification(
            "T_CONTAINED", block_len, identity,
            "target is strictly shorter and aligned end-to-end inside query",
        )

    # End-to-end overlap. We deliberately do not infer final orientation here;
    # the upstream OLC builder owns edge direction/orientation.
    if (q_left or q_right) and (t_left or t_right):
        return OverlapClassification(
            "DOVETAIL", block_len, identity,
            "alignment reaches an end of both reads",
        )

    return OverlapClassification(
        "INTERNAL", block_len, identity,
        "alignment is internal to at least one read and is not strict containment",
    )


# ---------------------------------------------------------------------------
# Graph normalization for this small preprocessor
# ---------------------------------------------------------------------------

def _to_multidigraph(graph: nx.DiGraph | nx.MultiDiGraph) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_nodes_from((n, copy.deepcopy(d)) for n, d in graph.nodes(data=True))

    if graph.is_multigraph():
        for u, v, key, data in graph.edges(keys=True, data=True):
            g.add_edge(u, v, key=key, **copy.deepcopy(dict(data)))
    else:
        for idx, (u, v, data) in enumerate(graph.edges(data=True)):
            key = data.get("edge_id", f"pre::{idx}")
            g.add_edge(u, v, key=key, **copy.deepcopy(dict(data)))

    return g


def _edge_snapshot(u: Node, v: Node, key: Hashable, data: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "u": u,
        "v": v,
        "key": key,
        "reason": reason,
        "data": copy.deepcopy(dict(data)),
    }


# ---------------------------------------------------------------------------
# 2) duplicate parallel-edge reduction
# ---------------------------------------------------------------------------

def _edge_quality(data: Mapping[str, Any], config: SafePreprocessConfig) -> tuple[float, float, float]:
    f = _alignment_fields(data, config)
    overlap = float(f["block_len"] or 0)
    identity = float(f["identity"] or 0.0)
    mapq = float(_as_float(_first(data, config.mapq_keys)) or 0.0)
    return overlap, identity, mapq


def deduplicate_parallel_edges_safe(
    graph: nx.MultiDiGraph,
    *,
    config: SafePreprocessConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    hifiasm/asg_arc_del_multi-inspired cleanup.

    For each directed pair u->v, keep one representative edge with the largest
    (overlap_length, identity, mapq). All discarded records are copied into the
    representative edge's `merged_parallel_edges` metadata.
    """

    if config is None:
        config = SafePreprocessConfig()

    removed: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []

    for u, v in list({(u, v) for u, v in graph.edges()}):
        parallel = list(graph.get_edge_data(u, v, default={}).items())
        if len(parallel) <= 1:
            continue

        parallel.sort(
            key=lambda item: _edge_quality(item[1], config),
            reverse=True,
        )
        keep_key, keep_data = parallel[0]
        merged_records: list[dict[str, Any]] = []

        for key, data in parallel[1:]:
            snap = _edge_snapshot(u, v, key, data, "DUPLICATE_PARALLEL_EDGE")
            removed.append(snap)
            merged_records.append(snap)
            graph.remove_edge(u, v, key=key)

        current = graph[u][v][keep_key]
        previous = list(current.get("merged_parallel_edges", []))
        current["merged_parallel_edges"] = previous + merged_records
        current["parallel_edge_count_before_merge"] = len(parallel)

        groups.append({
            "u": u,
            "v": v,
            "kept_key": keep_key,
            "removed_count": len(parallel) - 1,
        })

    return removed, groups


# ---------------------------------------------------------------------------
# 3) cycle-safe forced-chain / unitig-like contraction
# ---------------------------------------------------------------------------

def _forced_edge_graph(graph: nx.MultiDiGraph) -> nx.DiGraph:
    """
    Build F containing u->v iff that edge is structurally forced:
        out_degree_G(u) == 1 and in_degree_G(v) == 1.

    Parallel edges should already have been deduplicated.
    """

    f = nx.DiGraph()
    f.add_nodes_from(graph.nodes)

    for u, v in graph.edges():
        if u == v:
            continue
        if graph.out_degree(u) == 1 and graph.in_degree(v) == 1:
            f.add_edge(u, v)

    return f


def _linear_forced_components(graph: nx.MultiDiGraph) -> list[list[Node]]:
    forced = _forced_edge_graph(graph)
    result: list[list[Node]] = []

    for component_nodes in nx.weakly_connected_components(forced):
        sub = forced.subgraph(component_nodes).copy()
        if sub.number_of_edges() == 0:
            continue

        # Never contract a directed cycle.  Checking only the forced-edge
        # graph is not enough: the same vertex set may contain a non-forced
        # back edge in the original OLC graph.  Contracting such a component
        # would silently delete a real cycle.
        if not nx.is_directed_acyclic_graph(sub):
            continue
        original_induced = nx.DiGraph(graph.subgraph(component_nodes))
        if not nx.is_directed_acyclic_graph(original_induced):
            continue

        starts = [n for n in sub.nodes if sub.in_degree(n) == 0 and sub.out_degree(n) > 0]
        if len(starts) != 1:
            continue

        path = [starts[0]]
        current = starts[0]
        seen = {current}

        while sub.out_degree(current) == 1:
            nxt = next(iter(sub.successors(current)))
            if nxt in seen:
                path = []
                break
            path.append(nxt)
            seen.add(nxt)
            current = nxt

        if len(path) >= 2 and len(path) == sub.number_of_nodes():
            result.append(path)

    # Longest first; components are disjoint, but deterministic ordering helps logs.
    result.sort(key=lambda p: (-len(p), tuple(map(str, p))))
    return result


def contract_linear_unitigs_safe(
    graph: nx.MultiDiGraph,
) -> tuple[dict[Node, tuple[Node, ...]], dict[Node, Node], list[dict[str, Any]]]:
    """
    Contract maximal forced linear components without touching directed cycles.

    Boundary edges are rewired to the unitig node. Internal edge data is stored
    inside the unitig node under `unitig_internal_edges` for reconstruction.
    """

    unitigs: dict[Node, tuple[Node, ...]] = {}
    node_to_unitig: dict[Node, Node] = {}
    removed_internal_edges: list[dict[str, Any]] = []

    paths = _linear_forced_components(graph)

    for index, path in enumerate(paths, start=1):
        # Previous contractions are disjoint by construction, but recheck.
        if not all(node in graph for node in path):
            continue

        path_set = set(path)
        unitig_id = f"__UNITIG__::{index:06d}"
        while unitig_id in graph:
            index += 1
            unitig_id = f"__UNITIG__::{index:06d}"

        incoming: list[tuple[Node, Node, Hashable, dict[str, Any]]] = []
        outgoing: list[tuple[Node, Node, Hashable, dict[str, Any]]] = []
        internal: list[dict[str, Any]] = []

        for u, v, key, data in list(graph.edges(keys=True, data=True)):
            u_in = u in path_set
            v_in = v in path_set

            if u_in and v_in:
                snap = _edge_snapshot(u, v, key, data, "UNITIG_INTERNAL_EDGE")
                internal.append(snap)
                removed_internal_edges.append(snap)
            elif (not u_in) and v_in:
                incoming.append((u, v, key, copy.deepcopy(dict(data))))
            elif u_in and (not v_in):
                outgoing.append((u, v, key, copy.deepcopy(dict(data))))

        member_node_data = {
            node: copy.deepcopy(dict(graph.nodes[node]))
            for node in path
        }

        graph.remove_nodes_from(path)
        graph.add_node(
            unitig_id,
            kind="safe_unitig",
            unitig_members=tuple(path),
            unitig_member_data=member_node_data,
            unitig_internal_edges=internal,
        )

        # Preserve boundary edge keys/data, only rewrite the endpoint.
        for u, old_v, key, data in incoming:
            data.setdefault("pre_unitig_v", old_v)
            graph.add_edge(u, unitig_id, key=key, **data)

        for old_u, v, key, data in outgoing:
            data.setdefault("pre_unitig_u", old_u)
            graph.add_edge(unitig_id, v, key=key, **data)

        unitigs[unitig_id] = tuple(path)
        for node in path:
            node_to_unitig[node] = unitig_id

    return unitigs, node_to_unitig, removed_internal_edges


# ---------------------------------------------------------------------------
# Integrated small function requested by the project
# ---------------------------------------------------------------------------

def safe_preprocess_before_planb(
    graph: nx.DiGraph | nx.MultiDiGraph,
    *,
    config: SafePreprocessConfig | None = None,
) -> SafePreprocessResult:
    """
    Run ONLY conservative preprocessing before the first Plan-B compression.

    Pipeline:
        classify overlaps
        -> remove strict full-contained reads
        -> remove internal / explicitly-too-short alignments
        -> deduplicate parallel u->v edges
        -> cycle-safe forced-chain contraction

    No bubble popping, no best-overlap path choice, no transitive reduction,
    no cycle breaking, and no Plan-B macro-edge processing are performed.
    """

    if config is None:
        config = SafePreprocessConfig()

    g = _to_multidigraph(graph)

    result = SafePreprocessResult(graph=g)
    initial_nodes = g.number_of_nodes()
    initial_edges = g.number_of_edges()

    contained: set[Node] = set()
    edges_to_remove: list[tuple[Node, Node, Hashable, str]] = []

    # ---- classify all current overlap edges ----
    for u, v, key, data in list(g.edges(keys=True, data=True)):
        cls = classify_overlap_safe(data, config=config)
        data["safe_overlap_class"] = cls.kind
        data["safe_overlap_reason"] = cls.reason

        if cls.kind == "Q_CONTAINED":
            contained.add(u)
        elif cls.kind == "T_CONTAINED":
            contained.add(v)
        elif cls.kind == "INTERNAL" and config.remove_internal_alignments:
            edges_to_remove.append((u, v, key, "INTERNAL_ALIGNMENT"))
        elif cls.kind == "TOO_SHORT":
            edges_to_remove.append((u, v, key, "EXPLICIT_FILTER_THRESHOLD"))

    # Remove explicitly invalid edges first so the log is clear.
    for u, v, key, reason in edges_to_remove:
        if g.has_edge(u, v, key=key):
            data = copy.deepcopy(dict(g[u][v][key]))
            result.removed_edge_records.append(
                _edge_snapshot(u, v, key, data, reason)
            )
            g.remove_edge(u, v, key=key)

    # Strict contained reads are removed as vertices. Preserve every incident edge.
    for node in sorted(contained, key=str):
        if node not in g:
            continue
        for u, v, key, data in list(g.in_edges(node, keys=True, data=True)):
            result.removed_edge_records.append(
                _edge_snapshot(u, v, key, data, "INCIDENT_TO_FULL_CONTAINED_READ")
            )
        for u, v, key, data in list(g.out_edges(node, keys=True, data=True)):
            # Self-loop may already have been recorded from in_edges.
            if not (u == node and v == node):
                result.removed_edge_records.append(
                    _edge_snapshot(u, v, key, data, "INCIDENT_TO_FULL_CONTAINED_READ")
                )
        g.remove_node(node)

    result.removed_contained_nodes = {node for node in contained if node in set(graph.nodes)}

    # hifiasm-like multi-edge cleanup.
    if config.deduplicate_parallel_edges:
        removed_dups, groups = deduplicate_parallel_edges_safe(g, config=config)
        result.removed_edge_records.extend(removed_dups)
        result.duplicate_groups = groups

    # Unitig-like forced linear contraction. Cycles are explicitly skipped.
    if config.contract_linear_unitigs:
        unitigs, node_to_unitig, internal_removed = contract_linear_unitigs_safe(g)
        result.unitigs = unitigs
        result.node_to_unitig = node_to_unitig
        result.removed_edge_records.extend(internal_removed)

    result.graph = g
    result.stats = {
        "initial_nodes": initial_nodes,
        "initial_edges": initial_edges,
        "final_nodes": g.number_of_nodes(),
        "final_edges": g.number_of_edges(),
        "removed_contained_nodes": len(result.removed_contained_nodes),
        "removed_edge_records": len(result.removed_edge_records),
        "duplicate_groups": len(result.duplicate_groups),
        "unitig_count": len(result.unitigs),
    }
    return result

# ============================================================================
# OLC_CLASSICAL
# ============================================================================

from dataclasses import dataclass, field
from math import inf
from typing import Any, Hashable, Iterable, Mapping

import networkx as nx



Node = Hashable


@dataclass(frozen=True)
class PathEdge:
    """Classical DAG solver で扱う1本の可視辺。"""

    edge_id: str
    u: Node
    v: Node
    weight: float
    kind: str
    block_id: str | None


@dataclass
class LocalPathProblem:
    """
    固定された macro edge p_alpha -> q_beta を評価するための局所 path 問題。

    solve_graph:
        実際に利用可能な辺だけを含むグラフ。
    certificate_graph:
        DAG 判定用。下位 block の全 candidate macro edge を、
        feasible / infeasible に関係なく構造として残す。

    complete_macro_groups が True なら、各 visible child block について
    candidate macro edge 群が丸ごと含まれている。
    この条件と certificate_graph が DAG であることから、同じ child block
    の macro edge を1本の有向 path が2回使えないことが保証される。
    """

    solve_graph: nx.MultiDiGraph
    certificate_graph: nx.MultiDiGraph
    source: Node
    target: Node
    path_edges: dict[str, PathEdge]
    expected_child_blocks: frozenset[str]
    complete_macro_groups: bool
    missing_feasible_child_blocks: frozenset[str] = frozenset()
    notes: list[str] = field(default_factory=list)

    @property
    def is_certificate_dag(self) -> bool:
        return nx.is_directed_acyclic_graph(self.certificate_graph)

    @property
    def can_use_fast_classical_solver(self) -> bool:
        return self.is_certificate_dag and self.complete_macro_groups


@dataclass
class ClassicalPathSolution:
    feasible: bool
    weight: float
    selected_actual_edge_ids: list[str]
    visited_node_count: int
    required_node_count: int
    used_child_blocks: tuple[str, ...]
    message: str = ""


@dataclass
class _DPState:
    visited: int
    weight: float
    prev_node: Node | None
    prev_macro_count: int | None
    prev_edge_id: str | None
    # Only used by the general DAG solver when a compressed child block could
    # structurally be traversed more than once.  Fast cases keep this at 0.
    prev_mask: int | None = None


def _macro_status(record: Any) -> str:
    """
    macro edge の解状態を統一して読む。

    新しい hybrid solver は record.data['solve_status'] を設定するが、
    旧コードとの互換性のため weight が既にある場合も feasible とみなす。
    """

    if record.kind != "macro":
        return "feasible" if record.weight is not None else "unresolved"

    status = str(record.data.get("solve_status", "")).strip().lower()
    if status in {"feasible", "infeasible", "unresolved"}:
        return status
    if record.weight is not None:
        return "feasible"
    return "unresolved"


def _add_graph_edge(
    graph: nx.MultiDiGraph,
    *,
    edge_id: str,
    u: Node,
    v: Node,
    weight: float | None,
    kind: str,
    block_id: str | None,
) -> None:
    graph.add_edge(
        u,
        v,
        key=edge_id,
        edge_id=edge_id,
        weight=weight,
        kind=kind,
        block_id=block_id,
    )


def _collect_complete_group_check(
    hierarchy: GraphHierarchy,
    visible_macro_edge_ids_by_block: Mapping[str, set[str]],
) -> tuple[bool, list[str]]:
    """visible child block が全 candidate macro edge を丸ごと含むか検査する。"""

    block_map = hierarchy.block_map()
    complete = True
    notes: list[str] = []

    for child_block_id, visible_ids in visible_macro_edge_ids_by_block.items():
        child = block_map.get(child_block_id)
        if child is None:
            complete = False
            notes.append(
                f"child block {child_block_id} の CompressionBlock が見つかりません。"
            )
            continue

        expected = set(child.macro_edge_ids)
        if visible_ids != expected:
            complete = False
            notes.append(
                f"child block {child_block_id} の macro edge 群が不完全です: "
                f"visible={len(visible_ids)}, expected={len(expected)}"
            )

    return complete, notes


def build_macro_edge_path_problem(
    hierarchy: GraphHierarchy,
    block_id: str,
    macro_edge_id: str,
) -> LocalPathProblem:
    """
    固定された第n層 macro edge の classical / DAG 判定用局所グラフを作る。

    QUBO 用の q -> r -> p はここでは絶対に追加しない。
    追加すると意図的に cycle が生じるため、DAG 判定には使えない。
    """

    block = hierarchy.block_map()[block_id]
    if macro_edge_id not in block.macro_edge_ids:
        raise ValueError("macro_edge_id は指定 block に属していません。")

    records = hierarchy.edge_records
    macro_record = records[macro_edge_id]
    if macro_record.input_edge_id is None or macro_record.output_edge_id is None:
        raise ValueError("macro edge に input/output edge ID がありません。")

    input_record = records[macro_record.input_edge_id]
    output_record = records[macro_record.output_edge_id]

    source = f"__PORT_IN__::{macro_edge_id}"
    target = f"__PORT_OUT__::{macro_edge_id}"

    solve_graph = nx.MultiDiGraph()
    certificate_graph = nx.MultiDiGraph()
    solve_graph.add_nodes_from(block.region_nodes)
    certificate_graph.add_nodes_from(block.region_nodes)
    solve_graph.add_nodes_from([source, target])
    certificate_graph.add_nodes_from([source, target])

    path_edges: dict[str, PathEdge] = {}

    def add_fixed_boundary(record: Any, u: Node, v: Node) -> None:
        if record.weight is None:
            status = _macro_status(record)
            if status == "infeasible":
                raise ValueError(
                    f"固定境界辺 {record.edge_id} は infeasible です。"
                )
            raise RuntimeError(
                f"固定境界辺 {record.edge_id} の weight が未確定です。"
            )

        edge = PathEdge(
            edge_id=record.edge_id,
            u=u,
            v=v,
            weight=float(record.weight),
            kind=record.kind,
            block_id=record.block_id,
        )
        path_edges[edge.edge_id] = edge
        _add_graph_edge(
            solve_graph,
            edge_id=edge.edge_id,
            u=edge.u,
            v=edge.v,
            weight=edge.weight,
            kind=edge.kind,
            block_id=edge.block_id,
        )
        _add_graph_edge(
            certificate_graph,
            edge_id=edge.edge_id,
            u=edge.u,
            v=edge.v,
            weight=edge.weight,
            kind=edge.kind,
            block_id=edge.block_id,
        )

    add_fixed_boundary(input_record, source, input_record.v)

    visible_macro_ids_by_block: dict[str, set[str]] = {}
    feasible_macro_count_by_block: dict[str, int] = {}

    for edge_id in block.internal_edge_ids:
        record = records[edge_id]

        # certificate graph には構造として全 candidate edge を残す。
        _add_graph_edge(
            certificate_graph,
            edge_id=edge_id,
            u=record.u,
            v=record.v,
            weight=record.weight,
            kind=record.kind,
            block_id=record.block_id,
        )

        if record.kind == "macro" and record.block_id is not None:
            visible_macro_ids_by_block.setdefault(record.block_id, set()).add(edge_id)

        if record.kind == "original":
            if record.weight is None:
                raise RuntimeError(
                    f"original edge {edge_id} の weight が未確定です。"
                )
            feasible = True
        else:
            status = _macro_status(record)
            if status == "unresolved":
                raise RuntimeError(
                    f"下位 macro edge {edge_id} が未解決です。"
                    "下層から順番に解いてください。"
                )
            feasible = status == "feasible" and record.weight is not None

        if not feasible:
            continue

        edge = PathEdge(
            edge_id=edge_id,
            u=record.u,
            v=record.v,
            weight=float(record.weight),
            kind=record.kind,
            block_id=record.block_id,
        )
        path_edges[edge.edge_id] = edge
        _add_graph_edge(
            solve_graph,
            edge_id=edge.edge_id,
            u=edge.u,
            v=edge.v,
            weight=edge.weight,
            kind=edge.kind,
            block_id=edge.block_id,
        )
        if edge.kind == "macro" and edge.block_id is not None:
            feasible_macro_count_by_block[edge.block_id] = (
                feasible_macro_count_by_block.get(edge.block_id, 0) + 1
            )

    add_fixed_boundary(output_record, output_record.u, target)

    # atomicity が守られていれば、ここは通常 True になる。
    complete_groups, notes = _collect_complete_group_check(
        hierarchy,
        visible_macro_ids_by_block,
    )

    expected_child_blocks = frozenset(visible_macro_ids_by_block)
    missing = frozenset(
        block_id
        for block_id in expected_child_blocks
        if feasible_macro_count_by_block.get(block_id, 0) == 0
    )

    return LocalPathProblem(
        solve_graph=solve_graph,
        certificate_graph=certificate_graph,
        source=source,
        target=target,
        path_edges=path_edges,
        expected_child_blocks=expected_child_blocks,
        complete_macro_groups=complete_groups,
        missing_feasible_child_blocks=missing,
        notes=notes,
    )


def build_top_path_problem(
    hierarchy: GraphHierarchy,
    source: Node,
    target: Node,
) -> LocalPathProblem:
    """最終圧縮グラフの classical / DAG 判定用問題を作る。"""

    if source not in hierarchy.current_graph:
        raise ValueError("source が最終圧縮グラフにありません。")
    if target not in hierarchy.current_graph:
        raise ValueError("target が最終圧縮グラフにありません。")

    records = hierarchy.edge_records
    solve_graph = nx.MultiDiGraph()
    certificate_graph = nx.MultiDiGraph()
    solve_graph.add_nodes_from(hierarchy.current_graph.nodes)
    certificate_graph.add_nodes_from(hierarchy.current_graph.nodes)

    path_edges: dict[str, PathEdge] = {}
    visible_macro_ids_by_block: dict[str, set[str]] = {}
    feasible_macro_count_by_block: dict[str, int] = {}

    for u, v, _key, data in hierarchy.current_graph.edges(keys=True, data=True):
        edge_id = data["edge_id"]
        record = records[edge_id]

        _add_graph_edge(
            certificate_graph,
            edge_id=edge_id,
            u=u,
            v=v,
            weight=record.weight,
            kind=record.kind,
            block_id=record.block_id,
        )

        if record.kind == "macro" and record.block_id is not None:
            visible_macro_ids_by_block.setdefault(record.block_id, set()).add(edge_id)

        if record.kind == "original":
            if record.weight is None:
                raise RuntimeError(f"original edge {edge_id} の weight が未確定です。")
            feasible = True
        else:
            status = _macro_status(record)
            if status == "unresolved":
                raise RuntimeError(
                    f"最終グラフの macro edge {edge_id} が未解決です。"
                )
            feasible = status == "feasible" and record.weight is not None

        if not feasible:
            continue

        edge = PathEdge(
            edge_id=edge_id,
            u=u,
            v=v,
            weight=float(record.weight),
            kind=record.kind,
            block_id=record.block_id,
        )
        path_edges[edge_id] = edge
        _add_graph_edge(
            solve_graph,
            edge_id=edge_id,
            u=u,
            v=v,
            weight=edge.weight,
            kind=edge.kind,
            block_id=edge.block_id,
        )
        if edge.kind == "macro" and edge.block_id is not None:
            feasible_macro_count_by_block[edge.block_id] = (
                feasible_macro_count_by_block.get(edge.block_id, 0) + 1
            )

    complete_groups, notes = _collect_complete_group_check(
        hierarchy,
        visible_macro_ids_by_block,
    )
    expected_child_blocks = frozenset(visible_macro_ids_by_block)
    missing = frozenset(
        block_id
        for block_id in expected_child_blocks
        if feasible_macro_count_by_block.get(block_id, 0) == 0
    )

    return LocalPathProblem(
        solve_graph=solve_graph,
        certificate_graph=certificate_graph,
        source=source,
        target=target,
        path_edges=path_edges,
        expected_child_blocks=expected_child_blocks,
        complete_macro_groups=complete_groups,
        missing_feasible_child_blocks=missing,
        notes=notes,
    )


def _is_better(candidate: _DPState, incumbent: _DPState | None) -> bool:
    """
    同じ (node, macro_count) では、まず通過頂点数を最大化し、
    同じ頂点数なら weight を最大化する。

    DAG では現在 node より前に飛ばした頂点を後から取り戻せないため、
    visited が大きい状態が小さい状態を支配する。
    """

    if incumbent is None:
        return True
    if candidate.visited != incumbent.visited:
        return candidate.visited > incumbent.visited
    return candidate.weight > incumbent.weight


def _ambiguous_macro_blocks_in_dag(problem: LocalPathProblem) -> set[str]:
    """
    Return child blocks for which the feasible DAG contains a directed route
    allowing two different macro edges of the same block to be used in order.

    For a block B with feasible macro edges e1=(u1,v1), e2=(u2,v2), a single
    directed path can potentially use e1 and later e2 when v1 can reach u2.
    Such a block needs an explicit one-use bit in the exact DAG DP.

    Blocks not returned here are structurally non-repeatable, so a simple
    macro-edge count is sufficient for them.
    """

    graph = nx.DiGraph(problem.solve_graph)
    by_block: dict[str, list[PathEdge]] = {}

    for edge in problem.path_edges.values():
        if (
            edge.kind == "macro"
            and edge.block_id is not None
            and edge.block_id in problem.expected_child_blocks
        ):
            by_block.setdefault(edge.block_id, []).append(edge)

    ambiguous: set[str] = set()
    descendants_cache: dict[Node, set[Node]] = {}

    def reachable_from(node: Node) -> set[Node]:
        if node not in descendants_cache:
            reached = {node}
            reached.update(nx.descendants(graph, node))
            descendants_cache[node] = reached
        return descendants_cache[node]

    for block_id, edges in by_block.items():
        if len(edges) <= 1:
            continue

        for first in edges:
            reached = reachable_from(first.v)
            if any(
                second is not first and second.u in reached
                for second in edges
            ):
                ambiguous.add(block_id)
                break

    return ambiguous


def solve_dag_path_problem(problem: LocalPathProblem) -> ClassicalPathSolution:
    """
    Solve the DAG source->target Hamiltonian-path problem exactly.

    Every visible child block must be used exactly once.  The old implementation
    needed a complete a*b macro-edge group and then tracked only the total
    number of used macro edges.  Adaptive pruning intentionally removes
    infeasible macro edges, so that completeness assumption no longer holds.

    This implementation keeps the fast count-only DP for blocks that are
    structurally non-repeatable in the feasible DAG.  Only blocks that *could*
    be traversed twice receive a bit in a small exact mask.

    Complexity:
        O( 2^A * K * (|V| + |E|) )

    where K is the number of visible child blocks and A<=K is the number of
    structurally ambiguous blocks.  In the common case A=0 this reduces to the
    previous O(K(|V|+|E|)) algorithm, while remaining exact after infeasible
    macro-edge pruning.
    """

    graph = problem.solve_graph

    if not nx.is_directed_acyclic_graph(graph):
        return ClassicalPathSolution(
            feasible=False,
            weight=-inf,
            selected_actual_edge_ids=[],
            visited_node_count=0,
            required_node_count=graph.number_of_nodes(),
            used_child_blocks=(),
            message="solve graph が DAG ではありません。",
        )

    if problem.missing_feasible_child_blocks:
        missing = ", ".join(sorted(problem.missing_feasible_child_blocks))
        return ClassicalPathSolution(
            feasible=False,
            weight=-inf,
            selected_actual_edge_ids=[],
            visited_node_count=0,
            required_node_count=graph.number_of_nodes(),
            used_child_blocks=(),
            message=f"feasible macro edge が1本もない child block: {missing}",
        )

    if problem.source not in graph or problem.target not in graph:
        return ClassicalPathSolution(
            feasible=False,
            weight=-inf,
            selected_actual_edge_ids=[],
            visited_node_count=0,
            required_node_count=graph.number_of_nodes(),
            used_child_blocks=(),
            message="source または target が solve graph にありません。",
        )

    child_blocks = set(problem.expected_child_blocks)
    required_macro_count = len(child_blocks)
    required_node_count = graph.number_of_nodes()

    ambiguous_blocks = sorted(_ambiguous_macro_blocks_in_dag(problem))
    bit_of = {
        block_id: 1 << index
        for index, block_id in enumerate(ambiguous_blocks)
    }
    required_mask = (1 << len(ambiguous_blocks)) - 1

    # states[node][(macro_count, ambiguous_mask)] = best state
    states: dict[Node, dict[tuple[int, int], _DPState]] = {
        node: {} for node in graph.nodes
    }
    states[problem.source][(0, 0)] = _DPState(
        visited=1,
        weight=0.0,
        prev_node=None,
        prev_macro_count=None,
        prev_edge_id=None,
        prev_mask=None,
    )

    topo = list(nx.topological_sort(graph))

    for u in topo:
        if not states[u]:
            continue

        current_states = list(states[u].items())

        for _u, v, edge_key, data in graph.out_edges(u, keys=True, data=True):
            edge_id = str(data.get("edge_id", edge_key))
            edge = problem.path_edges.get(edge_id)
            if edge is None:
                continue

            block_id = (
                edge.block_id
                if edge.kind == "macro" and edge.block_id in child_blocks
                else None
            )
            increment = int(block_id is not None)

            for (macro_count, mask), state in current_states:
                new_count = macro_count + increment
                if new_count > required_macro_count:
                    continue

                new_mask = mask
                if block_id is not None and block_id in bit_of:
                    bit = bit_of[block_id]
                    if mask & bit:
                        # Exact one-use enforcement for ambiguous blocks.
                        continue
                    new_mask |= bit

                candidate = _DPState(
                    visited=state.visited + 1,
                    weight=state.weight + edge.weight,
                    prev_node=u,
                    prev_macro_count=macro_count,
                    prev_edge_id=edge_id,
                    prev_mask=mask,
                )
                key = (new_count, new_mask)
                incumbent = states[v].get(key)
                if _is_better(candidate, incumbent):
                    states[v][key] = candidate

    final_state = states[problem.target].get(
        (required_macro_count, required_mask)
    )
    if final_state is None or final_state.visited != required_node_count:
        visited = 0 if final_state is None else final_state.visited
        return ClassicalPathSolution(
            feasible=False,
            weight=-inf,
            selected_actual_edge_ids=[],
            visited_node_count=visited,
            required_node_count=required_node_count,
            used_child_blocks=(),
            message=(
                "DAG 上で全 explicit 頂点の通過 + 全 visible child block を"
                "ちょうど1回使用する path は見つかりません。"
            ),
        )

    selected_reversed: list[str] = []
    node = problem.target
    macro_count = required_macro_count
    mask = required_mask

    while node != problem.source:
        state = states[node].get((macro_count, mask))
        if (
            state is None
            or state.prev_edge_id is None
            or state.prev_node is None
            or state.prev_macro_count is None
            or state.prev_mask is None
        ):
            return ClassicalPathSolution(
                feasible=False,
                weight=-inf,
                selected_actual_edge_ids=[],
                visited_node_count=final_state.visited,
                required_node_count=required_node_count,
                used_child_blocks=(),
                message="DP predecessor の復元に失敗しました。",
            )

        selected_reversed.append(state.prev_edge_id)
        node = state.prev_node
        macro_count = state.prev_macro_count
        mask = state.prev_mask

    selected = list(reversed(selected_reversed))

    used_by_block: dict[str, int] = {}
    for edge_id in selected:
        edge = problem.path_edges[edge_id]
        if edge.kind == "macro" and edge.block_id in child_blocks:
            assert edge.block_id is not None
            used_by_block[edge.block_id] = used_by_block.get(edge.block_id, 0) + 1

    bad_blocks = {
        block_id: used_by_block.get(block_id, 0)
        for block_id in child_blocks
        if used_by_block.get(block_id, 0) != 1
    }
    if bad_blocks:
        return ClassicalPathSolution(
            feasible=False,
            weight=-inf,
            selected_actual_edge_ids=[],
            visited_node_count=final_state.visited,
            required_node_count=required_node_count,
            used_child_blocks=tuple(sorted(used_by_block)),
            message=f"child block one-use 検証に失敗しました: {bad_blocks}",
        )

    return ClassicalPathSolution(
        feasible=True,
        weight=final_state.weight,
        selected_actual_edge_ids=selected,
        visited_node_count=final_state.visited,
        required_node_count=required_node_count,
        used_child_blocks=tuple(sorted(used_by_block)),
        message=(
            "" if not ambiguous_blocks
            else f"exact DAG DP used {len(ambiguous_blocks)} ambiguous block bits"
        ),
    )

# ============================================================================
# OLC_QA
# ============================================================================

from dataclasses import dataclass, field
from math import comb, exp, inf, isfinite, log
from random import Random
from typing import Any, Hashable, Mapping, Protocol, Sequence

import networkx as nx

Node = Hashable


# ============================================================
# 式(23)–(25)
# ============================================================

def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return -inf

    maximum = max(values)
    if maximum == -inf:
        return -inf

    return maximum + log(sum(exp(value - maximum) for value in values))


def log_N_s(l: int, d: int, s: int = 4) -> float:
    """
    log N_s(l,d) をlog-spaceで計算する。

    N_s(l,d) = sum_{0<=i<=d} C(l+d,i)(s-1)^i
    """

    if l < 0 or d < 0:
        raise ValueError("l,dは0以上である必要があります。")
    if s < 2:
        raise ValueError("alphabet size sは2以上である必要があります。")

    terms: list[float] = []

    for i in range(d + 1):
        coefficient = comb(l + d, i)
        if coefficient == 0:
            continue

        term = log(coefficient)
        if i > 0:
            term += i * log(s - 1)
        terms.append(term)

    return _logsumexp(terms)


def likelihood_weight(l: int, d: int, s: int = 4) -> float:
    """
    式(25):
        L(l,d) = -log_s P_s(l,d)

    P_s(l,d)
      = [sum_{0<=i<=d} N_s(l,i)N_s(l,d-i)]
        / [(d+1)s^(l+d)]

    大きなl,dでもoverflowしないようlog-spaceで計算する。
    """

    if l < 0 or d < 0:
        raise ValueError("l,dは0以上である必要があります。")
    if s < 2:
        raise ValueError("alphabet size sは2以上である必要があります。")

    numerator_logs = [
        log_N_s(l, i, s) + log_N_s(l, d - i, s)
        for i in range(d + 1)
    ]
    log_numerator = _logsumexp(numerator_logs)
    log_denominator = log(d + 1) + (l + d) * log(s)
    log_probability = log_numerator - log_denominator

    return -log_probability / log(s)


def ensure_original_edge_weights(
    hierarchy: GraphHierarchy,
    *,
    match_key: str = "l",
    error_key: str = "d",
    alphabet_size: int = 4,
) -> None:
    """
    original edgeにweightが無い場合、式(25)のl,dから計算する。
    """

    for record in hierarchy.edge_records.values():
        if record.kind != "original" or record.weight is not None:
            continue

        if match_key not in record.data or error_key not in record.data:
            raise ValueError(
                f"元辺{record.edge_id}にweightも"
                f"{match_key},{error_key}もありません。"
            )

        record.weight = likelihood_weight(
            int(record.data[match_key]),
            int(record.data[error_key]),
            alphabet_size,
        )


# ============================================================
# QUBO表現
# ============================================================

@dataclass
class QUBOModel:
    linear: dict[str, float] = field(default_factory=dict)
    quadratic: dict[tuple[str, str], float] = field(default_factory=dict)
    offset: float = 0.0

    def add_linear(self, variable: str, coefficient: float) -> None:
        self.linear[variable] = (
            self.linear.get(variable, 0.0) + float(coefficient)
        )

    def add_quadratic(
        self,
        first: str,
        second: str,
        coefficient: float,
    ) -> None:
        if first == second:
            # binary変数ではx^2=x
            self.add_linear(first, coefficient)
            return

        key = tuple(sorted((first, second)))
        self.quadratic[key] = (
            self.quadratic.get(key, 0.0) + float(coefficient)
        )

    def add_square(
        self,
        coefficients: Mapping[str, float],
        constant: float,
        penalty: float,
    ) -> None:
        """
        penalty * (constant + sum_i c_i x_i)^2 を追加する。
        """

        penalty = float(penalty)
        self.offset += penalty * constant * constant

        items = list(coefficients.items())

        for variable, coefficient in items:
            self.add_linear(
                variable,
                penalty * (
                    coefficient * coefficient
                    + 2.0 * constant * coefficient
                ),
            )

        for index, (first, first_coefficient) in enumerate(items):
            for second, second_coefficient in items[index + 1:]:
                self.add_quadratic(
                    first,
                    second,
                    2.0
                    * penalty
                    * first_coefficient
                    * second_coefficient,
                )

    @property
    def variables(self) -> list[str]:
        result = set(self.linear)
        for first, second in self.quadratic:
            result.add(first)
            result.add(second)
        return sorted(result)

    def energy(self, sample: Mapping[str, int]) -> float:
        value = self.offset

        for variable, coefficient in self.linear.items():
            value += coefficient * int(sample.get(variable, 0))

        for (first, second), coefficient in self.quadratic.items():
            value += (
                coefficient
                * int(sample.get(first, 0))
                * int(sample.get(second, 0))
            )

        return value

    def to_qubo_dict(self) -> dict[tuple[str, str], float]:
        qubo: dict[tuple[str, str], float] = {}

        for variable, coefficient in self.linear.items():
            qubo[(variable, variable)] = coefficient

        for key, coefficient in self.quadratic.items():
            qubo[key] = coefficient

        return qubo


@dataclass
class LocalEdge:
    variable: str
    u: Node
    v: Node
    weight: float
    actual_edge_id: str | None
    kind: str
    block_id: str | None


@dataclass
class LocalQUBOProblem:
    model: QUBOModel
    local_graph: nx.MultiDiGraph
    local_edges: dict[str, LocalEdge]
    actual_edge_variable: dict[str, str]

    source_port_node: Node
    target_port_node: Node
    auxiliary_node: Node

    a_penalty: float
    port_penalty: float
    link_penalty: float
    expected_child_blocks: tuple[str, ...] = ()


@dataclass
class QASolution:
    feasible: bool
    energy: float
    weight: float
    sample: dict[str, int]
    selected_actual_edge_ids: list[str]
    selected_variables: list[str]
    message: str = ""


class QUBOSampler(Protocol):
    def sample_qubo(
        self,
        qubo: Mapping[tuple[str, str], float],
        *,
        num_reads: int,
        seed: int | None,
    ) -> tuple[dict[str, int], float]:
        ...


class InfeasibleQUBOProblem(ValueError):
    """構造上、必要な child block を通る feasible edge が存在しない。"""


class ClassicalAnnealingDebugSampler:
    """
    外部QAライブラリが無い環境でのdebug用古典SA。

    これは量子アニーリングではない。
    実機またはOpenJij/D-Waveを使う場合はQUBOSamplerを差し替える。
    """

    def __init__(
        self,
        *,
        sweeps: int = 2000,
        beta_start: float = 0.05,
        beta_end: float = 8.0,
    ) -> None:
        self.sweeps = sweeps
        self.beta_start = beta_start
        self.beta_end = beta_end

    def sample_qubo(
        self,
        qubo: Mapping[tuple[str, str], float],
        *,
        num_reads: int = 100,
        seed: int | None = None,
    ) -> tuple[dict[str, int], float]:
        random = Random(seed)

        variables = sorted({
            variable
            for pair in qubo
            for variable in pair
        })

        def energy(sample: Mapping[str, int]) -> float:
            total = 0.0
            for (first, second), coefficient in qubo.items():
                total += (
                    coefficient
                    * sample[first]
                    * sample[second]
                )
            return total

        best_sample: dict[str, int] = {variable: 0 for variable in variables}
        best_energy = inf

        for _read in range(num_reads):
            sample = {
                variable: random.randint(0, 1)
                for variable in variables
            }
            current_energy = energy(sample)

            for sweep in range(max(1, self.sweeps)):
                progress = sweep / max(1, self.sweeps - 1)
                beta = (
                    self.beta_start
                    + progress * (self.beta_end - self.beta_start)
                )

                variable = random.choice(variables)
                sample[variable] ^= 1
                candidate_energy = energy(sample)
                delta = candidate_energy - current_energy

                if delta <= 0.0 or random.random() < exp(-beta * delta):
                    current_energy = candidate_energy
                else:
                    sample[variable] ^= 1

            if current_energy < best_energy:
                best_energy = current_energy
                best_sample = dict(sample)

        return best_sample, best_energy


# ============================================================
# 一つのmacro edge p_alpha -> q_beta のQUBO
# ============================================================

def _edge_variable(edge_id: str) -> str:
    return f"x::{edge_id}"


def _port_variable(block_id: str, direction: str, port_id: str) -> str:
    return f"{direction}::{block_id}::{port_id}"


def _record_status(record: EdgeRecord) -> str:
    """旧データとの互換性を保ちながら macro edge の解状態を読む。"""
    if record.kind != "macro":
        return "feasible" if record.weight is not None else "unresolved"

    status = str(record.data.get("solve_status", "")).strip().lower()
    if status in {"feasible", "infeasible", "unresolved"}:
        return status
    if record.weight is not None:
        return "feasible"
    return "unresolved"


def _build_internal_local_edges(
    hierarchy: GraphHierarchy,
    block: CompressionBlock,
    macro_edge_id: str,
) -> tuple[
    nx.MultiDiGraph,
    dict[str, LocalEdge],
    dict[str, str],
    Node,
    Node,
    Node,
    tuple[str, ...],
]:
    """
    選択された input/output port に仮想 p,q を作り、
        q -> r -> p
    の附加点 r を加えた QA 用局所有向グラフを構成する。

    Hybrid 版では、下位 macro edge が明示的に infeasible と確定している
    場合、その edge は QUBO 変数から除外する。ただし、その child block
    自体は expected_child_blocks に残し、少なくとも1本 feasible candidate が
    必要であることを build_macro_edge_qubo 側で検査する。
    """

    records = hierarchy.edge_records
    macro_record = records[macro_edge_id]

    if macro_record.input_edge_id is None:
        raise ValueError("macro edgeにinput_edge_idがありません。")
    if macro_record.output_edge_id is None:
        raise ValueError("macro edgeにoutput_edge_idがありません。")

    input_record = records[macro_record.input_edge_id]
    output_record = records[macro_record.output_edge_id]

    source_port = f"__PORT_IN__::{macro_edge_id}"
    target_port = f"__PORT_OUT__::{macro_edge_id}"
    auxiliary = f"__AUX__::{macro_edge_id}"

    graph = nx.MultiDiGraph()
    graph.add_nodes_from(block.region_nodes)
    graph.add_nodes_from([source_port, target_port, auxiliary])

    local_edges: dict[str, LocalEdge] = {}
    actual_to_variable: dict[str, str] = {}
    expected_child_blocks: set[str] = set()

    def add_local_edge(local_edge: LocalEdge) -> None:
        local_edges[local_edge.variable] = local_edge
        graph.add_edge(
            local_edge.u,
            local_edge.v,
            key=local_edge.variable,
            variable=local_edge.variable,
        )
        if local_edge.actual_edge_id is not None:
            actual_to_variable[local_edge.actual_edge_id] = local_edge.variable

    def require_fixed_edge(record: EdgeRecord, label: str) -> None:
        if record.weight is not None:
            return
        if record.kind == "macro" and _record_status(record) == "infeasible":
            raise InfeasibleQUBOProblem(
                f"{label} {record.edge_id} は infeasible です。"
            )
        raise RuntimeError(f"{label} {record.edge_id} のweightが未確定です。")

    # 選択した入口境界辺
    require_fixed_edge(input_record, "入口辺")
    add_local_edge(
        LocalEdge(
            variable=_edge_variable(input_record.edge_id),
            u=source_port,
            v=input_record.v,
            weight=float(input_record.weight),
            actual_edge_id=input_record.edge_id,
            kind=input_record.kind,
            block_id=input_record.block_id,
        )
    )
    if input_record.kind == "macro" and input_record.block_id is not None:
        expected_child_blocks.add(input_record.block_id)

    # region内部の普通辺・下位macro edge
    for edge_id in block.internal_edge_ids:
        record = records[edge_id]

        if record.kind == "macro" and record.block_id is not None:
            expected_child_blocks.add(record.block_id)

        if record.weight is None:
            if record.kind == "macro" and _record_status(record) == "infeasible":
                # infeasible candidate は QUBO から除外する。
                continue
            raise RuntimeError(
                f"内部辺{edge_id}のweightが未確定です。"
                "下層から先に解いてください。"
            )

        add_local_edge(
            LocalEdge(
                variable=_edge_variable(edge_id),
                u=record.u,
                v=record.v,
                weight=float(record.weight),
                actual_edge_id=edge_id,
                kind=record.kind,
                block_id=record.block_id,
            )
        )

    # 選択した出口境界辺
    require_fixed_edge(output_record, "出口辺")
    add_local_edge(
        LocalEdge(
            variable=_edge_variable(output_record.edge_id),
            u=output_record.u,
            v=target_port,
            weight=float(output_record.weight),
            actual_edge_id=output_record.edge_id,
            kind=output_record.kind,
            block_id=output_record.block_id,
        )
    )
    if output_record.kind == "macro" and output_record.block_id is not None:
        expected_child_blocks.add(output_record.block_id)

    # q -> r -> p の附加辺。weightは0。
    add_local_edge(
        LocalEdge(
            variable=f"aux::{macro_edge_id}::q_to_r",
            u=target_port,
            v=auxiliary,
            weight=0.0,
            actual_edge_id=None,
            kind="auxiliary",
            block_id=None,
        )
    )
    add_local_edge(
        LocalEdge(
            variable=f"aux::{macro_edge_id}::r_to_p",
            u=auxiliary,
            v=source_port,
            weight=0.0,
            actual_edge_id=None,
            kind="auxiliary",
            block_id=None,
        )
    )

    return (
        graph,
        local_edges,
        actual_to_variable,
        source_port,
        target_port,
        auxiliary,
        tuple(sorted(expected_child_blocks)),
    )

def build_macro_edge_qubo(
    hierarchy: GraphHierarchy,
    block_id: str,
    macro_edge_id: str,
    *,
    a_penalty: float | None = None,
    port_penalty: float | None = None,
    link_penalty: float | None = None,
) -> LocalQUBOProblem:
    """
    一つの第n層macro edge p_alpha -> q_beta のweightを求めるQUBO。

    含むもの:
      1. q -> auxiliary -> p
      2. A条件: 全頂点の入次数=1, 出次数=1
      3. 式(25)の普通辺weight
      4. 見えている下位blockの
           sum_i p_i = 1
           sum_j q_j = 1
         およびmacro edgeとの一致条件

    subtour除去は今回入れない。
    """

    block = hierarchy.block_map()[block_id]

    if macro_edge_id not in block.macro_edge_ids:
        raise ValueError("macro_edge_idは指定blockに属していません。")

    (
        local_graph,
        local_edges,
        actual_to_variable,
        source_port,
        target_port,
        auxiliary,
        expected_child_blocks,
    ) = _build_internal_local_edges(
        hierarchy,
        block,
        macro_edge_id,
    )

    total_abs_weight = sum(
        abs(edge.weight)
        for edge in local_edges.values()
        if edge.kind != "auxiliary"
    )

    default_penalty = max(10.0, 4.0 * total_abs_weight + 1.0)
    a_penalty = default_penalty if a_penalty is None else a_penalty
    port_penalty = (
        default_penalty if port_penalty is None else port_penalty
    )
    link_penalty = (
        default_penalty if link_penalty is None else link_penalty
    )

    model = QUBOModel()

    # 目的関数: weight最大化なので -weight*x
    for edge in local_edges.values():
        model.add_linear(edge.variable, -edge.weight)

    # A条件: 有向グラフの各頂点で入る辺1本、出る辺1本
    for node in local_graph.nodes:
        incoming_variables = {
            data["variable"]: -1.0
            for _, _, _, data in local_graph.in_edges(
                node, keys=True, data=True
            )
        }
        outgoing_variables = {
            data["variable"]: -1.0
            for _, _, _, data in local_graph.out_edges(
                node, keys=True, data=True
            )
        }

        # (1 - sum incoming)^2
        model.add_square(
            incoming_variables,
            constant=1.0,
            penalty=a_penalty,
        )
        # (1 - sum outgoing)^2
        model.add_square(
            outgoing_variables,
            constant=1.0,
            penalty=a_penalty,
        )

    # 現在の局所グラフに見えている下位blockを集める
    child_edges_by_block: dict[str, list[LocalEdge]] = {}

    for edge in local_edges.values():
        if edge.kind == "macro" and edge.block_id is not None:
            child_edges_by_block.setdefault(
                edge.block_id, []
            ).append(edge)

    records = hierarchy.edge_records

    missing_child_blocks = [
        block_id
        for block_id in expected_child_blocks
        if block_id not in child_edges_by_block
    ]
    if missing_child_blocks:
        raise InfeasibleQUBOProblem(
            "feasible macro edge が1本もない child block: "
            + ", ".join(sorted(missing_child_blocks))
        )

    for child_block_id, child_edges in child_edges_by_block.items():
        # input_edge_idごと、output_edge_idごとにportを分類する
        rows: dict[str, list[LocalEdge]] = {}
        columns: dict[str, list[LocalEdge]] = {}

        for edge in child_edges:
            assert edge.actual_edge_id is not None
            child_macro = records[edge.actual_edge_id]

            if child_macro.input_edge_id is None:
                raise RuntimeError("child macro edgeにinput portがありません。")
            if child_macro.output_edge_id is None:
                raise RuntimeError("child macro edgeにoutput portがありません。")

            rows.setdefault(
                child_macro.input_edge_id, []
            ).append(edge)
            columns.setdefault(
                child_macro.output_edge_id, []
            ).append(edge)

        p_variables: list[str] = []
        q_variables: list[str] = []

        # p_i = sum_j z_ij
        for input_port_id, row_edges in rows.items():
            p_variable = _port_variable(
                child_block_id,
                "p",
                input_port_id,
            )
            p_variables.append(p_variable)

            coefficients = {p_variable: 1.0}
            for edge in row_edges:
                coefficients[edge.variable] = (
                    coefficients.get(edge.variable, 0.0) - 1.0
                )

            model.add_square(
                coefficients,
                constant=0.0,
                penalty=link_penalty,
            )

        # q_j = sum_i z_ij
        for output_port_id, column_edges in columns.items():
            q_variable = _port_variable(
                child_block_id,
                "q",
                output_port_id,
            )
            q_variables.append(q_variable)

            coefficients = {q_variable: 1.0}
            for edge in column_edges:
                coefficients[edge.variable] = (
                    coefficients.get(edge.variable, 0.0) - 1.0
                )

            model.add_square(
                coefficients,
                constant=0.0,
                penalty=link_penalty,
            )

        # sum_i p_i = 1
        model.add_square(
            {variable: -1.0 for variable in p_variables},
            constant=1.0,
            penalty=port_penalty,
        )

        # sum_j q_j = 1
        model.add_square(
            {variable: -1.0 for variable in q_variables},
            constant=1.0,
            penalty=port_penalty,
        )

    return LocalQUBOProblem(
        model=model,
        local_graph=local_graph,
        local_edges=local_edges,
        actual_edge_variable=actual_to_variable,
        source_port_node=source_port,
        target_port_node=target_port,
        auxiliary_node=auxiliary,
        a_penalty=float(a_penalty),
        port_penalty=float(port_penalty),
        link_penalty=float(link_penalty),
        expected_child_blocks=tuple(expected_child_blocks),
    )


def _check_local_solution(
    problem: LocalQUBOProblem,
    sample: Mapping[str, int],
    hierarchy: GraphHierarchy,
) -> tuple[bool, str]:
    # A条件
    for node in problem.local_graph.nodes:
        incoming = sum(
            int(sample.get(data["variable"], 0))
            for _, _, _, data in problem.local_graph.in_edges(
                node, keys=True, data=True
            )
        )
        outgoing = sum(
            int(sample.get(data["variable"], 0))
            for _, _, _, data in problem.local_graph.out_edges(
                node, keys=True, data=True
            )
        )

        if incoming != 1 or outgoing != 1:
            return False, f"A条件違反: node={node}, in={incoming}, out={outgoing}"

    # child blockごとに選択macro edgeが1本であることを確認
    selected_by_block: dict[str, int] = {
        block_id: 0 for block_id in problem.expected_child_blocks
    }

    for edge in problem.local_edges.values():
        if edge.kind != "macro" or edge.block_id is None:
            continue

        selected_by_block.setdefault(edge.block_id, 0)
        selected_by_block[edge.block_id] += int(
            sample.get(edge.variable, 0)
        )

    for block_id, count in selected_by_block.items():
        if count != 1:
            return False, (
                f"下位block制約違反: block={block_id}, "
                f"selected_macro_edges={count}"
            )

    # A条件だけでは複数のdisjoint subtourを許してしまう。
    # Adaptive pruningでは誤ったfeasible判定がそのまま辺削除につながるため、
    # 選択辺が全頂点を含む1個の強連結cycleになっていることを後検査する。
    selected_graph = nx.DiGraph()
    selected_graph.add_nodes_from(problem.local_graph.nodes)
    for edge in problem.local_edges.values():
        if int(sample.get(edge.variable, 0)) == 1:
            selected_graph.add_edge(edge.u, edge.v)

    if selected_graph.number_of_nodes() > 0 and not nx.is_strongly_connected(selected_graph):
        return False, "subtour違反: 選択辺が単一cycleになっていません。"

    return True, ""


def solve_one_macro_edge_qa(
    hierarchy: GraphHierarchy,
    block_id: str,
    macro_edge_id: str,
    *,
    sampler: QUBOSampler | None = None,
    num_reads: int = 200,
    seed: int | None = None,
    a_penalty: float | None = None,
    port_penalty: float | None = None,
    link_penalty: float | None = None,
) -> QASolution:
    """
    一つのmacro edgeのweightをQA/QUBO samplerで求める。

    sampler=Noneの場合はdebug用古典SAを使用する。
    """

    try:
        problem = build_macro_edge_qubo(
            hierarchy,
            block_id,
            macro_edge_id,
            a_penalty=a_penalty,
            port_penalty=port_penalty,
            link_penalty=link_penalty,
        )
    except InfeasibleQUBOProblem as exc:
        return QASolution(
            feasible=False,
            energy=inf,
            weight=-inf,
            sample={},
            selected_actual_edge_ids=[],
            selected_variables=[],
            message=str(exc),
        )

    if sampler is None:
        sampler = ClassicalAnnealingDebugSampler()

    sample, sampled_energy = sampler.sample_qubo(
        problem.model.to_qubo_dict(),
        num_reads=num_reads,
        seed=seed,
    )

    # QUBOのoffsetを戻した完全energy
    total_energy = sampled_energy + problem.model.offset

    feasible, message = _check_local_solution(
        problem,
        sample,
        hierarchy,
    )

    selected_variables = [
        variable
        for variable in problem.model.variables
        if int(sample.get(variable, 0)) == 1
    ]

    selected_actual_edge_ids: list[str] = []
    positive_weight = 0.0

    for edge in problem.local_edges.values():
        if int(sample.get(edge.variable, 0)) != 1:
            continue
        if edge.actual_edge_id is not None:
            selected_actual_edge_ids.append(edge.actual_edge_id)
            positive_weight += edge.weight

    return QASolution(
        feasible=feasible,
        energy=total_energy,
        weight=positive_weight if feasible else -inf,
        sample=dict(sample),
        selected_actual_edge_ids=selected_actual_edge_ids,
        selected_variables=selected_variables,
        message=message,
    )


def solve_all_macro_weights_qa(
    hierarchy: GraphHierarchy,
    *,
    sampler: QUBOSampler | None = None,
    num_reads: int = 200,
    seed: int | None = None,
    match_key: str = "l",
    error_key: str = "d",
    alphabet_size: int = 4,
) -> dict[str, QASolution]:
    """
    圧縮後のhierarchyについて、全macro edgeのweightを
    第1層、第2層、...の順で確定する。

    圧縮そのものは行わない。
    """

    ensure_original_edge_weights(
        hierarchy,
        match_key=match_key,
        error_key=error_key,
        alphabet_size=alphabet_size,
    )

    results: dict[str, QASolution] = {}

    for layer in range(1, hierarchy.max_layer + 1):
        for block in hierarchy.blocks_by_layer.get(layer, []):
            for macro_edge_id in block.macro_edge_ids:
                solution = solve_one_macro_edge_qa(
                    hierarchy,
                    block.block_id,
                    macro_edge_id,
                    sampler=sampler,
                    num_reads=num_reads,
                    seed=seed,
                )
                results[macro_edge_id] = solution

                record = hierarchy.edge_records[macro_edge_id]

                record.data["solver_type"] = "qa"
                if solution.feasible:
                    record.weight = solution.weight
                    record.expansion_edge_ids = list(
                        solution.selected_actual_edge_ids
                    )
                    record.data["solve_status"] = "feasible"
                    record.data["qa_feasible"] = True
                    record.data["qa_energy"] = solution.energy
                else:
                    record.weight = None
                    record.expansion_edge_ids = None
                    record.data["solve_status"] = "infeasible"
                    record.data["qa_feasible"] = False
                    record.data["qa_message"] = solution.message

    return results


def solve_compressed_graph_path_qa(
    hierarchy: GraphHierarchy,
    source: Node,
    target: Node,
    *,
    sampler: QUBOSampler | None = None,
    num_reads: int = 500,
    seed: int | None = None,
) -> QASolution:
    """
    全macro edgeのweight確定後、最終圧縮グラフのsource->targetをQAで解く。

    Hybrid 対応:
      - infeasible と確定した macro edge は QUBO から除外する。
      - ただし visible block ごとに少なくとも1本 feasible macro edge が必要。
      - visible block ごとに選択 macro edge はちょうど1本。

    subtour除去は従来コードと同様、まだ入れていない。
    """

    if source not in hierarchy.current_graph:
        raise ValueError("sourceが最終圧縮グラフにありません。")
    if target not in hierarchy.current_graph:
        raise ValueError("targetが最終圧縮グラフにありません。")

    local_graph = nx.MultiDiGraph()
    local_graph.add_nodes_from(hierarchy.current_graph.nodes)
    local_edges: dict[str, LocalEdge] = {}

    expected_visible_blocks: set[str] = set()
    feasible_count_by_block: dict[str, int] = {}

    for u, v, _key, data in hierarchy.current_graph.edges(
        keys=True, data=True
    ):
        edge_id = data["edge_id"]
        record = hierarchy.edge_records[edge_id]

        if record.kind == "macro" and record.block_id is not None:
            expected_visible_blocks.add(record.block_id)

        if record.weight is None:
            if record.kind == "macro" and _record_status(record) == "infeasible":
                continue
            raise RuntimeError(
                f"最終グラフの辺{edge_id}のweightが未確定です。"
            )

        variable = _edge_variable(edge_id)
        local_edges[variable] = LocalEdge(
            variable=variable,
            u=u,
            v=v,
            weight=float(record.weight),
            actual_edge_id=edge_id,
            kind=record.kind,
            block_id=record.block_id,
        )
        local_graph.add_edge(
            u,
            v,
            key=variable,
            variable=variable,
        )

        if record.kind == "macro" and record.block_id is not None:
            feasible_count_by_block[record.block_id] = (
                feasible_count_by_block.get(record.block_id, 0) + 1
            )

    missing = [
        block_id
        for block_id in expected_visible_blocks
        if feasible_count_by_block.get(block_id, 0) == 0
    ]
    if missing:
        return QASolution(
            feasible=False,
            energy=inf,
            weight=-inf,
            sample={},
            selected_actual_edge_ids=[],
            selected_variables=[],
            message=(
                "最終グラフで feasible macro edge が1本もない block: "
                + ", ".join(sorted(missing))
            ),
        )

    auxiliary = "__FINAL_AUXILIARY__"
    local_graph.add_node(auxiliary)

    aux_q_to_r = LocalEdge(
        variable="aux::final::target_to_r",
        u=target,
        v=auxiliary,
        weight=0.0,
        actual_edge_id=None,
        kind="auxiliary",
        block_id=None,
    )
    aux_r_to_p = LocalEdge(
        variable="aux::final::r_to_source",
        u=auxiliary,
        v=source,
        weight=0.0,
        actual_edge_id=None,
        kind="auxiliary",
        block_id=None,
    )

    for edge in (aux_q_to_r, aux_r_to_p):
        local_edges[edge.variable] = edge
        local_graph.add_edge(
            edge.u,
            edge.v,
            key=edge.variable,
            variable=edge.variable,
        )

    total_abs_weight = sum(
        abs(edge.weight)
        for edge in local_edges.values()
        if edge.kind != "auxiliary"
    )
    penalty = max(10.0, 4.0 * total_abs_weight + 1.0)

    model = QUBOModel()
    for edge in local_edges.values():
        model.add_linear(edge.variable, -edge.weight)

    for node in local_graph.nodes:
        incoming = {
            data["variable"]: -1.0
            for _, _, _, data in local_graph.in_edges(
                node, keys=True, data=True
            )
        }
        outgoing = {
            data["variable"]: -1.0
            for _, _, _, data in local_graph.out_edges(
                node, keys=True, data=True
            )
        }
        model.add_square(incoming, constant=1.0, penalty=penalty)
        model.add_square(outgoing, constant=1.0, penalty=penalty)

    visible_by_block: dict[str, list[LocalEdge]] = {
        block_id: [] for block_id in expected_visible_blocks
    }
    for edge in local_edges.values():
        if edge.kind == "macro" and edge.block_id is not None:
            visible_by_block.setdefault(edge.block_id, []).append(edge)

    records = hierarchy.edge_records
    for block_id in sorted(expected_visible_blocks):
        edges = visible_by_block.get(block_id, [])
        if not edges:
            return QASolution(
                feasible=False,
                energy=inf,
                weight=-inf,
                sample={},
                selected_actual_edge_ids=[],
                selected_variables=[],
                message=f"block {block_id} に feasible macro edge がありません。",
            )

        rows: dict[str, list[LocalEdge]] = {}
        columns: dict[str, list[LocalEdge]] = {}

        for edge in edges:
            assert edge.actual_edge_id is not None
            record = records[edge.actual_edge_id]
            assert record.input_edge_id is not None
            assert record.output_edge_id is not None
            rows.setdefault(record.input_edge_id, []).append(edge)
            columns.setdefault(record.output_edge_id, []).append(edge)

        p_vars: list[str] = []
        q_vars: list[str] = []

        for port, row_edges in rows.items():
            p_var = _port_variable(block_id, "p", port)
            p_vars.append(p_var)
            expression = {p_var: 1.0}
            for edge in row_edges:
                expression[edge.variable] = expression.get(edge.variable, 0.0) - 1.0
            model.add_square(expression, constant=0.0, penalty=penalty)

        for port, column_edges in columns.items():
            q_var = _port_variable(block_id, "q", port)
            q_vars.append(q_var)
            expression = {q_var: 1.0}
            for edge in column_edges:
                expression[edge.variable] = expression.get(edge.variable, 0.0) - 1.0
            model.add_square(expression, constant=0.0, penalty=penalty)

        model.add_square(
            {variable: -1.0 for variable in p_vars},
            constant=1.0,
            penalty=penalty,
        )
        model.add_square(
            {variable: -1.0 for variable in q_vars},
            constant=1.0,
            penalty=penalty,
        )

    if sampler is None:
        sampler = ClassicalAnnealingDebugSampler()

    sample, sampled_energy = sampler.sample_qubo(
        model.to_qubo_dict(),
        num_reads=num_reads,
        seed=seed,
    )

    selected_actual: list[str] = []
    positive_weight = 0.0
    selected_by_block = {block_id: 0 for block_id in expected_visible_blocks}

    for edge in local_edges.values():
        if int(sample.get(edge.variable, 0)) != 1:
            continue
        if edge.actual_edge_id is not None:
            selected_actual.append(edge.actual_edge_id)
            positive_weight += edge.weight
        if edge.kind == "macro" and edge.block_id is not None:
            selected_by_block[edge.block_id] = selected_by_block.get(edge.block_id, 0) + 1

    feasible = True
    message = ""

    for node in local_graph.nodes:
        incoming_count = sum(
            int(sample.get(data["variable"], 0))
            for _, _, _, data in local_graph.in_edges(
                node, keys=True, data=True
            )
        )
        outgoing_count = sum(
            int(sample.get(data["variable"], 0))
            for _, _, _, data in local_graph.out_edges(
                node, keys=True, data=True
            )
        )
        if incoming_count != 1 or outgoing_count != 1:
            feasible = False
            message = (
                f"A条件違反: node={node}, "
                f"in={incoming_count}, out={outgoing_count}"
            )
            break

    if feasible:
        for block_id, count in selected_by_block.items():
            if count != 1:
                feasible = False
                message = (
                    f"下位block制約違反: block={block_id}, "
                    f"selected_macro_edges={count}"
                )
                break

    if feasible:
        selected_graph = nx.DiGraph()
        selected_graph.add_nodes_from(local_graph.nodes)
        for edge in local_edges.values():
            if int(sample.get(edge.variable, 0)) == 1:
                selected_graph.add_edge(edge.u, edge.v)
        if selected_graph.number_of_nodes() > 0 and not nx.is_strongly_connected(selected_graph):
            feasible = False
            message = "subtour違反: 選択辺が単一cycleになっていません。"

    return QASolution(
        feasible=feasible,
        energy=sampled_energy + model.offset,
        weight=positive_weight if feasible else -inf,
        sample=dict(sample),
        selected_actual_edge_ids=selected_actual,
        selected_variables=[
            variable
            for variable in model.variables
            if int(sample.get(variable, 0)) == 1
        ],
        message=message,
    )

def expand_selected_edges(
    hierarchy: GraphHierarchy,
    selected_edge_ids: Sequence[str],
) -> list[str]:
    """
    選択されたmacro edgeを再帰的に元のlayer-0辺へ展開する。
    """

    expanded: list[str] = []

    def expand_one(edge_id: str, stack: set[str]) -> None:
        if edge_id in stack:
            raise RuntimeError(f"復元cycleを検出しました: {edge_id}")

        record = hierarchy.edge_records[edge_id]

        if record.kind == "original":
            expanded.append(edge_id)
            return

        if record.expansion_edge_ids is None:
            raise RuntimeError(
                f"macro edge {edge_id}の内部解が未確定です。"
            )

        stack.add(edge_id)
        for child_edge_id in record.expansion_edge_ids:
            expand_one(child_edge_id, stack)
        stack.remove(edge_id)

    for edge_id in selected_edge_ids:
        expand_one(edge_id, set())

    return expanded


def order_selected_edge_ids_as_path(
    hierarchy: GraphHierarchy,
    selected_edge_ids: Sequence[str],
    *,
    source: Node,
    target: Node,
) -> list[str]:
    """Order a selected edge set into one directed source->target path."""

    remaining = set(selected_edge_ids)
    outgoing: dict[Node, list[str]] = {}
    for edge_id in selected_edge_ids:
        record = hierarchy.edge_records[edge_id]
        outgoing.setdefault(record.u, []).append(edge_id)

    ordered: list[str] = []
    current = source

    while current != target:
        candidates = [
            edge_id for edge_id in outgoing.get(current, [])
            if edge_id in remaining
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "selected edges do not define a unique directed path: "
                f"node={current}, outgoing_candidates={candidates}"
            )
        edge_id = candidates[0]
        remaining.remove(edge_id)
        ordered.append(edge_id)
        current = hierarchy.edge_records[edge_id].v

        if len(ordered) > len(selected_edge_ids):
            raise RuntimeError("cycle detected while ordering selected edges")

    if remaining:
        raise RuntimeError(
            "selected edge set contains edges outside the source->target path: "
            + ", ".join(sorted(remaining))
        )

    return ordered


def expand_selected_edges_ordered(
    hierarchy: GraphHierarchy,
    selected_edge_ids: Sequence[str],
    *,
    source: Node,
    target: Node,
) -> list[str]:
    """
    Recursively expand a solved top path while preserving traversal order.

    Each macro edge is expanded from record.u to record.v.  This is stronger
    than expand_selected_edges(), whose historical behavior preserved whatever
    list order the solver happened to return.
    """

    def expand_one(edge_id: str, stack: set[str]) -> list[str]:
        if edge_id in stack:
            raise RuntimeError(f"macro expansion cycle detected: {edge_id}")

        record = hierarchy.edge_records[edge_id]
        if record.kind == "original":
            return [edge_id]
        if record.expansion_edge_ids is None:
            raise RuntimeError(
                f"macro edge {edge_id} has no solved expansion"
            )

        children = order_selected_edge_ids_as_path(
            hierarchy,
            record.expansion_edge_ids,
            source=record.u,
            target=record.v,
        )
        stack.add(edge_id)
        result: list[str] = []
        for child in children:
            result.extend(expand_one(child, stack))
        stack.remove(edge_id)
        return result

    top_order = order_selected_edge_ids_as_path(
        hierarchy, selected_edge_ids, source=source, target=target
    )
    expanded: list[str] = []
    for edge_id in top_order:
        expanded.extend(expand_one(edge_id, set()))
    return expanded


def _unitig_chain_edge_ids(
    preprocess: SafePreprocessResult,
    unitig_id: Node,
    start_member: Node,
    end_member: Node,
) -> tuple[list[str], list[Node]]:
    """Return raw internal unitig edges/nodes from start_member to end_member."""

    members = list(preprocess.unitigs.get(unitig_id, ()))
    if not members:
        raise RuntimeError(f"unknown unitig: {unitig_id}")
    try:
        i = members.index(start_member)
        j = members.index(end_member)
    except ValueError as exc:
        raise RuntimeError(
            f"unitig endpoint is not a member: {unitig_id}, "
            f"{start_member}->{end_member}"
        ) from exc
    if i > j:
        raise RuntimeError(
            f"unitig traversal reverses forced direction: {start_member}->{end_member}"
        )
    if i == j:
        return [], [start_member]

    node_data = preprocess.graph.nodes[unitig_id]
    snapshots = list(node_data.get("unitig_internal_edges", []))
    by_pair: dict[tuple[Node, Node], list[dict[str, Any]]] = {}
    for snap in snapshots:
        by_pair.setdefault((snap["u"], snap["v"]), []).append(snap)

    edge_ids: list[str] = []
    nodes = [start_member]
    for left, right in zip(members[i:j], members[i + 1:j + 1]):
        candidates = by_pair.get((left, right), [])
        if len(candidates) != 1:
            raise RuntimeError(
                f"cannot uniquely restore unitig edge {left}->{right} in {unitig_id}"
            )
        snap = candidates[0]
        data = snap.get("data", {})
        edge_id = str(data.get("edge_id", snap.get("key")))
        edge_ids.append(edge_id)
        nodes.append(right)

    return edge_ids, nodes


def expand_preprocessed_path_to_raw(
    preprocess: SafePreprocessResult,
    hierarchy: GraphHierarchy,
    preprocessed_edge_ids: Sequence[str],
    *,
    raw_source: Node | None = None,
    raw_target: Node | None = None,
) -> tuple[list[str], list[Node]]:
    """
    Restore safe unitig contractions after Plan-B macro expansion.

    Returns (raw_edge_ids, raw_node_path).  Removed full-contained reads and
    rejected edges are intentionally not restored because they were declared
    unusable during conservative preprocessing.
    """

    if not preprocessed_edge_ids:
        if raw_source is not None and raw_target is not None:
            if raw_source == raw_target:
                return [], [raw_source]
            unitig_a = preprocess.node_to_unitig.get(raw_source)
            unitig_b = preprocess.node_to_unitig.get(raw_target)
            if unitig_a is not None and unitig_a == unitig_b:
                return _unitig_chain_edge_ids(
                    preprocess, unitig_a, raw_source, raw_target
                )
        return [], []

    def raw_endpoints(edge_id: str) -> tuple[Node, Node, str]:
        record = hierarchy.edge_records[edge_id]
        data = record.data
        u = data.get("pre_unitig_u", record.u)
        v = data.get("pre_unitig_v", record.v)
        raw_edge_id = str(data.get("edge_id", record.edge_id))
        return u, v, raw_edge_id

    first_u, _first_v, _ = raw_endpoints(preprocessed_edge_ids[0])
    current = first_u if raw_source is None else raw_source
    raw_nodes: list[Node] = [current]
    raw_edges: list[str] = []

    def bridge_inside_unitig(destination: Node) -> None:
        nonlocal current
        if current == destination:
            return
        unitig_a = preprocess.node_to_unitig.get(current)
        unitig_b = preprocess.node_to_unitig.get(destination)
        if unitig_a is None or unitig_a != unitig_b:
            raise RuntimeError(
                "raw path has a gap that is not explained by one safe unitig: "
                f"{current}->{destination}"
            )
        edge_ids, nodes = _unitig_chain_edge_ids(
            preprocess, unitig_a, current, destination
        )
        raw_edges.extend(edge_ids)
        raw_nodes.extend(nodes[1:])
        current = destination

    for edge_id in preprocessed_edge_ids:
        u, v, raw_edge_id = raw_endpoints(edge_id)
        bridge_inside_unitig(u)
        raw_edges.append(raw_edge_id)
        raw_nodes.append(v)
        current = v

    if raw_target is not None:
        bridge_inside_unitig(raw_target)

    return raw_edges, raw_nodes

# ============================================================================
# OLC_HYBRID
# ============================================================================

from dataclasses import dataclass, field
from math import inf
from typing import Hashable

Node = Hashable


@dataclass(frozen=True)
class RoutingDecision:
    solver_type: str              # "classical" or "qa"
    certificate_is_dag: bool
    complete_macro_groups: bool
    expected_child_block_count: int
    missing_feasible_child_blocks: tuple[str, ...]
    reason: str


@dataclass
class HybridSolution:
    feasible: bool
    solver_type: str
    weight: float
    selected_actual_edge_ids: list[str]
    message: str = ""
    certificate_is_dag: bool | None = None
    complete_macro_groups: bool | None = None
    energy: float | None = None
    qa_sample: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class HybridBatchResult:
    solutions: dict[str, HybridSolution]
    classical_count: int
    qa_count: int
    feasible_count: int
    infeasible_count: int


def _decision(problem: LocalPathProblem) -> RoutingDecision:
    is_dag = nx.is_directed_acyclic_graph(problem.solve_graph)
    complete = problem.complete_macro_groups

    if is_dag:
        reason = (
            "feasible solve graph が DAG なので exact classical DAG solver を使用する。"
            " macro group が不完全でも、必要な block だけを bitmask で追跡して"
            " one-use 条件を厳密に保つ。"
        )
        solver = "classical"
    else:
        reason = "feasible solve graph が non-DAG なので QUBO/QA に送る。"
        solver = "qa"

    return RoutingDecision(
        solver_type=solver,
        certificate_is_dag=is_dag,
        complete_macro_groups=complete,
        expected_child_block_count=len(problem.expected_child_blocks),
        missing_feasible_child_blocks=tuple(
            sorted(problem.missing_feasible_child_blocks)
        ),
        reason=reason,
    )

def classify_one_macro_edge(
    hierarchy: GraphHierarchy,
    block_id: str,
    macro_edge_id: str,
) -> RoutingDecision:
    """一つの macro edge を classical / QA のどちらへ送るか判定する。"""

    problem = build_macro_edge_path_problem(
        hierarchy,
        block_id,
        macro_edge_id,
    )
    return _decision(problem)


def classify_top_graph(
    hierarchy: GraphHierarchy,
    source: Node,
    target: Node,
) -> RoutingDecision:
    problem = build_top_path_problem(hierarchy, source, target)
    return _decision(problem)


def _from_classical(
    solution: ClassicalPathSolution,
    decision: RoutingDecision,
) -> HybridSolution:
    return HybridSolution(
        feasible=solution.feasible,
        solver_type="classical",
        weight=solution.weight,
        selected_actual_edge_ids=list(solution.selected_actual_edge_ids),
        message=solution.message,
        certificate_is_dag=decision.certificate_is_dag,
        complete_macro_groups=decision.complete_macro_groups,
        energy=None,
        qa_sample={},
        metadata={
            "visited_node_count": solution.visited_node_count,
            "required_node_count": solution.required_node_count,
            "used_child_blocks": solution.used_child_blocks,
            "routing_reason": decision.reason,
        },
    )


def _from_qa(
    solution: QASolution,
    decision: RoutingDecision,
) -> HybridSolution:
    return HybridSolution(
        feasible=solution.feasible,
        solver_type="qa",
        weight=solution.weight,
        selected_actual_edge_ids=list(solution.selected_actual_edge_ids),
        message=solution.message,
        certificate_is_dag=decision.certificate_is_dag,
        complete_macro_groups=decision.complete_macro_groups,
        energy=solution.energy,
        qa_sample=dict(solution.sample),
        metadata={
            "routing_reason": decision.reason,
            "selected_variables": list(solution.selected_variables),
        },
    )


def solve_one_macro_edge_hybrid(
    hierarchy: GraphHierarchy,
    block_id: str,
    macro_edge_id: str,
    *,
    sampler: QUBOSampler | None = None,
    num_reads: int = 200,
    seed: int | None = None,
    a_penalty: float | None = None,
    port_penalty: float | None = None,
    link_penalty: float | None = None,
) -> HybridSolution:
    """
    固定された macro edge を自動振り分けして解く。

    1. q->r->p を加える前の certificate graph を作る。
    2. DAG + complete macro groups なら classical exact DP。
    3. それ以外は既存 QUBO/QA。
    """

    problem = build_macro_edge_path_problem(
        hierarchy,
        block_id,
        macro_edge_id,
    )
    decision = _decision(problem)

    if decision.solver_type == "classical":
        classical = solve_dag_path_problem(problem)
        return _from_classical(classical, decision)

    qa = solve_one_macro_edge_qa(
        hierarchy,
        block_id,
        macro_edge_id,
        sampler=sampler,
        num_reads=num_reads,
        seed=seed,
        a_penalty=a_penalty,
        port_penalty=port_penalty,
        link_penalty=link_penalty,
    )
    return _from_qa(qa, decision)


def _write_solution_to_macro_record(
    hierarchy: GraphHierarchy,
    macro_edge_id: str,
    solution: HybridSolution,
) -> None:
    record = hierarchy.edge_records[macro_edge_id]
    record.data["solver_type"] = solution.solver_type
    record.data["certificate_is_dag"] = solution.certificate_is_dag
    record.data["complete_macro_groups"] = solution.complete_macro_groups
    record.data["hybrid_message"] = solution.message

    if solution.feasible:
        record.weight = float(solution.weight)
        record.expansion_edge_ids = list(solution.selected_actual_edge_ids)
        record.data["solve_status"] = "feasible"
    else:
        record.weight = None
        record.expansion_edge_ids = None
        record.data["solve_status"] = "infeasible"

    if solution.energy is not None:
        record.data["qa_energy"] = float(solution.energy)


def solve_all_macro_weights_hybrid(
    hierarchy: GraphHierarchy,
    *,
    sampler: QUBOSampler | None = None,
    num_reads: int = 200,
    seed: int | None = None,
    match_key: str = "l",
    error_key: str = "d",
    alphabet_size: int = 4,
) -> HybridBatchResult:
    """
    第1層から上へ、全 macro edge weight を hybrid に確定する。

    DAG candidate は classical、non-DAG candidate は QUBO/QA。
    どちらでも上位層に渡す形式は同じ:
        weight + expansion_edge_ids
    """

    ensure_original_edge_weights(
        hierarchy,
        match_key=match_key,
        error_key=error_key,
        alphabet_size=alphabet_size,
    )

    # original edge は常に解決済みとして明示する。
    for record in hierarchy.edge_records.values():
        if record.kind == "original" and record.weight is not None:
            record.data.setdefault("solve_status", "feasible")
            record.data.setdefault("solver_type", "input")

    solutions: dict[str, HybridSolution] = {}
    classical_count = 0
    qa_count = 0
    feasible_count = 0
    infeasible_count = 0

    for layer in range(1, hierarchy.max_layer + 1):
        for block in hierarchy.blocks_by_layer.get(layer, []):
            for macro_edge_id in block.macro_edge_ids:
                solution = solve_one_macro_edge_hybrid(
                    hierarchy,
                    block.block_id,
                    macro_edge_id,
                    sampler=sampler,
                    num_reads=num_reads,
                    seed=seed,
                )
                solutions[macro_edge_id] = solution
                _write_solution_to_macro_record(
                    hierarchy,
                    macro_edge_id,
                    solution,
                )

                if solution.solver_type == "classical":
                    classical_count += 1
                else:
                    qa_count += 1

                if solution.feasible:
                    feasible_count += 1
                else:
                    infeasible_count += 1

    return HybridBatchResult(
        solutions=solutions,
        classical_count=classical_count,
        qa_count=qa_count,
        feasible_count=feasible_count,
        infeasible_count=infeasible_count,
    )


def solve_compressed_graph_path_hybrid(
    hierarchy: GraphHierarchy,
    source: Node,
    target: Node,
    *,
    sampler: QUBOSampler | None = None,
    num_reads: int = 500,
    seed: int | None = None,
) -> HybridSolution:
    """最終圧縮グラフも同じルールで classical / QA を自動選択する。"""

    problem = build_top_path_problem(hierarchy, source, target)
    decision = _decision(problem)

    if decision.solver_type == "classical":
        classical = solve_dag_path_problem(problem)
        return _from_classical(classical, decision)

    qa = solve_compressed_graph_path_qa(
        hierarchy,
        source,
        target,
        sampler=sampler,
        num_reads=num_reads,
        seed=seed,
    )
    return _from_qa(qa, decision)



# ============================================================================
# RECURSIVE PLAN-B  f(B)
# ============================================================================

@dataclass
class RecursivePlanBConfig:
    """
    Configuration for the recursive Plan-B rule

        f(B) = classical(B)                         if B is DAG
             = QA(B)                                if B is non-DAG and QA-sized
             = f(child blocks inside B), then f(B) if B is non-DAG and too large

    A candidate boundary is preferably cycle-closed inside its parent scope:
    after replacing the candidate by p_i -> q_j macro edges, there should be
    no q_j ~> p_i return path in the part of the parent scope left outside the
    candidate.  This is the code version of avoiding the undesirable
    "exit reconnects to entry" cut.
    """

    # Main QA size limit.  None -> EvaluationConfig.preferred_local_qubo_variables
    qa_variable_limit: int | None = None
    qa_coupling_limit: int | None = None
    qa_density_limit: float | None = None
    qa_average_degree_limit: float | None = None

    # Candidate boundary / search settings.
    min_input_ports: int = 1
    min_output_ports: int = 1
    min_region_nodes: int = 2
    max_region_nodes: int = 20
    beam_width: int = 24
    max_candidates: int = 256

    # Recursive safety / termination.
    max_recursion_depth: int = 32
    max_total_blocks: int = 1000

    # Prefer a boundary with no exit->...->entry path inside the parent scope.
    require_cycle_closed_regions: bool = True
    auto_expand_cycle_closure: bool = True
    max_cycle_closure_rounds: int = 32


@dataclass
class RecursiveBlockResult:
    """Result of one completed f(B) call."""

    success: bool
    block_id: str | None
    layer: int | None
    branch: str                       # "classical", "qa", "recurse_failed"
    recursion_depth: int
    feasible_macro_count: int
    infeasible_macro_count: int
    message: str = ""


@dataclass
class RecursivePlanBResult:
    hierarchy: GraphHierarchy
    macro_batch: HybridBatchResult
    top_solution: HybridSolution | None
    stop_reason: str
    recursive_calls: int
    max_recursion_depth_reached: int
    total_blocks: int
    classical_blocks: int
    qa_blocks: int
    global_infeasible: bool = False


@dataclass
class _RecursiveRuntime:
    next_layer: int = 1
    recursive_calls: int = 0
    max_depth: int = 0
    total_blocks: int = 0
    classical_blocks: int = 0
    qa_blocks: int = 0
    solutions: dict[str, HybridSolution] = field(default_factory=dict)
    classical_macro_count: int = 0
    qa_macro_count: int = 0
    feasible_macro_count: int = 0
    infeasible_macro_count: int = 0
    global_infeasible: bool = False


def _recursive_qa_variable_limit(
    evaluation_config: EvaluationConfig,
    recursive_config: RecursivePlanBConfig,
) -> int:
    if recursive_config.qa_variable_limit is not None:
        return recursive_config.qa_variable_limit
    return evaluation_config.preferred_local_qubo_variables


def _region_fits_recursive_qa(
    evaluation: RegionEvaluation,
    *,
    evaluation_config: EvaluationConfig,
    recursive_config: RecursivePlanBConfig,
) -> bool:
    """True iff a non-DAG local region is small enough for direct QA."""

    if evaluation.is_dag:
        return False
    if evaluation.local_qubo_max_variables is None:
        return False

    if evaluation.local_qubo_max_variables > _recursive_qa_variable_limit(
        evaluation_config, recursive_config
    ):
        return False

    if (
        recursive_config.qa_coupling_limit is not None
        and evaluation.local_qubo_max_couplings is not None
        and evaluation.local_qubo_max_couplings
        > recursive_config.qa_coupling_limit
    ):
        return False

    density_limit = recursive_config.qa_density_limit
    if density_limit is None:
        density_limit = evaluation_config.preferred_local_qubo_density
    if (
        density_limit is not None
        and evaluation.local_qubo_max_density is not None
        and evaluation.local_qubo_max_density > density_limit
    ):
        return False

    degree_limit = recursive_config.qa_average_degree_limit
    if degree_limit is None:
        degree_limit = evaluation_config.preferred_local_qubo_avg_degree
    if (
        degree_limit is not None
        and evaluation.local_qubo_max_average_degree is not None
        and evaluation.local_qubo_max_average_degree > degree_limit
    ):
        return False

    return True


def _top_fits_recursive_qa(
    top: TopProblemEstimate,
    *,
    evaluation_config: EvaluationConfig,
    recursive_config: RecursivePlanBConfig,
) -> bool:
    """QA-size decision for the final visible top graph."""

    if top.estimated_qubo_variables > _recursive_qa_variable_limit(
        evaluation_config, recursive_config
    ):
        return False

    coupling_limit = recursive_config.qa_coupling_limit
    if coupling_limit is None:
        coupling_limit = evaluation_config.target_top_qubo_couplings
    if (
        coupling_limit is not None
        and top.estimated_qubo_couplings > coupling_limit
    ):
        return False

    density_limit = recursive_config.qa_density_limit
    if density_limit is None:
        density_limit = evaluation_config.target_top_qubo_density
    if (
        density_limit is not None
        and top.estimated_qubo_density > density_limit
    ):
        return False

    degree_limit = recursive_config.qa_average_degree_limit
    if (
        degree_limit is not None
        and top.estimated_qubo_average_degree > degree_limit
    ):
        return False

    return True


def _region_boundary_is_internal_to_scope(
    hierarchy: GraphHierarchy,
    region_nodes: set[Node],
    scope_nodes: set[Node],
) -> bool:
    """
    Nested Plan-B child regions must not cut through the parent boundary.

    Every outside endpoint of a child boundary edge must still be inside the
    current parent scope.  Therefore child compression changes only the inside
    representation of the parent and leaves the parent's external ports fixed.
    """

    _internal, incoming, outgoing = boundary_edge_ids(
        hierarchy.current_graph, region_nodes
    )
    records = hierarchy.edge_records

    for edge_id in incoming:
        if records[edge_id].u not in scope_nodes:
            return False
    for edge_id in outgoing:
        if records[edge_id].v not in scope_nodes:
            return False
    return True


def _prepare_recursive_candidate_in_scope(
    hierarchy: GraphHierarchy,
    region_nodes: set[Node],
    scope_nodes: set[Node],
    *,
    max_input_ports: int,
    max_output_ports: int,
    recursive_config: RecursivePlanBConfig,
    protected_nodes: set[Node],
) -> set[Node] | None:
    """Normalize and validate a child candidate relative to its parent scope."""

    if not region_nodes or region_nodes == scope_nodes:
        return None
    if region_nodes & protected_nodes:
        return None
    if not region_nodes < scope_nodes:
        return None

    scope_graph = hierarchy.current_graph.subgraph(scope_nodes).copy()
    region = set(region_nodes)

    if recursive_config.require_cycle_closed_regions:
        if recursive_config.auto_expand_cycle_closure:
            closure = cycle_close_region(
                scope_graph,
                region,
                max_region_nodes=recursive_config.max_region_nodes,
                max_rounds=recursive_config.max_cycle_closure_rounds,
            )
            if not closure.is_cycle_closed or closure.exceeded_node_limit:
                return None
            region = set(closure.region_nodes)
        elif not is_cycle_closed_region(scope_graph, region):
            return None

    if not region or region == scope_nodes or not region < scope_nodes:
        return None
    if region & protected_nodes:
        return None

    if not _region_boundary_is_internal_to_scope(
        hierarchy, region, scope_nodes
    ):
        return None

    if not _validate_region_for_compression(
        hierarchy,
        region,
        max_input_ports=max_input_ports,
        max_output_ports=max_output_ports,
        min_region_nodes=recursive_config.min_region_nodes,
        # For a nested child, closure was checked in the parent scope above.
        require_cycle_closed=False,
    ):
        return None

    _internal, incoming, outgoing = boundary_edge_ids(
        hierarchy.current_graph, region
    )
    if len(incoming) < recursive_config.min_input_ports:
        return None
    if len(outgoing) < recursive_config.min_output_ports:
        return None

    return region


def _find_recursive_child_region(
    hierarchy: GraphHierarchy,
    scope_nodes: set[Node],
    *,
    max_input_ports: int,
    max_output_ports: int,
    evaluation_config: EvaluationConfig,
    priority_model: PriorityModel,
    recursive_config: RecursivePlanBConfig,
    protected_nodes: set[Node],
) -> RegionEvaluation | None:
    """
    Find the next multi-entry/multi-exit subgraph inside a too-large parent B.

    Search happens on the induced parent scope so the recursive call really is
    f(B), not another unrelated global compression.
    """

    current_scope = set(scope_nodes) & set(hierarchy.current_graph.nodes)
    if len(current_scope) < recursive_config.min_region_nodes + 1:
        return None

    scope_graph = hierarchy.current_graph.subgraph(current_scope).copy()

    raw = find_bounded_boundary_regions(
        scope_graph,
        hierarchy.edge_records,
        max_input_ports=max_input_ports,
        max_output_ports=max_output_ports,
        min_region_nodes=recursive_config.min_region_nodes,
        max_region_nodes=min(
            recursive_config.max_region_nodes,
            max(recursive_config.min_region_nodes, len(current_scope) - 1),
        ),
        beam_width=recursive_config.beam_width,
        max_candidates=recursive_config.max_candidates,
        require_cycle_closed=recursive_config.require_cycle_closed_regions,
        auto_expand_cycle_closure=recursive_config.auto_expand_cycle_closure,
        max_cycle_closure_rounds=recursive_config.max_cycle_closure_rounds,
    )

    prepared: list[set[Node]] = []
    seen: set[frozenset[Node]] = set()

    for candidate in raw:
        region = _prepare_recursive_candidate_in_scope(
            hierarchy,
            set(candidate),
            current_scope,
            max_input_ports=max_input_ports,
            max_output_ports=max_output_ports,
            recursive_config=recursive_config,
            protected_nodes=protected_nodes,
        )
        if region is None:
            continue
        frozen = frozenset(region)
        if frozen in seen:
            continue
        seen.add(frozen)
        prepared.append(region)

    if not prepared:
        return None

    evaluations = evaluate_candidates(
        hierarchy,
        prepared,
        config=evaluation_config,
        priority_model=priority_model,
    )
    evaluations = [ev for ev in evaluations if ev.eligible]
    if not evaluations:
        return None

    # DefaultPriorityModel puts DAG regions first, then QA-sized non-DAG,
    # then oversized non-DAG.  That exactly matches the recursive Plan-B
    # preference: cheap classical simplification first when available.
    return evaluations[0]


def _solve_new_recursive_block(
    hierarchy: GraphHierarchy,
    block: CompressionBlock,
    *,
    expected_branch: str,
    recursion_depth: int,
    runtime: _RecursiveRuntime,
    sampler: QUBOSampler | None,
    num_reads: int,
    seed: int | None,
) -> tuple[int, int]:
    """Solve all p_i->q_j macro candidates of one newly compressed block now."""

    feasible = 0
    infeasible = 0
    actual_solver_types: list[str] = []

    for macro_edge_id in block.macro_edge_ids:
        solution = solve_one_macro_edge_hybrid(
            hierarchy,
            block.block_id,
            macro_edge_id,
            sampler=sampler,
            num_reads=num_reads,
            seed=seed,
        )
        _write_solution_to_macro_record(
            hierarchy, macro_edge_id, solution
        )
        runtime.solutions[macro_edge_id] = solution
        actual_solver_types.append(solution.solver_type)

        record = hierarchy.edge_records[macro_edge_id]
        record.data["recursive_depth"] = recursion_depth
        record.data["recursive_expected_branch"] = expected_branch

        if solution.solver_type == "classical":
            runtime.classical_macro_count += 1
        else:
            runtime.qa_macro_count += 1

        if solution.feasible:
            feasible += 1
            runtime.feasible_macro_count += 1
        else:
            infeasible += 1
            runtime.infeasible_macro_count += 1

    pruned = prune_infeasible_macro_edges(
        hierarchy, block_id=block.block_id
    )

    hierarchy.compression_log.append({
        "action": "recursive_block_solved",
        "block_id": block.block_id,
        "layer": block.layer,
        "recursive_depth": recursion_depth,
        "expected_branch": expected_branch,
        "actual_solver_types": actual_solver_types,
        "macro_total": len(block.macro_edge_ids),
        "macro_feasible": feasible,
        "macro_infeasible": infeasible,
        "pruned_macro_edge_ids": list(pruned),
    })

    return feasible, infeasible


def planb_f(
    hierarchy: GraphHierarchy,
    region_nodes: set[Node],
    *,
    parent_scope_nodes: set[Node] | None,
    recursion_depth: int,
    max_input_ports: int,
    max_output_ports: int,
    evaluation_config: EvaluationConfig,
    priority_model: PriorityModel,
    recursive_config: RecursivePlanBConfig,
    runtime: _RecursiveRuntime,
    sampler: QUBOSampler | None,
    num_reads: int,
    seed: int | None,
    protected_nodes: set[Node],
) -> RecursiveBlockResult:
    """
    Recursive Plan-B function f(B).

    For the current representation of B:
        1) if DAG -> solve B classically and compress it;
        2) else if direct QA fits -> solve B by QA and compress it;
        3) else find a child C inside B, call f(C), replace C by solved
           macro edges, and then re-evaluate B.

    Children are compressed before parents, so the ordinary `layer` index is
    assigned in post-order: inner packages receive lower layer numbers and an
    outer package receives a higher layer number.
    """

    runtime.recursive_calls += 1
    runtime.max_depth = max(runtime.max_depth, recursion_depth)

    if recursion_depth > recursive_config.max_recursion_depth:
        return RecursiveBlockResult(
            success=False,
            block_id=None,
            layer=None,
            branch="recurse_failed",
            recursion_depth=recursion_depth,
            feasible_macro_count=0,
            infeasible_macro_count=0,
            message="max_recursion_depth exceeded",
        )

    scope = set(region_nodes) & set(hierarchy.current_graph.nodes)
    if not scope:
        return RecursiveBlockResult(
            False, None, None, "recurse_failed", recursion_depth, 0, 0,
            "region disappeared before it could be solved",
        )

    while True:
        scope &= set(hierarchy.current_graph.nodes)

        if not scope:
            return RecursiveBlockResult(
                False, None, None, "recurse_failed", recursion_depth, 0, 0,
                "region became empty during recursive reduction",
            )

        # The parent boundary must remain fixed while children are processed.
        if parent_scope_nodes is not None and not _region_boundary_is_internal_to_scope(
            hierarchy, scope, parent_scope_nodes
        ):
            return RecursiveBlockResult(
                False, None, None, "recurse_failed", recursion_depth, 0, 0,
                "recursive scope cut through its parent boundary",
            )

        try:
            evaluation = evaluate_region_candidate(
                hierarchy,
                scope,
                config=evaluation_config,
                priority_model=priority_model,
            )
        except (ValueError, nx.NetworkXError) as exc:
            return RecursiveBlockResult(
                False, None, None, "recurse_failed", recursion_depth, 0, 0,
                f"cannot evaluate recursive region: {exc}",
            )

        if evaluation.input_count < recursive_config.min_input_ports:
            return RecursiveBlockResult(
                False, None, None, "recurse_failed", recursion_depth, 0, 0,
                "region has too few input ports",
            )
        if evaluation.output_count < recursive_config.min_output_ports:
            return RecursiveBlockResult(
                False, None, None, "recurse_failed", recursion_depth, 0, 0,
                "region has too few output ports",
            )

        if evaluation.is_dag:
            branch = "classical"
        elif _region_fits_recursive_qa(
            evaluation,
            evaluation_config=evaluation_config,
            recursive_config=recursive_config,
        ):
            branch = "qa"
        else:
            branch = "recurse"

        hierarchy.compression_log.append({
            "action": "recursive_decision",
            "recursive_depth": recursion_depth,
            "region_nodes": [str(x) for x in sorted(scope, key=str)],
            "branch": branch,
            "evaluation": evaluation.to_dict(),
        })

        if branch == "recurse":
            if runtime.total_blocks >= recursive_config.max_total_blocks:
                return RecursiveBlockResult(
                    False, None, None, "recurse_failed", recursion_depth, 0, 0,
                    "max_total_blocks reached before parent became solvable",
                )

            child = _find_recursive_child_region(
                hierarchy,
                scope,
                max_input_ports=max_input_ports,
                max_output_ports=max_output_ports,
                evaluation_config=evaluation_config,
                priority_model=priority_model,
                recursive_config=recursive_config,
                protected_nodes=protected_nodes,
            )
            if child is None:
                return RecursiveBlockResult(
                    False, None, None, "recurse_failed", recursion_depth, 0, 0,
                    "non-DAG region is too large for QA and no valid child block was found",
                )

            child_result = planb_f(
                hierarchy,
                set(child.region_nodes),
                parent_scope_nodes=set(scope),
                recursion_depth=recursion_depth + 1,
                max_input_ports=max_input_ports,
                max_output_ports=max_output_ports,
                evaluation_config=evaluation_config,
                priority_model=priority_model,
                recursive_config=recursive_config,
                runtime=runtime,
                sampler=sampler,
                num_reads=num_reads,
                seed=seed,
                protected_nodes=protected_nodes,
            )
            if not child_result.success:
                return child_result

            if child_result.feasible_macro_count == 0:
                runtime.global_infeasible = True
                return RecursiveBlockResult(
                    False, child_result.block_id, child_result.layer,
                    "recurse_failed", recursion_depth,
                    0, child_result.infeasible_macro_count,
                    "child block has no feasible macro transition",
                )

            # Child nodes were removed and replaced by solved macro edges whose
            # endpoints stay in the parent scope.  Re-evaluate the parent.
            scope &= set(hierarchy.current_graph.nodes)
            continue

        # ------------------------------------------------------------
        # B is now directly solvable: classical if DAG, QA otherwise.
        # ------------------------------------------------------------
        if runtime.total_blocks >= recursive_config.max_total_blocks:
            return RecursiveBlockResult(
                False, None, None, "recurse_failed", recursion_depth, 0, 0,
                "max_total_blocks reached",
            )

        if not _validate_region_for_compression(
            hierarchy,
            scope,
            max_input_ports=max_input_ports,
            max_output_ports=max_output_ports,
            min_region_nodes=1,
            # Closure was guaranteed relative to the parent scope when this
            # region was selected.  During recursive processing the boundary
            # ports do not move.
            require_cycle_closed=False,
        ):
            return RecursiveBlockResult(
                False, None, None, "recurse_failed", recursion_depth, 0, 0,
                "directly solvable region is no longer structurally compressible",
            )

        layer = runtime.next_layer
        runtime.next_layer += 1
        runtime.total_blocks += 1

        # Preserve recursive diagnostics in the evaluation snapshot.
        snapshot_evaluation = copy.deepcopy(evaluation)
        block = compress_one_region(
            hierarchy,
            scope,
            layer=layer,
            max_input_ports=max_input_ports,
            max_output_ports=max_output_ports,
            evaluation=snapshot_evaluation,
            require_cycle_closed=False,
        )
        block.evaluation_snapshot["recursive_depth"] = recursion_depth
        block.evaluation_snapshot["recursive_branch"] = branch

        hierarchy.blocks_by_layer.setdefault(layer, []).append(block)

        feasible, infeasible = _solve_new_recursive_block(
            hierarchy,
            block,
            expected_branch=branch,
            recursion_depth=recursion_depth,
            runtime=runtime,
            sampler=sampler,
            num_reads=num_reads,
            seed=seed,
        )

        hierarchy.graphs_by_layer[layer] = hierarchy.current_graph.copy()

        if branch == "classical":
            runtime.classical_blocks += 1
        else:
            runtime.qa_blocks += 1

        if feasible == 0:
            runtime.global_infeasible = True

        return RecursiveBlockResult(
            success=feasible > 0,
            block_id=block.block_id,
            layer=layer,
            branch=branch,
            recursion_depth=recursion_depth,
            feasible_macro_count=feasible,
            infeasible_macro_count=infeasible,
            message=(
                "" if feasible > 0
                else "the solved block has no feasible macro transition"
            ),
        )


def _find_root_recursive_region(
    hierarchy: GraphHierarchy,
    *,
    max_input_ports: int,
    max_output_ports: int,
    evaluation_config: EvaluationConfig,
    priority_model: PriorityModel,
    recursive_config: RecursivePlanBConfig,
    protected_nodes: set[Node],
) -> RegionEvaluation | None:
    """
    Find the next TOP-LEVEL non-DAG Plan-B block.

    At the top level Plan B exists to remove cyclic/non-DAG structure.  DAG
    candidates are therefore not selected here.  DAG child regions may still
    be selected *inside* planb_f() because compressing a cheap DAG child can
    reduce an oversized non-DAG parent before the parent is re-evaluated.
    """

    scope = set(hierarchy.current_graph.nodes)
    if len(scope) < recursive_config.min_region_nodes + 1:
        return None

    raw = find_bounded_boundary_regions(
        hierarchy.current_graph,
        hierarchy.edge_records,
        max_input_ports=max_input_ports,
        max_output_ports=max_output_ports,
        min_region_nodes=recursive_config.min_region_nodes,
        max_region_nodes=recursive_config.max_region_nodes,
        beam_width=recursive_config.beam_width,
        max_candidates=recursive_config.max_candidates,
        require_cycle_closed=recursive_config.require_cycle_closed_regions,
        auto_expand_cycle_closure=recursive_config.auto_expand_cycle_closure,
        max_cycle_closure_rounds=recursive_config.max_cycle_closure_rounds,
    )

    prepared: list[set[Node]] = []
    seen: set[frozenset[Node]] = set()

    for region in raw:
        candidate = set(region)
        if candidate & protected_nodes:
            continue
        if candidate == scope:
            continue
        if nx.is_directed_acyclic_graph(
            hierarchy.current_graph.subgraph(candidate)
        ):
            # A top-level DAG block does not eliminate any remaining cycle.
            continue
        if not _validate_region_for_compression(
            hierarchy,
            candidate,
            max_input_ports=max_input_ports,
            max_output_ports=max_output_ports,
            min_region_nodes=recursive_config.min_region_nodes,
            require_cycle_closed=recursive_config.require_cycle_closed_regions,
        ):
            continue

        _internal, incoming, outgoing = boundary_edge_ids(
            hierarchy.current_graph, candidate
        )
        if len(incoming) < recursive_config.min_input_ports:
            continue
        if len(outgoing) < recursive_config.min_output_ports:
            continue

        frozen = frozenset(candidate)
        if frozen in seen:
            continue
        seen.add(frozen)
        prepared.append(candidate)

    evaluations = evaluate_candidates(
        hierarchy,
        prepared,
        config=evaluation_config,
        priority_model=priority_model,
    ) if prepared else []
    evaluations = [
        ev for ev in evaluations
        if ev.eligible and not ev.is_dag
    ]
    if evaluations:
        return evaluations[0]

    # Fallback for the case that the cyclic parent itself is larger than the
    # ordinary beam-search max_region_nodes.  A cyclic SCC can be used as the
    # oversized parent B; planb_f(B) will then recursively find smaller children.
    # This is essential for the intended DFS design: the parent is allowed to
    # be too large precisely because recursion exists to reduce it.
    fallback_regions: list[set[Node]] = []
    for component in nx.strongly_connected_components(
        nx.DiGraph(hierarchy.current_graph)
    ):
        candidate = set(component)
        if len(candidate) == 1:
            node = next(iter(candidate))
            if not hierarchy.current_graph.has_edge(node, node):
                continue
        if candidate & protected_nodes:
            continue
        if candidate == scope:
            continue

        if recursive_config.require_cycle_closed_regions:
            if recursive_config.auto_expand_cycle_closure:
                closure = cycle_close_region(
                    hierarchy.current_graph,
                    candidate,
                    max_region_nodes=None,
                    max_rounds=recursive_config.max_cycle_closure_rounds,
                )
                if not closure.is_cycle_closed:
                    continue
                candidate = set(closure.region_nodes)
            elif not is_cycle_closed_region(
                hierarchy.current_graph, candidate
            ):
                continue

        if candidate & protected_nodes or candidate == scope:
            continue
        if nx.is_directed_acyclic_graph(
            hierarchy.current_graph.subgraph(candidate)
        ):
            continue
        if not _validate_region_for_compression(
            hierarchy,
            candidate,
            max_input_ports=max_input_ports,
            max_output_ports=max_output_ports,
            min_region_nodes=1,
            # Closure was checked above without an artificial size cap.
            require_cycle_closed=False,
        ):
            continue

        _internal, incoming, outgoing = boundary_edge_ids(
            hierarchy.current_graph, candidate
        )
        if len(incoming) < recursive_config.min_input_ports:
            continue
        if len(outgoing) < recursive_config.min_output_ports:
            continue
        fallback_regions.append(candidate)

    if not fallback_regions:
        return None

    fallback_evaluations = evaluate_candidates(
        hierarchy,
        fallback_regions,
        config=evaluation_config,
        priority_model=priority_model,
    )
    fallback_evaluations = [
        ev for ev in fallback_evaluations
        if ev.eligible and not ev.is_dag
    ]
    if not fallback_evaluations:
        return None

    # Prefer the smallest safe cyclic parent in the fallback set; its purpose is
    # to give DFS a parent scope, not to solve the whole SCC directly.
    fallback_evaluations.sort(
        key=lambda ev: (ev.node_count, ev.priority_tier, -ev.priority_score)
    )
    return fallback_evaluations[0]


def run_recursive_planb_on_preprocessed_graph(
    graph: nx.DiGraph | nx.MultiDiGraph,
    *,
    max_input_ports: int,
    max_output_ports: int,
    source: Node | None = None,
    target: Node | None = None,
    evaluation_config: EvaluationConfig | None = None,
    priority_model: PriorityModel | None = None,
    recursive_config: RecursivePlanBConfig | None = None,
    sampler: QUBOSampler | None = None,
    num_reads_macro: int = 200,
    num_reads_top: int = 500,
    seed: int | None = None,
    match_key: str = "l",
    error_key: str = "d",
    alphabet_size: int = 4,
    solve_top: bool = True,
) -> RecursivePlanBResult:
    """
    Execute the DFS-style whole-graph Plan-B design.

    There are two distinct phases.

    BEFORE Plan B is activated:
        top DAG                         -> final classical solve
        top non-DAG and directly QA-able -> direct final QA solve
        top non-DAG and too large      -> activate Plan B

    AFTER Plan B is activated:
        while the visible top graph is non-DAG:
            choose a cycle-closed non-DAG multi-port block B
            execute f(B) recursively
            replace B by its solved feasible macro alternatives
        final visible top graph is DAG -> one final classical solve

    The key point is that after Plan B starts, a later top-level QA-size check
    is diagnostic only.  It never steals control from Plan B.
    """

    if evaluation_config is None:
        evaluation_config = EvaluationConfig()
    if priority_model is None:
        priority_model = DefaultPriorityModel()
    if recursive_config is None:
        recursive_config = RecursivePlanBConfig()

    normalized, records = normalize_olc_graph(graph)
    hierarchy = GraphHierarchy(
        original_graph=normalized.copy(),
        current_graph=normalized.copy(),
        edge_records=records,
        graphs_by_layer={0: normalized.copy()},
    )

    ensure_original_edge_weights(
        hierarchy,
        match_key=match_key,
        error_key=error_key,
        alphabet_size=alphabet_size,
    )
    for record in hierarchy.edge_records.values():
        if record.kind == "original" and record.weight is not None:
            record.data.setdefault("solve_status", "feasible")
            record.data.setdefault("solver_type", "input")

    protected_nodes = {
        node for node in (source, target)
        if node is not None
    }
    missing_protected = protected_nodes - set(hierarchy.current_graph.nodes)
    if missing_protected:
        raise ValueError(
            "source/target not present in recursive Plan-B graph: "
            + ", ".join(sorted(map(str, missing_protected)))
        )

    runtime = _RecursiveRuntime()
    stop_reason = "MAX_TOTAL_BLOCKS_REACHED"
    top_solution: HybridSolution | None = None
    planb_activated = False
    first_top_decision = True

    while runtime.total_blocks < recursive_config.max_total_blocks:
        prune_infeasible_macro_edges(hierarchy)

        top = estimate_top_problem(hierarchy, evaluation_config)
        top_is_dag = nx.is_directed_acyclic_graph(hierarchy.current_graph)
        top_fits_qa = _top_fits_recursive_qa(
            top,
            evaluation_config=evaluation_config,
            recursive_config=recursive_config,
        )

        hierarchy.compression_log.append({
            "action": "recursive_top_decision",
            "top": asdict(top),
            "top_is_dag": top_is_dag,
            "top_fits_qa": top_fits_qa,
            "planb_activated": planb_activated,
            "first_top_decision": first_top_decision,
            "blocks_so_far": runtime.total_blocks,
        })

        # ------------------------------------------------------------
        # DFS head / terminal condition for the WHOLE graph.
        # ------------------------------------------------------------
        if top_is_dag:
            stop_reason = (
                "INITIAL_TOP_DAG_CLASSICAL"
                if not planb_activated
                else "PLANB_FINISHED_TOP_DAG_CLASSICAL"
            )
            if solve_top:
                if source is None or target is None:
                    raise ValueError(
                        "source and target are required when solve_top=True"
                    )
                top_solution = solve_compressed_graph_path_hybrid(
                    hierarchy,
                    source,
                    target,
                    sampler=sampler,
                    num_reads=num_reads_top,
                    seed=seed,
                )
                if top_solution.solver_type != "classical":
                    raise RuntimeError(
                        "final visible graph is DAG but top solver did not route to classical"
                    )
            break

        if first_top_decision and top_fits_qa:
            # If the original problem is already directly QA-solvable there is
            # no reason to pay the Plan-B decomposition overhead.
            stop_reason = "INITIAL_TOP_FITS_QA_DIRECT"
            if solve_top:
                if source is None or target is None:
                    raise ValueError(
                        "source and target are required when solve_top=True"
                    )
                top_solution = solve_compressed_graph_path_hybrid(
                    hierarchy,
                    source,
                    target,
                    sampler=sampler,
                    num_reads=num_reads_top,
                    seed=seed,
                )
                if top_solution.solver_type != "qa":
                    raise RuntimeError(
                        "initial non-DAG graph fits QA but top solver did not route to QA"
                    )
            break

        # Reaching here means the original top graph was too large and Plan B
        # is required.  Once activated it stays active until the top is DAG.
        if not planb_activated:
            planb_activated = True
            hierarchy.compression_log.append({
                "action": "planb_activated",
                "reason": "INITIAL_TOP_NON_DAG_TOO_LARGE_FOR_QA",
                "top": asdict(top),
            })

        first_top_decision = False

        # Even if the compressed top later becomes QA-sized, do NOT solve the
        # whole top by QA.  Plan B's job is now to eliminate the non-DAG parts.
        if top_fits_qa:
            hierarchy.compression_log.append({
                "action": "top_qa_shortcut_ignored_after_planb",
                "reason": "PLANB_ALREADY_ACTIVE_CONTINUE_UNTIL_DAG",
                "top": asdict(top),
            })

        candidate = _find_root_recursive_region(
            hierarchy,
            max_input_ports=max_input_ports,
            max_output_ports=max_output_ports,
            evaluation_config=evaluation_config,
            priority_model=priority_model,
            recursive_config=recursive_config,
            protected_nodes=protected_nodes,
        )
        if candidate is None:
            stop_reason = "PLANB_NON_DAG_NO_VALID_CYCLE_CLOSED_REGION"
            break

        result = planb_f(
            hierarchy,
            set(candidate.region_nodes),
            parent_scope_nodes=None,
            recursion_depth=1,
            max_input_ports=max_input_ports,
            max_output_ports=max_output_ports,
            evaluation_config=evaluation_config,
            priority_model=priority_model,
            recursive_config=recursive_config,
            runtime=runtime,
            sampler=sampler,
            num_reads=num_reads_macro,
            seed=seed,
            protected_nodes=protected_nodes,
        )

        if not result.success:
            stop_reason = (
                "GLOBAL_INFEASIBLE"
                if runtime.global_infeasible
                else "RECURSIVE_PLANB_FAILED: " + result.message
            )
            break

    # If the loop consumed its entire block budget, check whether the very last
    # compression happened to finish the DAG before declaring failure.
    if (
        runtime.total_blocks >= recursive_config.max_total_blocks
        and top_solution is None
        and not runtime.global_infeasible
        and nx.is_directed_acyclic_graph(hierarchy.current_graph)
    ):
        stop_reason = "PLANB_FINISHED_TOP_DAG_CLASSICAL_AT_BLOCK_LIMIT"
        if solve_top:
            if source is None or target is None:
                raise ValueError(
                    "source and target are required when solve_top=True"
                )
            top_solution = solve_compressed_graph_path_hybrid(
                hierarchy, source, target, sampler=sampler,
                num_reads=num_reads_top, seed=seed,
            )

    batch = HybridBatchResult(
        solutions=runtime.solutions,
        classical_count=runtime.classical_macro_count,
        qa_count=runtime.qa_macro_count,
        feasible_count=runtime.feasible_macro_count,
        infeasible_count=runtime.infeasible_macro_count,
    )

    hierarchy.compression_log.append({
        "action": "recursive_run_finished",
        "stop_reason": stop_reason,
        "planb_activated": planb_activated,
        "final_graph_is_dag": nx.is_directed_acyclic_graph(
            hierarchy.current_graph
        ),
        "total_blocks": runtime.total_blocks,
    })

    return RecursivePlanBResult(
        hierarchy=hierarchy,
        macro_batch=batch,
        top_solution=top_solution,
        stop_reason=stop_reason,
        recursive_calls=runtime.recursive_calls,
        max_recursion_depth_reached=runtime.max_depth,
        total_blocks=runtime.total_blocks,
        classical_blocks=runtime.classical_blocks,
        qa_blocks=runtime.qa_blocks,
        global_infeasible=runtime.global_infeasible,
    )

# ============================================================================
# ADAPTIVE CYCLE ELIMINATION
# ============================================================================

@dataclass
class AdaptiveCycleConfig:
    """
    Controls the new solve-immediately cycle elimination mode.

    The important distinction from the legacy hierarchy builder is that a
    compressed block is solved in the same iteration in which it is created.
    Infeasible macro edges are then removed before SCC/DAG analysis continues.
    """

    # A local cyclic candidate is sent to QA only if its largest fixed-port
    # QUBO estimate is within this logical-variable limit.  None means use
    # EvaluationConfig.preferred_local_qubo_variables.
    direct_qubo_variable_limit: int | None = None

    # Candidate search size used only when a whole SCC is too large to solve
    # directly.  Such a subregion is an internal Plan-B cut of that SCC.
    min_subregion_nodes: int = 2
    max_subregion_nodes: int = 20
    beam_width: int = 24
    max_candidates: int = 256

    # Number of solve->prune iterations.
    max_iterations: int = 1000

    # In adaptive mode we intentionally do NOT require cycle-closed cuts.
    # Because every new macro edge is solved immediately, an outside return
    # path is no longer a spurious unresolved cycle: only feasible transitions
    # survive into the next active graph.
    require_cycle_closed_subregions: bool = False


@dataclass
class AdaptiveCycleResult:
    hierarchy: GraphHierarchy
    macro_batch: HybridBatchResult
    iterations: int
    stop_reason: str
    final_graph_is_dag: bool
    global_infeasible: bool = False


def _cyclic_sccs(graph: nx.MultiDiGraph) -> list[set[Node]]:
    """Return all SCCs that contain a directed cycle."""

    simple = nx.DiGraph(graph)
    result: list[set[Node]] = []

    for component in nx.strongly_connected_components(simple):
        nodes = set(component)
        if len(nodes) > 1:
            result.append(nodes)
            continue
        if nodes:
            node = next(iter(nodes))
            if simple.has_edge(node, node):
                result.append(nodes)

    result.sort(key=lambda s: (len(s), tuple(sorted(map(str, s)))))
    return result


def _active_macro_status_counts(hierarchy: GraphHierarchy) -> dict[str, int]:
    counts = {"feasible": 0, "infeasible": 0, "unresolved": 0}
    for edge_id in graph_edge_ids(hierarchy.current_graph):
        record = hierarchy.edge_records[edge_id]
        if record.kind != "macro":
            continue
        status = _record_status(record)
        counts[status] = counts.get(status, 0) + 1
    return counts


def prune_infeasible_macro_edges(
    hierarchy: GraphHierarchy,
    *,
    block_id: str | None = None,
) -> list[str]:
    """
    Remove already-solved infeasible macro edges from the *visible* graph.

    EdgeRecord objects are intentionally retained for audit/reconstruction.
    This is what prevents fake p_i->q_j alternatives from creating higher-level
    cycles after their local block has already been solved.
    """

    removed: list[str] = []

    for u, v, key, data in list(
        hierarchy.current_graph.edges(keys=True, data=True)
    ):
        edge_id = data["edge_id"]
        record = hierarchy.edge_records[edge_id]

        if record.kind != "macro":
            continue
        if block_id is not None and record.block_id != block_id:
            continue
        if _record_status(record) != "infeasible":
            continue

        hierarchy.current_graph.remove_edge(u, v, key=key)
        record.data["pruned_from_visible_graph"] = True
        removed.append(edge_id)

    return removed


def _candidate_is_directly_solvable(
    evaluation: RegionEvaluation,
    *,
    qubo_limit: int,
) -> bool:
    if evaluation.is_dag or not evaluation.eligible:
        return False
    if evaluation.local_qubo_max_variables is None:
        return False
    return evaluation.local_qubo_max_variables <= qubo_limit


def _choose_adaptive_cyclic_region(
    hierarchy: GraphHierarchy,
    *,
    max_input_ports: int,
    max_output_ports: int,
    evaluation_config: EvaluationConfig,
    priority_model: PriorityModel,
    adaptive_config: AdaptiveCycleConfig,
    protected_nodes: set[Node],
) -> tuple[RegionEvaluation | None, str]:
    """
    Choose one cyclic region.

    Order of preference:
      1. a whole SCC, when it fits the local QA limit;
      2. otherwise a smaller cyclic Plan-B subregion inside an SCC.

    DAG-only regions are never compressed in this mode.
    """

    graph = hierarchy.current_graph
    sccs = _cyclic_sccs(graph)
    if not sccs:
        return None, "GRAPH_IS_DAG"

    qubo_limit = (
        adaptive_config.direct_qubo_variable_limit
        if adaptive_config.direct_qubo_variable_limit is not None
        else evaluation_config.preferred_local_qubo_variables
    )

    # ---- 1) Try to solve an entire SCC directly. ----
    whole_scc_candidates: list[set[Node]] = []
    for scc in sccs:
        if scc & protected_nodes:
            continue
        if not _validate_region_for_compression(
            hierarchy,
            scc,
            max_input_ports=max_input_ports,
            max_output_ports=max_output_ports,
            min_region_nodes=1,
            require_cycle_closed=False,
        ):
            continue
        whole_scc_candidates.append(scc)

    whole_evals = evaluate_candidates(
        hierarchy,
        whole_scc_candidates,
        config=evaluation_config,
        priority_model=priority_model,
    )
    whole_evals = [
        ev for ev in whole_evals
        if _candidate_is_directly_solvable(ev, qubo_limit=qubo_limit)
    ]
    if whole_evals:
        # Prefer eliminating more of one SCC when the QA cost is acceptable.
        whole_evals.sort(
            key=lambda ev: (
                -ev.node_reduction,
                ev.local_qubo_max_variables or 10**18,
                -ev.priority_score,
            )
        )
        return whole_evals[0], "WHOLE_SCC"

    # ---- 2) SCC too large: solve a smaller cyclic subregion immediately. ----
    raw = find_bounded_boundary_regions(
        graph,
        hierarchy.edge_records,
        max_input_ports=max_input_ports,
        max_output_ports=max_output_ports,
        min_region_nodes=adaptive_config.min_subregion_nodes,
        max_region_nodes=adaptive_config.max_subregion_nodes,
        beam_width=adaptive_config.beam_width,
        max_candidates=adaptive_config.max_candidates,
        require_cycle_closed=adaptive_config.require_cycle_closed_subregions,
        auto_expand_cycle_closure=False,
    )

    cyclic_node_union = set().union(*sccs)
    scc_lookup: dict[Node, int] = {}
    for index, scc in enumerate(sccs):
        for node in scc:
            scc_lookup[node] = index

    subregions: list[set[Node]] = []
    for region in raw:
        if region & protected_nodes:
            continue
        if not region <= cyclic_node_union:
            continue
        ids = {scc_lookup[node] for node in region if node in scc_lookup}
        if len(ids) != 1:
            continue
        if nx.is_directed_acyclic_graph(graph.subgraph(region)):
            continue
        if not _validate_region_for_compression(
            hierarchy,
            region,
            max_input_ports=max_input_ports,
            max_output_ports=max_output_ports,
            min_region_nodes=adaptive_config.min_subregion_nodes,
            require_cycle_closed=adaptive_config.require_cycle_closed_subregions,
        ):
            continue
        subregions.append(region)

    sub_evals = evaluate_candidates(
        hierarchy,
        subregions,
        config=evaluation_config,
        priority_model=priority_model,
    )
    sub_evals = [
        ev for ev in sub_evals
        if _candidate_is_directly_solvable(ev, qubo_limit=qubo_limit)
    ]

    if not sub_evals:
        return None, "NO_SOLVABLE_CYCLIC_REGION"

    # Existing priority score now ranks only genuinely cyclic candidates.
    return sub_evals[0], "SUBREGION_PLAN_B"


def eliminate_cycles_adaptively(
    graph: nx.DiGraph | nx.MultiDiGraph,
    *,
    max_input_ports: int,
    max_output_ports: int,
    evaluation_config: EvaluationConfig | None = None,
    priority_model: PriorityModel | None = None,
    adaptive_config: AdaptiveCycleConfig | None = None,
    sampler: QUBOSampler | None = None,
    num_reads: int = 200,
    seed: int | None = None,
    match_key: str = "l",
    error_key: str = "d",
    alphabet_size: int = 4,
    protected_nodes: Iterable[Node] = (),
) -> AdaptiveCycleResult:
    """
    New default Plan-B execution strategy.

    Algorithm:
        G_0 = preprocessed OLC graph
        while G_t contains a directed cycle:
            choose one directly solvable cyclic SCC / subregion B_t
            structurally compress B_t to candidate p_i -> q_j macro edges
            solve *all* those candidates immediately
            remove infeasible candidates from the visible graph
            G_{t+1} = the pruned feasible graph
        stop exactly when the active graph is a DAG

    The `layer` field is now only a dependency/reconstruction index.  It no
    longer means that the program first builds a complete hierarchy and then
    solves all layers afterwards.
    """

    if evaluation_config is None:
        evaluation_config = EvaluationConfig()
    if priority_model is None:
        priority_model = DefaultPriorityModel()
    if adaptive_config is None:
        adaptive_config = AdaptiveCycleConfig()

    normalized, records = normalize_olc_graph(graph)
    hierarchy = GraphHierarchy(
        original_graph=normalized.copy(),
        current_graph=normalized.copy(),
        edge_records=records,
        graphs_by_layer={0: normalized.copy()},
    )

    ensure_original_edge_weights(
        hierarchy,
        match_key=match_key,
        error_key=error_key,
        alphabet_size=alphabet_size,
    )
    for record in hierarchy.edge_records.values():
        if record.kind == "original" and record.weight is not None:
            record.data.setdefault("solve_status", "feasible")
            record.data.setdefault("solver_type", "input")

    protected = set(protected_nodes)
    unknown_protected = protected - set(hierarchy.current_graph.nodes)
    if unknown_protected:
        raise ValueError(
            "protected_nodes not present after preprocessing: "
            + ", ".join(sorted(map(str, unknown_protected)))
        )

    solutions: dict[str, HybridSolution] = {}
    classical_count = qa_count = feasible_count = infeasible_count = 0
    stop_reason = "MAX_ITERATIONS_REACHED"
    global_infeasible = False
    iterations = 0

    for iteration in range(1, adaptive_config.max_iterations + 1):
        iterations = iteration - 1

        # Defensive cleanup in case a caller supplied already-solved records.
        prune_infeasible_macro_edges(hierarchy)

        if nx.is_directed_acyclic_graph(hierarchy.current_graph):
            stop_reason = "ACTIVE_GRAPH_IS_DAG"
            break

        scc_before = _cyclic_sccs(hierarchy.current_graph)
        top_before = estimate_top_problem(hierarchy, evaluation_config)

        evaluation, selection_kind = _choose_adaptive_cyclic_region(
            hierarchy,
            max_input_ports=max_input_ports,
            max_output_ports=max_output_ports,
            evaluation_config=evaluation_config,
            priority_model=priority_model,
            adaptive_config=adaptive_config,
            protected_nodes=protected,
        )

        if evaluation is None:
            stop_reason = selection_kind
            hierarchy.compression_log.append({
                "step": iteration,
                "action": "stop",
                "reason": stop_reason,
                "cyclic_scc_count": len(scc_before),
                "top_before": asdict(top_before),
            })
            break

        region = set(evaluation.region_nodes)
        block = compress_one_region(
            hierarchy,
            region,
            layer=iteration,
            max_input_ports=max_input_ports,
            max_output_ports=max_output_ports,
            evaluation=evaluation,
            require_cycle_closed=False,
        )

        hierarchy.blocks_by_layer[iteration] = [block]

        block_feasible = 0
        block_infeasible = 0
        block_solver_types: list[str] = []

        # Critical new behavior: solve NOW, before the next SCC analysis.
        for macro_edge_id in block.macro_edge_ids:
            solution = solve_one_macro_edge_hybrid(
                hierarchy,
                block.block_id,
                macro_edge_id,
                sampler=sampler,
                num_reads=num_reads,
                seed=seed,
            )
            _write_solution_to_macro_record(
                hierarchy,
                macro_edge_id,
                solution,
            )
            solutions[macro_edge_id] = solution
            block_solver_types.append(solution.solver_type)

            if solution.solver_type == "classical":
                classical_count += 1
            else:
                qa_count += 1

            if solution.feasible:
                feasible_count += 1
                block_feasible += 1
            else:
                infeasible_count += 1
                block_infeasible += 1

        pruned = prune_infeasible_macro_edges(
            hierarchy,
            block_id=block.block_id,
        )

        # No boundary transition through this mandatory removed region survives.
        if block_feasible == 0:
            global_infeasible = True
            stop_reason = "BLOCK_HAS_NO_FEASIBLE_MACRO_EDGE"

        hierarchy.graphs_by_layer[iteration] = hierarchy.current_graph.copy()
        top_after = estimate_top_problem(hierarchy, evaluation_config)
        scc_after = _cyclic_sccs(hierarchy.current_graph)

        hierarchy.compression_log.append({
            "step": iteration,
            "action": "solve_compress_cycle",
            "selection_kind": selection_kind,
            "block_id": block.block_id,
            "region_nodes": [str(x) for x in sorted(region, key=str)],
            "evaluation": evaluation.to_dict(),
            "solver_types": block_solver_types,
            "macro_total": len(block.macro_edge_ids),
            "macro_feasible": block_feasible,
            "macro_infeasible": block_infeasible,
            "pruned_macro_edge_ids": list(pruned),
            "cyclic_scc_count_before": len(scc_before),
            "cyclic_scc_count_after": len(scc_after),
            "active_macro_status": _active_macro_status_counts(hierarchy),
            "top_before": asdict(top_before),
            "top_after": asdict(top_after),
        })

        iterations = iteration

        if global_infeasible:
            break

        if nx.is_directed_acyclic_graph(hierarchy.current_graph):
            stop_reason = "ACTIVE_GRAPH_IS_DAG"
            break

    else:
        iterations = adaptive_config.max_iterations

    batch = HybridBatchResult(
        solutions=solutions,
        classical_count=classical_count,
        qa_count=qa_count,
        feasible_count=feasible_count,
        infeasible_count=infeasible_count,
    )

    return AdaptiveCycleResult(
        hierarchy=hierarchy,
        macro_batch=batch,
        iterations=iterations,
        stop_reason=stop_reason,
        final_graph_is_dag=nx.is_directed_acyclic_graph(hierarchy.current_graph),
        global_infeasible=global_infeasible,
    )

# ============================================================================
# FULL PIPELINE WRAPPER
# ============================================================================

@dataclass
class FinalAssemblySolution:
    """Final solved assembly after top solve + recursive macro expansion."""

    feasible: bool
    reconstruction_ok: bool
    solver_type: str | None
    weight: float
    top_selected_edge_ids: list[str] = field(default_factory=list)
    preprocessed_edge_ids: list[str] = field(default_factory=list)
    raw_edge_ids: list[str] = field(default_factory=list)
    raw_node_path: list[Node] = field(default_factory=list)
    message: str = ""


def build_final_assembly_solution(
    preprocess: SafePreprocessResult,
    hierarchy: GraphHierarchy,
    top_solution: HybridSolution | None,
    *,
    mapped_source: Node | None,
    mapped_target: Node | None,
    raw_source: Node | None,
    raw_target: Node | None,
) -> FinalAssemblySolution:
    """
    Convert the final top solution into the actual solved OLC path.

    Stages:
        top selected edges
        -> ordered recursive macro expansion
        -> layer-0 preprocessed edges
        -> safe-unitig restoration
        -> raw OLC edge/node path
    """

    if top_solution is None:
        return FinalAssemblySolution(
            feasible=False, reconstruction_ok=False, solver_type=None,
            weight=-inf, message="top graph was not solved",
        )
    if not top_solution.feasible:
        return FinalAssemblySolution(
            feasible=False, reconstruction_ok=False,
            solver_type=top_solution.solver_type,
            weight=top_solution.weight,
            top_selected_edge_ids=list(top_solution.selected_actual_edge_ids),
            message=top_solution.message or "top solution is infeasible",
        )
    if mapped_source is None or mapped_target is None:
        return FinalAssemblySolution(
            feasible=False, reconstruction_ok=False,
            solver_type=top_solution.solver_type, weight=top_solution.weight,
            top_selected_edge_ids=list(top_solution.selected_actual_edge_ids),
            message="source/target are required for ordered final reconstruction",
        )

    try:
        top_order = order_selected_edge_ids_as_path(
            hierarchy,
            top_solution.selected_actual_edge_ids,
            source=mapped_source,
            target=mapped_target,
        )
        layer0 = expand_selected_edges_ordered(
            hierarchy,
            top_order,
            source=mapped_source,
            target=mapped_target,
        )
    except RuntimeError as exc:
        return FinalAssemblySolution(
            feasible=False, reconstruction_ok=False,
            solver_type=top_solution.solver_type, weight=top_solution.weight,
            top_selected_edge_ids=list(top_solution.selected_actual_edge_ids),
            message=f"Plan-B macro reconstruction failed: {exc}",
        )

    try:
        raw_edges, raw_nodes = expand_preprocessed_path_to_raw(
            preprocess,
            hierarchy,
            layer0,
            raw_source=raw_source,
            raw_target=raw_target,
        )
    except RuntimeError as exc:
        return FinalAssemblySolution(
            feasible=False, reconstruction_ok=False,
            solver_type=top_solution.solver_type, weight=top_solution.weight,
            top_selected_edge_ids=top_order,
            preprocessed_edge_ids=layer0,
            message=f"safe-unitig reconstruction failed: {exc}",
        )

    return FinalAssemblySolution(
        feasible=True,
        reconstruction_ok=True,
        solver_type=top_solution.solver_type,
        weight=top_solution.weight,
        top_selected_edge_ids=top_order,
        preprocessed_edge_ids=layer0,
        raw_edge_ids=raw_edges,
        raw_node_path=raw_nodes,
        message="",
    )


@dataclass
class FullOLCPipelineResult:
    preprocess: SafePreprocessResult
    hierarchy: GraphHierarchy
    macro_batch: HybridBatchResult | None = None
    top_solution: HybridSolution | None = None
    final_assembly: FinalAssemblySolution | None = None
    mapped_source: Node | None = None
    mapped_target: Node | None = None

    # Execution diagnostics.
    execution_mode: str = "recursive_planb"

    # Recursive Plan-B diagnostics.
    recursive_calls: int = 0
    recursive_max_depth: int = 0
    recursive_total_blocks: int = 0
    recursive_classical_blocks: int = 0
    recursive_qa_blocks: int = 0
    recursive_stop_reason: str = ""

    # Legacy adaptive-cycle diagnostics (kept for compatibility).
    adaptive_mode: bool = False
    adaptive_iterations: int = 0
    adaptive_stop_reason: str = ""
    final_graph_is_dag: bool = False
    global_infeasible: bool = False


def map_node_after_safe_preprocess(
    preprocess: SafePreprocessResult,
    node: Node | None,
) -> Node | None:
    """Map an original node to its safe-unitig representative when needed."""
    if node is None:
        return None
    return preprocess.node_to_unitig.get(node, node)


def build_hierarchy_from_raw_olc(
    graph: nx.DiGraph | nx.MultiDiGraph,
    *,
    preprocess_config: SafePreprocessConfig | None = None,
    max_input_ports: int,
    max_output_ports: int,
    evaluation_config: EvaluationConfig | None = None,
    priority_model: PriorityModel | None = None,
    max_compression_steps: int = 100,
    target_node_count: int | None = None,
    regions_by_step: Sequence[Sequence[Iterable[Node]]] | None = None,
    region_finder: Callable[
        [nx.MultiDiGraph, Mapping[str, EdgeRecord], int, int, int],
        Sequence[set[Node]],
    ] | None = None,
    min_region_nodes: int = 2,
    max_region_nodes: int = 20,
    beam_width: int = 24,
    max_candidates: int = 256,
    regions_per_step: int = 1,
    require_cycle_closed_regions: bool = True,
    auto_expand_cycle_closure: bool = True,
    max_cycle_closure_rounds: int = 32,
) -> tuple[SafePreprocessResult, GraphHierarchy]:
    """
    Complete structural front-end:

        raw OLC
          -> conservative hifiasm-inspired safe preprocessing
          -> cycle-closed priority-aware hierarchical Plan-B compression

    No macro-edge solving is performed here.
    """

    pre = safe_preprocess_before_planb(
        graph,
        config=preprocess_config,
    )

    hierarchy = compress_olc_hierarchy_prioritized(
        pre.graph,
        max_input_ports=max_input_ports,
        max_output_ports=max_output_ports,
        evaluation_config=evaluation_config,
        priority_model=priority_model,
        max_compression_steps=max_compression_steps,
        target_node_count=target_node_count,
        regions_by_step=regions_by_step,
        region_finder=region_finder,
        min_region_nodes=min_region_nodes,
        max_region_nodes=max_region_nodes,
        beam_width=beam_width,
        max_candidates=max_candidates,
        regions_per_step=regions_per_step,
        require_cycle_closed_regions=require_cycle_closed_regions,
        auto_expand_cycle_closure=auto_expand_cycle_closure,
        max_cycle_closure_rounds=max_cycle_closure_rounds,
    )

    return pre, hierarchy


def run_full_olc_pipeline(
    graph: nx.DiGraph | nx.MultiDiGraph,
    *,
    max_input_ports: int,
    max_output_ports: int,
    source: Node | None = None,
    target: Node | None = None,
    preprocess_config: SafePreprocessConfig | None = None,
    evaluation_config: EvaluationConfig | None = None,
    priority_model: PriorityModel | None = None,
    sampler: QUBOSampler | None = None,
    num_reads_macro: int = 200,
    num_reads_top: int = 500,
    seed: int | None = None,
    match_key: str = "l",
    error_key: str = "d",
    alphabet_size: int = 4,
    solve_macro_weights: bool = True,
    solve_top: bool = True,
    max_compression_steps: int = 100,
    target_node_count: int | None = None,
    regions_by_step: Sequence[Sequence[Iterable[Node]]] | None = None,
    min_region_nodes: int = 2,
    max_region_nodes: int = 20,
    beam_width: int = 24,
    max_candidates: int = 256,
    regions_per_step: int = 1,
    require_cycle_closed_regions: bool = True,
    auto_expand_cycle_closure: bool = True,
    max_cycle_closure_rounds: int = 32,
    # Recommended/default execution mode: recursive Plan-B f(B).
    recursive_planb: bool = True,
    recursive_config: RecursivePlanBConfig | None = None,
    # Older adaptive-cycle mode is kept only for comparison experiments.
    adaptive_cycle_elimination: bool = False,
    adaptive_config: AdaptiveCycleConfig | None = None,
) -> FullOLCPipelineResult:
    """
    End-to-end pipeline.

    Default (recursive_planb=True):
        safe preprocessing once
        -> original top DAG: classical final solve
        -> original top non-DAG but QA-sized: direct QA final solve
        -> otherwise activate Plan B
        -> repeatedly choose a cycle-closed non-DAG multi-port block B
        -> execute f(B):
              DAG                    -> classical
              non-DAG and QA-sized   -> QA
              non-DAG and too large  -> recursively apply f to a child of B
        -> after Plan B has started, continue until the TOP graph is DAG
        -> solve that final DAG once classically
        -> recursively expand selected macro edges back to the original path.

    adaptive_cycle_elimination=True is retained only as an older comparison
    mode.  Set recursive_planb=False to use it.
    """

    if evaluation_config is None:
        evaluation_config = EvaluationConfig()
    if priority_model is None:
        priority_model = DefaultPriorityModel()

    pre = safe_preprocess_before_planb(
        graph,
        config=preprocess_config,
    )

    mapped_source = map_node_after_safe_preprocess(pre, source)
    mapped_target = map_node_after_safe_preprocess(pre, target)

    if recursive_planb:
        if not solve_macro_weights:
            raise ValueError(
                "recursive_planb=True requires solve_macro_weights=True because "
                "every leaf/parent block must be solved before it can be packed "
                "into the next recursive level."
            )

        if recursive_config is None:
            recursive_config = RecursivePlanBConfig(
                min_region_nodes=min_region_nodes,
                max_region_nodes=max_region_nodes,
                beam_width=beam_width,
                max_candidates=max_candidates,
                max_total_blocks=max_compression_steps,
                require_cycle_closed_regions=require_cycle_closed_regions,
                auto_expand_cycle_closure=auto_expand_cycle_closure,
                max_cycle_closure_rounds=max_cycle_closure_rounds,
            )

        recursive = run_recursive_planb_on_preprocessed_graph(
            pre.graph,
            max_input_ports=max_input_ports,
            max_output_ports=max_output_ports,
            source=mapped_source,
            target=mapped_target,
            evaluation_config=evaluation_config,
            priority_model=priority_model,
            recursive_config=recursive_config,
            sampler=sampler,
            num_reads_macro=num_reads_macro,
            num_reads_top=num_reads_top,
            seed=seed,
            match_key=match_key,
            error_key=error_key,
            alphabet_size=alphabet_size,
            solve_top=solve_top,
        )

        final_assembly = (
            build_final_assembly_solution(
                pre,
                recursive.hierarchy,
                recursive.top_solution,
                mapped_source=mapped_source,
                mapped_target=mapped_target,
                raw_source=source,
                raw_target=target,
            )
            if solve_top else None
        )

        return FullOLCPipelineResult(
            preprocess=pre,
            hierarchy=recursive.hierarchy,
            macro_batch=recursive.macro_batch,
            top_solution=recursive.top_solution,
            final_assembly=final_assembly,
            mapped_source=mapped_source,
            mapped_target=mapped_target,
            execution_mode="recursive_planb",
            recursive_calls=recursive.recursive_calls,
            recursive_max_depth=recursive.max_recursion_depth_reached,
            recursive_total_blocks=recursive.total_blocks,
            recursive_classical_blocks=recursive.classical_blocks,
            recursive_qa_blocks=recursive.qa_blocks,
            recursive_stop_reason=recursive.stop_reason,
            adaptive_mode=False,
            final_graph_is_dag=nx.is_directed_acyclic_graph(
                recursive.hierarchy.current_graph
            ),
            global_infeasible=recursive.global_infeasible,
        )

    if adaptive_cycle_elimination:
        if not solve_macro_weights:
            raise ValueError(
                "adaptive_cycle_elimination=True requires solve_macro_weights=True "
                "because every compressed cyclic block must be solved before "
                "the next SCC analysis."
            )

        if adaptive_config is None:
            adaptive_config = AdaptiveCycleConfig(
                min_subregion_nodes=min_region_nodes,
                max_subregion_nodes=max_region_nodes,
                beam_width=beam_width,
                max_candidates=max_candidates,
                max_iterations=max_compression_steps,
            )

        protected = {
            node for node in (mapped_source, mapped_target)
            if node is not None
        }

        adaptive = eliminate_cycles_adaptively(
            pre.graph,
            max_input_ports=max_input_ports,
            max_output_ports=max_output_ports,
            evaluation_config=evaluation_config,
            priority_model=priority_model,
            adaptive_config=adaptive_config,
            sampler=sampler,
            num_reads=num_reads_macro,
            seed=seed,
            match_key=match_key,
            error_key=error_key,
            alphabet_size=alphabet_size,
            protected_nodes=protected,
        )
        hierarchy = adaptive.hierarchy
        batch: HybridBatchResult | None = adaptive.macro_batch

        top_solution: HybridSolution | None = None
        if solve_top and not adaptive.global_infeasible:
            if mapped_source is None or mapped_target is None:
                raise ValueError("source and target are required when solve_top=True")
            top_solution = solve_compressed_graph_path_hybrid(
                hierarchy,
                mapped_source,
                mapped_target,
                sampler=sampler,
                num_reads=num_reads_top,
                seed=seed,
            )

        final_assembly = (
            build_final_assembly_solution(
                pre, hierarchy, top_solution,
                mapped_source=mapped_source,
                mapped_target=mapped_target,
                raw_source=source, raw_target=target,
            )
            if solve_top else None
        )

        return FullOLCPipelineResult(
            preprocess=pre,
            hierarchy=hierarchy,
            macro_batch=batch,
            top_solution=top_solution,
            final_assembly=final_assembly,
            mapped_source=mapped_source,
            mapped_target=mapped_target,
            execution_mode="adaptive_cycle_elimination",
            adaptive_mode=True,
            adaptive_iterations=adaptive.iterations,
            adaptive_stop_reason=adaptive.stop_reason,
            final_graph_is_dag=adaptive.final_graph_is_dag,
            global_infeasible=adaptive.global_infeasible,
        )

    # ------------------------------------------------------------------
    # Legacy: build the whole structural hierarchy first, then solve layers.
    # ------------------------------------------------------------------
    hierarchy = compress_olc_hierarchy_prioritized(
        pre.graph,
        max_input_ports=max_input_ports,
        max_output_ports=max_output_ports,
        evaluation_config=evaluation_config,
        priority_model=priority_model,
        max_compression_steps=max_compression_steps,
        target_node_count=target_node_count,
        regions_by_step=regions_by_step,
        min_region_nodes=min_region_nodes,
        max_region_nodes=max_region_nodes,
        beam_width=beam_width,
        max_candidates=max_candidates,
        regions_per_step=regions_per_step,
        require_cycle_closed_regions=require_cycle_closed_regions,
        auto_expand_cycle_closure=auto_expand_cycle_closure,
        max_cycle_closure_rounds=max_cycle_closure_rounds,
    )

    batch = None
    if solve_macro_weights and hierarchy.max_layer > 0:
        batch = solve_all_macro_weights_hybrid(
            hierarchy,
            sampler=sampler,
            num_reads=num_reads_macro,
            seed=seed,
            match_key=match_key,
            error_key=error_key,
            alphabet_size=alphabet_size,
        )
    elif solve_macro_weights:
        ensure_original_edge_weights(
            hierarchy,
            match_key=match_key,
            error_key=error_key,
            alphabet_size=alphabet_size,
        )

    top_solution = None
    if solve_top:
        if mapped_source is None or mapped_target is None:
            raise ValueError("source and target are required when solve_top=True")
        top_solution = solve_compressed_graph_path_hybrid(
            hierarchy,
            mapped_source,
            mapped_target,
            sampler=sampler,
            num_reads=num_reads_top,
            seed=seed,
        )

    final_assembly = (
        build_final_assembly_solution(
            pre, hierarchy, top_solution,
            mapped_source=mapped_source,
            mapped_target=mapped_target,
            raw_source=source, raw_target=target,
        )
        if solve_top else None
    )

    return FullOLCPipelineResult(
        preprocess=pre,
        hierarchy=hierarchy,
        macro_batch=batch,
        top_solution=top_solution,
        final_assembly=final_assembly,
        mapped_source=mapped_source,
        mapped_target=mapped_target,
        execution_mode="legacy_hierarchy",
        adaptive_mode=False,
        adaptive_iterations=0,
        adaptive_stop_reason="LEGACY_HIERARCHY_MODE",
        final_graph_is_dag=nx.is_directed_acyclic_graph(hierarchy.current_graph),
        global_infeasible=False,
    )

# ============================================================================
# INTEGRATED SELF TESTS
# ============================================================================

def _test_safe_preprocess_integrated() -> None:
    g = nx.MultiDiGraph()
    attrs = dict(
        q_start=50, q_end=100, q_len=100,
        t_start=0, t_end=50, t_len=100,
        matches=50, block_len=50, mapq=60,
        weight=1.0,
    )
    g.add_edge("A", "B", key="ab1", edge_id="ab1", **attrs)
    g.add_edge("A", "B", key="ab2", edge_id="ab2", **attrs)
    g.add_edge("B", "C", key="bc", edge_id="bc", **attrs)

    contain = dict(
        q_start=0, q_end=50, q_len=50,
        t_start=20, t_end=70, t_len=100,
        matches=50, block_len=50, mapq=60,
        weight=1.0,
    )
    g.add_edge("X", "Y", key="xy", edge_id="xy", **contain)

    g.add_edge("P", "Q", key="pq", edge_id="pq", **attrs)
    g.add_edge("Q", "P", key="qp", edge_id="qp", **attrs)

    r = safe_preprocess_before_planb(g)
    assert "X" not in r.graph
    assert len(r.duplicate_groups) == 1
    assert any(set(members) == {"A", "B", "C"} for members in r.unitigs.values())
    assert r.graph.has_edge("P", "Q") and r.graph.has_edge("Q", "P")


def _test_cycle_closure_expands_boundary() -> None:
    g = nx.MultiDiGraph()
    for eid, u, v in [
        ("xp", "X", "P"),
        ("pa", "P", "A"),
        ("aq", "A", "Q"),
        ("qp", "Q", "P"),
        ("qy", "Q", "Y"),
    ]:
        g.add_edge(u, v, key=eid, edge_id=eid, weight=1.0)

    closure = cycle_close_region(g, {"A"}, max_region_nodes=10)
    assert closure.is_cycle_closed, closure
    assert set(closure.region_nodes) == {"A", "P", "Q"}, closure

    internal, incoming, outgoing = boundary_edge_ids(g, set(closure.region_nodes))
    assert len(incoming) == 1 and len(outgoing) == 1
    assert set(internal) == {"pa", "aq", "qp"}


def _test_cycle_closed_manual_candidate_is_auto_expanded() -> None:
    g = nx.MultiDiGraph()
    for eid, u, v in [
        ("xp", "X", "P"),
        ("pa", "P", "A"),
        ("aq", "A", "Q"),
        ("qp", "Q", "P"),
        ("qy", "Q", "Y"),
    ]:
        g.add_edge(u, v, key=eid, edge_id=eid, weight=1.0)

    cfg = EvaluationConfig(
        target_top_qubo_variables=None,
        stop_if_top_graph_is_dag=True,
    )
    h = compress_olc_hierarchy_prioritized(
        g,
        max_input_ports=2,
        max_output_ports=2,
        evaluation_config=cfg,
        max_compression_steps=1,
        regions_by_step=[[{"A"}]],
        max_region_nodes=10,
        require_cycle_closed_regions=True,
        auto_expand_cycle_closure=True,
    )

    assert h.max_layer == 1, h.compression_log
    block = h.blocks_by_layer[1][0]
    assert set(block.region_nodes) == {"A", "P", "Q"}
    assert nx.is_directed_acyclic_graph(h.current_graph)


def _build_two_layer_dag_integrated() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()

    def add(u: str, v: str, edge_id: str, weight: float = 1.0) -> None:
        g.add_edge(u, v, key=edge_id, edge_id=edge_id, weight=weight)

    add("S1", "U1", "e_S1_U1")
    add("S2", "U1", "e_S2_U1")
    add("V2", "T1", "e_V2_T1")
    add("V2", "T2", "e_V2_T2")
    add("S1", "S2", "e_S1_S2")
    add("T1", "T2", "e_T1_T2")
    add("U1", "U2", "e_U1_U2")
    add("V1", "V2", "e_V1_V2")
    add("U1", "A1", "e_U1_A1")
    add("U2", "A1", "e_U2_A1")
    add("A1", "A2", "e_A1_A2")
    add("A2", "A3", "e_A2_A3")
    add("A3", "V1", "e_A3_V1")
    add("A3", "V2", "e_A3_V2")
    return g


def _test_existing_hybrid_dag_route() -> None:
    # Disable cycle-closed auto expansion here because this is the original
    # hand-constructed compatibility test for the hybrid solver itself.
    hierarchy = compress_olc_hierarchy_prioritized(
        _build_two_layer_dag_integrated(),
        max_input_ports=2,
        max_output_ports=2,
        evaluation_config=EvaluationConfig(
            target_top_qubo_variables=None,
            stop_if_top_graph_is_dag=False,
        ),
        max_compression_steps=2,
        regions_by_step=[
            [{"A1", "A2", "A3"}],
            [{"U1", "U2", "V1", "V2"}],
        ],
        require_cycle_closed_regions=False,
    )

    batch = solve_all_macro_weights_hybrid(hierarchy)
    assert batch.qa_count == 0, batch
    assert batch.classical_count == 8, batch


class _ExactTinyQUBOSampler:
    """Deterministic exhaustive sampler used only by integrated self tests."""

    def sample_qubo(
        self,
        qubo: Mapping[tuple[str, str], float],
        *,
        num_reads: int = 1,
        seed: int | None = None,
    ) -> tuple[dict[str, int], float]:
        variables = sorted({v for pair in qubo for v in pair})
        if len(variables) > 22:
            raise RuntimeError("tiny exact test sampler received too many variables")

        best_sample: dict[str, int] = {}
        best_energy = inf
        for bits in range(1 << len(variables)):
            sample = {
                var: (bits >> i) & 1
                for i, var in enumerate(variables)
            }
            energy = sum(
                coefficient * sample[first] * sample[second]
                for (first, second), coefficient in qubo.items()
            )
            if energy < best_energy:
                best_energy = energy
                best_sample = sample
        return best_sample, best_energy



def _build_recursive_nested_test_graph() -> nx.MultiDiGraph:
    """Two serial cyclic children inside one larger parent block."""
    g = nx.MultiDiGraph()
    for edge_id, u, v in [
        ("s_p", "S", "P"),
        ("p_a", "P", "A"),
        ("a_b", "A", "B"),
        ("b_a", "B", "A"),
        ("b_q", "B", "Q"),
        ("q_c", "Q", "C"),
        ("c_d", "C", "D"),
        ("d_c", "D", "C"),
        ("d_r", "D", "R"),
        ("r_t", "R", "T"),
    ]:
        g.add_edge(u, v, key=edge_id, edge_id=edge_id, weight=1.0)
    return g


def _test_recursive_planb_f_keeps_nested_layers() -> None:
    """
    Parent B is too large for QA.

    f(B) must therefore:
      1. find a smaller child cycle and QA-solve it,
      2. find the second child cycle and QA-solve it,
      3. re-evaluate the parent, which is now DAG,
      4. classical-solve the parent.

    Children must receive lower layer numbers than the parent.
    """
    g = _build_recursive_nested_test_graph()
    normalized, records = normalize_olc_graph(g)
    hierarchy = GraphHierarchy(
        original_graph=normalized.copy(),
        current_graph=normalized.copy(),
        edge_records=records,
        graphs_by_layer={0: normalized.copy()},
    )
    ensure_original_edge_weights(hierarchy)
    for record in hierarchy.edge_records.values():
        if record.kind == "original":
            record.data["solve_status"] = "feasible"

    runtime = _RecursiveRuntime()
    evaluation_config = EvaluationConfig(
        target_top_qubo_variables=None,
        preferred_local_qubo_variables=64,
    )
    recursive_config = RecursivePlanBConfig(
        qa_variable_limit=8,
        min_region_nodes=2,
        max_region_nodes=2,
        max_recursion_depth=8,
        max_total_blocks=10,
        require_cycle_closed_regions=True,
    )

    result = planb_f(
        hierarchy,
        {"P", "A", "B", "Q", "C", "D", "R"},
        parent_scope_nodes=None,
        recursion_depth=1,
        max_input_ports=2,
        max_output_ports=2,
        evaluation_config=evaluation_config,
        priority_model=DefaultPriorityModel(),
        recursive_config=recursive_config,
        runtime=runtime,
        sampler=_ExactTinyQUBOSampler(),
        num_reads=1,
        seed=1,
        protected_nodes={"S", "T"},
    )

    assert result.success, result
    assert result.branch == "classical", result
    assert result.layer == 3, result
    assert runtime.qa_blocks == 2, runtime
    assert runtime.classical_blocks == 1, runtime
    assert runtime.max_depth == 2, runtime
    assert sorted(hierarchy.blocks_by_layer) == [1, 2, 3]

    branches = [
        hierarchy.blocks_by_layer[layer][0].evaluation_snapshot["recursive_branch"]
        for layer in [1, 2, 3]
    ]
    assert branches == ["qa", "qa", "classical"], branches

    visible = graph_edge_ids(hierarchy.current_graph)
    assert len(visible) == 1
    top_macro = hierarchy.edge_records[visible[0]]
    assert top_macro.kind == "macro"
    expanded = expand_selected_edges(hierarchy, visible)
    assert expanded == [
        "s_p", "p_a", "a_b", "b_q", "q_c", "c_d", "d_r", "r_t"
    ]
    assert "b_a" not in expanded and "d_c" not in expanded


def _test_recursive_pipeline_is_new_default() -> None:
    """The public full pipeline should now route through recursive Plan-B."""
    g = _build_recursive_nested_test_graph()
    result = run_full_olc_pipeline(
        g,
        max_input_ports=2,
        max_output_ports=2,
        source="S",
        target="T",
        sampler=_ExactTinyQUBOSampler(),
        preprocess_config=SafePreprocessConfig(contract_linear_unitigs=False),
        evaluation_config=EvaluationConfig(
            target_top_qubo_variables=None,
            preferred_local_qubo_variables=64,
        ),
        recursive_config=RecursivePlanBConfig(
            qa_variable_limit=8,
            min_region_nodes=2,
            max_region_nodes=2,
            max_total_blocks=10,
            require_cycle_closed_regions=True,
        ),
    )

    assert result.execution_mode == "recursive_planb"
    assert not result.adaptive_mode
    assert result.recursive_total_blocks >= 2
    assert result.recursive_qa_blocks == 2
    # The root selector is allowed to pack a cheap DAG bridge between the two
    # cyclic children before the second QA call.
    assert result.recursive_classical_blocks >= 0
    assert result.recursive_stop_reason == "PLANB_FINISHED_TOP_DAG_CLASSICAL"
    assert result.top_solution is not None and result.top_solution.feasible
    assert result.top_solution.solver_type == "classical"
    assert result.final_assembly is not None and result.final_assembly.feasible
    assert result.final_assembly.raw_edge_ids == [
        "s_p", "p_a", "a_b", "b_q", "q_c", "c_d", "d_r", "r_t"
    ]
    assert result.final_assembly.raw_node_path == [
        "S", "P", "A", "B", "Q", "C", "D", "R", "T"
    ]


def _test_initial_small_non_dag_is_direct_qa() -> None:
    """If the original non-DAG already fits QA, Plan B must not run."""
    g = nx.MultiDiGraph()
    for edge_id, u, v in [
        ("x_a", "X", "A"),
        ("a_b", "A", "B"),
        ("b_a", "B", "A"),
        ("b_y", "B", "Y"),
    ]:
        g.add_edge(u, v, key=edge_id, edge_id=edge_id, weight=1.0)

    result = run_full_olc_pipeline(
        g,
        max_input_ports=2,
        max_output_ports=2,
        source="X",
        target="Y",
        sampler=_ExactTinyQUBOSampler(),
        preprocess_config=SafePreprocessConfig(contract_linear_unitigs=False),
        evaluation_config=EvaluationConfig(
            target_top_qubo_variables=None,
            preferred_local_qubo_variables=64,
        ),
        recursive_config=RecursivePlanBConfig(
            qa_variable_limit=64,
            max_total_blocks=10,
        ),
    )

    assert result.recursive_total_blocks == 0
    assert result.recursive_stop_reason == "INITIAL_TOP_FITS_QA_DIRECT"
    assert result.top_solution is not None and result.top_solution.feasible
    assert result.top_solution.solver_type == "qa"
    assert result.final_assembly is not None and result.final_assembly.feasible
    assert result.final_assembly.raw_edge_ids == ["x_a", "a_b", "b_y"]


def _test_planb_ignores_later_top_qa_shortcut_until_dag() -> None:
    """Once Plan B starts, a later QA-sized top must not terminate DFS."""
    g = _build_recursive_nested_test_graph()
    result = run_full_olc_pipeline(
        g,
        max_input_ports=2,
        max_output_ports=2,
        source="S",
        target="T",
        sampler=_ExactTinyQUBOSampler(),
        preprocess_config=SafePreprocessConfig(contract_linear_unitigs=False),
        evaluation_config=EvaluationConfig(
            target_top_qubo_variables=None,
            preferred_local_qubo_variables=64,
        ),
        recursive_config=RecursivePlanBConfig(
            # Initial top Q=12, after the first cycle Q=11.
            # Therefore the second top decision fits QA, but Plan B must continue.
            qa_variable_limit=11,
            min_region_nodes=2,
            max_region_nodes=2,
            max_total_blocks=10,
            require_cycle_closed_regions=True,
        ),
    )

    assert result.recursive_stop_reason == "PLANB_FINISHED_TOP_DAG_CLASSICAL"
    assert result.final_graph_is_dag
    assert result.recursive_qa_blocks == 2
    assert result.top_solution is not None and result.top_solution.solver_type == "classical"
    assert any(
        item.get("action") == "top_qa_shortcut_ignored_after_planb"
        for item in result.hierarchy.compression_log
    )


def _test_safe_unitig_is_restored_in_final_answer() -> None:
    """Final answer must expand the safe preprocessing unitig too."""
    g = nx.MultiDiGraph()
    for edge_id, u, v in [
        ("s_a", "S", "A"),
        ("a_b", "A", "B"),
        ("b_t", "B", "T"),
    ]:
        g.add_edge(u, v, key=edge_id, edge_id=edge_id, weight=1.0)

    result = run_full_olc_pipeline(
        g,
        max_input_ports=2,
        max_output_ports=2,
        source="S",
        target="T",
        sampler=_ExactTinyQUBOSampler(),
        # default safe unitig contraction is intentionally ON
        evaluation_config=EvaluationConfig(target_top_qubo_variables=None),
    )

    assert result.top_solution is not None and result.top_solution.feasible
    assert result.top_solution.solver_type == "classical"
    assert result.final_assembly is not None and result.final_assembly.feasible
    assert result.final_assembly.raw_edge_ids == ["s_a", "a_b", "b_t"]
    assert result.final_assembly.raw_node_path == ["S", "A", "B", "T"]


def _test_adaptive_cycle_is_solved_then_pruned_to_dag() -> None:
    g = nx.MultiDiGraph()
    for edge_id, u, v in [
        ("x_a", "X", "A"),
        ("a_b", "A", "B"),
        ("b_a", "B", "A"),
        ("b_y", "B", "Y"),
    ]:
        g.add_edge(u, v, key=edge_id, edge_id=edge_id, weight=1.0)

    result = run_full_olc_pipeline(
        g,
        max_input_ports=2,
        max_output_ports=2,
        source="X",
        target="Y",
        sampler=_ExactTinyQUBOSampler(),
        recursive_planb=False,
        adaptive_cycle_elimination=True,
        evaluation_config=EvaluationConfig(
            target_top_qubo_variables=None,
            stop_if_top_graph_is_dag=True,
            preferred_local_qubo_variables=64,
        ),
        max_compression_steps=10,
        max_region_nodes=8,
    )

    assert result.adaptive_mode
    assert result.final_graph_is_dag, result.hierarchy.compression_log
    assert result.adaptive_iterations == 1, result.hierarchy.compression_log
    assert result.top_solution is not None and result.top_solution.feasible
    assert result.top_solution.solver_type == "classical"

    visible_macro = [
        result.hierarchy.edge_records[eid]
        for eid in graph_edge_ids(result.hierarchy.current_graph)
        if result.hierarchy.edge_records[eid].kind == "macro"
    ]
    assert len(visible_macro) == 1
    expanded = expand_selected_edges(
        result.hierarchy,
        result.top_solution.selected_actual_edge_ids,
    )
    assert set(expanded) == {"x_a", "a_b", "b_y"}
    assert "b_a" not in expanded


def _test_dag_solver_survives_incomplete_pruned_macro_group() -> None:
    # Construct a plain DAG problem with two macro alternatives from the same
    # block and mark the group incomplete.  The new exact DAG solver must not
    # fall back to QA merely because completeness was lost by pruning.
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(["S", "A", "B", "T"])
    edges = {
        "sa": PathEdge("sa", "S", "A", 1.0, "original", None),
        "m1": PathEdge("m1", "A", "B", 2.0, "macro", "B1"),
        "bt": PathEdge("bt", "B", "T", 1.0, "original", None),
    }
    for edge in edges.values():
        graph.add_edge(
            edge.u, edge.v, key=edge.edge_id,
            edge_id=edge.edge_id, weight=edge.weight,
            kind=edge.kind, block_id=edge.block_id,
        )

    problem = LocalPathProblem(
        solve_graph=graph,
        certificate_graph=graph.copy(),
        source="S",
        target="T",
        path_edges=edges,
        expected_child_blocks=frozenset({"B1"}),
        complete_macro_groups=False,
    )
    solution = solve_dag_path_problem(problem)
    assert solution.feasible, solution
    assert solution.selected_actual_edge_ids == ["sa", "m1", "bt"]



def _test_adaptive_prunes_infeasible_macro_before_dag_check() -> None:
    # One cyclic SCC has two possible exits.  Exactly one boundary pairing is
    # a Hamiltonian traversal; the other macro edge must be deleted before the
    # next DAG check.
    g = nx.MultiDiGraph()
    for edge_id, u, v in [
        ("s_p1", "S", "P1"),
        ("p1_a", "P1", "A"),
        ("a_b", "A", "B"),
        ("b_a", "B", "A"),
        ("b_q2", "B", "Q2"),
        ("q2_p2", "Q2", "P2"),
        ("p2_b", "P2", "B"),
        ("a_q1", "A", "Q1"),
        ("p2_q1_direct", "P2", "Q1"),
        ("q1_t", "Q1", "T"),
    ]:
        g.add_edge(u, v, key=edge_id, edge_id=edge_id, weight=1.0)

    result = run_full_olc_pipeline(
        g,
        max_input_ports=2,
        max_output_ports=2,
        source="S",
        target="T",
        sampler=_ExactTinyQUBOSampler(),
        recursive_planb=False,
        adaptive_cycle_elimination=True,
        preprocess_config=SafePreprocessConfig(contract_linear_unitigs=False),
        evaluation_config=EvaluationConfig(
            target_top_qubo_variables=None,
            preferred_local_qubo_variables=64,
        ),
        max_compression_steps=10,
        max_region_nodes=8,
    )

    assert result.final_graph_is_dag, result.hierarchy.compression_log
    assert result.adaptive_iterations == 1
    assert result.macro_batch is not None
    assert result.macro_batch.feasible_count == 1
    assert result.macro_batch.infeasible_count == 1

    log = result.hierarchy.compression_log[0]
    assert log["macro_total"] == 2
    assert log["macro_feasible"] == 1
    assert log["macro_infeasible"] == 1
    assert len(log["pruned_macro_edge_ids"]) == 1

    assert result.top_solution is not None
    assert result.top_solution.feasible
    assert result.top_solution.solver_type == "classical"


def _test_adaptive_large_scc_uses_nested_subregion_only_when_needed() -> None:
    # Whole SCC Q estimate is 9 (>8), while the inner A<->B cycle is 6.
    # Adaptive mode must solve A<->B first, recompute SCCs, then solve C<->D.
    g = nx.MultiDiGraph()
    for edge_id, u, v in [
        ("s_d", "S", "D"),
        ("a_b", "A", "B"),
        ("b_a", "B", "A"),
        ("b_c", "B", "C"),
        ("c_d", "C", "D"),
        ("d_a", "D", "A"),
        ("c_t", "C", "T"),
    ]:
        g.add_edge(u, v, key=edge_id, edge_id=edge_id, weight=1.0)

    result = run_full_olc_pipeline(
        g,
        max_input_ports=2,
        max_output_ports=2,
        source="S",
        target="T",
        sampler=_ExactTinyQUBOSampler(),
        recursive_planb=False,
        adaptive_cycle_elimination=True,
        preprocess_config=SafePreprocessConfig(contract_linear_unitigs=False),
        evaluation_config=EvaluationConfig(
            target_top_qubo_variables=None,
            preferred_local_qubo_variables=64,
        ),
        adaptive_config=AdaptiveCycleConfig(
            direct_qubo_variable_limit=8,
            min_subregion_nodes=2,
            max_subregion_nodes=2,
            max_iterations=10,
        ),
    )

    assert result.final_graph_is_dag, result.hierarchy.compression_log
    assert not result.global_infeasible
    assert result.adaptive_iterations == 2
    assert len(result.hierarchy.compression_log) == 2
    assert result.hierarchy.compression_log[0]["selection_kind"] == "SUBREGION_PLAN_B"
    assert result.hierarchy.compression_log[1]["selection_kind"] == "WHOLE_SCC"
    assert set(result.hierarchy.compression_log[0]["region_nodes"]) == {"A", "B"}
    assert set(result.hierarchy.compression_log[1]["region_nodes"]) == {"C", "D"}

    assert result.top_solution is not None and result.top_solution.feasible
    assert result.top_solution.solver_type == "classical"
    expanded = expand_selected_edges(
        result.hierarchy,
        result.top_solution.selected_actual_edge_ids,
    )
    assert expanded == ["s_d", "d_a", "a_b", "b_c", "c_t"]

def run_integrated_self_test() -> None:
    _test_safe_preprocess_integrated()
    _test_cycle_closure_expands_boundary()
    _test_cycle_closed_manual_candidate_is_auto_expanded()
    _test_existing_hybrid_dag_route()
    _test_recursive_planb_f_keeps_nested_layers()
    _test_recursive_pipeline_is_new_default()
    _test_initial_small_non_dag_is_direct_qa()
    _test_planb_ignores_later_top_qa_shortcut_until_dag()
    _test_safe_unitig_is_restored_in_final_answer()
    _test_adaptive_cycle_is_solved_then_pruned_to_dag()
    _test_adaptive_prunes_infeasible_macro_before_dag_check()
    _test_adaptive_large_scc_uses_nested_subregion_only_when_needed()
    _test_dag_solver_survives_incomplete_pruned_macro_group()
    print("ALL FULL-INTEGRATION TESTS PASSED")


if __name__ == "__main__":
    run_integrated_self_test()
