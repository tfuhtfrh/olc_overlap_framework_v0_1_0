"""Reference-free overlap-graph cleanup helpers."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from time import monotonic
from typing import Iterable, Literal

from .data import ContainmentEvidence, FullCoverageEvidence, OverlapEdge, Read
from .layout_solver import OverlapRewardScorer

MODULE_VERSION = "0.1.0"
OrientedRead = tuple[str, int]


@dataclass(frozen=True)
class OrientedSCCSelection:
    """The largest oriented strongly connected component of an overlap graph."""

    reads: list[Read]
    edges: list[OverlapEdge]
    oriented_nodes: frozenset[OrientedRead]
    physical_read_ids: frozenset[str]
    component_count: int


@dataclass(frozen=True)
class SingleOrientationSelection:
    """A physical-read graph with exactly one orientation per read."""

    reads: list[Read]
    edges: list[OverlapEdge]
    orientation_by_read: dict[str, int]
    retained_score: float
    rounds: int
    constraint_count: int = 0
    constraint_conflict_count: int = 0
    constraint_component_count: int = 0


@dataclass(frozen=True)
class ReadFilterResult:
    """Reads removed by an allowed node-level filter and remaining edges."""

    reads: list[Read]
    edges: list[OverlapEdge]
    removed_read_ids: frozenset[str]
    removal_reason_by_read: dict[str, str]


@dataclass(frozen=True)
class ReadOnlyGraphResult:
    """Final reads-only graph after node filtering and orientation projection."""

    reads: list[Read]
    edges: list[OverlapEdge]
    orientation_by_read: dict[str, int]
    full_coverage_read_ids: frozenset[str]
    contained_read_ids: frozenset[str]
    low_quality_read_ids: frozenset[str]
    removal_reason_by_read: dict[str, str]
    retained_score: float
    orientation_rounds: int
    orientation_constraint_count: int = 0
    orientation_constraint_conflict_count: int = 0
    orientation_constraint_component_count: int = 0


@dataclass(frozen=True)
class HamiltonCycleCheck:
    """Result of a graph-only Hamilton-cycle existence check."""

    status: Literal["yes", "no", "unknown_timeout"]
    node_count: int
    all_nodes_have_in_edge: bool
    all_nodes_have_out_edge: bool
    strongly_connected: bool
    cycle: tuple[OrientedRead, ...] = ()


def select_largest_oriented_scc(
    reads: list[Read],
    edges: list[OverlapEdge],
) -> OrientedSCCSelection:
    """Keep the largest bidirected SCC using only reads and overlap edges.

    This is intentionally a conservative graph-stage filter.  It does not use
    a reference, supplied truth cycle, or read-to-reference audit.  For a
    circular assembly, a valid oriented cycle must lie inside an SCC; the
    selected component can still contain branches and is therefore only a
    graph-stage diagnostic, not a solved layout.
    """
    read_by_id = {read.rid: read for read in reads}
    adjacency: dict[OrientedRead, set[OrientedRead]] = defaultdict(set)
    reverse: dict[OrientedRead, set[OrientedRead]] = defaultdict(set)
    for edge in edges:
        left = (edge.left_id, edge.left_orientation)
        right = (edge.right_id, edge.right_orientation)
        if left[0] not in read_by_id or right[0] not in read_by_id or left == right:
            continue
        adjacency[left].add(right)
        reverse[right].add(left)

    nodes = set(adjacency) | set(reverse)
    components = _strongly_connected_components(nodes, adjacency, reverse)
    nontrivial = [component for component in components if len(component) > 1]
    if not nontrivial:
        return OrientedSCCSelection(
            reads=[],
            edges=[],
            oriented_nodes=frozenset(),
            physical_read_ids=frozenset(),
            component_count=len(components),
        )

    read_rank = {read.rid: index for index, read in enumerate(reads)}
    selected = max(
        nontrivial,
        key=lambda component: (
            len({read_id for read_id, _ in component}),
            len(component),
            -min(read_rank.get(read_id, len(reads)) for read_id, _ in component),
        ),
    )
    physical_ids = frozenset(read_id for read_id, _ in selected)
    selected_reads = [read for read in reads if read.rid in physical_ids]
    selected_edges = [
        edge
        for edge in edges
        if (edge.left_id, edge.left_orientation) in selected
        and (edge.right_id, edge.right_orientation) in selected
    ]
    return OrientedSCCSelection(
        reads=selected_reads,
        edges=selected_edges,
        oriented_nodes=frozenset(selected),
        physical_read_ids=physical_ids,
        component_count=len(components),
    )


def select_one_orientation_per_read(
    reads: list[Read],
    edges: list[OverlapEdge],
    *,
    score_mode: str = "overlap_len_power2",
    max_rounds: int = 50,
) -> SingleOrientationSelection:
    """Choose one strand per read by propagating PAF relative-orientation constraints.

    An oriented overlap edge says that the selected orientations must satisfy
    ``orientation[right] = orientation[left] * right_orientation * left_orientation``.
    The constraints are undirected for orientation assignment: edge direction
    is retained separately, while the relative strand relation is propagated
    between the two physical reads.  A maximum-weight spanning forest supplies
    a deterministic assignment when redundant constraints contain conflicts;
    the conflicts are reported and incompatible directed edges are not kept.

    ``score_mode`` is used only to rank redundant constraints while constructing
    the forest.  It is not used as an independent orientation objective, and
    the reference is never consulted.
    """
    read_ids = [read.rid for read in reads]
    if not read_ids:
        return SingleOrientationSelection([], [], {}, 0.0, 0)

    read_id_set = set(read_ids)
    scorer = OverlapRewardScorer()
    scored_edges = [
        (edge, scorer.score(edge, score_mode))
        for edge in edges
        if edge.left_id in read_id_set
        and edge.right_id in read_id_set
        and edge.left_id != edge.right_id
    ]

    # Multiple PAF records can describe the same physical pair.  They carry
    # the same parity constraint in the normal case, so use the strongest
    # representative for the orientation forest while preserving all directed
    # overlap edges for the final compatibility filter.
    constraint_representatives: dict[
        tuple[str, str, int], tuple[OverlapEdge, float]
    ] = {}
    for edge, score in scored_edges:
        first, second = sorted((edge.left_id, edge.right_id))
        parity = edge.left_orientation * edge.right_orientation
        key = (first, second, parity)
        current = constraint_representatives.get(key)
        rank = (score, edge.overlap_len, edge.identity, edge.error_rate)
        if current is None:
            constraint_representatives[key] = (edge, score)
            continue
        old_edge, old_score = current
        old_rank = (
            old_score,
            old_edge.overlap_len,
            old_edge.identity,
            old_edge.error_rate,
        )
        if rank > old_rank:
            constraint_representatives[key] = (edge, score)

    # Build a maximum-weight spanning forest over physical read IDs.  The
    # forest, rather than all redundant edges, determines the orientation
    # assignment.  This makes contradictory cycles observable instead of
    # allowing traversal order to silently decide their result.
    parent = {read_id: read_id for read_id in read_ids}
    rank_by_root = {read_id: 0 for read_id in read_ids}

    def find(read_id: str) -> str:
        root = read_id
        while parent[root] != root:
            root = parent[root]
        while parent[read_id] != read_id:
            next_id = parent[read_id]
            parent[read_id] = root
            read_id = next_id
        return root

    def union(first: str, second: str) -> bool:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return False
        if rank_by_root[first_root] < rank_by_root[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        if rank_by_root[first_root] == rank_by_root[second_root]:
            rank_by_root[first_root] += 1
        return True

    forest: dict[str, list[tuple[str, int]]] = defaultdict(list)
    sorted_constraints = sorted(
        constraint_representatives.values(),
        key=lambda item: (
            -item[1],
            -item[0].overlap_len,
            -item[0].identity,
            item[0].left_id,
            item[0].right_id,
            item[0].left_orientation,
            item[0].right_orientation,
        ),
    )
    for edge, _ in sorted_constraints:
        parity = edge.left_orientation * edge.right_orientation
        if union(edge.left_id, edge.right_id):
            forest[edge.left_id].append((edge.right_id, parity))
            forest[edge.right_id].append((edge.left_id, parity))

    orientations = {}
    for seed in sorted(read_ids):
        if seed in orientations:
            continue
        orientations[seed] = +1
        queue = deque([seed])
        while queue:
            current = queue.popleft()
            for neighbor, parity in sorted(forest[current]):
                if neighbor not in orientations:
                    orientations[neighbor] = orientations[current] * parity
                    queue.append(neighbor)

    # Relative constraints leave one global +/- phase per physical component.
    # Pick the phase that keeps the stronger compatible directed evidence.  A
    # phase flip is not an independent scoring-based orientation decision: it
    # is the unavoidable reverse-complement symmetry of the same assignment.
    components: dict[str, list[str]] = defaultdict(list)
    for read_id in read_ids:
        components[find(read_id)].append(read_id)
    for component_read_ids in components.values():
        component_ids = set(component_read_ids)
        phase_scores = {+1: 0.0, -1: 0.0}
        phase_edge_counts = {+1: 0, -1: 0}
        phase_incoming: dict[int, set[str]] = {+1: set(), -1: set()}
        phase_outgoing: dict[int, set[str]] = {+1: set(), -1: set()}
        for edge, score in scored_edges:
            if edge.left_id not in component_ids or edge.right_id not in component_ids:
                continue
            for phase in (+1, -1):
                if (
                    phase * orientations[edge.left_id] == edge.left_orientation
                    and phase * orientations[edge.right_id] == edge.right_orientation
                ):
                    phase_scores[phase] += score
                    phase_edge_counts[phase] += 1
                    phase_outgoing[phase].add(edge.left_id)
                    phase_incoming[phase].add(edge.right_id)
        phase = max(
            (+1, -1),
            key=lambda value: (
                len(phase_incoming[value] & phase_outgoing[value]),
                len(phase_incoming[value] | phase_outgoing[value]),
                phase_scores[value],
                phase_edge_counts[value],
                value == +1,
            ),
        )
        if phase == -1:
            for read_id in component_read_ids:
                orientations[read_id] *= -1

    constraint_conflicts = 0
    for edge, _ in sorted_constraints:
        parity = edge.left_orientation * edge.right_orientation
        if orientations[edge.left_id] * orientations[edge.right_id] != parity:
            constraint_conflicts += 1

    retained_edges = [
        edge
        for edge, _ in scored_edges
        if orientations[edge.left_id] == edge.left_orientation
        and orientations[edge.right_id] == edge.right_orientation
    ]
    retained_score = sum(
        score
        for edge, score in scored_edges
        if orientations[edge.left_id] == edge.left_orientation
        and orientations[edge.right_id] == edge.right_orientation
    )
    return SingleOrientationSelection(
        reads=list(reads),
        edges=retained_edges,
        orientation_by_read=orientations,
        retained_score=retained_score,
        rounds=1,
        constraint_count=len(sorted_constraints),
        constraint_conflict_count=constraint_conflicts,
        constraint_component_count=len({find(read_id) for read_id in read_ids}),
    )


def remove_contained_reads(
    reads: list[Read],
    edges: list[OverlapEdge],
    containment_evidence: Iterable[ContainmentEvidence],
) -> ReadFilterResult:
    """Remove only reads with explicit full-read/internal-target evidence.

    This is a node filter, not a graph reduction.  All edges between retained
    reads are preserved, including transitive edges and branches.
    """
    read_by_id = {read.rid: read for read in reads}
    best_parent: dict[str, tuple[tuple[float, int, int, str], str]] = {}
    for evidence in containment_evidence:
        contained = read_by_id.get(evidence.contained_id)
        container = read_by_id.get(evidence.container_id)
        if contained is None or container is None or contained.rid == container.rid:
            continue
        if len(contained.seq) >= len(container.seq):
            continue
        score = (
            evidence.identity,
            evidence.alignment_length,
            len(container.seq),
            container.rid,
        )
        old = best_parent.get(contained.rid)
        if old is None or score > old[0]:
            best_parent[contained.rid] = (score, container.rid)

    removed = frozenset(best_parent)
    kept_ids = set(read_by_id) - removed
    return ReadFilterResult(
        reads=[read for read in reads if read.rid in kept_ids],
        edges=[
            edge
            for edge in edges
            if edge.left_id in kept_ids and edge.right_id in kept_ids
        ],
        removed_read_ids=removed,
        removal_reason_by_read={
            read_id: f"contained_in:{container_id}"
            for read_id, (_, container_id) in best_parent.items()
        },
    )


def remove_full_coverage_reads(
    reads: list[Read],
    edges: list[OverlapEdge],
    evidence: Iterable[FullCoverageEvidence],
) -> ReadFilterResult:
    """Keep one representative for each exact full-coverage read group.

    Equal-length end-to-end reads are included because they create redundant
    nodes without a positive-shift OLC edge.  Evidence is reference-free and
    must use exact PAF endpoints; no terminal tolerance is applied.
    """
    read_by_id = {read.rid: read for read in reads}
    parent = {read.rid: read.rid for read in reads}

    def find(read_id: str) -> str:
        root = read_id
        while parent[root] != root:
            root = parent[root]
        while parent[read_id] != read_id:
            next_id = parent[read_id]
            parent[read_id] = root
            read_id = next_id
        return root

    for item in evidence:
        if item.first_id not in read_by_id or item.second_id not in read_by_id:
            continue
        first_root = find(item.first_id)
        second_root = find(item.second_id)
        if first_root != second_root:
            parent[second_root] = first_root

    groups: dict[str, list[str]] = defaultdict(list)
    for read_id in read_by_id:
        groups[find(read_id)].append(read_id)

    incident = {read_id: 0 for read_id in read_by_id}
    for edge in edges:
        if edge.left_id in incident and edge.right_id in incident:
            incident[edge.left_id] += 1
            incident[edge.right_id] += 1

    removed_reason: dict[str, str] = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        representative = max(
            group,
            key=lambda read_id: (
                len(read_by_id[read_id].seq),
                incident[read_id],
                read_id,
            ),
        )
        for read_id in group:
            if read_id != representative:
                removed_reason[read_id] = f"full_coverage_with:{representative}"

    removed = frozenset(removed_reason)
    kept_ids = set(read_by_id) - removed
    return ReadFilterResult(
        reads=[read for read in reads if read.rid in kept_ids],
        edges=[
            edge
            for edge in edges
            if edge.left_id in kept_ids and edge.right_id in kept_ids
        ],
        removed_read_ids=removed,
        removal_reason_by_read=removed_reason,
    )


def remove_low_quality_reads(
    reads: list[Read],
    edges: list[OverlapEdge],
    *,
    min_in_support: int = 1,
    min_out_support: int = 1,
) -> ReadFilterResult:
    """Remove reads lacking sufficient accepted incoming/outgoing support.

    The default is only a high-quality-support screen: at least one incoming
    and one outgoing accepted overlap in the current graph.  It does not cap
    either degree and does not remove branches or transitive edges.  The
    filter is deliberately one-pass; later zero-degree nodes are reported by
    the Hamilton checker rather than recursively removed as graph topology.
    """
    if min_in_support < 0 or min_out_support < 0:
        raise ValueError("minimum support counts must be non-negative")

    read_by_id = {read.rid: read for read in reads}
    active_ids = set(read_by_id)
    incoming: dict[str, set[tuple[str, int, int]]] = {
        read_id: set() for read_id in active_ids
    }
    outgoing: dict[str, set[tuple[str, int, int]]] = {
        read_id: set() for read_id in active_ids
    }
    for edge in edges:
        if edge.left_id not in active_ids or edge.right_id not in active_ids:
            continue
        outgoing[edge.left_id].add(
            (edge.right_id, edge.left_orientation, edge.right_orientation)
        )
        incoming[edge.right_id].add(
            (edge.left_id, edge.left_orientation, edge.right_orientation)
        )

    reasons: dict[str, str] = {}
    for read_id in sorted(active_ids):
        has_in = len(incoming[read_id]) >= min_in_support
        has_out = len(outgoing[read_id]) >= min_out_support
        if has_in and has_out:
            continue
        if not has_in and not has_out:
            reason = "low_quality:no_in_or_out_support"
        elif not has_in:
            reason = "low_quality:no_in_support"
        else:
            reason = "low_quality:no_out_support"
        reasons[read_id] = reason
    active_ids -= reasons.keys()

    return ReadFilterResult(
        reads=[read for read in reads if read.rid in active_ids],
        edges=[
            edge
            for edge in edges
            if edge.left_id in active_ids and edge.right_id in active_ids
        ],
        removed_read_ids=frozenset(reasons),
        removal_reason_by_read=reasons,
    )


def build_read_only_graph(
    reads: list[Read],
    edges: list[OverlapEdge],
    containment_evidence: Iterable[ContainmentEvidence] = (),
    *,
    min_in_support: int = 1,
    min_out_support: int = 1,
    score_mode: str = "overlap_len_power2",
    max_orientation_rounds: int = 50,
    apply_low_quality_filter: bool = False,
    full_coverage_evidence: Iterable[FullCoverageEvidence] = (),
    apply_full_coverage_filter: bool = True,
) -> ReadOnlyGraphResult:
    """Apply the approved node-only graph preprocessing policy.

    Orientation projection is performed first from PAF relative-direction
    constraints.  Exact full-coverage groups are then reduced to one physical
    representative, followed by strict containment removal.  The one-pass
    low-support filter is optional and disabled by default, so unsupported
    reads remain available for diagnosis.  No SCC selection, transitive
    reduction, or exact-degree enforcement is performed here.
    """
    orientation = select_one_orientation_per_read(
        reads,
        edges,
        score_mode=score_mode,
        max_rounds=max_orientation_rounds,
    )
    if apply_full_coverage_filter:
        full_coverage = remove_full_coverage_reads(
            orientation.reads,
            orientation.edges,
            full_coverage_evidence,
        )
    else:
        full_coverage = ReadFilterResult(
            reads=orientation.reads,
            edges=orientation.edges,
            removed_read_ids=frozenset(),
            removal_reason_by_read={},
        )
    containment = remove_contained_reads(
        full_coverage.reads,
        full_coverage.edges,
        containment_evidence,
    )
    if apply_low_quality_filter:
        low_quality = remove_low_quality_reads(
            containment.reads,
            containment.edges,
            min_in_support=min_in_support,
            min_out_support=min_out_support,
        )
    else:
        low_quality = ReadFilterResult(
            reads=containment.reads,
            edges=containment.edges,
            removed_read_ids=frozenset(),
            removal_reason_by_read={},
        )
    kept_ids = {read.rid for read in low_quality.reads}
    orientations = {
        read_id: orientation_value
        for read_id, orientation_value in orientation.orientation_by_read.items()
        if read_id in kept_ids
    }
    reasons = {
        **full_coverage.removal_reason_by_read,
        **containment.removal_reason_by_read,
        **low_quality.removal_reason_by_read,
    }
    return ReadOnlyGraphResult(
        reads=low_quality.reads,
        edges=low_quality.edges,
        orientation_by_read=orientations,
        full_coverage_read_ids=full_coverage.removed_read_ids,
        contained_read_ids=containment.removed_read_ids,
        low_quality_read_ids=low_quality.removed_read_ids,
        removal_reason_by_read=reasons,
        retained_score=orientation.retained_score,
        orientation_rounds=orientation.rounds,
        orientation_constraint_count=orientation.constraint_count,
        orientation_constraint_conflict_count=orientation.constraint_conflict_count,
        orientation_constraint_component_count=orientation.constraint_component_count,
    )


def check_hamilton_cycle(
    nodes: Iterable[OrientedRead],
    edges: list[OverlapEdge],
    *,
    time_limit_sec: float = 120.0,
) -> HamiltonCycleCheck:
    """Check for a directed Hamilton cycle without imposing degree one.

    Zero in/out degree and a non-single SCC are conclusive negative tests.
    For a branching graph, a perfect cycle cover plus legal two-edge merges is
    tried first; a successful merge is an exact witness.  Otherwise an exact
    depth-first search is attempted until the time limit, and timeout is
    reported separately from infeasibility.
    """
    node_list = list(dict.fromkeys(nodes))
    node_set = set(node_list)
    adjacency: dict[OrientedRead, set[OrientedRead]] = {
        node: set() for node in node_list
    }
    reverse: dict[OrientedRead, set[OrientedRead]] = {
        node: set() for node in node_list
    }
    edge_rank: dict[tuple[OrientedRead, OrientedRead], tuple[int, int, float]] = {}
    for edge in edges:
        left = (edge.left_id, edge.left_orientation)
        right = (edge.right_id, edge.right_orientation)
        if left in node_set and right in node_set:
            adjacency[left].add(right)
            reverse[right].add(left)
            rank = (max(0, edge.shift), -edge.overlap_len, -edge.identity)
            key = (left, right)
            if key not in edge_rank or rank < edge_rank[key]:
                edge_rank[key] = rank

    all_in = all(reverse[node] for node in node_list)
    all_out = all(adjacency[node] for node in node_list)
    if not node_list or not all_in or not all_out:
        return HamiltonCycleCheck(
            status="no",
            node_count=len(node_list),
            all_nodes_have_in_edge=all_in,
            all_nodes_have_out_edge=all_out,
            strongly_connected=False,
        )

    components = _strongly_connected_components(node_set, adjacency, reverse)
    strongly_connected = len(components) == 1
    if not strongly_connected:
        return HamiltonCycleCheck(
            status="no",
            node_count=len(node_list),
            all_nodes_have_in_edge=all_in,
            all_nodes_have_out_edge=all_out,
            strongly_connected=False,
        )

    if all(len(adjacency[node]) == 1 and len(reverse[node]) == 1 for node in node_list):
        start = node_list[0]
        cycle = [start]
        current = start
        for _ in range(len(node_list)):
            current = next(iter(adjacency[current]))
            cycle.append(current)
        if current == start and len(set(cycle[:-1])) == len(node_list):
            return HamiltonCycleCheck(
                status="yes",
                node_count=len(node_list),
                all_nodes_have_in_edge=True,
                all_nodes_have_out_edge=True,
                strongly_connected=True,
                cycle=tuple(cycle),
            )

    # A bipartite perfect matching gives a directed cycle cover (one selected
    # incoming and outgoing edge per node).  Merge disjoint cycles whenever a
    # two-edge exchange exists.  A single merged cycle is an exact Hamilton
    # witness; failure to merge is inconclusive and falls through to DFS.
    cycle_cover = _mergeable_cycle_cover(node_list, adjacency, edge_rank)
    if cycle_cover is not None:
        return HamiltonCycleCheck(
            status="yes",
            node_count=len(node_list),
            all_nodes_have_in_edge=True,
            all_nodes_have_out_edge=True,
            strongly_connected=True,
            cycle=tuple(cycle_cover),
        )

    start = min(node_list, key=lambda node: (len(adjacency[node]), str(node)))
    path = [start]
    visited = {start}
    deadline = monotonic() + max(0.0, time_limit_sec)
    timed_out = False

    def search(current: OrientedRead):
        nonlocal timed_out
        if monotonic() >= deadline:
            timed_out = True
            return None
        if len(path) == len(node_list):
            return list(path) if start in adjacency[current] else None
        candidates = [node for node in adjacency[current] if node not in visited]
        # OLC edges with a shorter positive shift usually connect immediate
        # layout neighbors.  Trying those first changes only DFS search order,
        # not the exact Hamilton-cycle criterion.
        candidates.sort(key=lambda node: (
            edge_rank.get((current, node), (10**18, 0, 0.0)),
            len(adjacency[node] - visited),
            str(node),
        ))
        for neighbor in candidates:
            visited.add(neighbor)
            path.append(neighbor)
            result = search(neighbor)
            if result is not None:
                return result
            path.pop()
            visited.remove(neighbor)
            if timed_out:
                return None
        return None

    result = search(start)
    if result is None:
        return HamiltonCycleCheck(
            status="unknown_timeout" if timed_out else "no",
            node_count=len(node_list),
            all_nodes_have_in_edge=all_in,
            all_nodes_have_out_edge=all_out,
            strongly_connected=True,
        )
    return HamiltonCycleCheck(
        status="yes",
        node_count=len(node_list),
        all_nodes_have_in_edge=True,
        all_nodes_have_out_edge=True,
        strongly_connected=True,
        cycle=tuple(result + [start]),
    )


def _mergeable_cycle_cover(
    nodes: list[OrientedRead],
    adjacency: dict[OrientedRead, set[OrientedRead]],
    edge_rank: dict[tuple[OrientedRead, OrientedRead], tuple[int, int, float]],
) -> list[OrientedRead] | None:
    """Return a Hamilton cycle when a perfect cycle cover can be merged."""
    ordered_neighbors = {
        node: sorted(
            adjacency[node],
            key=lambda neighbor: (
                edge_rank.get((node, neighbor), (10**18, 0, 0.0)),
                str(neighbor),
            ),
        )
        for node in nodes
    }
    matched_left_by_right: dict[OrientedRead, OrientedRead] = {}

    def augment(left: OrientedRead, seen: set[OrientedRead]) -> bool:
        for right in ordered_neighbors[left]:
            if right in seen:
                continue
            seen.add(right)
            previous_left = matched_left_by_right.get(right)
            if previous_left is None or augment(previous_left, seen):
                matched_left_by_right[right] = left
                return True
        return False

    for left in sorted(nodes, key=lambda node: (len(adjacency[node]), str(node))):
        if not augment(left, set()):
            return None

    successor = {
        left: right for right, left in matched_left_by_right.items()
    }

    def cycles() -> list[list[OrientedRead]]:
        unseen = set(nodes)
        result: list[list[OrientedRead]] = []
        while unseen:
            start = min(unseen, key=str)
            cycle = []
            current = start
            while current in unseen:
                unseen.remove(current)
                cycle.append(current)
                current = successor[current]
            result.append(cycle)
        return result

    components = cycles()
    while len(components) > 1:
        merged = False
        for first_index, first in enumerate(components):
            for second in components[first_index + 1:]:
                for left in first:
                    left_next = successor[left]
                    for right in second:
                        right_next = successor[right]
                        if right_next in adjacency[left] and left_next in adjacency[right]:
                            successor[left], successor[right] = right_next, left_next
                            merged = True
                            break
                    if merged:
                        break
                if merged:
                    break
            if merged:
                break
        if not merged:
            return None
        components = cycles()

    start = min(nodes, key=str)
    cycle = [start]
    current = successor[start]
    while current != start:
        if current in cycle:
            return None
        cycle.append(current)
        current = successor[current]
    return cycle + [start] if len(cycle) == len(nodes) else None


def _strongly_connected_components(
    nodes: set[OrientedRead],
    adjacency: dict[OrientedRead, set[OrientedRead]],
    reverse: dict[OrientedRead, set[OrientedRead]],
) -> list[set[OrientedRead]]:
    """Iterative Kosaraju SCC implementation with no graph dependency."""
    visited: set[OrientedRead] = set()
    finish_order: list[OrientedRead] = []
    for start in nodes:
        if start in visited:
            continue
        stack: list[tuple[OrientedRead, bool]] = [(start, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                finish_order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for neighbor in adjacency.get(node, ()):
                if neighbor not in visited:
                    stack.append((neighbor, False))

    components: list[set[OrientedRead]] = []
    assigned: set[OrientedRead] = set()
    for start in reversed(finish_order):
        if start in assigned:
            continue
        component: set[OrientedRead] = set()
        stack = [start]
        assigned.add(start)
        while stack:
            node = stack.pop()
            component.add(node)
            for neighbor in reverse.get(node, ()):
                if neighbor not in assigned:
                    assigned.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components
