from __future__ import annotations

"""
Hierarchical OLC graph compressor
=================================

This module integrates twelve OLC graph-reduction methods and an optional
Plan-B decomposition for quantum-annealing workflows.

Core principle
--------------
The compressor does not irreversibly discard the internal structure of
bubble/SCC/repeat/partition blocks.  A compressed macro node stores its
original internal graph, while boundary edges store hierarchical port paths.

Implemented stages
------------------
1.  Duplicate-read merging
2.  Contained-read absorption
3.  Edge-quality filtering
4.  Internal-match removal
5.  Parallel-edge reduction
6.  Optional top-k pruning
7.  Two-step transitive-edge removal
8.  Unitig contraction
9.  Optional tip removal
10. Equivalent-bubble contraction
11. Hierarchical SCC contraction
12. Optional hierarchical repeat-like contraction

Plan B
------
If the compressed macro graph is still larger than ``max_block_size``, the
code searches for a two-terminal region S satisfying

    |delta^-(S)| = 1,  |delta^+(S)| = 1,

and contracts S to one partition block.  The unique entering and leaving
edges determine the start/end boundary conditions of the later subproblem.
The same operation is applied recursively to large internal blocks.

This module compresses and decomposes the graph.  It does not itself build or
solve a QUBO.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple
import copy


PortPath = Tuple[str, ...]
EdgeSignature = Tuple[Any, ...]


# ============================================================
# Data structures
# ============================================================

@dataclass
class Node:
    id: str
    seq: str = ""
    coverage: float = 1.0
    members: Tuple[str, ...] = field(default_factory=tuple)

    # read, unitig, bubble_block, scc_block,
    # repeat_block, partition_block, cycle_segment
    kind: str = "read"


@dataclass
class Edge:
    src: str
    dst: str
    overlap: int
    identity: float = 1.0
    coverage: float = 1.0
    suffix_prefix: bool = True

    # shift(i,j) = len(R_i) - overlap(i,j)
    shift: Optional[int] = None

    # Number of new bases added by appending dst after src.
    cost: Optional[int] = None

    label: str = ""

    # Hierarchical port paths.  Empty means the edge touches the macro node
    # itself.  Example: dst_port=("SCC_1", "read_7") means that after opening
    # nested blocks the edge enters read_7 through SCC_1.
    src_port: PortPath = field(default_factory=tuple)
    dst_port: PortPath = field(default_factory=tuple)


@dataclass
class OLCGraph:
    nodes: Dict[str, Node]
    edges: List[Edge]

    # Each key is also a macro-node id in ``nodes``.
    blocks: Dict[str, "GraphBlock"] = field(default_factory=dict)

    def copy(self) -> "OLCGraph":
        return copy.deepcopy(self)

    def out_edges(self) -> Dict[str, List[Edge]]:
        out: Dict[str, List[Edge]] = defaultdict(list)
        for edge in self.edges:
            out[edge.src].append(edge)
        return out

    def in_edges(self) -> Dict[str, List[Edge]]:
        inn: Dict[str, List[Edge]] = defaultdict(list)
        for edge in self.edges:
            inn[edge.dst].append(edge)
        return inn

    def remove_nodes(self, remove_set: Set[str]) -> "OLCGraph":
        nodes = {
            node_id: copy.deepcopy(node)
            for node_id, node in self.nodes.items()
            if node_id not in remove_set
        }
        edges = [
            copy.deepcopy(edge)
            for edge in self.edges
            if edge.src not in remove_set and edge.dst not in remove_set
        ]
        blocks = {
            block_id: copy.deepcopy(block)
            for block_id, block in self.blocks.items()
            if block_id not in remove_set
        }
        return OLCGraph(nodes=nodes, edges=edges, blocks=blocks)

    def induced_subgraph(self, vertex_set: Set[str]) -> "OLCGraph":
        nodes = {
            node_id: copy.deepcopy(self.nodes[node_id])
            for node_id in vertex_set
        }
        edges = [
            copy.deepcopy(edge)
            for edge in self.edges
            if edge.src in vertex_set and edge.dst in vertex_set
        ]
        blocks = {
            block_id: copy.deepcopy(self.blocks[block_id])
            for block_id in vertex_set
            if block_id in self.blocks
        }
        return OLCGraph(nodes=nodes, edges=edges, blocks=blocks)


@dataclass
class GraphBlock:
    """A reversible hierarchical macro node."""

    id: str
    kind: str
    internal_graph: OLCGraph

    # Original boundary edges before contraction.
    boundary_in_edges: List[Edge] = field(default_factory=list)
    boundary_out_edges: List[Edge] = field(default_factory=list)

    # Immediate internal vertices touched by boundary edges.
    entry_vertices: Set[str] = field(default_factory=set)
    exit_vertices: Set[str] = field(default_factory=set)

    # Allowed immediate (entry, exit) pairs when known.
    # Used especially for equivalent bubbles.
    feasible_port_pairs: Set[Tuple[str, str]] = field(default_factory=set)

    # Alternative internal paths, stored as immediate node-id sequences.
    alternatives: List[Tuple[str, ...]] = field(default_factory=list)

    # For an equivalent bubble this is the common spelled sequence.
    canonical_sequence: str = ""

    # For a simple directed cycle, one cyclic ordering of the vertices.
    cycle_order: Tuple[str, ...] = field(default_factory=tuple)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionStats:
    initial_nodes: int = 0
    initial_edges: int = 0
    final_nodes: int = 0
    final_edges: int = 0

    stage_nodes: Dict[str, int] = field(default_factory=dict)
    stage_edges: Dict[str, int] = field(default_factory=dict)

    duplicate_nodes_merged: int = 0
    contained_nodes_absorbed: int = 0
    quality_edges_removed: int = 0
    internal_edges_removed: int = 0
    parallel_edges_removed: int = 0
    top_k_edges_removed: int = 0
    transitive_edges_removed: int = 0
    unitig_merges: int = 0
    tip_nodes_removed: int = 0
    bubble_blocks_created: int = 0
    scc_blocks_created: int = 0
    repeat_blocks_created: int = 0
    partition_blocks_created: int = 0

    unresolved_large_blocks: List[str] = field(default_factory=list)


@dataclass
class CompressionResult:
    graph: OLCGraph
    absorbed_reads: Dict[str, str]
    stats: CompressionStats


@dataclass
class CompressConfig:
    # --------------------------------------------------------
    # 1-8: relatively safe reductions
    # --------------------------------------------------------
    remove_duplicates: bool = True
    remove_contained_reads: bool = True
    remove_internal_matches: bool = True
    reduce_multi_edges: bool = True
    remove_transitive_edges: bool = True
    contract_unitigs: bool = True

    # Quality filtering
    min_overlap: int = 0
    min_identity: float = 0.0
    remove_low_coverage_edges: bool = False
    min_edge_coverage: float = 1.0

    # Optional sparsification
    top_k_out: Optional[int] = None
    top_k_in: Optional[int] = None

    # Transitive reduction tolerance
    eps_shift: int = 0

    # Unitig settings
    contract_cycles: bool = False
    contract_macro_nodes: bool = False

    # --------------------------------------------------------
    # 9-12: structural reductions
    # --------------------------------------------------------
    remove_tips: bool = False
    tip_max_len: int = 500
    min_tip_coverage: float = 1.5

    # Exact-equivalent bubbles are contracted; unequal bubbles remain.
    contract_equivalent_bubbles: bool = True
    bubble_max_path_nodes: int = 100
    bubble_max_spelled_length: int = 2000

    # SCCs are represented by hierarchical macro nodes.
    compress_scc: bool = True
    scc_min_size: int = 2

    # Heuristic repeat-like blocks; disabled by default.
    compress_repeat_blocks: bool = False
    repeat_degree_threshold: int = 10

    # --------------------------------------------------------
    # Plan B: two-terminal decomposition
    # --------------------------------------------------------
    partition_large_graph: bool = True
    max_block_size: int = 50
    min_partition_size: int = 2
    max_partition_depth: int = 32


# ============================================================
# Validation and general helpers
# ============================================================

def normalize_port_path(value: Any) -> PortPath:
    if value is None:
        return tuple()
    if isinstance(value, tuple):
        return tuple(str(x) for x in value)
    if isinstance(value, list):
        return tuple(str(x) for x in value)
    if value == "":
        return tuple()
    return (str(value),)


def edge_signature(edge: Edge) -> EdgeSignature:
    """Signature used for port-aware parallel-edge reduction."""
    return (
        edge.src,
        edge.dst,
        normalize_port_path(edge.src_port),
        normalize_port_path(edge.dst_port),
        edge.label,
    )


def edge_instance_signature(edge: Edge) -> EdgeSignature:
    """A more detailed signature useful for boundary comparisons."""
    return edge_signature(edge) + (
        edge.overlap,
        edge.identity,
        edge.coverage,
        edge.shift,
        edge.cost,
        edge.suffix_prefix,
    )


def unique_block_id(graph: OLCGraph, prefix: str) -> str:
    index = 1
    while True:
        block_id = f"{prefix}_{index}"
        if block_id not in graph.nodes and block_id not in graph.blocks:
            return block_id
        index += 1


def node_members(node: Node) -> Tuple[str, ...]:
    return node.members if node.members else (node.id,)


def aggregate_members(nodes: Iterable[Node]) -> Tuple[str, ...]:
    result: Tuple[str, ...] = tuple()
    for node in nodes:
        result += node_members(node)
    return result


def weighted_average_coverage(nodes: Iterable[Node]) -> float:
    weighted_sum = 0.0
    total_weight = 0
    for node in nodes:
        weight = max(1, len(node_members(node)))
        weighted_sum += node.coverage * weight
        total_weight += weight
    return weighted_sum / max(1, total_weight)


def ensure_edge_fields(graph: OLCGraph) -> None:
    for edge in graph.edges:
        edge.src_port = normalize_port_path(edge.src_port)
        edge.dst_port = normalize_port_path(edge.dst_port)

        src_node = graph.nodes.get(edge.src)
        dst_node = graph.nodes.get(edge.dst)

        if src_node is not None and src_node.seq and edge.shift is None:
            edge.shift = len(src_node.seq) - edge.overlap

        if dst_node is not None and dst_node.seq and edge.cost is None:
            edge.cost = len(dst_node.seq) - edge.overlap


def validate_graph(graph: OLCGraph) -> None:
    for edge in graph.edges:
        if edge.src not in graph.nodes:
            raise ValueError(f"Edge source does not exist: {edge.src}")
        if edge.dst not in graph.nodes:
            raise ValueError(f"Edge destination does not exist: {edge.dst}")

    for block_id, block in graph.blocks.items():
        if block_id not in graph.nodes:
            raise ValueError(f"Block {block_id} has no macro node")
        validate_graph(block.internal_graph)


def record_stage(stats: CompressionStats, stage: str, graph: OLCGraph) -> None:
    stats.stage_nodes[stage] = len(graph.nodes)
    stats.stage_edges[stage] = len(graph.edges)


def edge_score(graph: OLCGraph, edge: Edge) -> float:
    cost = edge.cost if edge.cost is not None else 0
    return (
        1000.0 * edge.identity
        + 1.0 * edge.overlap
        + 0.1 * edge.coverage
        - 0.01 * cost
    )


def spell_concat(seq_a: str, seq_b: str, overlap: int) -> str:
    if not seq_a:
        return seq_b
    if not seq_b:
        return seq_a
    if overlap <= 0:
        return seq_a + seq_b
    if overlap > min(len(seq_a), len(seq_b)):
        raise ValueError(
            f"Invalid overlap={overlap} for sequence lengths "
            f"{len(seq_a)} and {len(seq_b)}"
        )
    return seq_a + seq_b[overlap:]


def spell_path(
    graph: OLCGraph,
    node_path: Sequence[str],
    edge_path: Sequence[Edge],
) -> str:
    if not node_path:
        return ""
    if len(edge_path) != len(node_path) - 1:
        raise ValueError("edge_path length must be len(node_path)-1")

    sequence = graph.nodes[node_path[0]].seq
    if not sequence:
        return ""

    for index, edge in enumerate(edge_path):
        next_sequence = graph.nodes[node_path[index + 1]].seq
        if not next_sequence:
            return ""
        sequence = spell_concat(sequence, next_sequence, edge.overlap)

    return sequence


def weak_components(
    graph: OLCGraph,
    excluded_edge_indices: Optional[Set[int]] = None,
) -> List[Set[str]]:
    excluded = excluded_edge_indices or set()
    adjacency: Dict[str, Set[str]] = defaultdict(set)

    for index, edge in enumerate(graph.edges):
        if index in excluded:
            continue
        adjacency[edge.src].add(edge.dst)
        adjacency[edge.dst].add(edge.src)

    components: List[Set[str]] = []
    seen: Set[str] = set()

    for start in graph.nodes:
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        component: Set[str] = set()

        while queue:
            vertex = queue.popleft()
            component.add(vertex)
            for neighbor in adjacency.get(vertex, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)

        components.append(component)

    return components


def path_exists(
    graph: OLCGraph,
    start: str,
    target: str,
    forbidden_signature: Optional[EdgeSignature] = None,
) -> bool:
    out = graph.out_edges()
    queue = deque([start])
    seen = {start}

    while queue:
        vertex = queue.popleft()
        if vertex == target:
            return True

        for edge in out.get(vertex, []):
            if (
                forbidden_signature is not None
                and edge_instance_signature(edge) == forbidden_signature
            ):
                continue
            if edge.dst not in seen:
                seen.add(edge.dst)
                queue.append(edge.dst)

    return False


# ============================================================
# Generic hierarchical block contraction
# ============================================================

def contract_vertex_set_to_block(
    graph: OLCGraph,
    vertex_set: Set[str],
    *,
    block_id: str,
    kind: str,
    feasible_port_pairs: Optional[Set[Tuple[str, str]]] = None,
    alternatives: Optional[List[Tuple[str, ...]]] = None,
    canonical_sequence: str = "",
    cycle_order: Sequence[str] = (),
    metadata: Optional[Dict[str, Any]] = None,
) -> OLCGraph:
    """
    Contract ``vertex_set`` to one macro node while preserving the complete
    internal graph and hierarchical boundary-port paths.
    """
    if not vertex_set:
        raise ValueError("Cannot contract an empty vertex set")
    if not vertex_set.issubset(graph.nodes):
        missing = vertex_set.difference(graph.nodes)
        raise ValueError(f"Unknown vertices in contraction set: {missing}")
    if block_id in graph.nodes and block_id not in vertex_set:
        raise ValueError(f"block_id already exists: {block_id}")

    internal_graph = graph.induced_subgraph(vertex_set)
    boundary_in: List[Edge] = []
    boundary_out: List[Edge] = []
    entry_vertices: Set[str] = set()
    exit_vertices: Set[str] = set()

    new_edges: List[Edge] = []

    for edge in graph.edges:
        src_inside = edge.src in vertex_set
        dst_inside = edge.dst in vertex_set

        if src_inside and dst_inside:
            continue

        if not src_inside and dst_inside:
            boundary_in.append(copy.deepcopy(edge))
            entry_vertices.add(edge.dst)

            new_edge = copy.deepcopy(edge)
            new_edge.dst = block_id
            new_edge.dst_port = (
                edge.dst,
            ) + normalize_port_path(edge.dst_port)
            new_edges.append(new_edge)
            continue

        if src_inside and not dst_inside:
            boundary_out.append(copy.deepcopy(edge))
            exit_vertices.add(edge.src)

            new_edge = copy.deepcopy(edge)
            new_edge.src = block_id
            new_edge.src_port = (
                edge.src,
            ) + normalize_port_path(edge.src_port)
            new_edges.append(new_edge)
            continue

        new_edges.append(copy.deepcopy(edge))

    internal_nodes = [graph.nodes[vertex] for vertex in vertex_set]
    macro_node = Node(
        id=block_id,
        # Keep macro-node sequence empty.  The canonical sequence (when one
        # exists, e.g. an equivalent bubble) belongs to GraphBlock metadata;
        # treating it as an ordinary read would make later overlap/unitig
        # operations duplicate boundary sequence.
        seq="",
        coverage=weighted_average_coverage(internal_nodes),
        members=aggregate_members(internal_nodes),
        kind=kind,
    )

    new_nodes = {
        node_id: copy.deepcopy(node)
        for node_id, node in graph.nodes.items()
        if node_id not in vertex_set
    }
    new_nodes[block_id] = macro_node

    new_blocks = {
        existing_id: copy.deepcopy(block)
        for existing_id, block in graph.blocks.items()
        if existing_id not in vertex_set
    }

    new_blocks[block_id] = GraphBlock(
        id=block_id,
        kind=kind,
        internal_graph=internal_graph,
        boundary_in_edges=boundary_in,
        boundary_out_edges=boundary_out,
        entry_vertices=entry_vertices,
        exit_vertices=exit_vertices,
        feasible_port_pairs=set(feasible_port_pairs or set()),
        alternatives=list(alternatives or []),
        canonical_sequence=canonical_sequence,
        cycle_order=tuple(cycle_order),
        metadata=dict(metadata or {}),
    )

    result = OLCGraph(nodes=new_nodes, edges=new_edges, blocks=new_blocks)
    ensure_edge_fields(result)
    return reduce_multi_edges(result)[0]


# ============================================================
# 1. Duplicate-read removal
# ============================================================

def remove_duplicate_reads(graph: OLCGraph) -> Tuple[OLCGraph, int]:
    seq_to_rep: Dict[str, str] = {}
    old_to_new: Dict[str, str] = {}
    new_nodes: Dict[str, Node] = {}
    merged_count = 0

    for node_id, node in graph.nodes.items():
        # Empty-sequence and hierarchical nodes are never merged here.
        if not node.seq or node.kind.endswith("_block"):
            old_to_new[node_id] = node_id
            new_nodes[node_id] = copy.deepcopy(node)
            continue

        representative = seq_to_rep.get(node.seq)
        if representative is None:
            seq_to_rep[node.seq] = node_id
            old_to_new[node_id] = node_id
            new_nodes[node_id] = copy.deepcopy(node)
            if not new_nodes[node_id].members:
                new_nodes[node_id].members = (node_id,)
            continue

        merged_count += 1
        old_to_new[node_id] = representative
        rep_node = new_nodes[representative]
        rep_node.coverage += node.coverage
        rep_node.members = node_members(rep_node) + node_members(node)

    new_edges: List[Edge] = []
    for edge in graph.edges:
        new_edge = copy.deepcopy(edge)
        new_edge.src = old_to_new[edge.src]
        new_edge.dst = old_to_new[edge.dst]
        if new_edge.src == new_edge.dst:
            continue
        new_edges.append(new_edge)

    # Blocks are not merged by this operation.
    result = OLCGraph(
        nodes=new_nodes,
        edges=new_edges,
        blocks=copy.deepcopy(graph.blocks),
    )
    return result, merged_count


# ============================================================
# 2. Contained-read absorption
# ============================================================

def remove_contained_reads(
    graph: OLCGraph,
) -> Tuple[OLCGraph, Dict[str, str]]:
    """
    Remove a concrete read if its sequence is a proper substring of a longer
    concrete read.  The returned map records absorbed_read -> container_read.

    Complexity is O(N^2) in the number of concrete reads.
    """
    concrete_ids = [
        node_id
        for node_id, node in graph.nodes.items()
        if node.seq and not node.kind.endswith("_block")
    ]
    concrete_ids.sort(
        key=lambda node_id: len(graph.nodes[node_id].seq),
        reverse=True,
    )

    absorbed: Dict[str, str] = {}

    for index_i, node_i in enumerate(concrete_ids):
        seq_i = graph.nodes[node_i].seq
        if node_i in absorbed:
            continue

        for index_j in range(0, index_i):
            node_j = concrete_ids[index_j]
            seq_j = graph.nodes[node_j].seq
            if len(seq_j) <= len(seq_i):
                continue
            if seq_i in seq_j:
                absorbed[node_i] = node_j
                break

    result = graph.remove_nodes(set(absorbed))
    return result, absorbed


# ============================================================
# 3-4. Edge filtering
# ============================================================

def filter_edges_by_quality(
    graph: OLCGraph,
    config: CompressConfig,
) -> Tuple[OLCGraph, int]:
    ensure_edge_fields(graph)
    new_edges: List[Edge] = []

    for edge in graph.edges:
        if edge.overlap < config.min_overlap:
            continue
        if edge.identity < config.min_identity:
            continue
        if (
            config.remove_low_coverage_edges
            and edge.coverage < config.min_edge_coverage
        ):
            continue
        new_edges.append(copy.deepcopy(edge))

    removed = len(graph.edges) - len(new_edges)
    return OLCGraph(
        nodes=copy.deepcopy(graph.nodes),
        edges=new_edges,
        blocks=copy.deepcopy(graph.blocks),
    ), removed


def remove_internal_matches(graph: OLCGraph) -> Tuple[OLCGraph, int]:
    new_edges = [
        copy.deepcopy(edge)
        for edge in graph.edges
        if edge.suffix_prefix
    ]
    removed = len(graph.edges) - len(new_edges)
    return OLCGraph(
        nodes=copy.deepcopy(graph.nodes),
        edges=new_edges,
        blocks=copy.deepcopy(graph.blocks),
    ), removed


# ============================================================
# 5. Port-aware parallel-edge reduction
# ============================================================

def reduce_multi_edges(graph: OLCGraph) -> Tuple[OLCGraph, int]:
    ensure_edge_fields(graph)
    best: Dict[EdgeSignature, Edge] = {}

    for edge in graph.edges:
        key = edge_signature(edge)
        if key not in best or edge_score(graph, edge) > edge_score(
            graph, best[key]
        ):
            best[key] = copy.deepcopy(edge)

    new_edges = list(best.values())
    removed = len(graph.edges) - len(new_edges)
    return OLCGraph(
        nodes=copy.deepcopy(graph.nodes),
        edges=new_edges,
        blocks=copy.deepcopy(graph.blocks),
    ), removed


# ============================================================
# 6. Optional top-k pruning
# ============================================================

def top_k_pruning(
    graph: OLCGraph,
    config: CompressConfig,
) -> Tuple[OLCGraph, int]:
    ensure_edge_fields(graph)
    edges = [copy.deepcopy(edge) for edge in graph.edges]

    if config.top_k_out is not None:
        out: Dict[str, List[Edge]] = defaultdict(list)
        for edge in edges:
            out[edge.src].append(edge)

        kept: List[Edge] = []
        for edge_list in out.values():
            edge_list.sort(
                key=lambda edge: edge_score(graph, edge),
                reverse=True,
            )
            kept.extend(edge_list[: config.top_k_out])
        edges = kept

    if config.top_k_in is not None:
        inn: Dict[str, List[Edge]] = defaultdict(list)
        for edge in edges:
            inn[edge.dst].append(edge)

        kept = []
        for edge_list in inn.values():
            edge_list.sort(
                key=lambda edge: edge_score(graph, edge),
                reverse=True,
            )
            kept.extend(edge_list[: config.top_k_in])
        edges = kept

    # Remove accidental duplicate object copies created by two pruning passes.
    temp = OLCGraph(
        nodes=copy.deepcopy(graph.nodes),
        edges=edges,
        blocks=copy.deepcopy(graph.blocks),
    )
    reduced, _ = reduce_multi_edges(temp)
    removed = len(graph.edges) - len(reduced.edges)
    return reduced, removed


# ============================================================
# 7. Two-step transitive-edge removal
# ============================================================

def remove_transitive_edges(
    graph: OLCGraph,
    config: CompressConfig,
) -> Tuple[OLCGraph, int]:
    reduced, _ = reduce_multi_edges(graph)
    ensure_edge_fields(reduced)

    out = reduced.out_edges()
    direct: Dict[Tuple[str, str], List[Edge]] = defaultdict(list)
    for edge in reduced.edges:
        direct[(edge.src, edge.dst)].append(edge)

    remove_signatures: Set[EdgeSignature] = set()

    for source in reduced.nodes:
        for edge_ij in out.get(source, []):
            # Transitive reduction is only reliable for ordinary sequence
            # edges with known shifts and no hierarchical ports.
            if edge_ij.shift is None or edge_ij.src_port or edge_ij.dst_port:
                continue

            middle = edge_ij.dst
            for edge_jk in out.get(middle, []):
                if (
                    edge_jk.shift is None
                    or edge_jk.src_port
                    or edge_jk.dst_port
                ):
                    continue

                target = edge_jk.dst
                if source == target:
                    continue

                for edge_ik in direct.get((source, target), []):
                    if (
                        edge_ik.shift is None
                        or edge_ik.src_port
                        or edge_ik.dst_port
                    ):
                        continue

                    predicted = edge_ij.shift + edge_jk.shift
                    if abs(edge_ik.shift - predicted) <= config.eps_shift:
                        remove_signatures.add(edge_instance_signature(edge_ik))

    new_edges = [
        copy.deepcopy(edge)
        for edge in reduced.edges
        if edge_instance_signature(edge) not in remove_signatures
    ]
    return OLCGraph(
        nodes=copy.deepcopy(reduced.nodes),
        edges=new_edges,
        blocks=copy.deepcopy(reduced.blocks),
    ), len(reduced.edges) - len(new_edges)


# ============================================================
# 8. Unitig contraction
# ============================================================

def merge_two_unitig_nodes(
    graph: OLCGraph,
    source: str,
    destination: str,
    connecting_edge: Edge,
) -> OLCGraph:
    source_node = graph.nodes[source]
    destination_node = graph.nodes[destination]

    new_id = unique_block_id(graph, "UNITIG")
    new_sequence = spell_concat(
        source_node.seq,
        destination_node.seq,
        connecting_edge.overlap,
    )
    merged_nodes = [source_node, destination_node]

    new_node = Node(
        id=new_id,
        seq=new_sequence,
        coverage=weighted_average_coverage(merged_nodes),
        members=aggregate_members(merged_nodes),
        kind="unitig",
    )

    new_nodes = {
        node_id: copy.deepcopy(node)
        for node_id, node in graph.nodes.items()
        if node_id not in {source, destination}
    }
    new_nodes[new_id] = new_node

    new_edges: List[Edge] = []
    for edge in graph.edges:
        new_edge = copy.deepcopy(edge)
        if new_edge.src in {source, destination}:
            new_edge.src = new_id
        if new_edge.dst in {source, destination}:
            new_edge.dst = new_id

        if new_edge.src == new_edge.dst:
            continue

        # Ports into ordinary unitig nodes are not needed because source and
        # destination had no macro structure.
        new_edges.append(new_edge)

    result = OLCGraph(
        nodes=new_nodes,
        edges=new_edges,
        blocks=copy.deepcopy(graph.blocks),
    )
    result, _ = reduce_multi_edges(result)
    ensure_edge_fields(result)
    return result


def contract_unitigs(
    graph: OLCGraph,
    config: CompressConfig,
) -> Tuple[OLCGraph, int]:
    result, _ = reduce_multi_edges(graph)
    merge_count = 0

    while True:
        out = result.out_edges()
        inn = result.in_edges()
        indegree = {
            node_id: len(inn.get(node_id, []))
            for node_id in result.nodes
        }
        outdegree = {
            node_id: len(out.get(node_id, []))
            for node_id in result.nodes
        }

        selected_edge: Optional[Edge] = None

        for edge in result.edges:
            source = edge.src
            destination = edge.dst
            if source == destination:
                continue

            source_node = result.nodes[source]
            destination_node = result.nodes[destination]

            if not config.contract_macro_nodes:
                if source in result.blocks or destination in result.blocks:
                    continue

            if not source_node.seq or not destination_node.seq:
                continue
            if edge.src_port or edge.dst_port:
                continue

            if outdegree[source] != 1 or indegree[destination] != 1:
                continue

            if not config.contract_cycles:
                if path_exists(
                    result,
                    destination,
                    source,
                    forbidden_signature=edge_instance_signature(edge),
                ):
                    continue

            selected_edge = edge
            break

        if selected_edge is None:
            break

        result = merge_two_unitig_nodes(
            result,
            selected_edge.src,
            selected_edge.dst,
            selected_edge,
        )
        merge_count += 1

    return result, merge_count


# ============================================================
# 9. Optional tip removal
# ============================================================

def remove_tips(
    graph: OLCGraph,
    config: CompressConfig,
) -> Tuple[OLCGraph, int]:
    result = graph.copy()
    total_removed = 0

    while True:
        out = result.out_edges()
        inn = result.in_edges()
        remove_set: Set[str] = set()

        for node_id, node in result.nodes.items():
            if node_id in result.blocks:
                continue

            indegree = len(inn.get(node_id, []))
            outdegree = len(out.get(node_id, []))
            is_terminal = indegree == 0 or outdegree == 0

            if (
                is_terminal
                and len(node.seq) <= config.tip_max_len
                and node.coverage < config.min_tip_coverage
            ):
                remove_set.add(node_id)

        if not remove_set:
            break

        total_removed += len(remove_set)
        result = result.remove_nodes(remove_set)

    return result, total_removed


# ============================================================
# 10. Equivalent-bubble contraction
# ============================================================

def _trace_linear_branch(
    graph: OLCGraph,
    source: str,
    first_edge: Edge,
    max_nodes: int,
) -> Tuple[List[str], List[Edge], str]:
    """
    Trace source -> ... until the first node that is not internally 1-in/1-out.

    Returns:
        node_path including source and terminal,
        edge_path,
        terminal node.
    """
    out = graph.out_edges()
    inn = graph.in_edges()

    node_path = [source, first_edge.dst]
    edge_path = [first_edge]
    current = first_edge.dst
    seen = {source, current}

    while len(node_path) <= max_nodes:
        indegree = len(inn.get(current, []))
        outdegree = len(out.get(current, []))

        if indegree != 1 or outdegree != 1:
            break

        next_edge = out[current][0]
        next_vertex = next_edge.dst
        if next_vertex in seen:
            break

        edge_path.append(next_edge)
        node_path.append(next_vertex)
        seen.add(next_vertex)
        current = next_vertex

    return node_path, edge_path, current


def find_one_equivalent_bubble(
    graph: OLCGraph,
    config: CompressConfig,
) -> Optional[Dict[str, Any]]:
    out = graph.out_edges()

    for source, source_edges in out.items():
        if len(source_edges) < 2:
            continue

        for first_edge, second_edge in combinations(source_edges, 2):
            if first_edge.dst == second_edge.dst:
                continue

            path_a, edges_a, terminal_a = _trace_linear_branch(
                graph,
                source,
                first_edge,
                config.bubble_max_path_nodes,
            )
            path_b, edges_b, terminal_b = _trace_linear_branch(
                graph,
                source,
                second_edge,
                config.bubble_max_path_nodes,
            )

            if terminal_a != terminal_b:
                continue

            sink = terminal_a
            if sink == source:
                continue

            internal_a = set(path_a[1:-1])
            internal_b = set(path_b[1:-1])
            if not internal_a or not internal_b:
                continue
            if internal_a.intersection(internal_b):
                continue

            # Do not absorb already hierarchical nodes into an exact-sequence
            # bubble because their concrete sequence may be undefined.
            if any(vertex in graph.blocks for vertex in internal_a | internal_b):
                continue

            sequence_a = spell_path(graph, path_a, edges_a)
            sequence_b = spell_path(graph, path_b, edges_b)
            if not sequence_a or sequence_a != sequence_b:
                continue
            if len(sequence_a) > config.bubble_max_spelled_length:
                continue

            return {
                "source": source,
                "sink": sink,
                "path_a": tuple(path_a),
                "path_b": tuple(path_b),
                "internal_vertices": internal_a | internal_b,
                "sequence": sequence_a,
            }

    return None


def contract_equivalent_bubbles(
    graph: OLCGraph,
    config: CompressConfig,
) -> Tuple[OLCGraph, int]:
    result = graph.copy()
    created = 0

    while True:
        bubble = find_one_equivalent_bubble(result, config)
        if bubble is None:
            break

        internal_vertices: Set[str] = bubble["internal_vertices"]
        path_a: Tuple[str, ...] = bubble["path_a"]
        path_b: Tuple[str, ...] = bubble["path_b"]

        # Pair the first and last internal vertices of each alternative.
        feasible_pairs = {
            (path_a[1], path_a[-2]),
            (path_b[1], path_b[-2]),
        }

        block_id = unique_block_id(result, "BUBBLE")
        result = contract_vertex_set_to_block(
            result,
            internal_vertices,
            block_id=block_id,
            kind="bubble_block",
            feasible_port_pairs=feasible_pairs,
            alternatives=[path_a[1:-1], path_b[1:-1]],
            canonical_sequence=bubble["sequence"],
            metadata={
                "source": bubble["source"],
                "sink": bubble["sink"],
                "equivalent": True,
            },
        )
        created += 1

    return result, created


# ============================================================
# 11. SCC detection and hierarchical contraction
# ============================================================

def strongly_connected_components(graph: OLCGraph) -> List[List[str]]:
    out = graph.out_edges()
    index = 0
    stack: List[str] = []
    on_stack: Set[str] = set()
    indices: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    components: List[List[str]] = []

    def strongconnect(vertex: str) -> None:
        nonlocal index

        indices[vertex] = index
        lowlink[vertex] = index
        index += 1
        stack.append(vertex)
        on_stack.add(vertex)

        for edge in out.get(vertex, []):
            neighbor = edge.dst
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlink[vertex] = min(
                    lowlink[vertex], lowlink[neighbor]
                )
            elif neighbor in on_stack:
                lowlink[vertex] = min(
                    lowlink[vertex], indices[neighbor]
                )

        if lowlink[vertex] == indices[vertex]:
            component: List[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == vertex:
                    break
            components.append(component)

    for vertex in graph.nodes:
        if vertex not in indices:
            strongconnect(vertex)

    return components


def classify_simple_cycle(
    graph: OLCGraph,
    component: Set[str],
) -> Tuple[bool, Tuple[str, ...]]:
    internal_out: Dict[str, List[str]] = defaultdict(list)
    internal_in: Dict[str, List[str]] = defaultdict(list)

    for edge in graph.edges:
        if edge.src in component and edge.dst in component:
            internal_out[edge.src].append(edge.dst)
            internal_in[edge.dst].append(edge.src)

    for vertex in component:
        if len(internal_out.get(vertex, [])) != 1:
            return False, tuple()
        if len(internal_in.get(vertex, [])) != 1:
            return False, tuple()

    start = min(component)
    order = [start]
    current = start

    for _ in range(len(component) - 1):
        current = internal_out[current][0]
        if current in order:
            return False, tuple()
        order.append(current)

    if internal_out[order[-1]][0] != start:
        return False, tuple()
    if set(order) != component:
        return False, tuple()

    return True, tuple(order)


def contract_scc_blocks(
    graph: OLCGraph,
    config: CompressConfig,
) -> Tuple[OLCGraph, int]:
    components = strongly_connected_components(graph)
    result = graph.copy()
    created = 0

    # Contract larger components first. Components are disjoint in the graph
    # on which they were detected, so ids remain valid until each contraction.
    components.sort(key=len, reverse=True)

    for component_list in components:
        component = set(component_list)
        if len(component) < config.scc_min_size:
            continue
        if not component.issubset(result.nodes):
            continue

        is_cycle, cycle_order = classify_simple_cycle(result, component)
        block_id = unique_block_id(result, "SCC")
        result = contract_vertex_set_to_block(
            result,
            component,
            block_id=block_id,
            kind="scc_block",
            cycle_order=cycle_order,
            metadata={
                "scc_type": "simple_cycle" if is_cycle else "general_scc",
                "requires_internal_qa": not is_cycle,
            },
        )
        created += 1

    return result, created


def split_simple_cycle_by_ports(
    block: GraphBlock,
    entry_vertex: str,
    exit_vertex: str,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Post-processing helper agreed for a simple-cycle SCC.

    Example cycle order (1,2,3,4), entry=1, exit=3 returns:
        (1,2), (3,4)

    These are hierarchical cycle segments; boundary ports must still be kept.
    """
    if block.metadata.get("scc_type") != "simple_cycle":
        raise ValueError("Block is not a simple-cycle SCC")
    order = list(block.cycle_order)
    if entry_vertex not in order or exit_vertex not in order:
        raise ValueError("entry/exit is not in the SCC")

    entry_index = order.index(entry_vertex)
    rotated = order[entry_index:] + order[:entry_index]
    exit_index = rotated.index(exit_vertex)

    first = tuple(rotated[:exit_index])
    second = tuple(rotated[exit_index:])
    return first, second


def selected_block_ports(
    incoming_macro_edge: Edge,
    outgoing_macro_edge: Edge,
    block_id: str,
) -> Tuple[str, str]:
    """Recover the immediate selected entry/exit vertices of a macro block."""
    if incoming_macro_edge.dst != block_id:
        raise ValueError("incoming_macro_edge does not enter the block")
    if outgoing_macro_edge.src != block_id:
        raise ValueError("outgoing_macro_edge does not leave the block")
    if not incoming_macro_edge.dst_port:
        raise ValueError("Incoming edge has no destination port")
    if not outgoing_macro_edge.src_port:
        raise ValueError("Outgoing edge has no source port")
    return incoming_macro_edge.dst_port[0], outgoing_macro_edge.src_port[0]


# ============================================================
# 12. Optional hierarchical repeat-like contraction
# ============================================================

def contract_repeat_like_blocks(
    graph: OLCGraph,
    config: CompressConfig,
) -> Tuple[OLCGraph, int]:
    out = graph.out_edges()
    inn = graph.in_edges()

    repeat_nodes = {
        vertex
        for vertex in graph.nodes
        if len(out.get(vertex, [])) >= config.repeat_degree_threshold
        and len(inn.get(vertex, [])) >= config.repeat_degree_threshold
    }

    if not repeat_nodes:
        return graph.copy(), 0

    adjacency: Dict[str, Set[str]] = defaultdict(set)
    for edge in graph.edges:
        if edge.src in repeat_nodes and edge.dst in repeat_nodes:
            adjacency[edge.src].add(edge.dst)
            adjacency[edge.dst].add(edge.src)

    components: List[Set[str]] = []
    seen: Set[str] = set()

    for start in repeat_nodes:
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        component: Set[str] = set()
        while queue:
            vertex = queue.popleft()
            component.add(vertex)
            for neighbor in adjacency.get(vertex, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        if len(component) > 1:
            components.append(component)

    result = graph.copy()
    created = 0
    components.sort(key=len, reverse=True)

    for component in components:
        if not component.issubset(result.nodes):
            continue
        block_id = unique_block_id(result, "REPEAT")
        result = contract_vertex_set_to_block(
            result,
            component,
            block_id=block_id,
            kind="repeat_block",
            metadata={
                "degree_threshold": config.repeat_degree_threshold,
                "heuristic": True,
            },
        )
        created += 1

    return result, created


# ============================================================
# Plan B: two-terminal subgraph contraction
# ============================================================

@dataclass(frozen=True)
class TwoTerminalCandidate:
    vertices: frozenset[str]
    incoming_edge_index: int
    outgoing_edge_index: int


def boundary_edge_indices(
    graph: OLCGraph,
    vertex_set: Set[str],
) -> Tuple[List[int], List[int]]:
    incoming: List[int] = []
    outgoing: List[int] = []

    for index, edge in enumerate(graph.edges):
        src_inside = edge.src in vertex_set
        dst_inside = edge.dst in vertex_set
        if not src_inside and dst_inside:
            incoming.append(index)
        elif src_inside and not dst_inside:
            outgoing.append(index)

    return incoming, outgoing


def find_two_terminal_candidates(
    graph: OLCGraph,
    min_size: int = 2,
) -> List[TwoTerminalCandidate]:
    """
    Find weakly connected vertex regions S with exactly one incoming and one
    outgoing boundary edge.

    The implementation removes a candidate pair of boundary edges and checks
    whether one weak component is isolated by exactly those two edges.  This is
    exact for regions whose undirected boundary cut consists of those two edge
    instances.
    """
    candidates: Dict[frozenset[str], TwoTerminalCandidate] = {}
    edge_count = len(graph.edges)

    for incoming_index in range(edge_count):
        incoming_edge = graph.edges[incoming_index]

        for outgoing_index in range(edge_count):
            if incoming_index == outgoing_index:
                continue
            outgoing_edge = graph.edges[outgoing_index]

            # Potential region must contain incoming.dst and outgoing.src,
            # while excluding incoming.src and outgoing.dst.
            excluded = {incoming_index, outgoing_index}
            components = weak_components(graph, excluded)

            for component in components:
                if len(component) < min_size or len(component) >= len(graph.nodes):
                    continue
                if incoming_edge.dst not in component:
                    continue
                if outgoing_edge.src not in component:
                    continue
                if incoming_edge.src in component:
                    continue
                if outgoing_edge.dst in component:
                    continue

                in_indices, out_indices = boundary_edge_indices(
                    graph, component
                )
                if in_indices != [incoming_index]:
                    continue
                if out_indices != [outgoing_index]:
                    continue

                key = frozenset(component)
                candidates[key] = TwoTerminalCandidate(
                    vertices=key,
                    incoming_edge_index=incoming_index,
                    outgoing_edge_index=outgoing_index,
                )

    return list(candidates.values())


def choose_two_terminal_candidate(
    graph: OLCGraph,
    candidates: Sequence[TwoTerminalCandidate],
    config: CompressConfig,
) -> Optional[TwoTerminalCandidate]:
    if not candidates:
        return None

    small_enough = [
        candidate
        for candidate in candidates
        if len(candidate.vertices) <= config.max_block_size
    ]

    if small_enough:
        # Contract the largest immediately solvable region first.
        return max(small_enough, key=lambda candidate: len(candidate.vertices))

    # Otherwise choose a reasonably balanced large region and recursively
    # decompose its stored internal graph.
    target = len(graph.nodes) / 2.0
    return min(
        candidates,
        key=lambda candidate: abs(len(candidate.vertices) - target),
    )


def apply_plan_b_two_terminal_partition(
    graph: OLCGraph,
    config: CompressConfig,
    stats: CompressionStats,
    *,
    depth: int = 0,
    block_path: str = "ROOT",
) -> OLCGraph:
    """
    Recursively contract two-terminal regions until each visible graph has at
    most ``max_block_size`` nodes, or no safe region is found.
    """
    result = graph.copy()

    if len(result.nodes) <= config.max_block_size:
        # Internal blocks may still be large because they were created before
        # Plan B. Process them recursively as well.
        for block_id, block in list(result.blocks.items()):
            if len(block.internal_graph.nodes) > config.max_block_size:
                block.internal_graph = apply_plan_b_two_terminal_partition(
                    block.internal_graph,
                    config,
                    stats,
                    depth=depth + 1,
                    block_path=f"{block_path}/{block_id}",
                )
        return result

    if depth >= config.max_partition_depth:
        stats.unresolved_large_blocks.append(
            f"{block_path}: depth limit, {len(result.nodes)} nodes"
        )
        return result

    while len(result.nodes) > config.max_block_size:
        candidates = find_two_terminal_candidates(
            result,
            min_size=config.min_partition_size,
        )
        candidate = choose_two_terminal_candidate(
            result, candidates, config
        )

        if candidate is None:
            stats.unresolved_large_blocks.append(
                f"{block_path}: no two-terminal region, "
                f"{len(result.nodes)} nodes"
            )
            break

        incoming_edge = copy.deepcopy(
            result.edges[candidate.incoming_edge_index]
        )
        outgoing_edge = copy.deepcopy(
            result.edges[candidate.outgoing_edge_index]
        )
        vertices = set(candidate.vertices)

        block_id = unique_block_id(result, "PARTITION")
        result = contract_vertex_set_to_block(
            result,
            vertices,
            block_id=block_id,
            kind="partition_block",
            feasible_port_pairs={(incoming_edge.dst, outgoing_edge.src)},
            metadata={
                "entry_vertex": incoming_edge.dst,
                "exit_vertex": outgoing_edge.src,
                "incoming_external_vertex": incoming_edge.src,
                "outgoing_external_vertex": outgoing_edge.dst,
                "plan_b": True,
            },
        )
        stats.partition_blocks_created += 1

        # Recursively decompose the new block's internal graph if necessary.
        block = result.blocks[block_id]
        if len(block.internal_graph.nodes) > config.max_block_size:
            block.internal_graph = apply_plan_b_two_terminal_partition(
                block.internal_graph,
                config,
                stats,
                depth=depth + 1,
                block_path=f"{block_path}/{block_id}",
            )

    # Process any other previously existing large blocks.
    for block_id, block in list(result.blocks.items()):
        if len(block.internal_graph.nodes) > config.max_block_size:
            block.internal_graph = apply_plan_b_two_terminal_partition(
                block.internal_graph,
                config,
                stats,
                depth=depth + 1,
                block_path=f"{block_path}/{block_id}",
            )

    return result


# ============================================================
# Hierarchy and subproblem helpers
# ============================================================

def iter_blocks_recursive(
    graph: OLCGraph,
    prefix: Tuple[str, ...] = (),
) -> Iterator[Tuple[Tuple[str, ...], GraphBlock]]:
    for block_id, block in graph.blocks.items():
        path = prefix + (block_id,)
        yield path, block
        yield from iter_blocks_recursive(block.internal_graph, path)


def leaf_qa_subproblems(
    graph: OLCGraph,
    max_nodes: int = 50,
) -> List[Tuple[Tuple[str, ...], OLCGraph, Dict[str, Any]]]:
    """
    Collect leaf internal graphs intended for later QA.

    Boundary conditions are read from block.metadata when available.
    The root graph is included if it has no child block and is small enough.
    """
    result: List[Tuple[Tuple[str, ...], OLCGraph, Dict[str, Any]]] = []

    def visit(
        current: OLCGraph,
        path: Tuple[str, ...],
        metadata: Dict[str, Any],
    ) -> None:
        if current.blocks:
            for block_id, block in current.blocks.items():
                visit(
                    block.internal_graph,
                    path + (block_id,),
                    dict(block.metadata),
                )
            return

        if len(current.nodes) <= max_nodes:
            result.append((path, current.copy(), metadata))

    visit(graph, tuple(), {})
    return result


def hierarchy_summary(graph: OLCGraph) -> Dict[str, Any]:
    def build(current: OLCGraph) -> Dict[str, Any]:
        return {
            "node_count": len(current.nodes),
            "edge_count": len(current.edges),
            "blocks": {
                block_id: {
                    "kind": block.kind,
                    "entry_vertices": sorted(block.entry_vertices),
                    "exit_vertices": sorted(block.exit_vertices),
                    "metadata": copy.deepcopy(block.metadata),
                    "internal": build(block.internal_graph),
                }
                for block_id, block in current.blocks.items()
            },
        }

    return build(graph)


# ============================================================
# Main integrated compressor
# ============================================================

def compress_olc_graph(
    graph: OLCGraph,
    config: Optional[CompressConfig] = None,
) -> CompressionResult:
    """
    Run all configured OLC graph reductions and return a reversible hierarchy.

    The function does not build a QUBO.  ``partition_block`` and ``scc_block``
    metadata provide the boundary conditions needed by a later QA layer.
    """
    if config is None:
        config = CompressConfig()

    result = graph.copy()
    ensure_edge_fields(result)
    validate_graph(result)

    stats = CompressionStats(
        initial_nodes=len(result.nodes),
        initial_edges=len(result.edges),
    )
    absorbed_reads: Dict[str, str] = {}
    record_stage(stats, "input", result)

    # 1. Duplicate reads
    if config.remove_duplicates:
        result, count = remove_duplicate_reads(result)
        stats.duplicate_nodes_merged += count
    record_stage(stats, "01_duplicates", result)

    # 2. Contained reads
    if config.remove_contained_reads:
        result, absorbed = remove_contained_reads(result)
        absorbed_reads.update(absorbed)
        stats.contained_nodes_absorbed += len(absorbed)
    record_stage(stats, "02_contained", result)

    # 3. Quality filtering
    result, removed = filter_edges_by_quality(result, config)
    stats.quality_edges_removed += removed
    record_stage(stats, "03_quality", result)

    # 4. Internal matches
    if config.remove_internal_matches:
        result, removed = remove_internal_matches(result)
        stats.internal_edges_removed += removed
    record_stage(stats, "04_internal_matches", result)

    # 5. Parallel edges
    if config.reduce_multi_edges:
        result, removed = reduce_multi_edges(result)
        stats.parallel_edges_removed += removed
    record_stage(stats, "05_parallel_edges", result)

    # 6. Optional top-k
    if config.top_k_out is not None or config.top_k_in is not None:
        result, removed = top_k_pruning(result, config)
        stats.top_k_edges_removed += removed
    record_stage(stats, "06_top_k", result)

    # 7. Transitive edges
    if config.remove_transitive_edges:
        result, removed = remove_transitive_edges(result, config)
        stats.transitive_edges_removed += removed
    record_stage(stats, "07_transitive", result)

    # 8. Unitigs
    if config.contract_unitigs:
        result, merges = contract_unitigs(result, config)
        stats.unitig_merges += merges
    record_stage(stats, "08_unitigs", result)

    # 9. Tips
    if config.remove_tips:
        result, removed = remove_tips(result, config)
        stats.tip_nodes_removed += removed
    record_stage(stats, "09_tips", result)

    # 10. Exact-equivalent bubbles
    if config.contract_equivalent_bubbles:
        result, created = contract_equivalent_bubbles(result, config)
        stats.bubble_blocks_created += created
    record_stage(stats, "10_bubbles", result)

    # 11. SCC hierarchy
    if config.compress_scc:
        result, created = contract_scc_blocks(result, config)
        stats.scc_blocks_created += created
    record_stage(stats, "11_scc", result)

    # 12. Repeat-like hierarchy
    if config.compress_repeat_blocks:
        result, created = contract_repeat_like_blocks(result, config)
        stats.repeat_blocks_created += created
    record_stage(stats, "12_repeats", result)

    # Final cleanup before Plan B.
    result, removed = reduce_multi_edges(result)
    stats.parallel_edges_removed += removed
    ensure_edge_fields(result)

    # Plan B: only after all 1-12 stages.
    if (
        config.partition_large_graph
        and len(result.nodes) > config.max_block_size
    ):
        result = apply_plan_b_two_terminal_partition(
            result,
            config,
            stats,
        )
    elif config.partition_large_graph:
        # Even if the root is small, older hierarchical blocks might be large.
        result = apply_plan_b_two_terminal_partition(
            result,
            config,
            stats,
        )

    result, removed = reduce_multi_edges(result)
    stats.parallel_edges_removed += removed
    ensure_edge_fields(result)
    validate_graph(result)

    stats.final_nodes = len(result.nodes)
    stats.final_edges = len(result.edges)
    record_stage(stats, "final", result)

    return CompressionResult(
        graph=result,
        absorbed_reads=absorbed_reads,
        stats=stats,
    )


# ============================================================
# Small construction helpers and demonstration
# ============================================================

def graph_from_sequences_and_overlaps(
    sequences: Mapping[str, str],
    overlaps: Iterable[Tuple[str, str, int]],
) -> OLCGraph:
    """Convenience constructor for tests and small examples."""
    nodes = {
        node_id: Node(id=node_id, seq=sequence, members=(node_id,))
        for node_id, sequence in sequences.items()
    }
    edges = [
        Edge(src=src, dst=dst, overlap=overlap)
        for src, dst, overlap in overlaps
    ]
    graph = OLCGraph(nodes=nodes, edges=edges)
    ensure_edge_fields(graph)
    return graph


def print_compression_summary(result: CompressionResult) -> None:
    stats = result.stats
    print(
        f"nodes: {stats.initial_nodes} -> {stats.final_nodes}, "
        f"edges: {stats.initial_edges} -> {stats.final_edges}"
    )
    print(f"duplicate nodes merged: {stats.duplicate_nodes_merged}")
    print(f"contained reads absorbed: {stats.contained_nodes_absorbed}")
    print(f"transitive edges removed: {stats.transitive_edges_removed}")
    print(f"unitig merges: {stats.unitig_merges}")
    print(f"bubble blocks: {stats.bubble_blocks_created}")
    print(f"SCC blocks: {stats.scc_blocks_created}")
    print(f"repeat blocks: {stats.repeat_blocks_created}")
    print(f"partition blocks: {stats.partition_blocks_created}")
    if stats.unresolved_large_blocks:
        print("unresolved large blocks:")
        for message in stats.unresolved_large_blocks:
            print(f"  - {message}")


if __name__ == "__main__":
    # Minimal smoke example.  Replace this with real overlap data.
    example = graph_from_sequences_and_overlaps(
        {
            "R1": "ACGTAA",
            "R2": "GTAACC",
            "R3": "AACCGG",
        },
        [
            ("R1", "R2", 4),
            ("R2", "R3", 4),
            ("R1", "R3", 2),
        ],
    )

    compressed = compress_olc_graph(example)
    print_compression_summary(compressed)
