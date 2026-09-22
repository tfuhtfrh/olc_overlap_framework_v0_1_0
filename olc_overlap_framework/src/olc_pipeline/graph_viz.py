"""
olc_pipeline.graph_viz
Version: 0.1.0

Optional overlap-graph visualization helpers.
"""

from __future__ import annotations

from pathlib import Path

from .data import Read, OverlapEdge
from .layout_solver import OverlapRewardScorer

MODULE_VERSION = "0.1.0"


def write_overlap_graph_dot(
    reads: list[Read],
    edges: list[OverlapEdge],
    path: Path,
    score_mode: str = "dp",
    normalize_rewards: bool = True,
    max_edges: int | None = None,
) -> Path:
    """
    Write a Graphviz DOT visualization of a directed overlap graph.

    Node order follows true_start when available. Edge labels include overlap
    length, shift, raw reward, and normalized reward. This function has no
    Graphviz dependency; install Graphviz separately only if you want to render
    the DOT file to SVG/PNG.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    scorer = OverlapRewardScorer()
    scored_edges = [
        (edge, scorer.score(edge, score_mode))
        for edge in edges
    ]
    scored_edges.sort(key=lambda item: item[1], reverse=True)
    if max_edges is not None:
        scored_edges = scored_edges[:max_edges]

    max_reward = max((score for _, score in scored_edges), default=0.0)
    read_order = sorted(reads, key=lambda read: read.true_start if read.true_start >= 0 else read.rid)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("digraph overlap_graph {\n")
        handle.write("  rankdir=LR;\n")
        handle.write("  node [shape=box, style=rounded];\n")
        for index, read in enumerate(read_order):
            label = f"{read.rid}\\nidx={index} start={read.true_start}"
            handle.write(f'  "{_dot_escape(read.rid)}" [label="{_dot_escape(label)}"];\n')

        for edge, raw_reward in scored_edges:
            norm_reward = (
                raw_reward / max_reward
                if normalize_rewards and max_reward > 0.0
                else raw_reward
            )
            penwidth = 1.0 + 4.0 * max(0.0, min(1.0, norm_reward))
            label = (
                f"L={edge.overlap_len}\\n"
                f"shift={edge.shift}\\n"
                f"reward={raw_reward:.3f}\\n"
                f"norm={norm_reward:.3f}"
            )
            handle.write(
                f'  "{_dot_escape(edge.left_id)}" -> "{_dot_escape(edge.right_id)}" '
                f'[label="{_dot_escape(label)}", penwidth={penwidth:.2f}];\n'
            )
        handle.write("}\n")

    return path


def write_oriented_overlap_graph_dot(
    reads: list[Read],
    edges: list[OverlapEdge],
    path: Path,
    score_mode: str = "dp",
    normalize_rewards: bool = True,
    max_edges: int | None = None,
    highlight_edges: set[tuple[str, str, int, int]] | None = None,
    rankdir: str = "LR",
    show_edge_labels: bool = True,
    orientation_by_read: dict[str, int] | None = None,
) -> Path:
    """Write a direction-aware overlap graph for visual inspection.

    The ordinary helper above intentionally renders physical read IDs only.
    This helper renders each endpoint as ``read_id+`` or ``read_id-`` and
    preserves the orientation carried by :class:`OverlapEdge`.  It is useful
    for the reverse-strand-aware reads-only graph and does not use reference
    coordinates or a reference sequence.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    scorer = OverlapRewardScorer()
    scored_edges = [(edge, scorer.score(edge, score_mode)) for edge in edges]
    scored_edges.sort(
        key=lambda item: (
            item[1],
            item[0].overlap_len,
            item[0].shift,
        ),
        reverse=True,
    )
    if max_edges is not None:
        scored_edges = scored_edges[:max_edges]

    max_reward = max((score for _, score in scored_edges), default=0.0)
    read_rank = {read.rid: index for index, read in enumerate(reads)}
    node_keys: set[tuple[str, int]] = set()
    if orientation_by_read is not None:
        node_keys.update(orientation_by_read.items())
    for edge, _ in scored_edges:
        node_keys.add((edge.left_id, edge.left_orientation))
        node_keys.add((edge.right_id, edge.right_orientation))
    node_order = sorted(
        node_keys,
        key=lambda key: (read_rank.get(key[0], len(read_rank)), key[0], -key[1]),
    )
    highlight_edges = highlight_edges or set()

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("digraph oriented_overlap_graph {\n")
        handle.write(f"  rankdir={_dot_escape(rankdir)};\n")
        handle.write("  graph [overlap=false, splines=true];\n")
        handle.write("  node [shape=box, style=rounded, style=filled];\n")
        for rid, orientation in node_order:
            suffix = _orientation_suffix(orientation)
            node_id = f"{rid}{suffix}"
            fill = "#d9edf7" if orientation > 0 else "#fde0dd"
            label = f"{rid}{suffix}"
            handle.write(
                f'  "{_dot_escape(node_id)}" '
                f'[label="{_dot_escape(label)}", fillcolor="{fill}"];\n'
            )

        for edge, raw_reward in scored_edges:
            norm_reward = (
                raw_reward / max_reward
                if normalize_rewards and max_reward > 0.0
                else raw_reward
            )
            penwidth = 1.0 + 4.0 * max(0.0, min(1.0, norm_reward))
            left_node = f"{edge.left_id}{_orientation_suffix(edge.left_orientation)}"
            right_node = f"{edge.right_id}{_orientation_suffix(edge.right_orientation)}"
            edge_key = (
                edge.left_id,
                edge.right_id,
                edge.left_orientation,
                edge.right_orientation,
            )
            color = "#d62728" if edge_key in highlight_edges else "#777777"
            width = max(2.5, penwidth) if edge_key in highlight_edges else penwidth
            label = ""
            if show_edge_labels:
                label = (
                    f"L={edge.overlap_len} shift={edge.shift} "
                    f"id={edge.identity:.3f} score={raw_reward:.3f}"
                )
            handle.write(
                f'  "{_dot_escape(left_node)}" -> "{_dot_escape(right_node)}" '
                f'[label="{_dot_escape(label)}", color="{color}", '
                f'penwidth={width:.2f}];\n'
            )
        handle.write("}\n")

    return path


def _orientation_suffix(orientation: int) -> str:
    return "+" if orientation >= 0 else "-"


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
