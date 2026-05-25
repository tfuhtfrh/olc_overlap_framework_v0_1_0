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


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
