from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

from demo_edge_path_parameter_sweep import selected_edge_truth_counts
from demo_sqa_parameter_sweep import build_overlap_graph, select_layout_inputs
from olc_pipeline.layout_solver import (
    EdgePathDAGHamiltonianConfig,
    EdgePathDAGQUBOHamiltonian,
    OpenJijSQAConfig,
    OpenJijSimulatedQuantumAnnealer,
    QUBOLayoutSolver,
    qubo_sample_for_order,
)


def simple_path_fragments(read_ids, selected_edges, rank):
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    nodes = set(read_ids)
    for left_id, right_id in selected_edges:
        outgoing[left_id].append(right_id)
        incoming[right_id].append(left_id)
        nodes.add(left_id)
        nodes.add(right_id)

    starts = sorted((node for node in nodes if not incoming[node]), key=rank.get)
    paths = []
    visited_edges = set()
    for start in starts:
        if not outgoing[start]:
            paths.append([start])
            continue
        for first in sorted(outgoing[start], key=rank.get):
            path = [start, first]
            visited_edges.add((start, first))
            seen = {start, first}
            current = first
            while len(outgoing[current]) == 1:
                nxt = outgoing[current][0]
                if nxt in seen:
                    break
                path.append(nxt)
                visited_edges.add((current, nxt))
                seen.add(nxt)
                current = nxt
            paths.append(path)

    for left_id, right_ids in outgoing.items():
        for right_id in right_ids:
            if (left_id, right_id) not in visited_edges:
                paths.append([left_id, right_id])

    paths.sort(key=lambda path: (-len(set(path)), rank[path[0]]))
    return paths, incoming, outgoing


def longest_selected_path(read_ids, selected_edges, rank):
    outgoing = defaultdict(list)
    for left_id, right_id in selected_edges:
        outgoing[left_id].append(right_id)

    best_len = {}
    best_next = {}
    for read_id in sorted(read_ids, key=rank.get, reverse=True):
        best_len[read_id] = 1
        best_next[read_id] = None
        for right_id in sorted(outgoing[read_id], key=rank.get):
            candidate = 1 + best_len.get(right_id, 1)
            if candidate > best_len[read_id]:
                best_len[read_id] = candidate
                best_next[read_id] = right_id

    start = max(read_ids, key=lambda read_id: (best_len[read_id], -rank[read_id]))
    path = [start]
    while best_next[path[-1]] is not None:
        path.append(best_next[path[-1]])
    return path


def main() -> None:
    args = SimpleNamespace(
        max_reads=100,
        read_counts=None,
        genome_len=52500,
        read_len=3000,
        step=500,
        mismatch_rate=0.0,
        ins_rate=0.0,
        del_rate=0.0,
        gc_fraction=0.65,
        describe_only=False,
        num_reads=100,
        num_sweeps=1000,
        simulation_seed=42,
        seed=42,
        annealer_seeds="40,42,45",
        count_penalties="40",
        degree_penalties="60",
        reward_scale=20.0,
        score_mode="overlap_len",
        trotter=32,
        beta=None,
        output="",
    )

    reads, edges = build_overlap_graph(args)
    reads, edges = select_layout_inputs(reads, edges, 100)
    read_ids = [read.rid for read in reads]
    true_order = [read.rid for read in sorted(reads, key=lambda item: item.true_start)]
    rank = {read_id: index for index, read_id in enumerate(true_order)}

    for seed in (40, 42, 45):
        hamiltonian = EdgePathDAGQUBOHamiltonian(
            EdgePathDAGHamiltonianConfig(
                edge_count_penalty=40,
                degree_penalty=60,
                edge_reward_scale=20,
                score_mode="overlap_len",
                normalize_rewards=True,
                require_hamiltonian_path=True,
            )
        )
        solver = QUBOLayoutSolver(
            hamiltonian=hamiltonian,
            annealer=OpenJijSimulatedQuantumAnnealer(
                OpenJijSQAConfig(
                    num_reads=100,
                    num_sweeps=1000,
                    seed=seed,
                    beta=None,
                    trotter=32,
                )
            ),
        )
        layout = solver.solve(reads, edges, weight_mode="overlap_len")
        model = solver.last_model
        assert model is not None
        true_energy = model.energy(qubo_sample_for_order(model, true_order))
        selected_edges = layout.metadata["selected_edges"]
        counts = selected_edge_truth_counts(reads, selected_edges)

        paths, incoming, outgoing = simple_path_fragments(read_ids, selected_edges, rank)
        used_nodes = {node for edge in selected_edges for node in edge}
        isolated = [rank[read_id] for read_id in true_order if read_id not in used_nodes]
        jumps = sorted(
            (rank[left_id], rank[right_id], rank[right_id] - rank[left_id])
            for left_id, right_id in selected_edges
            if rank[right_id] - rank[left_id] > 1
        )
        branches = sorted(
            (rank[left_id], sorted(rank[right_id] for right_id in right_ids))
            for left_id, right_ids in outgoing.items()
            if len(right_ids) > 1
        )
        merges = sorted(
            (rank[right_id], sorted(rank[left_id] for left_id in left_ids))
            for right_id, left_ids in incoming.items()
            if len(left_ids) > 1
        )
        path_spans = [(len(path), rank[path[0]], rank[path[-1]]) for path in paths]
        longest_path = longest_selected_path(read_ids, selected_edges, rank)
        longest_jumps = [
            rank[right_id] - rank[left_id]
            for left_id, right_id in zip(longest_path, longest_path[1:])
        ]

        print(f"seed,{seed}")
        print(f"energy_gap,{layout.objective_value - true_energy}")
        print(
            "valid,"
            f"{layout.metadata['valid_edge_path']},"
            f"selected_edges,{len(selected_edges)},"
            f"adjacent,{counts['adjacent_correct']},"
            f"jump,{counts['jump_correct']},"
            f"wrong,{counts['wrong']}"
        )
        print(
            "violations,"
            f"edge,{layout.metadata['edge_count_violation']},"
            f"in,{layout.metadata['in_degree_violations']},"
            f"out,{layout.metadata['out_degree_violations']}"
        )
        print(f"path_count,{len(paths)},path_spans,{path_spans}")
        print(
            "longest_path,"
            f"nodes,{len(longest_path)},"
            f"span,{rank[longest_path[0]]}-{rank[longest_path[-1]]},"
            f"missing_within_span,{(rank[longest_path[-1]] - rank[longest_path[0]] + 1) - len(longest_path)},"
            f"max_jump,{max(longest_jumps) if longest_jumps else 0}"
        )
        print(f"isolated_count,{len(isolated)},isolated_ranks,{isolated}")
        print(f"jumps,{jumps}")
        print(f"branches,{branches}")
        print(f"merges,{merges}")
        print()


if __name__ == "__main__":
    main()
