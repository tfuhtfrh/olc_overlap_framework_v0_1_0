"""Compatibility audit for the external single-file Plan-B implementation."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import sys
from pathlib import Path
from time import perf_counter

import networkx as nx

from demo_phi174_common import DATA_DIR, PROJECT_DIR
from demo_phi174_cycle742_cut_dag_qubo import _build_accepted_graph
from olc_pipeline.io_utils import read_fastq


DATASET = DATA_DIR / "SRR27862880_phiX174_OLC_graph812_i0.990_o80"
CYCLE_PATH = PROJECT_DIR / "debug" / "phi174_cycle742_graph" / "cycle742_hamilton_cycle.tsv"
PLANB_PATH = PROJECT_DIR / "tests" / "olc_planb_complete.py"
OUTPUT_DIR = PROJECT_DIR / "debug" / "planb_cycle812"
REPORT_PATH = OUTPUT_DIR / "planb_cycle812_audit.txt"
CUT_SEED = 20260904


def _load_planb_module():
    spec = importlib.util.spec_from_file_location("olc_planb_external", PLANB_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Plan-B module: {PLANB_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_int(row: dict[str, str], name: str) -> int:
    return int(row[name])


def _as_float(row: dict[str, str], name: str) -> float:
    return float(row[name])


def load_packaged_graph() -> nx.MultiDiGraph:
    reads = read_fastq(DATASET / "reads.fastq.gz")
    lengths = {read.rid: len(read.seq) for read in reads}
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(
        (read.rid, {"length": len(read.seq)})
        for read in reads
    )

    with (DATASET / "graph_edges.tsv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for index, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            left_id = row["left_id"]
            right_id = row["right_id"]
            edge_id = f"E{index:08d}"
            graph.add_edge(
                left_id,
                right_id,
                key=edge_id,
                edge_id=edge_id,
                q_start=_as_int(row, "left_start"),
                q_end=_as_int(row, "left_end"),
                q_len=lengths[left_id],
                t_start=_as_int(row, "right_start"),
                t_end=_as_int(row, "right_end"),
                t_len=lengths[right_id],
                block_len=_as_int(row, "overlap_len"),
                overlap_len=_as_int(row, "overlap_len"),
                matches=_as_int(row, "matches"),
                identity=_as_float(row, "identity"),
                mapq=_as_int(row, "mapq"),
                l=_as_int(row, "matches"),
                d=_as_int(row, "edit_distance"),
                left_orientation=_as_int(row, "left_orientation"),
                right_orientation=_as_int(row, "right_orientation"),
                shift=_as_int(row, "shift"),
            )

    if graph.number_of_nodes() != 812:
        raise ValueError(f"expected 812 nodes, got {graph.number_of_nodes()}")
    if graph.number_of_edges() != 8385:
        raise ValueError(f"expected 8385 edges, got {graph.number_of_edges()}")
    return graph


def _load_cycle() -> list[str]:
    with CYCLE_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    cycle = [row["read_id"] for row in rows]
    if len(cycle) != 812 or len(set(cycle)) != 812:
        raise ValueError("cycle certificate does not contain 812 unique reads")
    return cycle


def _packaged_edge_signatures(graph: nx.MultiDiGraph) -> set[tuple[object, ...]]:
    return {
        (
            left,
            data["left_orientation"],
            right,
            data["right_orientation"],
            data["q_start"],
            data["q_end"],
            data["t_start"],
            data["t_end"],
            data["overlap_len"],
            data["shift"],
            data["matches"],
            data["d"],
            round(data["identity"], 12),
            data["mapq"],
        )
        for left, right, data in graph.edges(data=True)
    }


def _rebuilt_edge_signatures(graph) -> set[tuple[object, ...]]:
    return {
        (
            edge.left_id,
            edge.left_orientation,
            edge.right_id,
            edge.right_orientation,
            edge.left_start,
            edge.left_end,
            edge.right_start,
            edge.right_end,
            edge.overlap_len,
            edge.shift,
            edge.matches,
            edge.edit_distance,
            round(edge.identity, 12),
            edge.mapq,
        )
        for edge in graph.edges
    }


def _safe_config(planb):
    return planb.SafePreprocessConfig(
        min_overlap_length=80,
        min_identity=0.990,
        remove_internal_alignments=True,
        deduplicate_parallel_edges=True,
        contract_linear_unitigs=True,
    )


def _hierarchy_for_estimate(planb, graph):
    normalized, records = planb.normalize_olc_graph(graph)
    return planb.GraphHierarchy(
        original_graph=normalized.copy(),
        current_graph=normalized.copy(),
        edge_records=records,
        graphs_by_layer={0: normalized.copy()},
    )


def audit_cyclic_input(planb, graph) -> list[str]:
    pre = planb.safe_preprocess_before_planb(
        graph,
        config=_safe_config(planb),
    )
    hierarchy = _hierarchy_for_estimate(planb, pre.graph)
    estimate = planb.estimate_top_problem(
        hierarchy,
        planb.EvaluationConfig(),
    )
    simple = nx.DiGraph(pre.graph)
    strong_components = list(nx.strongly_connected_components(simple))

    recursive = planb.run_recursive_planb_on_preprocessed_graph(
        pre.graph,
        max_input_ports=4,
        max_output_ports=4,
        solve_top=False,
        evaluation_config=planb.EvaluationConfig(),
        recursive_config=planb.RecursivePlanBConfig(
            max_region_nodes=20,
            beam_width=24,
            max_candidates=256,
            max_total_blocks=1,
            require_cycle_closed_regions=True,
            auto_expand_cycle_closure=True,
        ),
    )

    return [
        "== complete 812-node cyclic input ==",
        f"input_nodes: {graph.number_of_nodes()}",
        f"input_edges: {graph.number_of_edges()}",
        f"preprocess_stats: {pre.stats}",
        f"preprocessed_is_dag: {nx.is_directed_acyclic_graph(pre.graph)}",
        f"preprocessed_strong_components: {len(strong_components)}",
        f"largest_strong_component: {max(map(len, strong_components), default=0)}",
        f"estimated_top_qubo_variables: {estimate.estimated_qubo_variables}",
        f"estimated_top_qubo_couplings: {estimate.estimated_qubo_couplings}",
        f"default_recursive_stop_reason: {recursive.stop_reason}",
        f"default_recursive_blocks: {recursive.total_blocks}",
        f"default_recursive_global_infeasible: {recursive.global_infeasible}",
    ]


def run_cut_dag(planb, graph, rebuilt_graph, cut_seed: int) -> list[str]:
    cycle = _load_cycle()
    pairs = set(graph.edges())
    missing_cycle_edges = sum(
        (left, right) not in pairs
        for left, right in zip(cycle, cycle[1:] + cycle[:1])
    )
    if missing_cycle_edges:
        raise ValueError(f"cycle has {missing_cycle_edges} absent graph edges")

    cut_position = random.Random(cut_seed).randrange(len(cycle))
    rotated = cycle[cut_position + 1:] + cycle[:cut_position + 1]
    rank = {read_id: index for index, read_id in enumerate(rotated)}
    dag = nx.MultiDiGraph()
    dag.add_nodes_from(graph.nodes(data=True))
    for left, right, key, data in graph.edges(keys=True, data=True):
        if rank[left] < rank[right]:
            dag.add_edge(left, right, key=key, **dict(data))

    rebuilt_dag_pairs = {
        (edge.left_id, edge.right_id)
        for edge in rebuilt_graph.edges
        if rank[edge.left_id] < rank[edge.right_id]
    }
    planb_dag_pairs = set(dag.edges())

    solve_start = perf_counter()
    result = planb.run_full_olc_pipeline(
        dag,
        max_input_ports=4,
        max_output_ports=4,
        source=rotated[0],
        target=rotated[-1],
        preprocess_config=_safe_config(planb),
        solve_top=True,
        recursive_planb=True,
    )
    solve_seconds = perf_counter() - solve_start
    final = result.final_assembly
    if final is None:
        raise RuntimeError("Plan-B wrapper did not return a final assembly")

    return [
        "",
        "== known-cycle-guided cut-DAG compatibility test ==",
        "test_class: non_strict_known_witness_guided",
        f"cut_seed: {cut_seed}",
        f"cut_position: {cut_position}",
        f"path_source: {rotated[0]}",
        f"path_target: {rotated[-1]}",
        f"dag_nodes: {dag.number_of_nodes()}",
        f"dag_edges: {dag.number_of_edges()}",
        f"dag_pair_set_matches_qubo_demo_projection: {planb_dag_pairs == rebuilt_dag_pairs}",
        f"dag_is_dag: {nx.is_directed_acyclic_graph(dag)}",
        f"planb_preprocess_stats: {result.preprocess.stats}",
        f"execution_mode: {result.execution_mode}",
        f"recursive_stop_reason: {result.recursive_stop_reason}",
        f"top_solver_type: {result.top_solution.solver_type if result.top_solution else None}",
        f"top_energy: {result.top_solution.energy if result.top_solution else None}",
        f"top_qa_sample_variables: {len(result.top_solution.qa_sample) if result.top_solution else 0}",
        f"final_feasible: {final.feasible}",
        f"final_reconstruction_ok: {final.reconstruction_ok}",
        f"final_raw_nodes: {len(final.raw_node_path)}",
        f"final_raw_edges: {len(final.raw_edge_ids)}",
        f"matches_rotated_cycle_path: {final.raw_node_path == rotated}",
        f"planb_pipeline_sec: {solve_seconds:.6f}",
        f"message: {final.message}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("cyclic-audit", "cut-dag", "all"),
        default="all",
    )
    parser.add_argument("--cut-seed", type=int, default=CUT_SEED)
    args = parser.parse_args()

    planb = _load_planb_module()
    graph = load_packaged_graph()
    rebuilt_graph = _build_accepted_graph("minimap2")
    packaged_signatures = _packaged_edge_signatures(graph)
    rebuilt_signatures = _rebuilt_edge_signatures(rebuilt_graph)
    lines = [
        "== phi174 graph812 / external Plan-B compatibility audit ==",
        f"planb_module: {PLANB_PATH}",
        f"graph_tsv: {DATASET / 'graph_edges.tsv'}",
        "reference_used: no",
        f"packaged_edges_match_qubo_demo_rebuild: {packaged_signatures == rebuilt_signatures}",
        f"packaged_only_edge_records: {len(packaged_signatures - rebuilt_signatures)}",
        f"rebuilt_only_edge_records: {len(rebuilt_signatures - packaged_signatures)}",
    ]
    if args.mode in {"cyclic-audit", "all"}:
        lines.extend(["", *audit_cyclic_input(planb, graph)])
    if args.mode in {"cut-dag", "all"}:
        lines.extend(run_cut_dag(planb, graph, rebuilt_graph, args.cut_seed))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
