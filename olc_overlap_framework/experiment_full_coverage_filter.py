"""Sweep a reference-free filter for reads fully covered by another read."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from demo_phi174_common import DATA_DIR, run_reads_only_pipeline
from olc_pipeline.graph_builder import check_hamilton_cycle, select_one_orientation_per_read
from olc_pipeline.io_utils import read_fastq


DATASET = DATA_DIR / "SRR27862880_phiX174_OLC_cycle742_pilot"
PAF_PATH = DATASET / "all_vs_all.exact.paf"
OUTPUT_PATH = Path(__file__).resolve().parent / "debug" / "phi174_cycle742_graph" / "full_coverage_filter_sweep.txt"
THRESHOLDS = (0.995, 0.993, 0.990, 0.985)


class _DisjointSet:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            next_value = self.parent[value]
            self.parent[value] = root
            value = next_value
        return root

    def union(self, first, second):
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def _covered_read_pairs(threshold: float):
    """Return pairs where the shorter read is fully covered in PAF geometry."""
    pairs = {}
    with PAF_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 12:
                continue
            q_id, t_id = fields[0], fields[5]
            if q_id == t_id:
                continue
            q_len, q_st, q_en = int(fields[1]), int(fields[2]), int(fields[3])
            t_len, t_st, t_en = int(fields[6]), int(fields[7]), int(fields[8])
            matches, block = int(fields[9]), int(fields[10])
            if block <= 0 or matches / block < threshold:
                continue
            q_full = q_st == 0 and q_en == q_len
            t_full = t_st == 0 and t_en == t_len
            if q_len < t_len:
                q_covered = q_full
                t_covered = False
            elif t_len < q_len:
                q_covered = False
                t_covered = t_full
            else:
                q_covered = q_full and t_full
                t_covered = q_covered
            if not (q_covered or t_covered):
                continue
            key = tuple(sorted((q_id, t_id)))
            record = (q_id, t_id, fields[4], q_len, t_len, matches, block)
            old = pairs.get(key)
            if old is None or (matches / block, block) > (old[5] / old[6], old[6]):
                pairs[key] = record
    return pairs


def _groups(read_ids, pairs):
    dsu = _DisjointSet(read_ids)
    for first, second in pairs:
        dsu.union(first, second)
    grouped = defaultdict(list)
    for read_id in read_ids:
        grouped[dsu.find(read_id)].append(read_id)
    return [sorted(group) for group in grouped.values() if len(group) > 1]


def _keep_representatives(groups, result, reads):
    incident = {read.rid: 0 for read in reads}
    lengths = {read.rid: len(read.seq) for read in reads}
    for edge in result.edges:
        incident[edge.left_id] += 1
        incident[edge.right_id] += 1
    keep = {read.rid for read in reads}
    removed = {}
    for group in groups:
        representative = max(group, key=lambda read_id: (lengths[read_id], incident[read_id], read_id))
        for read_id in group:
            if read_id != representative:
                keep.remove(read_id)
                removed[read_id] = representative
    return keep, removed


def _audit_graph(reads, result, keep_ids, *, apply_low_support=False):
    selected = select_one_orientation_per_read(reads, result.edges)
    selected_reads = [read for read in reads if read.rid in keep_ids]
    selected_edges = [
        edge for edge in selected.edges
        if edge.left_id in keep_ids and edge.right_id in keep_ids
    ]
    orientations = {
        read_id: orientation
        for read_id, orientation in selected.orientation_by_read.items()
        if read_id in keep_ids
    }
    nodes = [(read.rid, orientations[read.rid]) for read in selected_reads]
    incoming = {node: set() for node in nodes}
    outgoing = {node: set() for node in nodes}
    for edge in selected_edges:
        left = (edge.left_id, edge.left_orientation)
        right = (edge.right_id, edge.right_orientation)
        if left in outgoing and right in incoming:
            outgoing[left].add(right)
            incoming[right].add(left)
    if apply_low_support:
        supported_ids = {
            node[0]
            for node in nodes
            if incoming[node] and outgoing[node]
        }
        selected_reads = [read for read in selected_reads if read.rid in supported_ids]
        selected_edges = [
            edge for edge in selected_edges
            if edge.left_id in supported_ids and edge.right_id in supported_ids
        ]
        nodes = [(read.rid, orientations[read.rid]) for read in selected_reads]
        incoming = {node: set() for node in nodes}
        outgoing = {node: set() for node in nodes}
        for edge in selected_edges:
            left = (edge.left_id, edge.left_orientation)
            right = (edge.right_id, edge.right_orientation)
            if left in outgoing and right in incoming:
                outgoing[left].add(right)
                incoming[right].add(left)
    hamilton = check_hamilton_cycle(nodes, selected_edges, time_limit_sec=10.0)
    return len(nodes), len(selected_edges), sum(not values for values in incoming.values()), sum(not values for values in outgoing.values()), hamilton.status


def main() -> None:
    reads = read_fastq(DATASET / "SRR27862880.stride500.unique.fastq.gz")
    lines = [
        "== reference-free full-coverage/co-linear filter sweep ==",
        f"reads: {len(reads)}",
        f"paf: {PAF_PATH}",
        "definition: the shorter read is exactly full-length; equal-length pairs require both reads exactly full-length",
        "representative: longest read, then highest accepted-edge incidence, then ID",
        "",
    ]
    for threshold in THRESHOLDS:
        pairs = _covered_read_pairs(threshold)
        groups = _groups([read.rid for read in reads], pairs)
        _, result = run_reads_only_pipeline(
            reads,
            minimap2_bin="minimap2",
            min_overlap=80,
            min_paf_identity=threshold,
            max_error_rate_hint=0.01,
            max_error_rate=0.01,
            overhang_tolerance=20,
            extra_args=("-k", "15", "-w", "5", "-m", "40", "-n", "2", "-X", "--secondary=yes", "-N", "1000", "-c", "--eqx"),
        )
        keep_ids, removed = _keep_representatives(groups, result, reads)
        nodes, edges, zero_in, zero_out, hamilton = _audit_graph(reads, result, keep_ids)
        low_nodes, low_edges, low_zero_in, low_zero_out, low_hamilton = _audit_graph(
            reads,
            result,
            keep_ids,
            apply_low_support=True,
        )
        lines.extend([
            f"threshold: {threshold:.3f}",
            f"full_coverage_pairs: {len(pairs)}",
            f"full_coverage_groups: {len(groups)}",
            f"full_coverage_group_sizes: {sorted((len(group) for group in groups), reverse=True)[:20]}",
            f"removed_nodes: {len(removed)}",
            f"retained_nodes: {nodes}",
            f"selected_edges: {edges}",
            f"zero_in_nodes: {zero_in}",
            f"zero_out_nodes: {zero_out}",
            f"hamilton_cycle: {hamilton}",
            f"after_one_pass_low_support_nodes: {low_nodes}",
            f"after_one_pass_low_support_edges: {low_edges}",
            f"after_one_pass_low_support_zero_in_nodes: {low_zero_in}",
            f"after_one_pass_low_support_zero_out_nodes: {low_zero_out}",
            f"after_one_pass_low_support_hamilton_cycle: {low_hamilton}",
            "",
        ])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
