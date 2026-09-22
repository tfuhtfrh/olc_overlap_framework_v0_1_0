from collections import deque

from demo_phi174_common import DATA_DIR, run_reads_only_pipeline
from olc_pipeline.graph_builder import remove_contained_reads, remove_low_quality_reads, select_one_orientation_per_read
from olc_pipeline.io_utils import read_fastq


DATASET = DATA_DIR / "SRR27862880_phiX174_OLC_cycle742_pilot"


def metrics(reads, edges, orientations):
    nodes = [(read.rid, orientations[read.rid]) for read in reads]
    node_set = set(nodes)
    adjacency = {node: set() for node in nodes}
    reverse = {node: set() for node in nodes}
    for edge in edges:
        left = (edge.left_id, edge.left_orientation)
        right = (edge.right_id, edge.right_orientation)
        if left in node_set and right in node_set and left != right:
            adjacency[left].add(right)
            reverse[right].add(left)
    unseen = set(nodes)
    components = []
    while unseen:
        start = min(unseen, key=str)
        component = {start}
        unseen.remove(start)
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node] | reverse[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return len(nodes), sum(len(value) for value in adjacency.values()), len(components), max((len(value) for value in components), default=0), sum(not reverse[node] for node in nodes), sum(not adjacency[node] for node in nodes)


def main():
    reads = read_fastq(DATASET / "SRR27862880.stride500.unique.fastq.gz")
    finder, result = run_reads_only_pipeline(reads, minimap2_bin="minimap2", min_overlap=80, min_paf_identity=0.995, max_error_rate_hint=0.01, max_error_rate=0.01, overhang_tolerance=20, extra_args=("-k", "15", "-w", "5", "-m", "40", "-n", "2", "-X", "--secondary=yes", "-N", "1000", "-c", "--eqx"))
    orientation = select_one_orientation_per_read(reads, result.edges)
    containment = remove_contained_reads(orientation.reads, orientation.edges, finder.last_containment_evidence)
    low = remove_low_quality_reads(containment.reads, containment.edges)
    selected_orientations = {read_id: orientation.orientation_by_read[read_id] for read_id in [read.rid for read in low.reads]}
    print("before", metrics(orientation.reads, orientation.edges, orientation.orientation_by_read))
    print("containment_removed", len(containment.removed_read_ids))
    print("low_quality_removed_after_orientation", len(low.removed_read_ids))
    print("after_low_quality", metrics(low.reads, low.edges, selected_orientations))


if __name__ == "__main__":
    main()
