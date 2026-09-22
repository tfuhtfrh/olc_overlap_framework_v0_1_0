from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from demo_phi174_common import DATA_DIR, run_reads_only_pipeline
from olc_pipeline.io_utils import read_fastq

dataset = DATA_DIR / 'SRR27862880_phiX174_OLC_cycle742_pilot'
reads = read_fastq(dataset / 'SRR27862880.stride500.unique.fastq.gz')
finder, result = run_reads_only_pipeline(
    reads,
    minimap2_bin='minimap2',
    min_overlap=80,
    min_paf_identity=0.995,
    max_error_rate_hint=0.01,
    max_error_rate=0.01,
    overhang_tolerance=20,
    extra_args=('-k', '15', '-w', '5', '-m', '40', '-n', '2', '-X', '--secondary=yes', '-N', '1000', '-c', '--eqx'),
)
edge_keys = {
    (edge.left_id, edge.right_id, edge.left_orientation, edge.right_orientation)
    for edge in result.edges
}
cycles = defaultdict(list)
with (dataset / 'truth_cycles.tsv').open(encoding='utf-8') as handle:
    next(handle)
    for line in handle:
        fields = line.rstrip().split('\t')
        cycles[int(fields[0])].append((fields[3], 1 if fields[4] == '+' else -1))

for cycle_id, cycle in sorted(cycles.items()):
    missing = []
    for index, left in enumerate(cycle):
        right = cycle[(index + 1) % len(cycle)]
        key = (left[0], right[0], left[1], right[1])
        if key not in edge_keys:
            missing.append((left, right))
    print('cycle_id', cycle_id, 'nodes', len(cycle), 'missing_edges', len(missing))
    if missing:
        print('first_missing', missing[:5])
print('refined_edges', len(result.edges))
print('filter_counts', dict(finder.last_filter_counts))
