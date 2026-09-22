# phi174 cycle742 reads-only graph workflow

Generated date: 2026-09-03 (Asia/Tokyo)

## What this tested

This note records the approved reference-free graph workflow for the
`SRR27862880_phiX174_OLC_cycle742_pilot` reads.  The current stage selects one
orientation side, analyzes that graph, and then applies node-level filters.

## Why it was tested

The PAF records already contain the relative orientation of the two reads.
The two orientation sides should therefore be reverse-complement equivalents.
This must be checked before choosing one side and screening nodes.

## Inputs and parameters

- Input reads: `SRR27862880.stride500.unique.fastq.gz`, 994 reads.
- Candidate discovery: minimap2 all-vs-all with `-k15 -w5 -m40 -n2 -X
  --secondary=yes -N1000 -c --eqx`.
- Candidate filters: PAF identity >= 0.995 and alignment block >= 80 bp.
- DP refinement: Parasail semi-global overlap refinement, minimum refined
  overlap 80 bp and maximum error rate 0.01.
- No reference coordinates are used for graph construction.

## Approved workflow

1. Parse PAF records and attach the relative orientation constraint between the
   two physical read IDs.
2. Generate the reciprocal reverse-complement candidate for each accepted
   candidate.
3. Verify that relative constraints are consistent and that the two oriented
   sides are symmetric.
4. Choose either member of each reverse-complement component pair.  The current
   implementation uses a deterministic phase; the global phase is arbitrary.
5. Analyze the selected one-side graph.
6. Apply node-only filtering: explicit containment removal first, followed by
   one non-recursive low-support screen.  No transitive reduction, SCC
   cropping, or exact in/out-degree enforcement is allowed.

## Symmetry and DP result

- Relative orientation conflicts: 0.
- Candidate-level reverse-complement asymmetry: 0.
- After the reverse-candidate coordinate fix, refined reciprocal pairs were
  either both accepted (11202 pairs) or both rejected (48 pairs); asymmetric
  acceptance: 0 pairs.
- The earlier 66 asymmetric refined edges came from swapped read-length bounds
  in `_reverse_complement_candidate`, not from DP itself.

## One-side graph before node filtering

- Reads: 994.
- Refined edges: 22404.
- Selected one-side edges: 11202.
- Relative-orientation constraints: 11202, with 0 conflicts.
- Constraint components: 27.
- Weak components after selecting one side: 27.
- Largest weak component: 967 nodes.
- Zero in-degree nodes: 52.
- Zero out-degree nodes: 51.

## Containment screening

Containment evidence is recorded only for an accepted PAF alignment where one
read is effectively full-length and the other alignment is internal, using a
2 bp terminal tolerance.  In this one-side baseline:

- containment evidence: 0
- contained reads removed: 0

Thus no read is removed by the containment stage for this dataset under the
current explicit evidence rule.

## Post-containment low-support screen

The one-pass low-support screen is applied only after direction selection and
containment screening.  It removes a read when its selected orientation has no
accepted incoming or outgoing support; it does not recursively peel nodes that
become unsupported after removal.

- low-support reads removed: 78
- final retained reads: 916
- final selected-side edges: 10864
- final weak components: 1
- largest weak component: 916
- final zero in-degree nodes: 0
- final zero out-degree nodes: 1
- Hamilton cycle check: `no`

The remaining one zero-out node is reported by the Hamilton checker and is not
silently removed as a degree-1 or topology-reduction operation.

Removed read IDs, grouped by the reason observed before removal:

### No incoming and no outgoing support (25)

```text
SRR27862880.107501_m2
SRR27862880.125501_m1
SRR27862880.174501_m2
SRR27862880.176001_m1
SRR27862880.196001_m2
SRR27862880.202001_m1
SRR27862880.202501_m1
SRR27862880.21001_m2
SRR27862880.212501_m2
SRR27862880.213001_m1
SRR27862880.218501_m1
SRR27862880.218501_m2
SRR27862880.219001_m1
SRR27862880.229001_m1
SRR27862880.229001_m2
SRR27862880.23001_m2
SRR27862880.231501_m1
SRR27862880.242501_m1
SRR27862880.244501_m1
SRR27862880.55001_m1
SRR27862880.55001_m2
SRR27862880.62001_m1
SRR27862880.66501_m1
SRR27862880.66501_m2
SRR27862880.67001_m2
```

### No incoming support (27)

```text
SRR27862880.104001_m1
SRR27862880.110001_m1
SRR27862880.124501_m2
SRR27862880.125001_m2
SRR27862880.130501_m2
SRR27862880.13501_m2
SRR27862880.135501_m1
SRR27862880.140001_m2
SRR27862880.146501_m1
SRR27862880.150501_m2
SRR27862880.153001_m2
SRR27862880.174501_m1
SRR27862880.208501_m2
SRR27862880.209501_m1
SRR27862880.213501_m1
SRR27862880.219501_m2
SRR27862880.223001_m2
SRR27862880.233001_m1
SRR27862880.3001_m2
SRR27862880.40001_m1
SRR27862880.4501_m2
SRR27862880.46501_m1
SRR27862880.52501_m2
SRR27862880.56001_m2
SRR27862880.60501_m1
SRR27862880.62001_m2
SRR27862880.68501_m2
```

### No outgoing support (26)

```text
SRR27862880.105501_m1
SRR27862880.110001_m2
SRR27862880.124001_m1
SRR27862880.125501_m2
SRR27862880.126001_m1
SRR27862880.133001_m1
SRR27862880.138501_m2
SRR27862880.14001_m2
SRR27862880.155001_m1
SRR27862880.159001_m1
SRR27862880.16001_m2
SRR27862880.162501_m1
SRR27862880.169501_m2
SRR27862880.177501_m2
SRR27862880.178001_m2
SRR27862880.241501_m2
SRR27862880.244001_m2
SRR27862880.245001_m2
SRR27862880.27001_m1
SRR27862880.38001_m1
SRR27862880.40001_m2
SRR27862880.40501_m2
SRR27862880.46501_m2
SRR27862880.63501_m2
SRR27862880.73501_m1
SRR27862880.98501_m1
```

## Output files

- `cycle742_candidate_overlaps_i0.995_o80.paf`: project candidate PAF.
- `cycle742_graph_audit.txt`: combined graph audit from the current demo.
- `../phi174_cycle12_graph/`: 12-read comparison outputs.
- Code: `src/olc_pipeline/candidate_finder.py`,
  `src/olc_pipeline/graph_builder.py`, and `demo_phi174_graph742.py`.

## Interpretation

The reciprocal direction sides are symmetric at the constraint and refined
edge acceptance level, so either side can be selected.  The selected-side
graph is now the input for node screening.  Containment removal is currently a
no-op on these reads.  The later low-support screen reduces the graph to one
weak component, but it does not establish a Hamilton cycle because one node
still has no outgoing edge.

This is an intermediate graph-processing record, not a final Hamilton-cycle
conclusion.  The supplied README's 742-node graph also used miniasm
containment handling and transitive reduction, which are intentionally excluded
from this workflow.

## User supplements

The user approved choosing either symmetric orientation side and requested that
containment be the next planned node filter.

## Superseding rule update (2026-09-03)

The user aligned the following processing rules:

1. A pre-PAF read-length screen is allowed for clearly short, long, or
   outlier reads, using the dataset length distribution as evidence.  It is
   skipped for this phi174 dataset because the current data quality is adequate.
2. The 78 reads lacking selected-side incoming or outgoing support are retained
   for diagnosis and are not deleted in the current formal graph.
3. Reference mapping is post-hoc audit only; it is not used to construct or
   filter the reads-only overlap graph.

The current formal 742-read graph therefore has 994 nodes and 11,202 selected
one-side edges.  It has 27 weak components, with a largest component of 967
nodes, 52 zero-in-degree nodes, and 51 zero-out-degree nodes.  Containment
evidence is 0, so containment removal removes 0 nodes.

The reference-only audit maps 984/994 reads and covers all 5,386 reference bp.
Among the 78 selected-side unsupported reads, 25 have neither selected-side
support, 27 have no selected incoming support, and 26 have no selected outgoing
support.  Before single-orientation projection, all 27 no-in and all 26 no-out
reads had physical support; this identifies their missing selected-side support
as an orientation-projection effect.  The remaining 25 had no physical support
even before projection.  Of those, 17 map to the reference with mean identity
0.98332534 (range 0.93918919--0.99337748), while 8 do not map.

The mapped unsupported reads are generally located inside covered reference
regions and are near other high-identity reads; therefore the 78-node set is
not explained by reference coverage gaps alone.  The detailed per-read
coordinates and nearby-read edge checks are in:

- `cycle742_reference_placements.tsv`
- `cycle742_unsupported_reads.tsv`
- `cycle742_unsupported_read_neighbors.tsv`
- `cycle742_unsupported_read_analysis.txt`

The prior 78-node deletion result remains an exploratory comparison only; it
is superseded by the retention rule above and must not be treated as the
current graph input.

## Orientation-projection interpretation correction (2026-09-03)

The earlier statement that the 53 reads with raw physical support but missing
selected-side support had thereby "lost" information was too strong.  In a
fully symmetric oriented graph, a raw edge `D -> A` is paired with its
reverse-complement edge `A -> D`.  Choosing the opposite global phase retains
`A -> D`, so the overlap information is preserved even though the selected
directed orientation is reversed.

The concrete experiment is recorded in
`cycle742_orientation_projection_abcd.txt`.  Its selected example is:

- `A = SRR27862880.208501_m2`: reference 4025--4175, `+`, selected in 0,
  selected out 16.
- `B = SRR27862880.170501_m2`: reference 4031--4181, `+`.
- `C = SRR27862880.71501_m1`: reference 4025--4175, `-`.
- `D = SRR27862880.99501_m1`: reference 4031--4181, `-`.

The selected graph contains `A -> B` and `C -> B`.  Before projection it
contains `D+ -> A-`; its symmetric counterpart is `A+ -> D-`, and the latter
is the edge retained by the selected orientation (`D -> A` is not selected,
`A -> D` is selected).  Thus A's selected zero in-degree in this example is a
direction-phase effect, not deletion of the D--A overlap.  Future audits must
distinguish missing selected-side degree from a genuinely missing symmetric
edge pair.

## Co-linear A/C redundancy (2026-09-03)

In the A/B/C/D experiment, A and C both cover reference coordinates
4025--4175, but with opposite reference strands.  The PAF contains a
full-length reverse-strand A/C alignment (`150/150` aligned bases, `149`
matches, both intervals spanning the complete reads).  It is not converted
to a directed overlap because its rough shift is 0, and its PAF identity
`149/150 = 0.993333...` was also below the then-current
`min_paf_identity=0.995` filter; it is now captured by the full-coverage
evidence filter at `0.990`.

This is a redundant full-coverage group: the two reads have the same external
reference span but cannot form a positive-shift dovetail edge.  The current
rule treats this structure uniformly with shorter-read full coverage.  At the
updated `0.990` threshold the pair is eligible, and the group representative
rule keeps the longer read (or the deterministic tie-breaker when equal).

## Full-coverage filter reconsideration (2026-09-03)

The user requested reconsidering node removal because retaining reads that have
no proper directed overlap prevents an all-node Hamilton cycle, and noted that
`min_paf_identity=0.995` may be too strict.  A reference-free experiment was
run against `all_vs_all.exact.paf` with the following conservative definition:

- one shorter read is exactly full-length in the PAF alignment, or
- equal-length reads are both exactly full-length aligned;
- both `+` and `-` PAF strands are accepted;
- no terminal tolerance is used, because a tolerance of 10 bp incorrectly
  grouped ordinary dovetail overlaps as full coverage;
- one representative is kept per connected full-coverage group, preferring the
  longer read, then higher accepted-edge incidence, then read ID.

Results before the one-pass low-support screen:

| PAF identity threshold | full-coverage groups | removed | retained | zero-in | zero-out |
|---:|---:|---:|---:|---:|---:|
| 0.995 | 111 | 137 | 857 | 52 | 51 |
| 0.993 | 118 | 147 | 847 | 31 | 35 |
| 0.990 | 118 | 148 | 846 | 20 | 27 |
| 0.985 | 122 | 154 | 840 | 18 | 24 |

After one pass removing remaining zero-in/zero-out nodes, the corresponding
node counts are 779, 799, 811, and 810.  The `0.993`, `0.990`, and `0.985`
cases have no remaining zero-in or zero-out nodes; the `0.995` case still has
one zero-out node.  Exact Hamilton search on the larger dense graphs was not
completed within the 10-second diagnostic limit, so these results establish
necessary degree conditions but not Hamilton-cycle existence.

The reproducible exploratory sweep is in `full_coverage_filter_sweep.txt`.
Its conclusions are superseded by the approved rule below; the sweep did not
perform transitive reduction.

## Current approved full-coverage reduction rule

The earlier experimental wording above is superseded by the implementation
used by the current demos.  The graph problem is treated uniformly: a strict
full-coverage/co-linear pair is a redundant node group whether it would be
called containment or a duplicate.  The criterion is reference-free:

- PAF query and/or target endpoints must be exactly the read endpoints;
- no terminal tolerance is applied;
- the shorter read may be fully covered by the longer read, or equal-length
  reads may both be fully covered (coextensive);
- the PAF total match ratio `n_match / aln_block_len` must be at least `0.990`;
- reverse-strand PAF records are included;
- each connected evidence group keeps the longest read; incident support and
  read ID are deterministic tie-breakers for equal lengths.

Only redundant nodes and their incident edges are removed.  Remaining edges
are not rewired, transitively reduced, or degree-normalized.  The Hamilton
checker therefore remains a diagnostic after filtering, not part of filtering.

The formal implementation is `FullCoverageEvidence` in `data.py`, evidence
collection in `candidate_finder.py`, and `remove_full_coverage_reads` in
`graph_builder.py`.  On the current PAF-direct 742-read run at `0.990`, it
records 169 full-coverage evidence pairs, removes 148 reads, and retains 846
reads.  The result still has 19 zero-in-degree and 26 zero-out-degree nodes, so
the graph does not contain a Hamilton cycle; this is a result of remaining
overlap support/topology, not an imposed degree filter.

## Remaining missing-side read audit (2026-09-03)

The phi174 preprocessing path now uses minimap2 PAF geometry directly; DP is
retained only as a future optional experiment and does not reject graph edges.
At the 80 bp baseline, this restores `SRR27862880.14001_m2` and leaves 34
physical reads missing at least one side: 11 have neither side, 8 lack incoming
support, and 15 lack outgoing support (45 missing directed sides total).

The primary causes from nearby reference-ordered reads and raw PAF stages are:

- 24 sides fail the `0.990` PAF identity threshold;
- 18 sides belong to 10 reads that cannot be mapped reliably to phiX174;
- 2 sides have exact local matches but no positive-shift dovetail geometry;
- 1 side has a best overlap shorter than 80 bp.

The read set has mean length 150.378 bp, median 151 bp, and range 140--151 bp.
Scanning minimum overlap 80, 70, 60, and 50 bp did not reduce the 34 abnormal
reads.  At 50 bp, two formerly unsupported physical reads gain partial raw
support, but the selected graph still has 34 abnormal reads and its edge count
grows from 8,506 to 12,070.  The 80 bp threshold therefore remains the current
safe baseline; identity and geometry, rather than overlap length, dominate the
remaining failures.

The 846-node graph consists of one 833-node weak component, eleven isolated
nodes, and one isolated two-read component (`SRR27862880.40001_m1/m2`).  Its
largest strongly connected component has 812 nodes; the other 34 nodes are
exactly the abnormal set.  Of these, 24 map to phiX174 and 10 do not.  The
mapped exceptional partial reads remain `SRR27862880.176001_m1` (2428--2518),
`SRR27862880.213001_m1` (512--598), and `SRR27862880.27001_m1` (888--992).

The full-coverage reduction still does not create any missing side: zero reads
lose their last incoming or outgoing edge between the pre-reduction selected
graph and the final graph.  The 812 fully supported reads all map and cover
5,386/5,386 reference bases.

Generated reports:

- `cycle742_unsupported_reads.tsv`: coordinates and pre/post-filter degrees;
- `cycle742_unsupported_read_neighbors.tsv`: all-mapped and retained-neighbor
  views with PAF-stage evidence;
- `cycle742_unsupported_missing_sides.tsv`: one row per missing directed side;
- `cycle742_unsupported_unmapped_reads.tsv`: PAF evidence for unmapped reads;
- `cycle742_unsupported_read_analysis.txt`: aggregate summary.

## Approved 34-node deletion and post-filter graph (2026-09-03)

The 80 bp demo now enables the existing one-pass low-support filter.  It
removes exactly the 34 reads identified above; the filter is not recursively
reapplied after their incident edges are removed.

Post-filter graph:

- retained physical reads: 812;
- selected directed edges: 8,385;
- weak components: 1 (812 nodes);
- strong components: 1 (812 nodes);
- zero-in and zero-out nodes: 0 and 0;
- newly exposed unsupported nodes after deletion: 0;
- Hamilton cycle: yes, 812 nodes.

Hamilton existence is certified using graph edges only.  The checker first
finds a one-in/one-out cycle cover, then merges its disjoint cycles using valid
two-edge exchanges; reference coordinates do not participate in that search.
The selected witness has mean overlap 137.191 bp, minimum overlap 80 bp, and
mean PAF identity 0.99942950.

Reference mapping is applied afterward as evaluation.  All 812 retained reads
map and cover 5,386/5,386 bp.  The graph contains 809/812 (`0.99630542`) of the
strict reference-order successor edges.  The current feasibility witness has
only `0.09729064` exact reference-successor accuracy because it is an arbitrary
merged cycle cover, not an optimized QUBO layout.  The three absent strict
reference-successor edges are listed separately.

Additional reports:

- `cycle742_hamilton_cycle.tsv`: the complete 812-edge Hamilton witness;
- `cycle742_reference_truth_missing_edges.tsv`: the three strict reference
  successor edges absent from the graph;
- `cycle742_graph_audit.txt`: final post-deletion graph and evaluation summary.

## Graph-processing milestone accepted (2026-09-03)

The user accepted the 80 bp result above as the completed graph-processing
baseline for this phiX174 dataset.  The frozen QUBO-input benchmark contains
812 physical-read nodes and 8,385 directed candidate edges, is one strongly
connected component, has no zero-in or zero-out nodes, and has a graph-only
812-node Hamilton-cycle certificate.

No further topology reduction is part of this accepted baseline.  Subsequent
work starts at the Hamiltonian/optimizer layer for cyclic candidate graphs.
The existing witness is a feasibility certificate only; its reference-order
score is not the target performance of the future optimized cyclic solver.
