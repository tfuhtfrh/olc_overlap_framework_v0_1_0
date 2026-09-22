# Edge weight specification

## Primary graph score

For every retained edge:

- `M = matching_bases`, counted from exact-CIGAR `=` operations;
- `B = alignment_block`;
- `d = errors = B - M`, counting mismatch and gap bases;
- `weight = M - 49*d`.

This is the primitive-integer form of the original 98% threshold-margin score:

`1000*M - 980*B = 20*(M - 49*d)`.

Dividing every edge weight by 20 leaves every Hamilton-path ordering, optimum,
and degeneracy unchanged.  The layout energy convention is
`E_layout = -sum(weight)` over selected edges.

The primary reference-path score is 1343093; the runner-up score
is 1338209; the certified margin is 4884.

## Where the values are stored

- `graph.graphml`, `graph_normalized.graphml`, `graph_with_rc.graphml`, and
  `graph_full_oriented.graphml`: edge attributes `weight`,
  `raw_quality_margin`, `weight_divisor`, `matching_bases`, and `errors`;
- `edges.tsv`: the same values as explicit columns;
- `weight_spec.json`: machine-readable formula and score convention;
- `weighted_certificate.json`: exhaustive 192-path scores;
- `final_summary.json`: primary and sensitivity-analysis score metadata;
- `verification/verify_complex144.py`: reconstructs weights from the PAF and
  asserts the formula independently.

Weights use read-read alignment evidence only. Reference coordinates and the
known path are not used in edge weights.
