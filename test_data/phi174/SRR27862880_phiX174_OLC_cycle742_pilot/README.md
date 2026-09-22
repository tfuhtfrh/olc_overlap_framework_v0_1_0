# SRR27862880 φX174 OLC cycle — reproducible pilot

This package is a **high-accuracy short-read OLC validation dataset**, not a
complex-topology benchmark.  It contains a reference-free constructed string
graph whose eligible physical reads admit an all-node Hamiltonian cycle.

## Source and scope

- SRA run: `SRR27862880`; BioProject `PRJNA815898`; sample `SAMN23371904`.
- Material: commercial φX174 RF I DNA; Illumina NovaSeq 6000.
- ENA reports 25,565,508 spots and 7,682,867,212 bases.  The archive exposes
  two approximately 150-bp FASTQ files.  Mates are treated as independent
  reads; pair information is not used in the OLC graph.
- The complete run is far too deep for direct OLC.  This pilot uses a fixed,
  reproducible **first 8 MiB compressed window** from each published FASTQ,
  retains every 500th complete record starting at record 1, and removes only
  incomplete/<140-bp terminal records introduced by the byte-window boundary.
  It does not use reference positions, read-to-reference identity, or an
  overlap score to choose reads.

This produces 994 reads / 149,476 bases.  Renaming `/1` and `/2` as `_m1` and
`_m2` is a parser compatibility change only: miniasm otherwise merges the two
mate names.  No bases are corrected or trimmed.

## Main graph specification

Candidate discovery was all-vs-all minimap2 (`-k15 -w5 -m40 -n2 -X
--secondary=yes -N1000 -c --eqx`).  Before string-graph construction, every
candidate PAF record was explicitly required to have:

- `matching_bases / alignment_block_length >= 0.995`;
- `alignment_block_length >= 80 bp`.

miniasm then applied dovetail/end geometry (`max terminal hang = 10 bp`,
end-to-end fraction >= 0.8), contained-read handling, and transitive
reduction.  The resulting graph has:

| property | result |
|---|---:|
| eligible physical reads | 742 |
| oriented nodes / arcs | 1,484 / 1,484 |
| directed components | two reciprocal 742-node cycles |
| split / merge / open ports | 0 / 0 / 0 |
| selected overlap length (min / median / mean / max) | 99 / 145 / 143.22 / 150 bp |

Thus either oriented component is an explicit 742-node Hamiltonian cycle.
`truth_cycles.tsv` lists both reciprocal orientations.

## Reference-only audit

`NC_001422.1.fasta` is supplied only for post-hoc validation.  All 742 retained
reads map to the reference.  Their weighted alignment identity is 99.9078%
(minimum individual identity 98.6667%); their reference union coverage is
5,386 / 5,386 bp.  The longest read is 2.804% of the genome, so no read is
near full length.

The reference-oriented cycle advances by 1–52 bp per edge (median 5 bp), with
the advances summing to exactly 5,386 bp.  It therefore makes one correct tour
of the reference rather than merely forming a topological cycle.

## Threshold stability

The same simple-cycle topology persists for all 12 combinations below.  The
number of retained physical nodes varies only with the explicit identity
threshold, not with the tested overlap minimum.

| explicit identity | overlap minimum (bp) | retained physical nodes |
|---:|---:|---:|
| 0.985 | 50, 60, 70, 80 | 773 |
| 0.990 | 50, 60, 70, 80 | 771 |
| 0.995 | 50, 60, 70, 80 | 742 |

## Interpretation and limits

This is a valid >=50-node real-read test of orientation, overlap selection,
transitive reduction, and an all-node Hamiltonian **cycle**.  It is deliberately
simple: it has no stable bubble, repeat branch, or biologically meaningful open
end.  It should complement, not replace, a long-read test such as T6 control
(83-node cycle) and a future single-sample complex graph.

The provided workset is a reproducible pilot based on the beginning of the
published compressed files, not a population-uniform full-run sample.  A final
publication-grade version should repeat the same fixed-rule experiment with a
streaming or full-file deterministic sampler over the complete accession.

## Files

- `SRR27862880.stride500.unique.fastq.gz`: 994 selected, uncorrected reads.
- `NC_001422.1.fasta`: φX174 reference for audit only.
- `all_vs_all.exact.paf`: raw exact all-vs-all candidates.
- `reliable_overlaps.i0.995.o80.paf`: explicitly filtered high-quality
  candidate evidence.
- `string_graph.i0.995.o80.gfa`: the final oriented string graph.
- `truth_cycles.tsv`: both reciprocal Hamiltonian cycles.
- `reads_to_reference_x2.audit_only.paf`: doubled-reference audit mapping.
- `NC_001422.1_x2.audit_only.fasta`: doubled reference used only for that
  circular-reference audit.
- `threshold_sweep.tsv`: all threshold results.
- `SHA256SUMS`: checksums for the package contents.
