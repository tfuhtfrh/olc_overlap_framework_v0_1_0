# CHM13 complex144 edge-ordered Hamilton-path QUBO

Generated: 2026-09-22 (Asia/Tokyo)

## What this tested

The 144-read, 261-edge CHM13 complex144 graph was loaded into the cyclic-graph
edge-ordered Hamilton-path QUBO.  OpenJij SQA was tested from random states, and
a certified feasible state was used separately to audit encoding and backend
stability.

## Why it was tested

This is the first nontrivial cyclic input for the new Hamiltonian.  The graph
has 192 certified Hamilton paths and a unique maximum-weight path, so topology,
feasibility, and weighted optimality can be checked independently.

## Inputs and parameters

- Dataset: `test_data/CHM13/CHM13_complex144`
- Reads: `reads.normalized.fasta`
- Graph: `nodes.tsv` and `edges.tsv`
- Nodes / directed edges: 144 / 261
- Weight mode: certified `weight_dp = M - 49*d`
- QUBO cost: shifted reward, `max(weight) - weight`
- Variables / quadratic terms: 2637 / 77843
- Position bits per edge: 8
- Degree, activation, and order penalties: 144 each
- Random SQA seed: 20260922

## Output files

- `sqa_random_r1_s100_t4_seed20260922.json`: default random baseline.
- `sqa_random_late_s080_r4_s500_t4_seed20260922.json`: intermediate late-start run.
- `sqa_random_late_s050_r8_s2000_t8_seed20260922.json`: largest random run in this test.
- `sqa_reference_init_r1_s100_t4_seed20260922.json`: reference initialization with the default schedule; diagnostic only.
- `sqa_reference_init_late_s099_b100_r1_s100_t4.json`: low-fluctuation reference-state stability control.

## Result statistics

Certified-path audit:

- Reference score: 1,343,093.
- Reference QUBO energy: 63.3092132807.
- Runner-up score: 1,338,209.
- Runner-up QUBO energy: 63.6000833809.
- QUBO optimum margin: 0.2908701003.
- All feasibility residuals on the reference sample: zero.
- Minimum one-bit energy increase from the reference sample: 288.

Random SQA baseline (1 read, 100 sweeps, Trotter 4):

- Energy: 70,177,104.7162.
- Valid Hamilton path: no.
- Selected edges / sources / sinks: 239 / 128 / 12.
- Order residual L1: 6,381.
- Downhill one-bit flips remaining: 965.

Largest random SQA run (8 reads, 2000 sweeps, Trotter 8, linear `s=0.5..1`):

- Runtime: 7.66 s.
- Energy: 16,423,462.2097.
- Valid Hamilton path: no.
- Selected edges / sources / sinks: 197 / 102 / 20.
- In/out residual L1: 211 / 113.
- Activation violations: 124.
- Order residual L1: 3,229.
- Energy components: objective 118.2097; degree 81,216; activation 17,856; order 16,324,272.
- Downhill one-bit flips remaining: zero; flat flips: 8.

Reference-state stability control (`s=0.99..1`, beta 100):

- Returned the reference bitstring unchanged.
- Energy: 63.3092132807.
- Selected edges / sources / sinks: 143 / 1 / 1.
- Traced path nodes: 144.
- All feasibility residuals: zero.

## Interpretation

The input adapter, weight transformation, Hamiltonian, and decoder are
consistent: the certified optimum is a feasible QUBO state, the runner-up has
the expected higher energy, and OpenJij preserves the optimum under a
low-fluctuation control schedule.

Random-start SQA did not solve this instance.  The larger run reached a
one-bit local minimum that is still highly infeasible.  Its energy is dominated
by the order term, not the edge-weight objective.  Binary position coefficients
produce QUBO coefficients up to 9,437,184, while the certified optimum margin is
only about 0.291 after normalization.  Moving between feasible paths therefore
requires coordinated changes to edge, endpoint, and multiple position bits;
single-bit SQA dynamics encounter large barriers.

## Main conclusion

The CHM13 graph and weighted optimum are represented correctly, but this
direct binary-order QUBO is not practically solved from random states by the
current OpenJij SQA schedule and budget.  The successful reference-state run is
an encoding/backend control, not evidence that SQA discovered the solution.

## Limitations

- Only one random seed was used in the reported parameter escalation.
- No classical repair, graph-derived warm start, reverse annealing, or
  multi-bit/domain-wall order encoding was applied.
- The reference state is used only for audit and stability controls.
- Runtime is local-machine specific.

## User supplements

None recorded for this experiment yet.

## Follow-up: feasibility-only and explicit void constraints

Generated: 2026-09-22 (Asia/Tokyo)

The edge objective was set to zero while all three original constraint
penalties remained 144.  The 8-read, 2000-sweep, Trotter-8 random SQA run
returned essentially the same infeasible state as the weighted run:

- Energy: 16,423,344, entirely from constraints.
- Selected edges / sources / sinks: 197 / 102 / 20.
- Order energy: 16,324,272.
- No downhill one-bit flips remained.

This confirms that the low weighted optimum margin is not the cause of the
feasibility failure.

Two redundant explicit constraints were then tested without changing the core
Hamiltonian:

`P_void * (sum(source)-1)^2 + P_void * (sum(sink)-1)^2`.

At `P_void=144`, selected sources fell from 102 to 32, but sinks remained 20.
At `P_void=10000`, selected sources fell to 2 and sinks to 14.  Neither run
formed a valid augmented cycle; the order term remained dominant.  The explicit
constraints add 20,592 quadratic terms for 144 reads, increasing the model from
77,843 to 98,435 terms.

Output files:

- `sqa_feasibility_only_random_s050_r8_s2000_t8.json`
- `sqa_feasibility_only_explicit_void_random_s050_r8_s2000_t8.json`
- `sqa_feasibility_only_explicit_void_p10000_random_s050_r8_s2000_t8.json`

Conclusion: exact feasibility already implies one source and one sink, but the
implicit relation gives poor local guidance.  Explicit one-hot endpoint terms
improve endpoint counts and are a reasonable strengthening, but do not by
themselves solve the binary-order local-minimum problem.
