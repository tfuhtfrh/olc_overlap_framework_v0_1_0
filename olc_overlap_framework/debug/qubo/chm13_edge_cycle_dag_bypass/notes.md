# CHM13 cyclic input with the legacy edge-cycle DAG QUBO

Generated: 2026-09-22 (Asia/Tokyo)

## What this tested

The legacy `EdgeCycleCoverDAGQUBOHamiltonian` was applied directly to the
cyclic CHM13 complex144 graph.  Its DAG validation was bypassed only in the
test harness; the Hamiltonian terms were not changed.

## Why it was tested

The edge-ordered cyclic Hamiltonian did not reach feasibility from random SQA
states.  This comparison tests whether the much smaller legacy degree-only
model converges, and shows precisely what is lost when its DAG assumption is
removed.

## Inputs and parameters

- Dataset: `test_data/CHM13/CHM13_complex144`
- Reads / directed edges: 144 / 261
- Variables: `E + 2N = 549`
- Quadratic terms: 21,434
- Degree penalty: 100
- Edge reward scale: 1
- Reward: normalized CHM13 `weight_dp`, maximum 16,826
- SQA seed: 20260922

## Output files

- `sqa_random_default_r8_s2000_t8_seed20260922.json`
- `sqa_random_s050_r8_s2000_t8_seed20260922.json`
- `sqa_random_s050_r32_s5000_t8_seed20260922.json`

## Result statistics

Certified Hamilton path plus void:

- Energy: -79.8224771188.
- Reward sum: 1,343,093.
- One cycle containing all 144 reads and the void node.
- All degree constraints satisfied.

Exact maximum-weight degree-feasible cycle cover:

- Energy: -79.8368596220.
- Reward sum: 1,343,335, which is 242 above the certified Hamilton path.
- Two cycles: a 141-read void component and one 3-read read-only cycle.
- The read-only cycle is
  `28903706 -> 89655134 -> 106236189 -> 28903706` using full read IDs in the JSON.

Default random SQA (8 reads, 2000 sweeps, Trotter 8):

- Runtime: 1.29 s.
- Energy: -76.8362653037.
- Reward sum: 1,292,847.
- All node and void degree constraints satisfied.
- Two cycles: a 133-read void component and one 11-read read-only cycle.
- Energy above the exact cycle cover: 3.0005943184.
- No downhill or flat one-bit flips remained.

Late-start SQA (`s=0.5..1`) returned a three-component cycle cover with sizes
140, 2, and 2 and energy -76.7160347082.  Increasing that schedule to 32 reads
and 5000 sweeps returned the same local minimum.

## Interpretation

The legacy model converges much more readily because it has no binary order
variables and explicitly enforces one virtual source and sink.  It successfully
reaches the degree-feasible cycle-cover manifold from a random state.

The result is not a Hamilton cycle once the DAG premise is removed.  On a DAG,
read-only directed cycles cannot exist, so a degree-feasible cover with one void
component must be the desired path.  On this cyclic graph, subtours are legal
and the exact weighted cycle-cover optimum is itself disconnected.  It scores
slightly better than the certified Hamilton path, so more successful
optimization would converge toward the wrong topology rather than repair it.

## Main conclusion

Ignoring only the DAG check makes the old QUBO easy enough for SQA to satisfy
all local degree constraints, but changes the solved problem from Hamilton path
to maximum-weight cycle cover.  The experiment confirms both why the old model
converges and why it cannot replace a subtour-eliminating cyclic Hamiltonian.

## Limitations

- The exact comparison solves the degree-feasible assignment/cycle-cover
  problem, not the Hamilton-cycle problem.
- Reported SQA runs use one seed.
- No subtour cuts, iterative cycle merging, or repair were applied.

## User supplements

The user explicitly requested bypassing the DAG check to observe convergence.
