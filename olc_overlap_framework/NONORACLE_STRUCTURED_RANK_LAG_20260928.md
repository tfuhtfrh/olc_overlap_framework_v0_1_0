# Non-oracle structured optimization and rank-certificate lag — 2026-09-28

## Summary

Starting from the reproduced CHM13 complex144 stuck checkpoint, a non-oracle
structured optimizer reaches the independently certified optimum without using
the target path or target edge set.

The move sequence is

    open alternating trail k=1
      -> R3
      -> compound R2 + R2

with path scores

    1,299,598
      -> 1,326,265
      -> 1,343,093

The final score is the certified optimum.

The optimizer uses only:
- current selected graph;
- residual tail/head bipartite geometry;
- current feasibility/topology;
- the intended weighted-path objective.

The value 1,343,093 is used only after the run to label a ground-state hit.

## Open-trail stage

From the stuck state:
- degree conflicts = 0
- one source
- one sink
- one cycle
- invalid Hamilton path

The residual graph contains only one shortest reachable sink-transfer candidate
from the current sink under the implemented shortest-path construction.

That open alternating trail has:
- residual length = 2
- k = 1
- one add + one remove

It produces immediately:
- degree conflicts = 0
- cycles = 0
- valid Hamilton path
- score = 1,299,598

Thus the endpoint-fiber repair is discoverable without oracle information.

## Feasible closed refinement

From the repaired Hamilton path, the optimizer enumerates:
- applicable R2 moves;
- applicable R3 moves;
- pairs of disjoint simultaneously-applicable R2 moves.

It keeps only outcomes that remain Hamilton paths and improve the weighted path
score.

Greedy best-improvement selection gives:

1. R3 template 65:
   - score 1,299,598 -> 1,326,265
   - six edge bits flipped

2. compound R2 pair templates 13 and 27:
   - score 1,326,265 -> 1,343,093
   - eight edge bits flipped

The latter is exactly the feasibility-preserving macro version of the two R2
components seen in the oracle symmetric-difference decomposition.

## Rank mechanism

Rank is not sampled as a QUBO variable in the projected solver.

At each outer step:
1. an integer rank certificate is fixed;
2. the inner edge-only BQM assigns a linear penalty to every selected edge that
   points backward under that fixed rank;
3. the inner solver changes only edge-selection variables;
4. after the inner solve, rank is recomputed from the selected graph using
   SCC condensation plus an Eades-style ordering inside nontrivial SCCs.

For a DAG Hamilton path, reprojecting rank yields a topological order and the
selected path has zero backward edges.

For a cyclic state, at least one backward edge remains in each nontrivial SCC
under the projected order, so the order term acts as a cycle-breaking
certificate/penalty.

## Rank-certificate lag

The fixed certificate can become stale during an inner move.  The new topology
may be globally better but temporarily look worse under the old rank.

The non-oracle successful route quantifies this directly at order penalty 4.0.

### Open k=1 endpoint repair

Under stale rank:
    Delta H = +3.63230301947

After independently reprojecting rank on both states:
    Delta H_self-projected = -4.36769698053

### R3 improvement

Under stale rank:
    Delta H = +2.41182776487

After rank reprojection:
    Delta H_self-projected = -1.58817223513

### Compound R2+R2 improvement

Under stale rank:
    Delta H = +6.99779643857

After rank reprojection:
    Delta H_self-projected = -1.00220356143

Therefore every useful macro move in this route is classified as uphill by the
old certificate and downhill by the reprojected certificate.

This is strong evidence for a **rank-certificate lag / stale-rank barrier**.

It is distinct from Trotter freezing:
- Trotter freezing is a path-integral sampling artifact caused by strong
  imaginary-time coupling and local slice updates;
- rank lag is an auxiliary-objective artifact caused by holding a classical
  topology certificate fixed while the edge topology changes.

## Consequences for algorithm design

Three regimes should be distinguished.

### 1. Static-QUBO / equilibrium-sampling interpretation

If rank is fixed, the inner sampler has a well-defined static BQM.  Reprojecting
rank after every proposed move changes the energy function itself and is not an
ordinary equilibrium Monte Carlo chain for one Hamiltonian.

Thus "dynamic rank after every proposal" should not be described as standard
SQA sampling of a fixed QUBO.

### 2. Hybrid optimization interpretation

For optimization, rank can be treated as a classical block variable/certificate.
Then the natural algorithm is block-coordinate or projected optimization:

    structured edge move
      -> project_rank
      -> rebuild edge energy
      -> structured edge move
      -> ...

This is exactly the interpretation already used by the projected edge hybrid,
but the current outer-loop granularity is too coarse for some topology changes.

### 3. Feasible-path refinement

Once a valid Hamilton path has been reached, its own projected rank has zero
backward-edge penalty.  At that point the rank term is unnecessary for moves
that are explicitly constrained to remain Hamilton-feasible.

A practical refinement phase can therefore:
- generate only feasible open/closed structured moves;
- compare them using the true weighted overlap objective;
- use rank only for fallback/cycle-repair proposals.

This removes the stale-rank barrier from the most important late optimization
phase.

## Recommended next implementation

Use a phase-aware projected optimizer:

1. **repair phase**
   - cardinality/degree repair;
   - open alternating trails for endpoint-fiber changes;
   - reproject rank immediately after accepted macro repairs.

2. **feasible refinement phase**
   - when current state is a Hamilton path, propose feasible R2/R3 and small
     compound reconnects;
   - rank candidates by true path score/cost;
   - do not let stale rank veto a feasible improving move.

3. **cycle excursion / fallback**
   - when no feasible direct improvement exists, optionally allow short
     controlled cycle excursions;
   - reproject rank at the macro boundary rather than waiting a full outer
     iteration.

For SQA studies, this should be described as a projected/hybrid algorithm, not
a pure fixed-Hamiltonian SQA.
