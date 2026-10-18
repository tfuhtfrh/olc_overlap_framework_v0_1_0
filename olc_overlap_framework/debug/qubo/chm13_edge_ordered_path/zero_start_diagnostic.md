# CHM13 complex144 zero-start QUBO diagnostic

Date: 2026-09-22

This is a **diagnostic experiment**, not the final cross-solver benchmark.
The Hamiltonian was reconstructed from the current repository implementation
and checked against the repository certificate:

- 144 reads, 261 directed edges
- 8 position bits per edge
- 2,637 logical variables
- 77,843 quadratic terms
- max |linear| = 4,718,736
- max |quadratic| = 9,437,184
- reference QUBO energy = 63.309213280677795 (exactly reproduced)
- reference sample has 641 one-bits / 2,637 variables = 24.31%

The local-search runs below use a lightweight custom one-flip tabu diagnostic
implemented only to study the QUBO landscape. They are **not** results from
D-Wave TabuSampler or another production solver.

## Single-flip structure at the all-zero state

At x=0, the exact one-bit energy deltas are:

| variable family | minimum delta | median delta | maximum delta |
|---|---:|---:|---:|
| edge selection x_e | -144 | -143.4197 | -143 |
| position bits z_e,k | 432 | 46,224 | 4,718,736 |
| source bits s_v | 0 | 0 | 0 |
| sink bits t_v | 2,985,840 | 2,985,840 | 2,985,840 |

Thus all 261 edge-selection moves are downhill from zero, but **every position
bit is uphill** and every sink bit has an enormous one-flip barrier. This
explains why a sparse/zero initialization naturally enters an edge-only
path-cover basin.

For the current penalties A_deg=A_act=A_ord=144, the sink barrier follows
approximately

    Delta E(t_v: 0->1) = A_ord * N^2 - A_deg
                         = 144 * 144^2 - 144
                         = 2,985,840

before any compensating position-bit changes.

## Zero-start tabu diagnostic

Using the current Hamiltonian and 30,000 tabu iterations, five seeds converged
to the same best energy:

- energy = 21,086.172056458818
- selected read edges = 142
- source bits = 0
- sink bits = 0
- degree residual L1 = 4
- activation violations = 0
- order residual L1 = 142
- objective contribution = 62.172056458817224

The selected 142 edges form **two directed path components**, of sizes 31 and
113 nodes. The only degree irregularities are their two starts and two ends.

Compared with the certified optimum bitstring, this state is 503 bits away,
but the decomposition is highly asymmetric:

- edge-selection Hamming distance = 5
- position-bit Hamming distance = 496
- source-bit Hamming distance = 1
- sink-bit Hamming distance = 1

So the graph/path structure is already close, while the binary-order encoding
makes the QUBO state very far away.

## Order-penalty sweep from zero

A_deg=A_act=144 and edge objective scale=1 were fixed. Only
A_ord/A_deg was varied.

| A_ord/A_deg | best energy | selected edges | sources | sinks | degree L1 | order L1 | valid |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 41,472 | 0 | 0 | 0 | 288 | 0 | no |
| 2 | 41,472 | 0 | 0 | 0 | 288 | 0 | no |
| 1 | 21,086.1721 | 142 | 0 | 0 | 4 | 142 | no |
| 0.5 | 10,575.2948 | 143 | 1 | 0 | 1 | 144 | no |
| 0.25 | 5,534.1721 | 142 | 2 | 0 | 2 | 144 | no |
| 0.125 | 2,942.1721 | 142 | 2 | 0 | 2 | 144 | no |
| 0.0625 | 1,646.1721 | 142 | 2 | 0 | 2 | 144 | no |
| 0.03125 | 998.1721 | 142 | 2 | 0 | 2 | 144 | no |

A scalar order-penalty change does not remove the basin. For
A_ord/A_deg >= 2, even selecting an edge ceases to be downhill from zero;
for smaller A_ord the search instead builds an edge/path-cover solution while
keeping all position bits at zero.

## Objective-scale sweep

With the constraint penalties unchanged, edge_cost_scale in
{0, 0.1, 1, 10, 100} gave essentially the same edge-only basin. Setting the
weighted objective completely to zero still produced a 143-edge, zero-position
state. This confirms that the current feasibility failure is not caused by the
edge-weight normalization.

## Explicit source/sink one-hot term

Adding

    P_void * (sum s - 1)^2 + P_void * (sum t - 1)^2

with P_void up to 10,000 did not activate a sink in the zero-start tabu
diagnostic. This is expected from the ~2.99e6 sink one-flip barrier above.

When P_void was increased to ~3e6, one source and one sink were finally forced,
but the state remained highly order-infeasible and did not approach a valid
Hamilton path. Therefore an endpoint one-hot term alone does not solve the
binary-order barrier.

## Sparse-start sweep

Very sparse random starts (0.2%--1% one-bits) returned to essentially the same
edge-only basin as all-zero. Starts with 2% or more random one-bits were worse.
This supports zero/sparse initialization as preferable to a dense random state,
but also shows that **zero is structurally pathological for the present binary
order encoding** because position and sink bits are individually uphill.

## Feasible runner-up warm start

The certified runner-up is a valid Hamilton path with

- energy = 63.600083380937576
- score = 1,338,209
- Hamming distance from the optimum = 24 bits
- minimum one-bit energy increase = 288

A 100,000-iteration custom tabu run from this valid runner-up did not improve
it in any of 10 seeds. Thus even two very nearby feasible paths are separated
by a multi-bit barrier in the current encoding.

## Interpretation

The experiments point to a representation/landscape issue more strongly than
to the weighted objective itself:

1. the QUBO/certificate mapping is consistent;
2. all-zero is much closer to the sparse ground state than a dense random
   bitstring, but it funnels one-flip dynamics into an x-only path-cover basin;
3. positive scalar tuning of A_ord cannot make the position bits downhill from
   zero;
4. the N-weighted sink term creates a ~3e6 one-flip barrier at the current
   scale;
5. feasible solutions that differ by only a few read adjacencies can differ by
   dozens of QUBO bits because every affected edge carries a binary position.

The next solver comparison should therefore keep two questions separate:

- Can a solver with collective/nonlocal dynamics (production tabu, simulated
  bifurcation, hybrid solvers) cross these barriers on the unchanged QUBO?
- If not, should the Hamiltonian replace the squared binary-integer order
  constraint with an encoding whose local coefficient range and Hamming
  geometry are better conditioned?
