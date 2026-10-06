# CHM13 latent-rank comparator experiments

Date: 2026-09-26

## Motivation

The edge selection variable should carry the overlap objective.  Order/rank
variables are only a certificate that the selected directed edges contain no
subtour.

Therefore an unselected candidate edge does not need to force its endpoint
rank variables to zero.  A selected edge only needs the implication

    x_uv = 1  =>  P_v > P_u

rather than the stronger consecutive condition P_v = P_u + 1.

## Exact ripple-borrow comparator

For every vertex v use K=ceil(log2 N) latent rank bits p[v,k].

For every candidate edge e=(u,v), compute whether P_v - P_u - 1 underflows.
Let b[e,0]=1 and let b[e,k+1] be the borrow out of bit k.  The Boolean
recurrence is

    b_out = majority(p_u, b_in, 1-p_v).

The majority relation has the exact quadratic zero-penalty representation

    R = u + b + c + u*b - u*v - b*v
        - 2*u*c - 2*b*c + 2*v*c,

where c is borrow_out.  R is nonnegative on binary assignments and is zero
exactly when c is the correct borrow bit.

These comparator relations are enforced for every candidate edge, whether
selected or not.  This does not constrain the latent ranks: for any pair of
rank values there is a unique zero-penalty borrow chain.

The final borrow b[e,K] equals 1 iff P_v <= P_u.  Therefore the only
selection-dependent order term is

    A_order * x_e * b[e,K].

A selected edge is thus required to point from lower rank to higher rank.
Unselected edges impose no rank-order requirement.

With one source/sink path-cover layer, strict rank increase eliminates every
directed subtour.

## Variable count

With source/sink variables:

    Q = M + 2N + NK + MK
      = M(K+1) + N(K+2).

For CHM13 complex144:
- N=144
- M=261
- K=8
- Q=3,789 variables.

Audit:
- 33,376 quadratic terms
- max |linear| = 2,016
- max |quadratic| = 576
- certified reference path has zero constraint residual
- consecutive ranks, shifted consecutive ranks, and nonuniform gapped ranks
  all have the same reference objective energy (up to floating roundoff).

Thus only rank ordering matters; absolute rank and spacing do not.

### Comparison

| formulation | variables | quadratic terms | max |J| |
|---|---:|---:|---:|
| current edge-position | 2,637 | 77,843 | 9,437,184 |
| pure selection + binary edge position | 3,420 | 99,380 | 9,437,184 |
| latent rank comparator + source/sink | 3,789 | 33,376 | 576 |
| bounded +1 vertex/carry prototype | 5,616 | 42,575 | 576 |
| earlier bounded vertex/carry | 7,704 | 44,924 | 576 |

The comparator needs only 369 more variables than the 3,420-variable
pure-selection/edge-position formulation while removing the power-of-two
squared coefficient pathology.

Relative to the 5,616-variable +1 formulation it saves

    M(K-1) = 261*7 = 1,827

variables.

## Important correction about the +1 prototype

The 5,616-variable conditional +1 prototype passes the known-reference audit,
but a QBSolv search found invalid assignments with energy below the certified
reference because finite Rosenberg/carry penalties can be exploited by the
expanded conditional arithmetic.  Therefore that specific implementation
should not yet be treated as a validated Hamiltonian competitor without a
stronger penalty proof or revised quadratization.

The comparator formulation avoids this particular loophole: its borrow
recurrence is itself an exact nonnegative quadratic relation and requires no
Rosenberg product auxiliaries.

## QBSolv quick test

Using the source/sink comparator model with the initial penalties
(degree=144, void=144, comparator=288, order gate=144), QBSolv found
near-complete edge sets but not a valid Hamilton path:

- subproblem 47: 138 selected edges, degree residual^2=9,
  62 selected-edge order violations;
- subproblem 100: 135 edges, degree residual^2=14,
  58 order violations;
- subproblem 200: 138 edges, degree residual^2=8,
  63 order violations.

Comparator recurrence violations were zero in every run.  Hence the solver can
maintain the borrow circuit; the remaining conflict is between completing the
edge path and arranging a globally compatible rank assignment.

Increasing the order-gate penalty removes selected-edge order violations, but
the solver then drops many selected edges (about 80--83 in the tested runs).
This is another manifestation of the topology/rank joint-update barrier rather
than a comparator-circuit correctness problem.

## Endpoint-free variant

Source/sink variables can also be removed.

Use:
1. at-most-one incoming selected edge per vertex;
2. at-most-one outgoing selected edge per vertex;
3. strict rank increase on every selected edge;
4. a dominant linear reward for selecting edges.

Conditions 1--3 make the selected graph a disjoint union of directed paths.
If a Hamilton path exists, maximizing selected-edge cardinality reaches N-1
edges, and N vertices with N-1 path-cover edges imply exactly one Hamilton
path.  The normalized overlap cost then breaks ties among maximum-cardinality
paths.

This gives

    Q = M + NK + MK
      = M(K+1) + NK.

For CHM13:

    Q = 3,501.

Audit:
- 12,262 quadratic terms
- max |linear| = 4,032
- max |quadratic| = 1,152
- certified reference path is valid with 143 edges and zero constraint
  violations
- consecutive and gapped ranks have identical objective energy.

The large reduction in couplers occurs because removing source/sink also
removes the two dense one-hot squares over all N endpoint variables.

A first generic QBSolv run did not complete the path: it settled at about
82--83 selected edges while satisfying degree-conflict and order constraints.
Thus this compact formulation is structurally clean but needs a search strategy
that can jointly grow/re-rank path components; simply making every hard
constraint strong causes the solver to prefer a smaller valid path cover.

## Current interpretation

For static single-QUBO formulations, the strict-rank comparator is currently
the best variable/coefficient compromise tested:

- independent weighted edge variables;
- unselected edges do not force ranks to zero;
- bounded local coefficients;
- no product auxiliaries;
- fewer variables and couplers than the +1/carry constructions.

The remaining solver problem is no longer arithmetic coefficient explosion.
It is the coordinated move needed to add/exchange edges while simultaneously
rearranging latent ranks.
