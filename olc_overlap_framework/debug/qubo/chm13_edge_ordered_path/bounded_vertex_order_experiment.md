# CHM13 bounded-coefficient vertex-order QUBO experiment

Date: 2026-09-22

This experiment follows the intended separation:

- real-edge variables are selection / weighted-objective variables and no longer
  encode the numeric order value;
- read-order variables use local bitwise successor relations whose coefficients
  do not grow with the numerical rank;
- source/sink variables are only virtual endpoint choices in the degree
  constraints. They have no edge reward and no position-order penalty.

The implementation is isolated in
`experimental_chm13_vertex_carry_qubo.py` on the benchmark branch.

## Formulation

For every read v, introduce K=ceil(log2 N) binary position bits p[v,k].

For every real candidate edge e=(u,v), x_e is the edge-selection variable and
carries the existing normalized overlap cost.

The exact path constraints are

    s_v + sum_{u->v} x_uv = 1
    t_v + sum_{v->w} x_vw = 1

and the virtual node has exactly one outgoing source edge and one incoming sink
edge.

No source/sink position is fixed. This is deliberate: once there is exactly one
source and one sink, and every selected real edge strictly increments the read
position, any disconnected read-only cycle is impossible. The absolute rank can
therefore float by a constant offset.

For a selected real edge u->v, enforce p_v = p_u + 1 without powers of two in
one squared integer residual. Rosenberg copies

    a[e,k] = x_e * p[u,k]
    b[e,k] = x_e * p[v,k]

and local carry bits implement

    a[e,k] + c[e,k] = b[e,k] + 2*c[e,k+1]

with c[e,0]=x_e and the final carry fixed to zero.

All local arithmetic coefficients are 1, 2, or 3 before multiplying by the
chosen penalty constants.

## Equal-penalty audit

Using 144 for degree, void-degree, successor, and product penalties:

| metric | current edge-position QUBO | bounded vertex-order QUBO |
|---|---:|---:|
| logical variables | 2,637 | 7,704 |
| quadratic terms | 77,843 | 44,924 |
| max absolute linear coefficient | 4,718,736 | 720 |
| max absolute quadratic coefficient | 9,437,184 | 576 |
| certified optimum energy | 63.30921328 | 63.30921327 |

The new construction uses about 2.92x as many variables but only about 57.7% as
many quadratic couplings. The largest quadratic coefficient is reduced by a
factor of 16,384.

The encoded certified optimum has zero constraint energy. The certified
runner-up has energy 63.60008338, so the weighted feasible-path ordering is
preserved.

The optimum bitstring is still sparse: about 1,766 / 7,704 bits are one
(~22.9%).

## All-zero local geometry

With equal 144 penalties, at the all-zero bitstring:

- every real edge-selection x_e has a downhill one-flip move (about -144 plus
  its normalized edge cost);
- every source bit has delta E = -288;
- every sink bit has delta E = -288;
- every read-position bit is exactly neutral before a selected edge couples to
  it;
- product auxiliaries and carry bits are uphill.

This removes the previous ~2.99e6 one-flip sink barrier and the 432..4,718,736
position-bit barrier created by the squared binary integer order residual.

## Interactive zero-start diagnostic

A lightweight sampled one-flip tabu routine was used only to inspect the
landscape; this is not a production TabuSampler result.

With equal penalties (144 each), the search activates positions but does not
maintain the degree constraints strongly enough.

A more useful diagnostic setting was:

    degree penalty       = 1000
    void-degree penalty  = 2000
    successor penalty    = 72
    product penalty      = 144

This remains a bounded-coefficient QUBO:

- 7,704 variables
- 45,185 quadratic terms
- max |linear| = 3,000
- max |quadratic| = 4,000

From the all-zero state, a 150,000-iteration diagnostic run reached:

- energy = 4,277.4252
- selected real edges = 141
- source count = 1
- sink count = 1
- read degree residual square = 4
- virtual-node residual = 0
- Rosenberg product violations = 0
- bitwise successor residual square = 3
- only 3 selected edges disagree with the decoded +1 successor relation
- 133 distinct decoded position values

The selected graph consists of three directed path components of sizes
19, 12, and 113. The source is the first node of the 19-node component. The
remaining degree violations are exactly the four open component-boundary
vertices.

The three components are structurally close to the certified/reference order:

    0 -> 1 -> ... -> 18

    19 -> 20 -> ... -> 30

    32 -> 33 -> 31 -> 34 -> ... -> 143

so the remaining graph repair is concentrated around the component joins and
the known 31/32/33/34 ordering alternative.

## Structured component-position projection

As a diagnostic only, the three path components above were concatenated in
their discovered order and their position/carry auxiliaries were initialized
consistently, without using the certified optimum order.

Before further search this gives:

- energy = 4,061.4252
- degree residual square = 4
- product violations = 0
- successor residual = 0
- 144 distinct positions

Five additional 150,000-iteration one-flip tabu runs improved this to a common
best basin around:

- energy = 2,134.1721
- 142 selected edges
- one source and one sink
- degree residual square = 2
- product violations = 0
- successor residual square = 1

The remaining graph has two components of sizes 31 and 113, with the only
degree break between the local region ending at node 30 and the component
starting at node 32. This is already much closer to a valid Hamilton path than
the original binary-edge-order zero-start basin.

## Feasible-state barrier

The bounded-coefficient formulation does not remove all one-flip barriers.

The certified runner-up is a zero-constraint feasible state with energy
63.60008338. Its Hamming distance from the certified optimum in the new
encoding is 72 bits. Both feasible states are strict one-flip local minima;
the minimum one-bit energy increase is 144.

A short custom tabu diagnostic started at the runner-up did not reach the
ground state. Thus bounded local coefficients substantially improve the
zero-start feasibility landscape, but a solver with genuinely nonlocal /
collective moves is still desirable for transitions between complete feasible
paths.

## Interpretation

The experiment supports separating two issues that were mixed in the original
edge-position QUBO.

1. **Coefficient conditioning.** The powers-of-two squared integer residual
   creates coefficient ranges up to ~1e7 and a huge sink barrier. The local
   carry construction removes that scaling completely.

2. **Combinatorial move size.** Even after coefficients are bounded, changing
   one valid Hamilton path into another may require a coordinated change of
   several edge, rank, copy, and carry bits. A pure one-flip optimizer can still
   be trapped.

The next useful comparison is therefore:

- production Tabu / Simulated Bifurcation / hybrid solver on the unchanged
  current QUBO;
- the same solvers on this bounded-coefficient QUBO;
- compare feasibility hit rate first, then weighted score among feasible
  samples.

This experiment does not yet establish that the larger 7,704-variable
formulation is globally easier; it establishes that it more closely matches the
desired Hamiltonian semantics and removes the most obvious coefficient-growth
pathology.
