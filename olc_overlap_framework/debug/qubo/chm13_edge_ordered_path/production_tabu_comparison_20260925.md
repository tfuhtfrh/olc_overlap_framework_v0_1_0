# CHM13 production TabuSampler comparison

Date: 2026-09-25

Solver:
- package: dwave-samplers 1.8.0
- class: dwave.samplers.TabuSampler
- algorithm: MST2 multistart tabu search
- this is a production solver package, not the earlier handwritten one-flip tabu diagnostic.

Ground/reference energies:
- current edge-position QUBO: 63.309213280677795
- bounded vertex-order QUBO: 63.30921326903626 (equal-144 penalties)
- bounded vertex-order QUBO: 63.30921326891985 (balanced penalties)

## Quick test, default MST2 settings

One read per initialization; approximately 2 s per read, with an additional
approximately 5 s zero-start run.

### Current edge-position Hamiltonian

Model:
- 2,637 variables
- 77,843 quadratic terms
- max |linear| = 4,718,736
- max |quadratic| = 9,437,184

Best short results:
- zero start, ~2 s:
  - energy 24,690.7735
  - 117 selected edges
  - source/sink = 0/0
  - read-in/read-out violations = 27/27
  - activation violations = 0
  - order residual L1 = 117
- sparse 1% start, ~2 s:
  - energy 1,573,850.1575
  - 129 selected edges
  - source/sink = 47/2
  - order residual L1 = 646
- random start, ~2 s:
  - energy 12,576,801.0883
  - 212 selected edges
  - source/sink = 114/19
  - order residual L1 = 2658
- zero start, ~5 s:
  - same best basin as the ~2 s run: energy 24,690.7735

No run reached a valid Hamilton path.

Interpretation:
- production Tabu strongly confirms that zero/sparse initialization is preferable
  to a dense random bitstring for the current encoding;
- however, the short production Tabu run did not outperform the earlier
  handwritten diagnostic tabu result (energy ~21,086 with 142 selected edges);
- the binary edge-position formulation remains trapped in an edge-only /
  order-infeasible basin.

### Bounded vertex-order Hamiltonian, equal penalties

Penalties:
- A_degree = 144
- A_void = 144
- A_successor = 144
- A_product = 144

Model:
- 7,704 variables
- 44,924 quadratic terms
- max |linear| = 720
- max |quadratic| = 576

Best short results:
- zero start, ~2 s:
  - energy 21,643.2828
  - 82 selected edges
  - source/sink = 2/2
  - degree residual square = 120
  - product violations = 16
  - successor residual square = 12
- sparse 1% start, ~2 s:
  - energy 21,360.6386
  - 78 selected edges
  - source/sink = 2/2
  - degree residual square = 128
  - product violations = 7
  - successor residual square = 11
- random start, ~2 s:
  - energy 45,360
  - 0 selected edges
  - source/sink = 5/5
- zero start, ~5 s:
  - energy 22,219.7440
  - 76 selected edges
  - source/sink = 2/2

No run reached a valid Hamilton path.

Interpretation:
- bounded coefficients alone do not make the 7,704-variable formulation easy
  for generic MST2 tabu;
- equal penalties underweight the degree/path-completion part relative to the
  expanded auxiliary-variable search space;
- the solver spends effort satisfying local product/successor relations while
  leaving many reads disconnected.

### Bounded vertex-order Hamiltonian, diagnostic balanced penalties

Penalties:
- A_degree = 1000
- A_void = 2000
- A_successor = 72
- A_product = 144

Model:
- 7,704 variables
- 45,185 quadratic terms
- max |linear| = 3,000
- max |quadratic| = 4,000

Best short results:
- zero start, ~2 s:
  - 115 selected edges
  - source/sink = 1/1
  - degree residual square = 56
  - product violations = 0
  - successor residual square = 115
- sparse 1% start, ~2 s:
  - 123 selected edges
  - source/sink = 1/1
  - degree residual square = 40
  - product violations = 0
  - successor residual square = 130
- random start, ~2 s:
  - 131 selected edges
  - source/sink = 1/1
  - degree residual square = 50
  - product violations = 20
  - successor residual square = 408
- zero start, ~5 s:
  - same basin as the ~2 s zero-start run.

No run reached a valid Hamilton path.

Important: absolute QUBO energies are not directly comparable with the equal-144
model because the penalty scales are different.

Interpretation:
- the stronger degree/void penalties do exactly what they were intended to do:
  source/sink become 1/1 and product constraints can be satisfied;
- the remaining failure is concentrated in path completion and successor
  consistency;
- this formulation is therefore easier to diagnose term-by-term, but its larger
  auxiliary-variable space still defeats a short generic tabu search.

## Controlled-work quick rerun

A second run limited MST2 work with:
- coefficient_z_first = 500
- coefficient_z_restart = 125
- lower_bound_z = 100000
- num_restarts = 25

and 1.5--3 s timeouts.

The same qualitative basins were recovered:
- current formulation zero-start: energy 24,690.7735, 117 edges, 0/0 endpoints;
- bounded equal-144: roughly 89 edges, 2/2 endpoints, with remaining degree/product/successor residuals;
- bounded balanced: 115 edges, 1/1 endpoints, product residual 0, successor residual ~115.

So the main short-run conclusion is not an artifact of the default MST2 internal
work limits.

## Current conclusion

The mature TabuSampler improves the situation relative to random-start SQA, but
it does not solve either CHM13 QUBO directly in a few seconds.

For the current 2,637-variable edge-position QUBO, zero start is clearly the
right initialization but the order encoding still traps the search.

For the 7,704-variable bounded-coefficient QUBO, coefficient conditioning is
much healthier, but the larger auxiliary-variable space requires either:
1. better penalty balancing;
2. structured initialization/repair;
3. longer or more problem-aware tabu neighborhoods;
4. a solver with more collective/nonlocal dynamics.

This result should not be interpreted as showing that the bounded formulation
is worse in general: the two formulations have very different variable counts,
penalty structures, and local-search geometries, and neither penalty set has
been optimized.
