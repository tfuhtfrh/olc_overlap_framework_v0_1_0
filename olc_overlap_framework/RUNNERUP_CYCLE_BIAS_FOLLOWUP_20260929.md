# Runner-up geometry and smarter acyclicity bias — 2026-09-29

## 1. Runner-up -> certified ground geometry

Exact Hamilton-path enumeration gives:

- ground score: 1,343,093
- runner-up score: 1,338,209
- gap: 4,884

The two paths differ in exactly six edge bits:
- 3 removed
- 3 added

Their symmetric difference is one connected alternating 6-cycle in the
tail/head bipartite representation.

Equivalent interpretations:
- one genuine R3 reconnect;
- one single-read relocation;
- relocate read m64062_190803_042216/28903706/ccs from path index 31 to 33.

The direct structured move is R3 template 18.

### R2 alternative

Ground is also reachable using R2 only, but the shortest route requires two
R2 moves and necessarily leaves the Hamilton-path manifold:

    runner-up HP
      -> one-cycle state
      -> ground HP

There is no feasible-only R2 route between these two paths.

Rank-free fixed-cardinality energies along one shortest R2 route:

1. runner -> cycle:
   Delta H = -0.3052825919

2. cycle -> ground:
   Delta H = +0.0144124829

Net runner -> ground:
   Delta H = -0.2908701090

Therefore the rank-free weight+degree objective actually assigns the cyclic
intermediate a slightly lower energy than the ground Hamilton path.  This is a
concrete counterexample to relying on weight+degree+count alone for acyclicity.

It also sets the relevant cycle-penalty scale: for this local transition an
exact full-cycle penalty only needs to exceed about 0.0144 to reverse the
cycle-vs-ground preference.  Penalties of order 1--16 are therefore very large
relative to this local barrier.

## 2. Rank and Trotter barriers on the direct R3

Direct runner -> ground R3:

- rank-free fixed-cardinality Delta H: -0.2908701090
- stale runner rank with A_order=4: +3.7091298910

Thus stale rank flips a genuinely downhill improvement into a substantial
uphill transition.

At beta=4, P=8, Gamma=0.03:

- K = 2.099890037
- direct six-bit single-slice action estimate: +50.2519
- whole-worldline direct action: -1.16348

This separates two artificial barriers:
- stale rank certificate;
- single-slice Trotter/domain-wall cost.

## 3. Direct runner-up R3 path-integral test

Start all P=8 slices exactly at the runner-up on the rank-free
fixed-cardinality BQM.  Use the static R3 template family as symmetric
involutive proposals.

24 seeds, 4000 attempts:

### single-slice R3
- ground hits: 17/24 = 70.8%
- median first ground hit: 66 proposals
- total proposals that were exactly the ground R3: 321
- accepted ground transitions: 17

### whole-worldline R3
- ground hits: 24/24 = 100%
- median first ground hit: 37 proposals
- exact ground R3 proposed: 24
- exact ground R3 accepted: 24

Whole-worldline R3 applies the same six-bit involution to every Trotter slice,
so the Trotter interaction is invariant.  This resolves the runner-up -> ground
transition reliably.

The result does not mean the full solver has already reached ground from random
starts.  It establishes that the final runner-up barrier is solvable by the
structured path-integral kernel.

## 4. Smarter acyclicity bias: first same-seed benchmark

OpenJij SQA, 12 outer iterations, four identical seeds.

Compared:
- none: no acyclicity term
- rank: original projected backward-edge rank
- avg_1 / avg_4: cycle-averaged rank bias
- slack_4: exact cycle cut using bounded slack
- break_4: exact cycle cut using symmetric break-choice auxiliaries

### Cycle-averaged rank

For a simple directed L-cycle, all L cyclic rotations of the linear rank are
equivalent.  Each edge is backward in exactly one rotation.  Averaging over
these rank certificates gives

    H_avg = (A/L) sum_{e in C} x_e.

This removes the arbitrary choice of a single backward edge and can be viewed
as the marginal probability that an edge is backward under an ensemble of
equivalent ranks.

Results:

| method | feasible | ground | best score | max vars |
|---|---:|---:|---:|---:|
| none | 3/4 | 0 | 1,338,209 | 261 |
| rank | 1/4 | 0 | 1,324,816 | 261 |
| avg_1 | 1/4 | 0 | 1,296,205 | 261 |
| avg_4 | 3/4 | 0 | 1,325,446 | 261 |
| slack_4 | 3/4 | 0 | 1,309,005 | 275 |
| break_4 | 3/4 | 0 | 1,325,688 | 375 |

Interpretation:
- deterministic stale rank is the weakest feasibility guide in this batch;
- distributing rank pressure over the whole cycle (avg_4) restores 3/4
  feasibility without auxiliary variables;
- exact symmetric cuts also restore 3/4 feasibility, but their quadratizations
  add search overhead;
- no-acyclicity remains best on weighted score for this particular benchmark,
  reaching the exact runner-up, but the R2 route above proves that this is not a
  generally valid Hamilton-path energy model.

## 5. Exact cut encodings

### Slack cut

    A (sum x_e + s - (L-1))^2

is exact: all states with <=L-1 selected cycle edges can have zero penalty;
full-cycle selection cannot.

It uses few auxiliary bits but creates relatively dense couplings among cycle
edges after squaring.

### Break-choice cut

Introduce one b_e per cycle edge:

    A (sum b_e - 1)^2 + A sum b_e x_e.

If any cycle edge is absent, choose b on that edge for zero penalty.  If the
entire cycle is selected, minimum penalty is A.

It avoids choosing a predetermined break edge but used substantially more
variables in the first benchmark.

### Sparse Rosenberg product cut

A further test was added using a sparse quadratization of

    A product_{e in C} x_e.

The product is represented by a chain of AND auxiliaries with Rosenberg
penalties.  This preserves the exact "penalize only the full cycle" semantics
while avoiding the original-edge clique produced by the slack square.

## 6. Current interpretation

Acyclicity and motion should remain separate concepts:

- acyclicity bias: cycle-level and symmetric, not one arbitrary backward edge;
- dynamics: endpoint transfer, R2/R3, compound reconnect, worldline/cluster
  updates.

For rank-like methods, the promising generalization is not a single Eades
ordering but an ensemble/marginal feedback score:

    p_e = fraction of low-feedback orderings in which e is backward,
    H = A sum_e p_e x_e.

For a simple cycle this reduces exactly to the uniform cycle-average bias.
For more complicated SCCs it can express which edges are consistently part of
feedback structure without declaring one arbitrary edge to be "the" bad edge.

The runner-up analysis also suggests that acyclicity penalties should be set at
the local objective scale rather than using very large fixed schedules.  The
observed cycle-to-ground barrier is only 0.0144 in the rank-free projected BQM.
