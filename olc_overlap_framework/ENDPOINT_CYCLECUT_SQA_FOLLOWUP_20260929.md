# Endpoint-transfer and symmetric cycle-cut SQA follow-up — 2026-09-29

## 1. Symmetric endpoint-transfer proposal

For candidate edges sharing a head,

    (a -> v) <-> (b -> v),

a two-bit swap preserves the incoming degree of v and total edge count while
transferring one unit of outgoing degree between a and b.  In a degree-correct
N-1-edge state this moves the sink defect.

The source analogue uses edges sharing a tail,

    (u -> a) <-> (u -> b),

and transfers the source defect.

If unordered templates are precomputed statically and sampled uniformly, each
proposal is an involution and has the same forward/reverse proposal
probability.  Ineligible templates are null proposals.  Thus the kernel is a
clean symmetric MC move; no oracle endpoint is needed.

## 2. MC proposal-mass benchmark from the stuck checkpoint

Rank was removed (order penalty 0) so that the experiment isolates proposal
geometry.  The fixed-cardinality objective retained weighted edge cost and
degree-conflict penalties.

24 seeds, 8000 proposals per run:

### Uniform 1<->1 exchange

- HP hits: 9/24 = 37.5%
- median first HP proposal among hits: 5120
- best first repaired HP score: 1,299,598

### Same-head sink-transfer templates

- HP hits: 24/24 = 100%
- median first HP proposal: 96
- best repaired HP score: 1,299,598

### Same-head + same-tail endpoint union

- HP hits: 24/24 = 100%
- median first HP proposal: 197
- best repaired HP score: 1,299,598

The result is insensitive to beta in {4,1,0.25} for this checkpoint because the
useful rank-free endpoint transfer is downhill and most arbitrary exchanges are
destroyed by the large degree penalty.

The main result is proposal mass: explicit endpoint geometry improves the
median hit time by roughly two orders of magnitude relative to uniform
exchange.

## 3. Endpoint move inside a P=8 path-integral sampler

The same static same-head templates were tested at

- P = 8
- beta = 4
- Gamma: 3 -> 0.03
- 24 seeds
- 8000 proposal attempts

Two update styles were compared.

### Single-slice endpoint swap

- HP hits: 24/24
- median first HP attempt: 183
- mean accepted moves before stopping: 1
- minimum observed action delta: +0.221

The highly focused two-bit proposal is still thermally accessible even with the
single-slice Trotter term once the stale-rank penalty is removed.

Note: the stopping criterion is that one slice becomes a valid Hamilton path;
the other slices need not agree.

### Whole-worldline endpoint swap

Apply the same two-bit involution to every Trotter slice.  If the two bits are
complementary in every slice, fixed cardinality is preserved slice by slice and

    Delta S_Trotter = 0

exactly, because each affected worldline is globally inverted.

Results:

- HP hits: 24/24
- median first HP attempt: 194
- mean accepted moves before stopping: 1
- minimum observed action delta: -1.471

All slices move together, so the repaired topology is replica-consistent.

This is a valid nonlocal path-integral MC update for the static rank-free
objective.  It should not be interpreted as a physical transverse-field
tunneling process; it is an improved Monte Carlo update kernel.

## 4. Exact symmetric cycle cut

The current projected rank penalty chooses a feedback ordering and linearly
penalizes whichever selected edges happen to be backward under that ordering.
The backward-edge audit showed that the successful structured route does not
follow those selected backward edges; in later valid-path states, useful new
edges are themselves backward under the old rank.

A symmetric alternative was implemented as the lazy inequality

    sum_{e in C} x_e <= |C|-1

for each discovered directed cycle C.

The inequality is encoded exactly as QUBO using a bounded nonnegative slack:

    A_cut (sum_{e in C} x_e + s_C - (|C|-1))^2.

Consequences:
- selecting the entire known cycle costs at least A_cut;
- any subset of at most |C|-1 cycle edges can achieve zero cut penalty;
- no particular edge is chosen as the one that must be removed.

One cycle is separated from each nontrivial SCC after an outer SQA solve and
cuts are accumulated lazily.

## 5. First OpenJij SQA rank-vs-cycle-cut benchmark

Common setting:
- CHM13 complex144
- OpenJij SQA
- 261 edge variables before cut auxiliaries
- count and degree penalties retained
- 12 outer iterations
- 4 seeds per configuration

Results:

### Stale projected rank

- feasible runs: 1/4
- ground hits: 0
- best score: 1,319,932
- variables: 261

### Cycle cut, A_cut=4

- feasible runs: 2/4
- ground hits: 0
- best score: 1,291,414
- typical final cuts: about 9
- maximum variables observed: 309

### Cycle cut, A_cut=16

- feasible runs: 1/4
- ground hits: 0
- best score: 1,301,902
- typical final cuts: about 9
- maximum variables observed: 291

Interpretation:
- symmetric cycle cuts show a modest feasibility signal (2/4 versus 1/4 at
  A_cut=4), but the sample is small;
- no configuration reaches the certified optimum;
- accumulated slack variables/couplers increase SQA search burden;
- cycle cuts are better aligned with the semantics "forbid this cycle" but are
  not by themselves a weight-refinement mechanism.

Therefore cycle cuts should not simply replace rank and be expected to solve
the whole problem.  They are best used as symmetric acyclicity constraints,
while endpoint/open-trail and R2/R3/compound proposals handle topology motion
and weighted refinement.

## 6. Updated SQA direction

The evidence supports separating three effects:

1. **proposal sparsity**
   - solved strongly by endpoint-transfer / alternating-trail proposals;

2. **stale-rank artificial barriers**
   - avoid using a fixed old ordering as a direction signal;
   - cycle cuts are a more symmetric representation of known cyclic defects;

3. **Trotter freezing**
   - use worldline/cluster updates for structured multi-bit moves where
     appropriate.

Near-term SQA architecture:

    edge/count/degree energy
      + lazy symmetric cycle cuts
      + static symmetric endpoint-transfer proposals
      + closed R2/R3 structured proposals
      + worldline/cluster path-integral updates

This is an SQA-assisted hybrid sampler.  The cycle cuts modify the QUBO between
outer iterations, while the structured kernels improve mixing within an
iteration.

The next benchmark should combine cycle cuts and endpoint-transfer kernels in a
single run from multiple non-hand-picked states, rather than testing them only
in isolation.
