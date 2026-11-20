# Fixed-cardinality kernel experiments (2026-09-28)

## Question

Keep the projected edge objective but enforce exactly N-1 selected edges in the
sampler state space instead of with the dense count-square penalty.  Compare
symmetric multi-bit proposal kernels.

CHM13 complex144:
- N=144
- M=261
- fixed Hamming weight = 143

The sampled energy is therefore

    H = H_weight + H_degree_conflict + H_order(rank)

because H_count is identically zero on the fixed-cardinality manifold.

## Symmetric kernels

### Exchange

Uniformly choose one selected and one unselected edge and swap them.

At fixed cardinality, forward and reverse proposal probabilities are both

    1 / (143 * 118).

This is a 2-bit move and can change the in/out-degree profile.

### Reconnect

Precompute a static set of directed 2-switch templates

    (a->b, c->d) <-> (a->d, c->b).

A template is chosen uniformly from the same fixed set in every state.  If the
current state occupies exactly one side of the template, flip all four bits;
otherwise perform a null proposal.

This is an involution, symmetric without a Hastings correction, preserves edge
count, and also preserves every vertex's in/out degree.

There are 41 such templates in complex144.

## Initial benchmark

Settings:
- Trotter slices: 8
- sweeps: 500
- reads: 2
- beta: 4
- gamma: 3 -> 0.03
- 12 outer projected-rank iterations
- 8 seeds/configuration

Configurations:
1. exchange only
2. static 50/50 exchange/reconnect
3. reconnect probability 0.8 -> 0.2 within each anneal
4. reconnect probability 0.2 -> 0.8 within each anneal

No configuration reached a Hamilton path at this budget.

Final mean degree conflicts:
- exchange only: 4.875
- static 50/50: 8.0
- reconnect 0.8 -> 0.2: 8.875
- reconnect 0.2 -> 0.8: 8.375

Exchange-only was clearly better at repairing a random 143-edge state's degree
profile.  Reconnect cannot repair degree counts by construction, so using it
heavily before the degree profile is close to a path wastes proposals.

Reconnect-template eligibility was about 19-22% of reconnect attempts.
Conditional acceptance of eligible reconnects was much higher than exchange,
but they solve a different structural problem.

## Long/phase benchmark

Settings:
- 1500 sweeps
- 2 reads
- 18 outer iterations
- 4 seeds/mode

Modes:
1. exchange_long: exchange only throughout
2. time_phase: exchange only for iterations 0..11, then static 50/50
3. threshold_phase: switch to static 50/50 after previous degree conflicts <=2

Aggregate:
- exchange_long: 0/4 feasible, minimum degree conflicts 1
- time_phase: 1/4 feasible, best score 1,343,093 (certified optimum)
- threshold_phase: 0/4 feasible, minimum degree conflicts 1

The informative time-phase seed was 20370928:
- iteration 4: degree conflicts 1, cycles 0
- iteration 9 (exchange only): first Hamilton path, score 1,326,008
- iterations 12-13 after enabling reconnect: temporarily leaves Hamilton
  feasibility by creating cycles while preserving degree=0
- iteration 14: returns to a Hamilton path with score 1,343,093
- iterations 15-17: remains at the certified optimum.

Two other time-phase seeds reached degree conflicts 0 but remained trapped with
1-2 directed cycles.  This is exactly the regime where a degree-preserving
reconnect kernel is complementary to exchange.

## Interpretation

For random fixed-cardinality initialization, the useful phase order is not
"reconnect first, exchange later".  The data support:

    exchange / degree repair
        -> reconnect / topology and cycle rewiring
        -> low-temperature refinement

The kernel probabilities can vary with sweep or outer iteration while retaining
a symmetric proposal at each fixed time, provided the probability schedule is
a function of time only.  If probabilities depend on the current state (for
example current degree residual), a Metropolis-Hastings correction is required
for a strict equilibrium sampler.

## SQA terminology

Multi-bit Monte Carlo proposals alone do not necessarily imply a different
quantum driver: cluster/multi-spin updates can sample the same path-integral
distribution if detailed balance is preserved.

However, restricting every slice to fixed Hamming weight changes the sampled
state space.  The standard independent transverse-field driver does not conserve
Hamming weight.  Therefore this benchmark should be called a
"fixed-cardinality constrained path-integral sampler" rather than an exact
standard-transverse-field SQA implementation unless a number-conserving driver
mapping is derived.
