# Current-only / gated cycle-cut SQA follow-up — 2026-09-29

## Motivation

Two questions were tested:

1. A known directed cycle C only violates acyclicity when every edge in C is
   simultaneously selected.  Therefore the exact logical constraint is

       sum_{e in C} x_e <= |C|-1,

   equivalently "penalize only sum x_e = |C|".

2. Unlike classical cutting-plane methods, SQA may not benefit from retaining
   every historical cycle cut.  Historical cuts remain mathematically valid,
   but their auxiliary variables and couplers can worsen the sampling
   landscape.  This motivates current-cycle-only cuts, analogous to projected
   rank using only the current topology.

The QUBO encoding used bounded nonnegative slack:

    A_cut (sum_{e in C} x_e + s_C - (|C|-1))^2.

For every state with at most |C|-1 selected cycle edges there exists a slack
assignment with exactly zero cut penalty.  Only selection of the complete
cycle is penalized.

## Experiment 1: same-seed comparison

CHM13 complex144, OpenJij SQA, 12 outer iterations, four identical seeds per
configuration.

Configurations:
- none: weight + degree + count only, no order/cycle term
- rank: previous projected stale-rank schedule
- current_4: discard old cuts; cut only cycles seen in the immediately previous
  state, A_cut=4
- current_16: same, A_cut=16
- accum_4: retain all historical cuts, A_cut=4

Results:

| config | feasible | ground | best score | max vars |
|---|---:|---:|---:|---:|
| none | 3/4 | 0 | 1,338,209 | 261 |
| rank | 1/4 | 0 | 1,324,816 | 261 |
| current_4 | 3/4 | 0 | 1,309,005 | 275 |
| current_16 | 0/4 | 0 | none | 275 |
| accum_4 | 2/4 | 0 | 1,278,651 | 294 |

Important observations:

- The rank-free/no-cycle baseline was the strongest configuration in this
  small batch.  It reached the independently known runner-up score 1,338,209.
- Current-only A=4 retained the same 3/4 feasibility rate as no-cut, but its
  best weighted path was worse.
- A_cut=16 was clearly too disruptive for this SQA budget.
- Accumulating cuts reduced feasibility and increased variables to 294.
- Current-only cuts bounded the variable overhead (max 275) and therefore
  behaved better as an SQA representation than historical accumulation, but
  they did not outperform using no cycle penalty.

## Why historical cuts are different from stale rank

A historical cycle cut is still a globally valid constraint: no Hamilton path
can contain every edge of a directed cycle.  Therefore retaining an old cut
does not create the same kind of *incorrect* barrier as retaining an old rank
ordering.

The disadvantage of historical cuts is instead sampler geometry:
- more auxiliary variables;
- more couplers;
- larger coefficient structure;
- repeated constraints that may be irrelevant to the current local topology.

Thus current-only versus accumulated cuts is a resource/mixing tradeoff, not a
correctness tradeoff.

## Experiment 2: phase-gated current cuts

To avoid disturbing degree/cardinality repair, a current cycle cut was
activated only when the previous state satisfied:

    selected_count = N-1
    degree_conflicts = 0
    cycle_count > 0.

Cuts were removed when the state was not on this manifold or when it became a
Hamilton path.

Same four seeds, same SQA budget:

| config | feasible | ground | best score | mean cut iterations | max vars |
|---|---:|---:|---:|---:|---:|
| none | 3/4 | 0 | 1,338,209 | 0 | 261 |
| gated_1 | 1/4 | 0 | 1,263,875 | 9.25 | 276 |
| gated_2 | 2/4 | 0 | 1,313,703 | 7.75 | 275 |
| gated_4 | 2/4 | 0 | 1,336,620 | 5.5 | 274 |
| current_4 | 3/4 | 0 | 1,309,005 | 8.75 | 275 |

Gating improves the interpretation of the cut but does not make it a robust
winner.  Gated A=4 can produce a high-quality path close to the runner-up, but
the simple rank-free baseline still has better feasibility and best score in
this batch.

## Revised interpretation

The experiments do not support using either stale rank or cycle cuts as the
main search-direction mechanism.

Evidence now favors:

1. keep the core edge energy as simple as possible:
       H_weight + H_degree + H_count;

2. use topology-aware reversible proposal kernels for motion:
       source/sink transfer,
       open alternating trails,
       R2/R3 reconnect,
       small compound reconnects;

3. use whole-worldline / imaginary-time cluster versions of these proposals
   where appropriate in the path-integral sampler;

4. treat a symmetric current-cycle cut only as an optional fallback when the
   sampler repeatedly returns a degree-correct cyclic state.

This is consistent with the endpoint-transfer benchmark:
- uniform exchange: 9/24 HP hits, median first hit 5120 proposals;
- same-head sink-transfer: 24/24, median 96;
- P=8 endpoint-transfer path-integral update: 24/24 for both single-slice and
  whole-worldline variants on the studied stuck state.

The strongest current signal is therefore not "find a better cycle penalty",
but "remove arbitrary direction penalties and put structural knowledge into
the reversible move kernel".

## Next benchmark target

The next combined SQA benchmark should use:
- no stale rank;
- no persistent cycle cut;
- rank-free weight/degree/count BQM;
- endpoint-transfer proposals;
- R2/R3 structured proposals;
- worldline/cluster updates;
- optional current-cycle cut only after repeated degree-correct cyclic returns.

The key metrics should be feasible-hit rate, ground-hit rate, time/proposals to
first HP, and best weighted path across multiple non-hand-picked starts.
