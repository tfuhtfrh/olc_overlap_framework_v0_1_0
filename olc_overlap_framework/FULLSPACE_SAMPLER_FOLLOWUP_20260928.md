# Full-space and late multi-k follow-up — 2026-09-28

This note records the first follow-up experiments after the fixed-cardinality
exchange/reconnect work.  It complements
`SAMPLER_DYNAMICS_RESEARCH_NOTE.md`.

## Late repeated k=2,3 on the fixed-cardinality branch

To test whether useful larger-k moves were simply too rare, the late
degree-repair phase made several independent k=2 and k=3 Metropolis proposals
per slice/sweep.

Configurations:
- baseline: k=1 only before reconnect;
- moderate retries: 2 k=2 + 4 k=3 attempts;
- heavy retries: 4 k=2 + 8 k=3 attempts.

Native proposals/run:
- baseline: 345,600;
- moderate: 806,400;
- heavy: 1,267,200.

In this 4-seed batch none reached a Hamilton path.  All approached degree
residual 1--2.  Extra random k>1 work therefore did not solve the final barrier.

Interpretation: the issue is not just that random k=3 draws are too rare.
Uniform random larger-k moves are too unstructured.  Future k>1 proposals
should target actual structural defects, while preserving proposal symmetry
or using a Hastings correction.

## First full-space sampler benchmark

The priority full-space route restored the count penalty

    A_count (sum_e x_e - 143)^2

and allowed all binary edge configurations.

Compared:
- single-bit only;
- single-bit + 1<->1 exchange;
- dynamic single/exchange/reconnect mixture.

All variants readily reached degree-conflict-free states, but typically stopped
below 143 selected edges:
- single only: usually 139--140;
- single + exchange: as close as 142;
- dynamic mixture: as close as 142.

No Hamilton path was obtained in the first 4-seed batch.

This reveals a full-space-specific barrier: dropping edges removes expensive
degree conflicts, but once the chain becomes under-filled, balanced exchange
and reconnect moves cannot change cardinality.  A single edge addition may
have to cross a local count/degree barrier.

## Structured cardinality-changing move

A symmetric three-bit insertion/removal template was tested:

    {a->b} <-> {a->c, c->b}

There are 134 static templates in complex144.  Choosing uniformly from this
fixed template set and flipping only when exactly one side is occupied is an
involution and therefore a symmetric proposal.

The move was eligible often (~64%) but rarely accepted:
- A_count=32: ~0.054% acceptance given eligibility;
- A_count=64: ~0.084%.

The closest observed count error improved to one edge, but no Hamilton path was
found.  Raising A_count also began to trade count repair against degree
conflicts.

## Current implication

Do not simply increase A_count or blindly add more random k>1 moves.

For the full-space route, the next useful move should jointly:
- change cardinality when needed;
- repair or preserve the relevant local degree structure;
- remain symmetric by construction, or include the exact Hastings ratio.

Examples worth testing:
- endpoint-aware 1-to-2 / 2-to-1 reroutes;
- component-endpoint reconnection;
- defect-local k=2/k=3 exchanges;
- bounded local repair moves with a statically enumerable reverse operation.

The single-bit channel should remain present so the full binary state space
stays connected.

These observations reinforce the effective-exchange idea: a strong count
penalty creates a low-energy near-fixed-cardinality sector, while transitions
between useful states can require correlated motion that a local single-bit
sampler reaches only through costly intermediate configurations.
