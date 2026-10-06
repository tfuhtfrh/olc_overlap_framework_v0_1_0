# SQA freezing and open alternating-trail follow-up — 2026-09-28

## Main finding

The CHM13 complex144 stuck checkpoint separates three different effects:
Trotter freezing, classical energy barriers/proposal sparsity, and structural
invariants of degree-preserving reconnect moves.

Checkpoint:
- 143 selected edges
- degree conflicts 0
- one source, one sink
- cycle count 1
- topology = one path + one cycle

## Classical barrier

Full-space count-penalized BQM:
- all 261 single-bit neighbors are uphill
- minimum Delta H = 27.5322494193
- median Delta H = 31.7182419153

Fixed-cardinality BQM:
- all 16,874 possible 1-to-1 exchanges are uphill
- minimum Delta H = 3.63230301947
- median Delta H = 576.423887797
- p90 Delta H = 580.308796379

Thus useful exchange moves are extremely sparse among mostly destructive
degree-conflict moves.

## Trotter freezing

For P=8, beta=4, Gamma=0.03, the extra single-bit single-slice Trotter action
cost when neighboring slices agree is about 4K = 8.39956.

Controlled comparison:
- P=1 classical Metropolis has small but nonzero mobility when heated
- P=8 single-slice proposals had zero acceptance in every tested snapshot
- whole-worldline moves recover approximately P=1 mobility
- a single-bit imaginary-time cluster move likewise recovers approximately
  P=1 mobility at beta=0.25

However, no variant found a lower classical state or Hamilton path from this
checkpoint.  Therefore Trotter freezing is real but is not the only barrier.

## Heating closed R2/R3 moves

A separate fixed-cardinality experiment removed the Trotter term and used only
static reversible closed alternating-cycle moves:
- R2 templates = 41
- R3 templates = 66
- 80,000 proposals per run
- beta in 0.02, 0.05, 0.10, 0.25, 0.50, 1, 4
- four seeds plus a beta 0.02 to 4 anneal

At beta=0.02 a run accepts roughly 10,900 moves, about half uphill, and can
reach symmetric-difference distance 36 from the initial edge set.  Nevertheless:
- Hamilton paths = 0
- minimum cycle count = 1
- best energy delta = 0

The failure is therefore not simply because the chain is too cold.

## Endpoint-fiber invariant

After mapping oriented graph IDs back to normalized read IDs, exact enumeration
shows:
- exact Hamilton paths = 192
- distinct endpoint pairs = 1
- the stuck source equals the unique Hamilton-path source
- the stuck sink differs from the unique Hamilton-path sink
- Hamilton paths with the stuck endpoint pair = 0

Every closed alternating cycle R_k preserves every in-degree and out-degree and
therefore preserves the source/sink identities.

Consequently, no sequence of closed R_k moves, for any k, can solve this
particular checkpoint.  Increasing R2/R3 to R4, R5, and beyond cannot repair
the wrong endpoint fiber.

## Open alternating sink transfer

Build the residual bipartite graph:
- unselected u->v gives u_out -> v_in, meaning add
- selected u->v gives v_in -> u_out, meaning remove

A residual path from the current sink out-copy to the target sink out-copy
transfers the out-degree defect while preserving cardinality, all indegrees,
and all other outdegrees.

For this checkpoint, BFS finds the shortest possible useful path:
- residual arc length = 2
- one add + one remove
- equivalent to a structured 1-to-1 exchange
- fixed-cardinality Delta H = +3.63230301947

The move directly gives:
- degree conflicts = 0
- correct source and sink
- cycle count = 0
- valid Hamilton path = yes
- score = 1,299,598

This Delta H is exactly the minimum exchange delta from the exhaustive
16,874-exchange audit.

At beta=0.25 its conditional acceptance is about 0.403, but under uniform
exchange it receives only 1/16,874 proposal mass.  The useful transition is
therefore not large; it is rare.

## Revised move framework

Use alternating components rather than only flip count:

1. degree/cardinality repair;
2. open alternating trails to move endpoint/degree defects and change fibers;
3. closed alternating cycles R2/R3/... to rewire topology and optimize weights
   inside the correct fiber.

For path-integral experiments, structured classical proposals should be paired
with whole-worldline or proper imaginary-time cluster updates rather than only
single-slice flips.

The immediate implementation target is a residual-graph proposal kernel that
generates short defect-to-defect open alternating trails and uses an exactly
reversible construction or the corresponding Hastings correction.
