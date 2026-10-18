# Projected edge-only hybrid search (2026-09-26)

## Classification

This is **not** a pure full-Hamiltonian SQA solve.

The full static comparator QUBOs (3,789-variable source/sink model and
3,501-variable endpoint-free model) remain useful baselines.  In the projected
hybrid solver, however, only the real edge-selection bits x_e are sent to the
QUBO backend.

The latent rank is stored classically as an integer ordering/permutation.  The
borrow/comparator bits disappear entirely because they are deterministic once
the ranks are known.  Source/sink can also be updated classically, or omitted
in the exact-count variant.

If the inner edge-only QUBO is sampled by OpenJij SQASampler, that inner step is
genuine simulated quantum annealing.  The overall algorithm should be described
as **SQA-assisted alternating hybrid optimization** (or projected SQA), not as
pure SQA on one fixed QUBO.

## Edge-only subproblem

For a fixed classical rank r^(t), the order certificate becomes a linear edge
penalty:

    A_order * sum_{(u,v): r_v <= r_u} x_uv.

The endpoint-free exact-count variant solves

    H_t(x) =
        H_weight(x)
      + H_degree-conflict(x)
      + A_count (sum_e x_e - (N-1))^2
      + A_order * sum_{backward under r^(t)} x_e.

Only M=261 binary variables are sampled.

The source/sink variant instead fixes the classically updated endpoint pair and
uses the corresponding in/out degree squares over the edge variables.

## Classical projection

After every edge-QUBO solve:

1. Build the selected directed graph.
2. Compute strongly connected components.
3. Topologically order the condensation DAG.
4. For a nontrivial SCC, use an Eades-style feedback-arc ordering heuristic.
   For simple path/cycle structures this gives the expected one-break ordering.
5. Assign integer ranks 0..N-1 from the resulting order.
6. Rebuild the next edge-only QUBO.

Thus rank variables need not be binary in the hybrid algorithm; they are simply
classical integer/order data.  Borrow bits do not exist in the search state.

The order penalty was continued through

    0, 0.25, 0.5, 1, 2, 4, 8, 16, 32.

## Endpoint correction

An initial source/sink projection independently chose the same vertex as both
source and sink.  This exposed the pathological basin "isolated vertex + cycle
on all remaining vertices".

The projected solver was corrected to require distinct source and sink.
The static 3,789-variable comparator formulation was also strengthened with

    A_st * sum_v s_v t_v,

which adds N=144 couplers and no variables.

## Production Tabu inner solver

With D-Wave MST2 Tabu as the edge-only QUBO backend:

### source/sink projected variant

After requiring distinct endpoints, the run settled at:
- 141 selected edges,
- zero degree conflicts,
- zero cycles,
- three path components,
- no backward selected edges.

It did not merge the final three components.

### exact-count projected variant

The result was much stronger:

- iteration 0, A_order=0:
  - 143 edges,
  - zero degree conflicts,
  - one cycle.
- iteration 1, A_order=0.25:
  - same one-cycle structure.
- iteration 2, A_order=0.5:
  - 143 edges,
  - zero degree conflicts,
  - zero cycles,
  - one source / one sink,
  - valid Hamilton path,
  - score = 1,343,093.

This equals the certified CHM13 weighted optimum.  This is currently a
single-seed proof-of-concept, not yet a robustness result.

## OpenJij SQA inner solver

The same projected exact-count algorithm was then run with OpenJij SQA
(8 reads, 1200 sweeps, trotter=8).

The SQA sequence reached:

- iteration 0: 143 edges, 3 cycles;
- iteration 1: 143 edges, 2 cycles;
- iteration 2: 143 edges, 1 cycle;
- iteration 5, A_order=4:
  - 143 edges,
  - zero degree conflicts,
  - zero cycles,
  - one source / one sink,
  - valid Hamilton path,
  - score = 1,275,862.

So the hybrid algorithm also works with SQA as the actual QUBO search engine.
The first feasible SQA path was not the certified weighted optimum; further
feasible-path optimization and multi-seed tests are still needed.

## Interpretation

These results strongly support the earlier diagnosis that the static
rank/borrow encoding creates a solver-neighborhood mismatch.

Once rank/comparator variables are projected classically:
- the QUBO search dimension drops to the 261 real edge bits;
- the solver can alter edge topology without simultaneously traversing hundreds
  of certificate-bit flips;
- rank consistency is restored in one classical projection.

The cost is methodological: this is no longer a single static QUBO solve.
It is an alternating hybrid method derived from the QUBO constraints.
