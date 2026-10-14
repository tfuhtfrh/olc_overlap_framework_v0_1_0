# CHM13 bounded-QUBO penalty tuning summary

Date: 2026-09-25

This note summarizes production `dwave.samplers.TabuSampler` tests on the
7,704-variable bounded-coefficient vertex-order QUBO.

## 1. Static penalty scaling from all-zero does not solve the topology basin

A broad static sweep was run over:
- degree penalty from 4 up to 8,000,
- void penalty = 2 * degree,
- successor penalty from 1 up to 72,
- product penalty from 2 up to 144.

Across this wide range, zero-start Tabu repeatedly returned essentially the
same maximal-matching-like state:
- 115 selected edges,
- source/sink = 1/1,
- read-degree residual square = 56,
- product violations = 0,
- successor residual square = 115,
- all read positions collapsed to one value.

Therefore the main barrier is not a simple global coefficient-scale problem.
Once the one-flip search enters this matching basin, multiplying penalties does
not alter the required alternating multi-edge exchange.

## 2. Penalty continuation changes the basin

A degree-only warmup was followed by weak order/product penalties, reusing the
previous sample as the next initial state.

Representative fixed-degree chain (degree=16, void=32):
- degree-only stage: about 127 selected edges, degree residual ~32;
- weak order stage (successor=0.25, product=0.5): as low as
  - 149 selected edges,
  - degree residual square = 14,
  - source/sink = 1/1,
  - about 103 distinct read positions;
- increasing successor/product afterwards can drive product violations to zero
  and reduce successor residual (roughly 460 -> ~205), while the graph-degree
  residual remains stuck around the same local topology.

This is substantially better than the static zero-start basin (degree residual
56 and only 115 edges), so penalty scheduling matters even when the final
Hamiltonian coefficients are unchanged.

## 3. Raising degree penalties after the weak-order stage mostly freezes the topology

Two-phase/adaptive schedules were tested by increasing degree penalties from
tens to thousands after a weak-order state had formed.

Typical result:
- selected edge set remains almost unchanged,
- source/sink remain correct,
- product/successor auxiliaries improve,
- degree residual does not fall to zero.

Thus simply making the degree term increasingly dominant after the search has
entered a particular edge basin does not repair the alternating-edge topology.

## 4. Focused weak-entry search

A focused degree-only -> weak-order search tested:
- degree = 8, 12, 16, 20, 24, 32,
- void = 2*degree,
- weak (successor, product) in {(0.125,0.25), (0.25,0.5)}.

Best observed degree residual in this search:
- degree=8, void=16, successor=0.25, product=0.5:
  - 148 selected edges,
  - read-degree residual square = 16,
  - product violations = 6,
  - successor residual square = 422,
  - 105 distinct positions.

A particularly interesting state:
- degree=24, void=48, successor=0.125, product=0.25:
  - exactly 143 selected real edges,
  - source/sink = 1/1,
  - but read-degree residual square = 18,
  - product violations = 22,
  - successor residual square = 497.

So the correct total edge count alone is not enough; degree conflicts remain.

Across all penalty-continuation runs so far, the best single observed read-degree
residual was 14 (degree=16, void=32, successor=0.25, product=0.5 in an earlier
seed).

## Current working recommendation

Do not choose a single very large static penalty set yet.

For further experiments, treat penalty handling as a continuation process:

1. **Topology entry stage**
   - degree ~ 8--24
   - void ~ 2*degree
   - successor ~ 0.125--0.25
   - product ~ 0.25--0.5
   - preferably preceded by a degree-only warmup.

2. **Auxiliary cleanup stage**
   - hold the edge topology near the best weak-order state,
   - gradually increase product and successor penalties,
   - do not immediately increase degree by orders of magnitude.

3. **Topology repair**
   - remaining degree residual is now dominated by an alternating multi-edge
     local-search barrier rather than penalty scale.
   - solving this likely needs either a nonlocal move/repair operator or a
     solver whose neighborhood can directly modify an alternating edge set.

The present evidence therefore separates two issues:
- coefficient conditioning has been substantially improved by the bounded
  formulation;
- the remaining degree/topology barrier cannot be removed by static penalty
  scaling alone.
