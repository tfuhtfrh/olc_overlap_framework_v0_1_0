# Follow-up: projected SQA optimum search and source/sink basin diagnostic

Date: 2026-09-26

## Extended projected SQA search

The projected exact-count hybrid was rerun with OpenJij SQA for 10 independent
outer runs, 18 iterations per run.  The order-penalty schedule

    0, 0.25, 0.5, 1, 2, 4

was repeated three times per run so that the algorithm periodically resets the
rank-induced order bias and can escape the topology of a previously feasible
path.

The search repeatedly found valid 143-edge Hamilton paths.  The best observed
score was

    1,327,035

at run 7, iteration 14, A_order=0.5.

The certified optimum is

    1,343,093,

so the best projected-SQA gap in this experiment was 16,058 (about 1.20%).
The certified optimum was NOT reached in this 10x18 experiment.

This is substantially better than the first one-shot feasible SQA path
(score 1,275,862), but still below the projected-Tabu run, which reached the
certified optimum.

## Why the source/sink projected variant stopped at 141 edges

The final projected source/sink state was reproduced and inspected.

It contains:
- 141 selected edges,
- three disjoint directed path components of sizes 3, 39, and 102,
- fixed source/sink on the 3-vertex middle component,
- fixed-endpoint degree residual square = 4.

With A_degree=144 this means a remaining degree penalty of 576.

There is no single unselected edge whose addition reduces this fixed-endpoint
degree residual.  Therefore increasing A_degree does not create a useful
one-edge descent direction.

There is, however, a direct two-connector merge of all three components:

    component(39) -> component(3) -> component(102).

The required candidate connector edges both exist.

The key issue is that the current fixed source and sink are the endpoints of
the 3-vertex middle component.  Adding either connector alone repairs one
component endpoint but simultaneously turns a fixed endpoint into an internal
vertex, so the fixed-endpoint degree residual does not improve.  After BOTH
connectors are inserted and the endpoints are re-projected classically, the
outer source/sink pair could move to the true ends of the merged path.

Thus this basin is not primarily caused by insufficient degree-penalty
magnitude.  It is a coordinated move involving:
- two edge activations, and
- simultaneous source/sink relocation.

The exact-count projected formulation avoids this particular endpoint-locking
barrier, which explains why it was much more effective in the current hybrid
tests.
