"""Low-strength phase-gated cycle-cut sweep.

Motivated by the exact runner-up R2 route, whose cyclic intermediate is only
0.0144125 lower than the ground state in the rank-free fixed-cardinality BQM.
Test cut penalties near that intrinsic energy scale rather than A=1..16.
"""
import benchmark_chm13_sqa_gated_cyclecut as base

base.OUT=base.Path("debug/qubo/chm13_sqa_low_cyclecut_20260929.json")
base.CONFIGS=("none","gated_0.025","gated_0.05","gated_0.1","gated_0.25","gated_0.5")

if __name__=="__main__":
    base.main()
