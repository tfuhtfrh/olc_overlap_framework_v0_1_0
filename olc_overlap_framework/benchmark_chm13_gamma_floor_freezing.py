"""Focused gamma-floor freezing sweep.

Keep the successful large initial transverse field Gamma0=3 so K starts small,
and only raise the final Gamma floor to prevent late-stage Trotter freezing.
Compare fixed beta=4 schedules:
  3 -> 0.03 (baseline)
  3 -> 0.10
  3 -> 0.30
  3 -> 0.50
plus one warmer beta=2 control at 3 -> 0.30.

Reuses the Stage-2 endpoint-transfer and Stage-3 runner-up R3 diagnostics.
"""
import benchmark_chm13_schedule_cluster_freezing as base

base.OUT=base.Path("debug/qubo/chm13_gamma_floor_freezing_20260929.json")
base.SCHEDULES={
  "baseline":{"beta0":4.0,"beta1":4.0,"g0":3.0,"g1":0.03},
  "floor_0.10":{"beta0":4.0,"beta1":4.0,"g0":3.0,"g1":0.10},
  "floor_0.30":{"beta0":4.0,"beta1":4.0,"g0":3.0,"g1":0.30},
  "floor_0.50":{"beta0":4.0,"beta1":4.0,"g0":3.0,"g1":0.50},
  "warm_floor_0.30":{"beta0":2.0,"beta1":2.0,"g0":3.0,"g1":0.30},
}
base.SEEDS=tuple(202609400+i for i in range(8))

if __name__=="__main__":
    base.main()
