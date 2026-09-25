"""Penalty-continuation experiment for bounded CHM13 QUBO.

Stage 0 solves only the graph degree/void structure (order/product penalties off).
Subsequent stages gradually turn on product and successor penalties while
re-using the previous Tabu solution as the initial state.

This tests whether static penalties fail because zero-start Tabu first falls into
a maximal-matching basin before the position variables can organize.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dwave.samplers import TabuSampler

from benchmark_chm13_tabu_two_hamiltonians import SEED, bounded_eval, build_bounded_problem
from experimental_chm13_vertex_carry_qubo import BoundedVertexOrderConfig

OUT = Path("debug/qubo/chm13_bounded_penalty_continuation_20260925.json")

STAGES = [
    (0.0, 0.0),
    (0.25, 0.5),
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 4.0),
    (4.0, 8.0),
    (8.0, 16.0),
]


def solve(problem, init, seed, timeout=3000):
    bqm = problem["model"].to_dimod_bqm()
    sampler = TabuSampler()
    t0 = time.perf_counter()
    ss = sampler.sample(
        bqm,
        initial_states=[init],
        initial_states_generator="none",
        num_reads=1,
        timeout=timeout,
        seed=seed,
        coefficient_z_first=1000,
        coefficient_z_restart=250,
        lower_bound_z=150000,
        num_restarts=60,
    )
    sec = time.perf_counter() - t0
    datum = next(ss.data(fields=["sample", "energy"], sorted_by="energy"))
    sample = [int(datum.sample.get(i, 0)) for i in range(problem["model"].num_variables)]
    return sample, sec


def run_chain(degree):
    current = None
    rows = []
    for stage, (succ, prod) in enumerate(STAGES):
        cfg = BoundedVertexOrderConfig(
            degree_penalty=float(degree),
            void_degree_penalty=float(2 * degree),
            successor_penalty=float(succ),
            product_penalty=float(prod),
            edge_cost_scale=1.0,
        )
        problem = build_bounded_problem(
            f"continuation_d{degree}_s{succ}_p{prod}", cfg
        )
        if current is None:
            current = [0] * problem["model"].num_variables
        sample, seconds = solve(
            problem, current, SEED + int(degree) * 100 + stage, timeout=3000
        )
        ev = bounded_eval(problem, sample)
        row = {
            "degree": degree,
            "void": 2 * degree,
            "stage": stage,
            "successor": succ,
            "product": prod,
            "seconds": seconds,
            **ev,
        }
        rows.append(row)
        current = sample
        print(json.dumps(row), flush=True)
    return rows


def main():
    rows = []
    for degree in (4, 8, 16, 32):
        print(f"CHAIN degree={degree}", flush=True)
        rows.extend(run_chain(degree))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print("\nFINAL_STAGES")
    finals = [r for r in rows if r["stage"] == len(STAGES)-1]
    print(json.dumps(finals, indent=2), flush=True)


if __name__ == "__main__":
    main()
