"""Focused penalty sweep for bounded CHM13 vertex-order QUBO using production TabuSampler."""

from __future__ import annotations

import json
import time
from pathlib import Path

from dwave.samplers import TabuSampler

from benchmark_chm13_tabu_two_hamiltonians import (
    SEED,
    bounded_eval,
    build_bounded_problem,
)
from experimental_chm13_vertex_carry_qubo import BoundedVertexOrderConfig


OUT = Path("debug/qubo/chm13_bounded_penalty_sweep_20260925.json")


def run():
    rows = []
    degrees = [1000.0, 2000.0, 4000.0, 8000.0]
    successors = [18.0, 36.0, 72.0]

    for d in degrees:
        for s in successors:
            cfg = BoundedVertexOrderConfig(
                degree_penalty=d,
                void_degree_penalty=2*d,
                successor_penalty=s,
                product_penalty=144.0,
                edge_cost_scale=1.0,
            )
            problem = build_bounded_problem(
                f"bounded_d{int(d)}_s{int(s)}",
                cfg,
            )
            bqm = problem["model"].to_dimod_bqm()
            init = [[0] * problem["model"].num_variables]

            sampler = TabuSampler()
            t0 = time.perf_counter()
            ss = sampler.sample(
                bqm,
                initial_states=init,
                initial_states_generator="none",
                num_reads=1,
                timeout=1500,
                seed=SEED + int(d) + int(s),
                coefficient_z_first=500,
                coefficient_z_restart=125,
                lower_bound_z=100000,
                num_restarts=25,
            )
            seconds = time.perf_counter() - t0

            datum = next(ss.data(fields=["sample", "energy"], sorted_by="energy"))
            sample = [int(datum.sample.get(i, 0)) for i in range(problem["model"].num_variables)]
            ev = bounded_eval(problem, sample)
            row = {
                "degree": d,
                "void": 2*d,
                "successor": s,
                "product": 144.0,
                "seconds": seconds,
                **ev,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    rows.sort(key=lambda r: (
        not r["valid_path"],
        r["read_degree_residual_sq"],
        r["successor_residual_sq"],
        r["product_violations"],
        r["energy"],
    ))
    payload = {"rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\nTOP")
    print(json.dumps(rows[:6], indent=2), flush=True)


if __name__ == "__main__":
    run()
