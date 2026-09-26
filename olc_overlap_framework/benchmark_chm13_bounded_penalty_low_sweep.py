"""Low-scale penalty sweep for the bounded CHM13 QUBO.

The previous sweep (degree=1000..8000) returned the same maximal-matching-like
basin. This sweep deliberately brings penalties down near the normalized edge
objective scale to test whether excessive penalties are freezing augmenting
moves.
"""

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

OUT = Path("debug/qubo/chm13_bounded_penalty_low_sweep_20260925.json")

CONFIGS = [
    (4, 8, 1, 2),
    (8, 16, 1, 2),
    (8, 16, 2, 4),
    (16, 32, 2, 4),
    (16, 32, 4, 8),
    (32, 64, 4, 8),
    (32, 64, 8, 16),
    (64, 128, 8, 16),
    (64, 128, 16, 32),
    (128, 256, 16, 32),
    (128, 256, 32, 64),
    (256, 512, 32, 64),
    (256, 512, 64, 128),
]


def run_one(d, v, s, p, seed):
    cfg = BoundedVertexOrderConfig(
        degree_penalty=float(d),
        void_degree_penalty=float(v),
        successor_penalty=float(s),
        product_penalty=float(p),
        edge_cost_scale=1.0,
    )
    problem = build_bounded_problem(
        f"bounded_d{d}_v{v}_s{s}_p{p}",
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
        timeout=2500,
        seed=seed,
        coefficient_z_first=500,
        coefficient_z_restart=125,
        lower_bound_z=100000,
        num_restarts=40,
    )
    seconds = time.perf_counter() - t0
    datum = next(ss.data(fields=["sample", "energy"], sorted_by="energy"))
    sample = [int(datum.sample.get(i, 0)) for i in range(problem["model"].num_variables)]
    ev = bounded_eval(problem, sample)
    return {
        "degree": d,
        "void": v,
        "successor": s,
        "product": p,
        "seconds": seconds,
        **ev,
    }


def main():
    rows = []
    for i, cfg in enumerate(CONFIGS):
        row = run_one(*cfg, SEED + i)
        rows.append(row)
        print(json.dumps(row), flush=True)

    rows.sort(key=lambda r: (
        not r["valid_path"],
        r["read_degree_residual_sq"] + r["void_residual_sq"],
        r["successor_residual_sq"] + r["product_violations"],
        abs(r["selected_edges"] - 143),
        r["energy"],
    ))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print("\nTOP")
    print(json.dumps(rows[:8], indent=2), flush=True)


if __name__ == "__main__":
    main()
