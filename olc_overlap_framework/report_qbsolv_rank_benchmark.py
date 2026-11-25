"""QbSolv (classical tabu) on the same order-variable QUBO used for the stage-report SQA runs.

The QUBO is built by ``experimental_chm13_latent_rank_comparator_qubo.build_model()``
with its default configuration, exactly as in ``report_sqa_benchmark.py``.
Seeds are the six formal seeds of the stage report. No solution repair is applied.
The reference path is read only after all sampling has finished, for offline evaluation.

Standalone by design: dwave-qbsolv 0.3.4 pins an old dimod and runs on Python 3.10.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

from dwave_qbsolv import QBSolv, ENERGY_IMPACT

from experimental_chm13_latent_rank_comparator_qubo import build_model, audit
from demo_chm13_edge_ordered_path_qubo import DATASET_DIR

SEEDS = (2026101801, 2026101802, 2026101803, 2026101804, 2026101805, 2026101806)
NUM_REPEATS = 50          # QbSolv default stopping rule: 50 repeats without improvement
SUBPROBLEM_SIZE = 47      # QbSolv default sub-QUBO size
TIMEOUT = 600             # seconds per seed (safety limit)


def path_metrics(bits, model, edges, rewards):
    """Same definition as report_sqa_benchmark.path_metrics."""
    selected = [e for i, e in enumerate(edges) if bits[i]]
    indeg = dict.fromkeys(model.read_ids, 0)
    outdeg = dict.fromkeys(model.read_ids, 0)
    nxt = {}
    for u, v in selected:
        indeg[v] += 1
        outdeg[u] += 1
        nxt[u] = v
    conflicts = sum(d * (d - 1) // 2 for d in (*indeg.values(), *outdeg.values()))
    sources = [v for v in model.read_ids if indeg[v] == 0]
    sinks = [v for v in model.read_ids if outdeg[v] == 0]
    order, seen = [], set()
    if not conflicts and len(sources) == len(sinks) == 1:
        v = sources[0]
        while v not in seen:
            order.append(v)
            seen.add(v)
            if v not in nxt:
                break
            v = nxt[v]
    valid = (not conflicts and len(selected) == len(model.read_ids) - 1
             and len(sources) == len(sinks) == 1 and len(order) == len(model.read_ids)
             and order[-1] == sinks[0])
    return {"valid_path": bool(valid), "path_score": int(sum(rewards[e] for e in selected)) if valid else None,
            "selected_edges": len(selected), "sources": len(sources), "sinks": len(sinks),
            "degree_conflicts": conflicts, "edge_count_residual": len(selected) - (len(model.read_ids) - 1)}


def environment():
    pk = {}
    for name in ("dwave-qbsolv", "dimod", "numpy"):
        try:
            pk[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pk[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": pk,
            "commit": os.environ.get("GITHUB_SHA"), "run_id": os.environ.get("GITHUB_RUN_ID")}


def run_seed(seed, model, rewards, cfg):
    edges = model.edge_pairs
    bqm = model.to_dimod_bqm()
    t0 = time.perf_counter()
    response = QBSolv().sample(bqm, num_repeats=NUM_REPEATS, seed=seed % (2 ** 31), algorithm=ENERGY_IMPACT,
                               verbosity=-1, timeout=TIMEOUT, solver_limit=SUBPROBLEM_SIZE, solver="tabu")
    seconds = time.perf_counter() - t0
    rows = []
    best_score, best_energy_row = None, None
    for dat in response.data(fields=["sample", "energy", "num_occurrences"], sorted_by="energy"):
        bits = [int(dat.sample.get(j, 0)) for j in range(model.num_variables)]
        met = path_metrics(bits, model, edges, rewards)
        aud = audit(bits, model, rewards, cfg)
        row = {"energy": float(model.energy(bits)), "occurrences": int(dat.num_occurrences),
               "path": met, "rank_audit": {k: v for k, v in aud.items() if not isinstance(v, (list, dict))}}
        rows.append(row)
        if best_energy_row is None:
            best_energy_row = row
        if met["valid_path"] and (best_score is None or met["path_score"] > best_score):
            best_score = met["path_score"]
    return {"seed": seed, "seconds": seconds, "returned_samples": len(rows),
            "valid_returned_samples": sum(r["occurrences"] for r in rows if r["path"]["valid_path"]),
            "best_score": best_score, "ever_feasible": best_score is not None,
            "lowest_energy_sample": best_energy_row, "samples": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="qbsolv-out")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, rewards, cfg = build_model()
    results = []
    for seed in SEEDS:
        r = run_seed(seed, model, rewards, cfg)
        print(json.dumps({k: v for k, v in r.items() if k != "samples"}, ensure_ascii=False), flush=True)
        results.append(r)
    # Offline evaluation only: the reference certificate is read after all sampling has finished.
    ref = json.loads((DATASET_DIR / "reference_path.json").read_text())["normalized_nodes"]
    ref_score = int(sum(rewards[e] for e in zip(ref, ref[1:])))
    for r in results:
        r["optimum_hit"] = r["best_score"] == ref_score
        r["relative_gap"] = None if r["best_score"] is None else (ref_score - r["best_score"]) / ref_score
    feasible = [r for r in results if r["ever_feasible"]]
    summary = {
        "solver": "dwave-qbsolv (classical tabu sub-solver, energy-impact decomposition)",
        "num_repeats": NUM_REPEATS, "subproblem_size": SUBPROBLEM_SIZE, "timeout_per_seed": TIMEOUT,
        "seeds": list(SEEDS), "qubo_config": cfg.__dict__, "variables": model.num_variables,
        "quadratic_terms": len(model.quadratic), "reference_score": ref_score,
        "trials_with_valid_path": len(feasible), "trials": len(results),
        "optimum_hits": sum(r["optimum_hit"] for r in results),
        "best_score": max((r["best_score"] for r in feasible), default=None),
        "per_seed": [{k: r[k] for k in ("seed", "seconds", "ever_feasible", "best_score", "optimum_hit", "relative_gap",
                                         "valid_returned_samples", "returned_samples")}
                     | {"lowest_energy": r["lowest_energy_sample"]["energy"],
                        "lowest_energy_path": r["lowest_energy_sample"]["path"],
                        "lowest_energy_audit": r["lowest_energy_sample"]["rank_audit"]} for r in results],
        "environment": environment(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "samples.json").write_text(json.dumps(results, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("trials_with_valid_path", "trials", "optimum_hits", "best_score")}))


if __name__ == "__main__":
    main()
