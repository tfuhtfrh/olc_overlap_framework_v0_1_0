"""SQA vs QbSolv on smaller induced subgraphs of CHM13 complex144 (order-variable QUBO).

Subgraph: the last N reads of the reference order (complex144 indices 144-N .. 143), with all
candidate edges among them. Every window keeps local cycles (the 11-read cyclic component is the
tail of the reference). The QUBO is built by the unmodified
``experimental_chm13_latent_rank_comparator_qubo.build_model()`` with default coefficients; only the
graph loader is restricted to the window.

SQA reproduces the stage-report rank run (commit e866a52, Actions 39137698403):
``report_sqa_benchmark.run("rank", seed, 0, out, reads=32, iterations=1, sweeps=20800)``
-> num_reads 32, 20,800 sweeps in one call, beta 5, gamma 1, trotter 8, random initial state.
QbSolv uses ``report_qbsolv_rank_benchmark.run_seed`` (defaults: num_repeats 50, sub-QUBO 47).

No repair. The exact optimum of each window is computed by exhaustive Hamilton-path enumeration
only after all sampling has finished, for offline evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import experimental_chm13_latent_rank_comparator_qubo as rankmod
from demo_chm13_edge_ordered_path_qubo import DATASET_DIR, load_chm13_graph

SEEDS = (2026101801, 2026101802, 2026101803, 2026101804, 2026101805, 2026101806)


def window_nodes(n):
    ref = json.loads((DATASET_DIR / "reference_path.json").read_text())["normalized_nodes"]
    if not 2 <= n <= len(ref):
        raise ValueError(f"window size {n} outside 2..{len(ref)}")
    return ref[len(ref) - n:]


def restrict_loader(keep):
    keep = set(keep)

    def loader(dataset_dir):
        reads, edges, reward = load_chm13_graph(dataset_dir)
        reads = [r for r in reads if r.rid in keep]
        edges = [e for e in edges if e.left_id in keep and e.right_id in keep]
        reward = {p: w for p, w in reward.items() if p[0] in keep and p[1] in keep}
        return reads, edges, reward

    rankmod.load_chm13_graph = loader


def exact_optimum(nodes, reward):
    """Enumerate all Hamilton paths of the (small, sparse) window graph."""
    out = {v: [] for v in nodes}
    for (u, v), w in reward.items():
        out[u].append((v, w))
    n = len(nodes)
    best, count = None, 0
    sys.setrecursionlimit(10000)

    def reach_ok(v, seen):
        stack, vis = [v], {v}
        while stack:
            x = stack.pop()
            for y, _ in out[x]:
                if y not in seen and y not in vis:
                    vis.add(y)
                    stack.append(y)
        return len(vis) - 1 == n - len(seen)

    def dfs(v, seen, score):
        nonlocal best, count
        if len(seen) == n:
            count += 1
            best = score if best is None or score > best else best
            return
        if not reach_ok(v, seen):
            return
        for y, w in out[v]:
            if y not in seen:
                seen.add(y)
                dfs(y, seen, score + w)
                seen.remove(y)

    for s in nodes:
        dfs(s, {s}, 0)
    return best, count


def run_sqa(n, out):
    import report_sqa_benchmark as R
    tmp = Path(tempfile.mkdtemp())
    (tmp / "reference_path.json").write_text(json.dumps({"normalized_nodes": window_nodes(n)}))
    R.DATASET_DIR = tmp  # run() reads only reference_path.json from here, after sampling
    rows = []
    for seed in SEEDS:
        r = R.run("rank", seed, 0, out / f"seed_{seed}", reads=32, iterations=1, sweeps=20800)
        rows.append({"seed": seed, "ever_feasible": r["ever_feasible"], "best_score": r["best_score"],
                     "feasible_returned_reads": r["feasible_returned_reads"], "seconds": r["sampling_seconds"]})
    return rows, R.environment()


def run_qbsolv(n, out):
    import report_qbsolv_rank_benchmark as Q
    model, rewards, cfg = rankmod.build_model()
    rows = []
    for seed in SEEDS:
        r = Q.run_seed(seed, model, rewards, cfg)
        low = r["lowest_energy_sample"]
        rows.append({"seed": seed, "ever_feasible": r["ever_feasible"], "best_score": r["best_score"],
                     "feasible_returned_reads": r["valid_returned_samples"], "seconds": r["seconds"],
                     "lowest_energy": low["energy"], "lowest_energy_path": low["path"],
                     "lowest_energy_audit": low["rank_audit"]})
    return rows, Q.environment()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--solver", choices=("sqa", "qbsolv"), required=True)
    ap.add_argument("--out", default="subgraph-out")
    a = ap.parse_args()
    out = Path(a.out) / f"{a.solver}_n{a.n:03d}"
    out.mkdir(parents=True, exist_ok=True)
    nodes = window_nodes(a.n)
    restrict_loader(nodes)
    model, rewards, cfg = rankmod.build_model()
    t0 = time.perf_counter()
    rows, env = run_sqa(a.n, out) if a.solver == "sqa" else run_qbsolv(a.n, out)
    wall = time.perf_counter() - t0
    # Offline evaluation only (after all sampling).
    opt, npaths = exact_optimum(list(model.read_ids), rewards)
    ref_nodes = nodes
    ref_score = int(sum(rewards[e] for e in zip(ref_nodes, ref_nodes[1:])))
    ref_energy = float(model.energy(rankmod.encode_order(model, ref_nodes)))
    for r in rows:
        r["optimum_hit"] = r["best_score"] == opt
        r["relative_gap"] = None if r["best_score"] is None else (opt - r["best_score"]) / opt
    feas = [r for r in rows if r["ever_feasible"]]
    summary = {
        "solver": a.solver, "n_reads": a.n, "window": "last N reads of the complex144 reference order",
        "variables": model.num_variables, "quadratic_terms": len(model.quadratic), "edges": len(model.edge_pairs),
        "rank_bits": model.k if hasattr(model, "k") else None, "qubo_config": cfg.__dict__,
        "hamilton_paths": npaths, "optimum_score": opt, "reference_path_score": ref_score,
        "reference_path_energy": ref_energy,
        "trials": len(rows), "trials_with_valid_path": len(feas), "optimum_hits": sum(r["optimum_hit"] for r in rows),
        "best_score": max((r["best_score"] for r in feas), default=None),
        "seconds_per_trial": [r["seconds"] for r in rows], "wall_seconds": wall,
        "per_seed": rows, "environment": env,
    }
    (Path(a.out) / f"summary_{a.solver}_n{a.n:03d}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("solver", "n_reads", "variables", "hamilton_paths", "optimum_score",
                                               "trials_with_valid_path", "optimum_hits", "best_score")}))


if __name__ == "__main__":
    main()
