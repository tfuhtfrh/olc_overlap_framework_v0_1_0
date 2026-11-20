"""Temporary accumulated-cycle test; reuse existing samplers, models and decoder."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import report_sqa_benchmark as sqa
import report_tabu_benchmark as tabu

BACKENDS = {"sqa": sqa, "tabu": tabu}
CYCLE_PENALTY = 0.1

def run(backend, seed, out, smoke=False):
    module = BACKENDS[backend]
    out = Path(out)
    original = module.edge_bqm
    counts = Counter()
    history = []

    def accumulated_bqm(model, edges, costs, cycles=(), cycle_penalty=0.0):
        # Repeated detection contributes again: sum of every previous H_cyc.
        for cycle in cycles:
            counts.update(cycle)
        bqm = original(model, edges, costs)
        for index, count in counts.items():
            bqm.add_linear(index, cycle_penalty * count)
        history.append({
            "iteration": len(history), "new_cycle_count": len(cycles),
            "cumulative_edge_occurrences": sum(counts.values()),
            "penalized_edges": len(counts),
            "max_cumulative_linear_penalty": cycle_penalty * max(counts.values(), default=0),
            "edge_detection_counts": dict(sorted(counts.items())),
        })
        module.write_json(out / "cumulative_penalties.json", history)
        print("CUMULATIVE", json.dumps({"backend": backend, **history[-1]}), flush=True)
        return bqm

    module.edge_bqm = accumulated_bqm
    try:
        kwargs = {"reads": 2, "iterations": 2} if smoke else {}
        if smoke and backend == "sqa":
            kwargs["sweeps"] = 50
        result = module.run("edge", seed, CYCLE_PENALTY, out, **kwargs)
    finally:
        module.edge_bqm = original
    result.update(backend=backend, cycle_update="sum_every_iteration_with_multiplicity",
                  cycle_penalty=CYCLE_PENALTY, cumulative_penalty_history=history,
                  baseline_run=39140273162 if backend == "sqa" else 39140828555)
    module.write_json(out / "summary.json", result)
    return result

def preflight(out):
    out = Path(out)
    model, rewards, cfg, edges, costs = sqa.problem()
    encoded = sqa.encode_order(model, json.loads(
        (sqa.DATASET_DIR / "reference_path.json").read_text())["normalized_nodes"])
    audit = sqa.rank_audit(encoded, model, rewards, cfg)
    assert audit["valid_path"] and model.num_variables == 3789 and len(edges) == 261
    base = sqa.edge_bqm(model, edges, costs)
    # Independent check: the same cycle seen twice contributes 2*A to each edge.
    test_cycle = (0, 1)
    repeated = sqa.edge_bqm(model, edges, costs, [test_cycle, test_cycle], CYCLE_PENALTY)
    assert all(abs(repeated.linear[i] - base.linear[i] - 0.2) < 1e-8 for i in test_cycle)
    assert dict(repeated.quadratic) == dict(base.quadratic)
    metrics, _ = sqa.path_metrics(encoded[:len(edges)], model, edges, rewards)
    assert metrics["valid_path"]
    for backend in BACKENDS:
        run(backend, sqa.PILOT_SEEDS[0], out / ("smoke_" + backend), smoke=True)
    sqa.write_json(out / "audit.json", {
        "reference": audit, "edge_reference": metrics,
        "repeated_cycle_accumulation_checked": True,
        "cycle_penalty": CYCLE_PENALTY, "formal_seeds": sqa.SEEDS,
        "environment": sqa.environment()})

def aggregate(root, out):
    rows = [json.loads(p.read_text()) for p in Path(root).rglob("summary.json")]
    rows = [r for r in rows if r["seed"] in sqa.SEEDS and r["reads"] == 50]
    expected = {(b, seed) for b in BACKENDS for seed in sqa.SEEDS}
    assert len(rows) == 12 and {(r["backend"], r["seed"]) for r in rows} == expected
    result = {}
    for backend in BACKENDS:
        trials = [r for r in rows if r["backend"] == backend]
        scores = [r["best_score"] for r in trials if r["ever_feasible"]]
        result[backend] = {
            "runs": len(trials), "feasible_runs": len(scores),
            "optimum_hits": sum(r["optimum_hit"] for r in trials),
            "best": max(scores, default=None),
            "sampling_seconds_by_seed": {r["seed"]: r["sampling_seconds"] for r in trials},
        }
    sqa.write_json(out, {"cycle_update": "sum_every_iteration_with_multiplicity",
                        "aggregate": result, "trials": rows})
    print(json.dumps(result, indent=2), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run", "aggregate"))
    parser.add_argument("--backend", choices=tuple(BACKENDS))
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight(args.out)
    elif args.mode == "aggregate":
        aggregate(args.input, args.out)
    else:
        run(args.backend, sqa.SEEDS[args.seed_index], args.out)
