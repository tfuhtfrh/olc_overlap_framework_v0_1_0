"""Fixed-cardinality path-integral sampler for CHM13 projected edge QUBO.

Purpose
-------
Change ONLY the sampling dynamics while keeping the projected edge objective:
    H = H_weight + H_degree_conflict + H_order(rank)
The exact N-1 edge-count condition is enforced by the sampler state space rather
than by the dense quadratic count penalty.

Every Trotter slice always contains exactly N-1 selected edges.

Two symmetric proposal kernels are mixed:
1) exchange: one selected edge <-> one unselected edge (2-bit move);
2) reconnect: a directed 2-switch
       (a->b, c->d) <-> (a->d, c->b)
   chosen from a STATIC precomputed template set.  The static-template proposal
   is an involution and is symmetric; ineligible templates are null proposals.

The reconnect probability may vary with anneal time.  Because p(t) depends only
on sweep index, not on the current state, each fixed-time proposal kernel remains
symmetric.  State-dependent p would require a Hastings correction.

This is best described as a fixed-cardinality constrained/path-integral sampler.
It is NOT claimed to be an exact simulation of the standard independent
transverse-field driver, because the sampled classical state space is restricted
to a fixed Hamming-weight sector.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import dimod

from benchmark_chm13_projected_edge_hybrid import (
    load_problem,
    project_rank,
    graph_metrics,
)

OUT = Path("debug/qubo/chm13_fixed_cardinality_kernel_benchmark_20260928.json")
BASE_SEED = 20260928
CERT_SCORE = 1_343_093

# Keep the same outer order-bias idea as projected SQA.
ORDER_SCHEDULE = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
OUTER_ITERS = 12

# Moderate benchmark budget; enough to compare kernels without making CI huge.
TROTTER = 8
SWEEPS = 500
READS = 2
BETA = 4.0
GAMMA_MAX = 3.0
GAMMA_MIN = 0.03
N_SEEDS = 8

CONFIGS = (
    "exchange_only",
    "static_50",
    "reconnect_high_to_low",  # user's proposed direction: 0.8 -> 0.2
    "reconnect_low_to_high",  # control: 0.2 -> 0.8
)


def reconnect_probability(config: str, frac: float) -> float:
    if config == "exchange_only":
        return 0.0
    if config == "static_50":
        return 0.5
    if config == "reconnect_high_to_low":
        return 0.8 - 0.6 * frac
    if config == "reconnect_low_to_high":
        return 0.2 + 0.6 * frac
    raise ValueError(config)


def build_bqm_fixed_cardinality(ep, cost, incoming, outgoing, ranks,
                                degree_conflict_penalty=288.0,
                                order_penalty=0.0):
    """Same projected edge energy, with no count-square term.

    On the sampler manifold sum_e x_e=N-1, the old count term is identically 0.
    """
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
    for e in ep:
        backward = 1.0 if ranks[e[1]] <= ranks[e[0]] else 0.0
        bqm.add_variable(e, float(cost[e]) + order_penalty * backward)
    for r in incoming:
        for group in (incoming[r], outgoing[r]):
            for i, e1 in enumerate(group):
                for e2 in group[i + 1:]:
                    bqm.add_interaction(e1, e2, degree_conflict_penalty)
    return bqm


@dataclass
class ArrayBQM:
    h: list[float]
    nbrs: list[dict[int, float]]
    offset: float

    @classmethod
    def from_bqm(cls, bqm, ep):
        idx = {e: i for i, e in enumerate(ep)}
        h = [float(bqm.linear.get(e, 0.0)) for e in ep]
        nbrs = [dict() for _ in ep]
        for (u, v), bias in bqm.quadratic.items():
            i, j = idx[u], idx[v]
            w = float(bias)
            nbrs[i][j] = nbrs[i].get(j, 0.0) + w
            nbrs[j][i] = nbrs[j].get(i, 0.0) + w
        return cls(h, nbrs, float(bqm.offset))

    def energy(self, x: list[int]) -> float:
        e = self.offset + sum(hv * xv for hv, xv in zip(self.h, x))
        for i, adj in enumerate(self.nbrs):
            if not x[i]:
                continue
            for j, w in adj.items():
                if j > i and x[j]:
                    e += w
        return e

    def delta_flipset(self, x: list[int], flips: tuple[int, ...]) -> float:
        """Exact classical-energy delta by sequential local-field flips."""
        y = x[:]  # flips are tiny (2 or 4); n=261, still cheap and robust.
        delta = 0.0
        for i in flips:
            old = y[i]
            d = 1 - 2 * old
            local = self.h[i]
            for j, w in self.nbrs[i].items():
                local += w * y[j]
            delta += d * local
            y[i] = 1 - old
        return delta


def build_reconnect_templates(ep):
    """Static symmetric directed 2-switch templates.

    Each template is ((i,j),(k,l)), where the two edge pairs have identical
    tail and head multisets:
        (a->b, c->d) <-> (a->d, c->b).
    Choosing uniformly from this static set makes the proposal symmetric.
    """
    edge_idx = {e: i for i, e in enumerate(ep)}
    templates = set()
    m = len(ep)
    for i in range(m):
        a, b = ep[i]
        for j in range(i + 1, m):
            c, d = ep[j]
            if a == c or b == d:
                continue
            e3 = (a, d)
            e4 = (c, b)
            if e3 not in edge_idx or e4 not in edge_idx:
                continue
            k, l = edge_idx[e3], edge_idx[e4]
            if k == l:
                continue
            side1 = tuple(sorted((i, j)))
            side2 = tuple(sorted((k, l)))
            if side1 == side2 or len(set(side1 + side2)) != 4:
                continue
            template = tuple(sorted((side1, side2)))
            templates.add(template)
    return tuple(sorted(templates))


def random_fixed_state(m, k, rng):
    ones = set(rng.sample(range(m), k))
    return [1 if i in ones else 0 for i in range(m)]


def propose_exchange(x, rng):
    ones = [i for i, v in enumerate(x) if v]
    zeros = [i for i, v in enumerate(x) if not v]
    i = rng.choice(ones)
    j = rng.choice(zeros)
    return (i, j), True


def propose_reconnect(x, templates, rng):
    if not templates:
        return (), False
    side1, side2 = rng.choice(templates)
    a = all(x[i] for i in side1) and all(not x[i] for i in side2)
    b = all(x[i] for i in side2) and all(not x[i] for i in side1)
    if a or b:
        return tuple(side1 + side2), True
    return (), False


def trotter_delta(slice_x, prev_x, next_x, flips, kt):
    # -K s_m s_{m+1} = constant + 2K * Hamming(x_m,x_{m+1}).
    dd = 0
    for i in flips:
        old = slice_x[i]
        dd += (1 - 2 * (old != prev_x[i]))
        dd += (1 - 2 * (old != next_x[i]))
    return 2.0 * kt * dd


def one_read(abqm, templates, k_select, init, seed, config):
    rng = random.Random(seed)
    m = len(abqm.h)
    if init is None:
        slices = [random_fixed_state(m, k_select, rng) for _ in range(TROTTER)]
    else:
        base = [1 if i in init else 0 for i in range(m)]
        assert sum(base) == k_select
        slices = [base[:] for _ in range(TROTTER)]

    attempts = {"exchange": 0, "reconnect": 0}
    eligible = {"exchange": 0, "reconnect": 0}
    accepted = {"exchange": 0, "reconnect": 0}

    for sw in range(SWEEPS):
        frac = sw / max(1, SWEEPS - 1)
        # Standard SQA-like transverse-field schedule for the Trotter coupling.
        gamma = GAMMA_MAX * ((GAMMA_MIN / GAMMA_MAX) ** frac)
        xarg = max(1e-12, min(50.0, BETA * gamma / TROTTER))
        kt = -0.5 * math.log(max(1e-300, math.tanh(xarg)))
        p_rec = reconnect_probability(config, frac)

        order = list(range(TROTTER))
        rng.shuffle(order)
        for s in order:
            use_rec = rng.random() < p_rec
            kernel = "reconnect" if use_rec else "exchange"
            attempts[kernel] += 1
            if use_rec:
                flips, ok = propose_reconnect(slices[s], templates, rng)
            else:
                flips, ok = propose_exchange(slices[s], rng)
            if not ok:
                continue
            eligible[kernel] += 1

            dclass = abqm.delta_flipset(slices[s], flips)
            dtrot = trotter_delta(
                slices[s],
                slices[(s - 1) % TROTTER],
                slices[(s + 1) % TROTTER],
                flips,
                kt,
            )
            daction = (BETA / TROTTER) * dclass + dtrot
            if daction <= 0.0 or rng.random() < math.exp(-min(700.0, daction)):
                for i in flips:
                    slices[s][i] ^= 1
                accepted[kernel] += 1

    best_x = min(slices, key=abqm.energy)
    assert sum(best_x) == k_select
    return best_x, abqm.energy(best_x), attempts, eligible, accepted


def sample_fixed_cardinality(bqm, ep, templates, k_select, init_edges, seed, config):
    abqm = ArrayBQM.from_bqm(bqm, ep)
    init = None
    if init_edges is not None:
        idx = {e: i for i, e in enumerate(ep)}
        init = {idx[e] for e in init_edges}
        assert len(init) == k_select

    best = None
    agg = {
        "attempts": {"exchange": 0, "reconnect": 0},
        "eligible": {"exchange": 0, "reconnect": 0},
        "accepted": {"exchange": 0, "reconnect": 0},
    }
    t0 = time.perf_counter()
    for rd in range(READS):
        x, en, attempts, eligible, accepted = one_read(
            abqm, templates, k_select, init, seed + 100003 * rd, config
        )
        for bucket, vals in (("attempts", attempts), ("eligible", eligible), ("accepted", accepted)):
            for key, val in vals.items():
                agg[bucket][key] += val
        if best is None or en < best[1]:
            best = (x, en)
    sec = time.perf_counter() - t0
    selected = {ep[i] for i, v in enumerate(best[0]) if v}
    return selected, sec, best[1], agg


def run_one(config, seed):
    rids, ep, reward, cost, incoming, outgoing = load_problem()
    templates = build_reconnect_templates(ep)
    rng = random.Random(seed)
    ranks = {r: i for i, r in enumerate(rng.sample(rids, len(rids)))}
    selected = None
    rows = []

    for it in range(OUTER_ITERS):
        op = ORDER_SCHEDULE[it % len(ORDER_SCHEDULE)]
        bqm = build_bqm_fixed_cardinality(
            ep, cost, incoming, outgoing, ranks,
            degree_conflict_penalty=288.0,
            order_penalty=op,
        )
        selected, sec, energy, stats = sample_fixed_cardinality(
            bqm, ep, templates, len(rids) - 1,
            selected, seed + 1000 * it, config,
        )
        ranks = project_rank(rids, selected)
        metrics = graph_metrics(rids, selected, ranks, reward)
        row = {
            "config": config,
            "seed": seed,
            "iteration": it,
            "order_penalty": op,
            "seconds": sec,
            "inner_energy_no_count": energy,
            "reconnect_templates": len(templates),
            **stats,
            **metrics,
        }
        rows.append(row)
        print("ROW", json.dumps(row), flush=True)

    feasible = [r for r in rows if r["valid_path"]]
    best_feasible = max(feasible, key=lambda r: r["path_score"]) if feasible else None
    return rows, best_feasible


def main():
    all_rows = []
    summaries = []
    for ci, config in enumerate(CONFIGS):
        for s in range(N_SEEDS):
            seed = BASE_SEED + 100000 * ci + 10000 * s
            rows, best = run_one(config, seed)
            all_rows.extend(rows)
            summaries.append({
                "config": config,
                "seed": seed,
                "ever_feasible": best is not None,
                "best_score": None if best is None else best["path_score"],
                "score_gap": None if best is None else CERT_SCORE - best["path_score"],
                "first_feasible_iteration": next(
                    (r["iteration"] for r in rows if r["valid_path"]), None
                ),
                "final_degree_conflicts": rows[-1]["degree_conflicts"],
                "final_cycle_count": rows[-1]["cycle_count"],
            })
            print("SUMMARY", json.dumps(summaries[-1]), flush=True)

    aggregate = {}
    for config in CONFIGS:
        ss = [s for s in summaries if s["config"] == config]
        feasible = [s for s in ss if s["ever_feasible"]]
        scores = [s["best_score"] for s in feasible]
        aggregate[config] = {
            "runs": len(ss),
            "feasible_runs": len(feasible),
            "P_feasible": len(feasible) / len(ss),
            "best_score": max(scores) if scores else None,
            "median_score": None if not scores else sorted(scores)[len(scores)//2],
            "ground_hits": sum(1 for x in scores if x == CERT_SCORE),
        }
    result = {
        "sampler": "fixed-cardinality constrained path-integral sampler",
        "state_cardinality": 143,
        "variables": 261,
        "trotter": TROTTER,
        "sweeps": SWEEPS,
        "reads": READS,
        "beta": BETA,
        "gamma_max": GAMMA_MAX,
        "gamma_min": GAMMA_MIN,
        "configs": CONFIGS,
        "aggregate": aggregate,
        "summaries": summaries,
        "rows": all_rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("AGGREGATE", json.dumps(aggregate), flush=True)


if __name__ == "__main__":
    main()
