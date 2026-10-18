"""Diagnose classical-vs-Trotter freezing at the CHM13 complex144 stuck state.

This benchmark deliberately separates three barriers:
1) classical QUBO barrier from count/degree penalties;
2) extra imaginary-time barrier from single-slice path-integral updates;
3) proposal-kernel freezing, tested with whole-worldline and Trotter-cluster moves.

It is a dynamics diagnostic, not a claim that the custom kernels reproduce
real-time quantum dynamics.

Two targets are used:
A. full-space count-penalized BQM, probed with single-edge flips;
B. fixed-cardinality BQM (count term removed on the 143-edge manifold),
   probed with 1<->1 exchange.

For P=8 we compare:
- slice_local: ordinary single-slice Metropolis update;
- whole_worldline: apply the same classical move to every Trotter slice;
- trotter_cluster (single-bit target only): Swendsen-Wang-style bonds along
  imaginary time for one edge bit, then Metropolis-correct only the classical
  slice-energy change.  This is designed to remove artificial freezing from
  the ferromagnetic Trotter coupling.
"""
from __future__ import annotations

import json
import math
import random
import statistics
import time
from pathlib import Path

from dwave.samplers import TabuSampler

import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import (
    build_bqm_count,
    graph_metrics,
)
from diagnose_chm13_move_basis_proposal import replay_stuck_checkpoint

OUT = Path("debug/qubo/chm13_sqa_freezing_diagnostic_20260928.json")

P = 8
ATTEMPTS = 12000
SEEDS = (202609281, 202609282, 202609283, 202609284)
BETAS = (0.25, 1.0, 4.0)
SQA_SNAPSHOTS = (
    ("actual_early", 4.0, 3.0),
    ("actual_mid", 4.0, 0.3),
    ("actual_late", 4.0, 0.03),
    ("warm_late", 1.0, 0.03),
    ("hot_late", 0.25, 0.03),
)


def accept_log(log_ratio: float, rng: random.Random) -> bool:
    return log_ratio >= 0.0 or math.log(max(rng.random(), 1e-300)) < log_ratio


def trotter_k(beta: float, gamma: float, p: int = P) -> float:
    xarg = max(1e-12, min(50.0, beta * gamma / p))
    return -0.5 * math.log(max(1e-300, math.tanh(xarg)))


def dtrot_local(slice_x, prev_x, next_x, flips, kt):
    # Effective action contribution for -K s_s s_{s+1}.
    dd = 0
    for i in flips:
        old = slice_x[i]
        dd += 1 - 2 * (old != prev_x[i])
        dd += 1 - 2 * (old != next_x[i])
    return 2.0 * kt * dd


def edge_set(x, ep):
    return {ep[i] for i, v in enumerate(x) if v}


def state_summary(x, abqm, ep, rids, ranks, reward, e0):
    selected = edge_set(x, ep)
    return {
        "energy": abqm.energy(x),
        "energy_delta_from_start": abqm.energy(x) - e0,
        "hamming_from_start": None,  # filled by caller when useful
        **graph_metrics(rids, selected, ranks, reward),
    }


def delta_stats(vals):
    vals = sorted(float(v) for v in vals)
    if not vals:
        return {}
    n = len(vals)
    return {
        "n": n,
        "min": vals[0],
        "p10": vals[max(0, int(0.10 * (n - 1)))],
        "median": statistics.median(vals),
        "p90": vals[min(n - 1, int(0.90 * (n - 1)))],
        "max": vals[-1],
        "nonpositive": sum(v <= 1e-12 for v in vals),
    }


def propose_exchange(x, rng):
    ones = [i for i, v in enumerate(x) if v]
    zeros = [i for i, v in enumerate(x) if not v]
    if not ones or not zeros:
        return ()
    return (rng.choice(ones), rng.choice(zeros))


def run_p1(abqm, x0, beta, seed, proposal):
    rng = random.Random(seed)
    x = x0[:]
    best = x[:]
    best_e = abqm.energy(x)
    accepted = 0
    uphill_accepted = 0
    max_hamming = 0
    for _ in range(ATTEMPTS):
        if proposal == "single":
            flips = (rng.randrange(len(x)),)
        elif proposal == "exchange":
            flips = propose_exchange(x, rng)
        else:
            raise ValueError(proposal)
        if not flips:
            continue
        de = abqm.delta_flipset(x, flips)
        if accept_log(-beta * de, rng):
            if de > 1e-12:
                uphill_accepted += 1
            for i in flips:
                x[i] ^= 1
            accepted += 1
            e = abqm.energy(x)
            if e < best_e:
                best_e = e
                best = x[:]
            max_hamming = max(max_hamming, sum(a != b for a, b in zip(x, x0)))
    return {
        "kernel": "p1_metropolis_" + proposal,
        "beta": beta,
        "seed": seed,
        "attempts": ATTEMPTS,
        "accepted": accepted,
        "acceptance": accepted / ATTEMPTS,
        "uphill_accepted": uphill_accepted,
        "best_energy": best_e,
        "best_energy_delta": best_e - abqm.energy(x0),
        "final_energy_delta": abqm.energy(x) - abqm.energy(x0),
        "max_hamming_from_start": max_hamming,
        "best_x": best,
    }


def run_p8_local(abqm, x0, beta, gamma, seed, proposal):
    rng = random.Random(seed)
    slices = [x0[:] for _ in range(P)]
    kt = trotter_k(beta, gamma)
    accepted = uphill_classical = 0
    best = x0[:]
    best_e = abqm.energy(x0)
    max_hamming = 0
    for _ in range(ATTEMPTS):
        s = rng.randrange(P)
        if proposal == "single":
            flips = (rng.randrange(len(x0)),)
        elif proposal == "exchange":
            flips = propose_exchange(slices[s], rng)
        else:
            raise ValueError(proposal)
        if not flips:
            continue
        dc = abqm.delta_flipset(slices[s], flips)
        dt = dtrot_local(
            slices[s],
            slices[(s - 1) % P],
            slices[(s + 1) % P],
            flips,
            kt,
        )
        da = (beta / P) * dc + dt
        if accept_log(-da, rng):
            if dc > 1e-12:
                uphill_classical += 1
            for i in flips:
                slices[s][i] ^= 1
            accepted += 1
            e = abqm.energy(slices[s])
            if e < best_e:
                best_e = e
                best = slices[s][:]
            max_hamming = max(
                max_hamming,
                sum(a != b for a, b in zip(slices[s], x0)),
            )
    return {
        "kernel": "p8_slice_local_" + proposal,
        "beta": beta,
        "gamma": gamma,
        "kt": kt,
        "seed": seed,
        "attempts": ATTEMPTS,
        "accepted": accepted,
        "acceptance": accepted / ATTEMPTS,
        "uphill_classical_accepted": uphill_classical,
        "best_energy": best_e,
        "best_energy_delta": best_e - abqm.energy(x0),
        "max_hamming_from_start": max_hamming,
        "best_x": best,
    }


def run_p8_worldline(abqm, x0, beta, gamma, seed, proposal):
    # The slices remain synchronized, so the Trotter term is exactly invariant.
    rng = random.Random(seed)
    x = x0[:]
    best = x[:]
    best_e = abqm.energy(x)
    accepted = uphill = 0
    max_hamming = 0
    for _ in range(ATTEMPTS):
        if proposal == "single":
            flips = (rng.randrange(len(x)),)
        elif proposal == "exchange":
            flips = propose_exchange(x, rng)
        else:
            raise ValueError(proposal)
        if not flips:
            continue
        de = abqm.delta_flipset(x, flips)
        # Applying the same move to all P slices gives beta * Delta H and
        # leaves every imaginary-time disagreement unchanged.
        if accept_log(-beta * de, rng):
            if de > 1e-12:
                uphill += 1
            for i in flips:
                x[i] ^= 1
            accepted += 1
            e = abqm.energy(x)
            if e < best_e:
                best_e = e
                best = x[:]
            max_hamming = max(max_hamming, sum(a != b for a, b in zip(x, x0)))
    return {
        "kernel": "p8_whole_worldline_" + proposal,
        "beta": beta,
        "gamma": gamma,
        "kt": trotter_k(beta, gamma),
        "seed": seed,
        "attempts": ATTEMPTS,
        "accepted": accepted,
        "acceptance": accepted / ATTEMPTS,
        "uphill_classical_accepted": uphill,
        "best_energy": best_e,
        "best_energy_delta": best_e - abqm.energy(x0),
        "max_hamming_from_start": max_hamming,
        "best_x": best,
    }


def cluster_slices_for_bit(slices, i, kt, rng):
    """SW-style imaginary-time cluster for one binary variable."""
    p_bond = 1.0 - math.exp(-2.0 * kt)
    bonds = [False] * P
    for s in range(P):
        t = (s + 1) % P
        if slices[s][i] == slices[t][i] and rng.random() < p_bond:
            bonds[s] = True  # bond s -- s+1
    seed = rng.randrange(P)
    seen = {seed}
    stack = [seed]
    while stack:
        s = stack.pop()
        # bond s joins s to s+1
        if bonds[s]:
            t = (s + 1) % P
            if t not in seen:
                seen.add(t)
                stack.append(t)
        # bond s-1 joins s-1 to s
        p = (s - 1) % P
        if bonds[p] and p not in seen:
            seen.add(p)
            stack.append(p)
    return tuple(sorted(seen)), p_bond


def run_p8_cluster_single(abqm, x0, beta, gamma, seed):
    rng = random.Random(seed)
    slices = [x0[:] for _ in range(P)]
    kt = trotter_k(beta, gamma)
    accepted = uphill = 0
    sizes = []
    best = x0[:]
    best_e = abqm.energy(x0)
    max_hamming = 0
    last_pb = None
    for _ in range(ATTEMPTS):
        i = rng.randrange(len(x0))
        cluster, pb = cluster_slices_for_bit(slices, i, kt, rng)
        last_pb = pb
        dc_sum = 0.0
        for s in cluster:
            dc_sum += abqm.delta_flipset(slices[s], (i,))
        # Cluster construction accounts for the Trotter ferromagnetic bonds.
        # Metropolis correction therefore uses only the classical slice action.
        da_class = (beta / P) * dc_sum
        if accept_log(-da_class, rng):
            if dc_sum > 1e-12:
                uphill += 1
            for s in cluster:
                slices[s][i] ^= 1
            accepted += 1
            sizes.append(len(cluster))
            for s in cluster:
                e = abqm.energy(slices[s])
                if e < best_e:
                    best_e = e
                    best = slices[s][:]
                max_hamming = max(
                    max_hamming,
                    sum(a != b for a, b in zip(slices[s], x0)),
                )
    return {
        "kernel": "p8_trotter_cluster_single",
        "beta": beta,
        "gamma": gamma,
        "kt": kt,
        "bond_probability": last_pb,
        "seed": seed,
        "attempts": ATTEMPTS,
        "accepted": accepted,
        "acceptance": accepted / ATTEMPTS,
        "uphill_classical_accepted": uphill,
        "mean_accepted_cluster_size": (
            statistics.mean(sizes) if sizes else 0.0
        ),
        "best_energy": best_e,
        "best_energy_delta": best_e - abqm.energy(x0),
        "max_hamming_from_start": max_hamming,
        "best_x": best,
    }


def tabu_baseline(bqm, ep, x0):
    sampler = TabuSampler()
    init = {ep[i]: int(x0[i]) for i in range(len(ep))}
    ss = sampler.sample(
        bqm,
        num_reads=1,
        timeout=1500,
        seed=20260928,
        coefficient_z_first=500,
        coefficient_z_restart=125,
        lower_bound_z=100000,
        num_restarts=30,
        initial_states=[init],
        initial_states_generator="none",
    )
    s = ss.first.sample
    x = [int(s[e]) for e in ep]
    return {
        "energy": float(ss.first.energy),
        "x": x,
    }


def strip_x(row, abqm, ep, rids, ranks, reward, x0):
    x = row.pop("best_x")
    met = state_summary(x, abqm, ep, rids, ranks, reward, abqm.energy(x0))
    met["hamming_from_start"] = sum(a != b for a, b in zip(x, x0))
    row["best_state"] = met
    return row


def aggregate(rows):
    out = {}
    keys = sorted({
        (r["target"], r["kernel"], r["beta"], r.get("gamma"))
        for r in rows
    }, key=str)
    for key in keys:
        rr = [
            r for r in rows
            if (r["target"], r["kernel"], r["beta"], r.get("gamma")) == key
        ]
        out[str(key)] = {
            "runs": len(rr),
            "mean_acceptance": statistics.mean(r["acceptance"] for r in rr),
            "mean_best_energy_delta": statistics.mean(
                r["best_energy_delta"] for r in rr
            ),
            "best_energy_delta": min(r["best_energy_delta"] for r in rr),
            "max_hamming_from_start": max(r["max_hamming_from_start"] for r in rr),
            "valid_path_runs": sum(r["best_state"]["valid_path"] for r in rr),
            "min_degree_conflicts": min(
                r["best_state"]["degree_conflicts"] for r in rr
            ),
            "min_cycle_count": min(r["best_state"]["cycle_count"] for r in rr),
        }
    return out


def main():
    (
        rids, ep, reward, cost, incoming, outgoing,
        selected, ranks, order_penalty,
    ) = replay_stuck_checkpoint()

    idx = {e: i for i, e in enumerate(ep)}
    x0 = [0] * len(ep)
    for e in selected:
        x0[idx[e]] = 1

    full_bqm = build_bqm_count(
        ep, cost, incoming, outgoing, ranks, len(rids),
        degree_conflict_penalty=288.0,
        count_penalty=32.0,
        order_penalty=order_penalty,
    )
    fixed_bqm = fc.build_bqm_fixed_cardinality(
        ep, cost, incoming, outgoing, ranks,
        degree_conflict_penalty=288.0,
        order_penalty=order_penalty,
    )
    full = fc.ArrayBQM.from_bqm(full_bqm, ep)
    fixed = fc.ArrayBQM.from_bqm(fixed_bqm, ep)

    single_deltas = [full.delta_flipset(x0, (i,)) for i in range(len(ep))]
    exchange_deltas = []
    ones = [i for i, v in enumerate(x0) if v]
    zeros = [i for i, v in enumerate(x0) if not v]
    for i in ones:
        for j in zeros:
            exchange_deltas.append(fixed.delta_flipset(x0, (i, j)))

    barrier = {
        "start_metrics": graph_metrics(rids, selected, ranks, reward),
        "order_penalty": order_penalty,
        "full_single_bit_delta": delta_stats(single_deltas),
        "fixed_exchange_delta": delta_stats(exchange_deltas),
        "single_bit_best_acceptance_by_beta": {
            str(beta): math.exp(-beta * max(0.0, min(single_deltas)))
            for beta in BETAS
        },
        "exchange_best_acceptance_by_beta": {
            str(beta): math.exp(-beta * max(0.0, min(exchange_deltas)))
            for beta in BETAS
        },
        "late_single_slice_trotter_cost_per_flipped_bit_if_slices_agree": (
            4.0 * trotter_k(4.0, 0.03)
        ),
    }
    print("BARRIER", json.dumps(barrier), flush=True)

    tb = tabu_baseline(full_bqm, ep, x0)
    tabu_metrics = state_summary(
        tb["x"], full, ep, rids, ranks, reward, full.energy(x0)
    )
    tabu_metrics["hamming_from_start"] = sum(
        a != b for a, b in zip(tb["x"], x0)
    )
    tabu = {"energy_reported": tb["energy"], **tabu_metrics}
    print("TABU", json.dumps(tabu), flush=True)

    rows = []
    # Classical P=1 reference.
    for beta in BETAS:
        for seed in SEEDS:
            for target, abqm, proposal in (
                ("full_count", full, "single"),
                ("fixed_cardinality", fixed, "exchange"),
            ):
                r = run_p1(abqm, x0, beta, seed, proposal)
                r["target"] = target
                rows.append(strip_x(r, abqm, ep, rids, ranks, reward, x0))

    # P=8 snapshot diagnostics.
    for label, beta, gamma in SQA_SNAPSHOTS:
        for seed in SEEDS:
            for target, abqm, proposal in (
                ("full_count", full, "single"),
                ("fixed_cardinality", fixed, "exchange"),
            ):
                for runner in (run_p8_local, run_p8_worldline):
                    r = runner(abqm, x0, beta, gamma, seed, proposal)
                    r["target"] = target
                    r["snapshot"] = label
                    rows.append(
                        strip_x(r, abqm, ep, rids, ranks, reward, x0)
                    )
            r = run_p8_cluster_single(full, x0, beta, gamma, seed)
            r["target"] = "full_count"
            r["snapshot"] = label
            rows.append(strip_x(r, full, ep, rids, ranks, reward, x0))

    agg = aggregate(rows)
    result = {
        "purpose": "separate classical penalty barrier from Trotter freezing",
        "P": P,
        "attempts_per_run": ATTEMPTS,
        "seeds": list(SEEDS),
        "barrier_audit": barrier,
        "tabu_from_same_state": tabu,
        "aggregate": agg,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("AGGREGATE", json.dumps(agg), flush=True)


if __name__ == "__main__":
    main()
