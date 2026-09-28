"""Non-oracle structured optimizer and rank-lag diagnostic for CHM13 complex144.

Starting from the known *stuck checkpoint* only (not the known optimum), this
script tests whether the newly identified geometry can discover a route to the
certified ground state without target-edge information.

Proposal stages
---------------
1. If the state is degree-correct but not a Hamilton path, build the residual
   tail/head bipartite graph and enumerate one shortest open alternating trail
   from the current sink to every reachable candidate sink.  Rank candidates by
   topology (valid path first, then fewer cycles) and intrinsic weighted cost.
2. Once a Hamilton path is reached, enumerate applicable R2, R3, and compound
   pairs of simultaneously-applicable R2 moves.  Keep only Hamilton-path
   outcomes that improve the true overlap score, and greedily take the best.

Rank diagnostic
---------------
For every candidate actually chosen, report:
- stale_rank_delta: change under the fixed rank certificate used before move;
- self_projected_delta: difference between state energies after independently
  projecting rank on each state.

The latter is not a single static Hamiltonian; it diagnoses certificate lag.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import (
    graph_metrics,
    project_rank,
    selected_degrees,
)
from diagnose_chm13_move_basis_proposal import (
    applicable,
    build_r3_templates,
    replay_stuck_checkpoint,
)

OUT = Path("debug/qubo/chm13_nonoracle_structured_optimizer_20260928.json")
CERT = 1_343_093


def state_metrics(rids, state, reward):
    ranks = project_rank(rids, state)
    m = graph_metrics(rids, state, ranks, reward)
    indeg, outdeg = selected_degrees(rids, state)
    m["sources"] = [r for r in rids if indeg[r] == 0 and outdeg[r] <= 1]
    m["sinks"] = [r for r in rids if outdeg[r] == 0 and indeg[r] <= 1]
    return ranks, m


def build_abqm(ep, cost, incoming, outgoing, ranks, op):
    bqm = fc.build_bqm_fixed_cardinality(
        ep, cost, incoming, outgoing, ranks,
        degree_conflict_penalty=288.0,
        order_penalty=op,
    )
    return fc.ArrayBQM.from_bqm(bqm, ep)


def energy(abqm, ep, state):
    s = set(state)
    x = [1 if e in s else 0 for e in ep]
    return abqm.energy(x)


def self_projected_energy(rids, ep, cost, incoming, outgoing, state, op):
    ranks = project_rank(rids, state)
    abqm = build_abqm(ep, cost, incoming, outgoing, ranks, op)
    return energy(abqm, ep, state)


def residual_shortest_paths(ep, state, current_sink):
    """One shortest residual path from current_sink_out to each reachable out node."""
    selected = set(state)
    adj = {}
    def add(a, b, ei):
        adj.setdefault(a, []).append((b, ei))
    for i, (u, v) in enumerate(ep):
        if (u, v) in selected:
            add(("in", v), ("out", u), i)   # remove
        else:
            add(("out", u), ("in", v), i)  # add

    start = ("out", current_sink)
    q = deque([start])
    prev = {start: None}
    how = {}
    while q:
        a = q.popleft()
        for b, ei in adj.get(a, ()):
            if b in prev:
                continue
            prev[b] = a
            how[b] = ei
            q.append(b)

    out = {}
    for goal in prev:
        if goal[0] != "out" or goal == start:
            continue
        cur = goal
        eis = []
        while cur != start:
            eis.append(how[cur])
            cur = prev[cur]
        eis.reverse()
        # start/end are both out copies: even residual length, equal add/remove.
        if len(eis) % 2 == 0:
            out[goal[1]] = tuple(eis)
    return out


def apply_flip_indices(state, ep, flip_indices):
    ns = set(state)
    for i in flip_indices:
        e = ep[i]
        if e in ns:
            ns.remove(e)
        else:
            ns.add(e)
    return ns


def open_candidates(rids, ep, reward, state):
    ranks, met = state_metrics(rids, state, reward)
    if met["degree_conflicts"] != 0 or len(met["sinks"]) != 1:
        return []
    sink = met["sinks"][0]
    paths = residual_shortest_paths(ep, state, sink)
    rows = []
    for target_sink, flips in paths.items():
        ns = apply_flip_indices(state, ep, flips)
        nr, nm = state_metrics(rids, ns, reward)
        rows.append({
            "family": "open_trail",
            "target_sink": target_sink,
            "residual_length": len(flips),
            "k": len(flips) // 2,
            "flips": flips,
            "state": ns,
            "metrics": nm,
        })
    return rows


def closed_candidates(rids, ep, reward, state, r2, r3):
    idx = {e:i for i,e in enumerate(ep)}
    st = frozenset(idx[e] for e in state)
    rows = []

    applicable_r2 = []
    for ti, tpl in enumerate(r2):
        nb = applicable(st, tpl)
        if nb is None:
            continue
        applicable_r2.append((ti, tpl, nb))
        ns = {ep[i] for i in nb}
        _, nm = state_metrics(rids, ns, reward)
        rows.append({
            "family": "r2", "templates": [ti],
            "k": 2, "state": ns, "metrics": nm,
            "flips": tuple(tpl[0] + tpl[1]),
        })

    for ti, tpl in enumerate(r3):
        nb = applicable(st, tpl)
        if nb is None:
            continue
        ns = {ep[i] for i in nb}
        _, nm = state_metrics(rids, ns, reward)
        rows.append({
            "family": "r3", "templates": [ti],
            "k": 3, "state": ns, "metrics": nm,
            "flips": tuple(tpl[0] + tpl[1]),
        })

    # Compound proposal: two disjoint R2 templates simultaneously applicable
    # in the current state.  This can preserve HP feasibility even when either
    # R2 alone temporarily creates a cycle.
    for a in range(len(applicable_r2)):
        tia, tpla, nba = applicable_r2[a]
        fa = set(tpla[0] + tpla[1])
        for b in range(a + 1, len(applicable_r2)):
            tib, tplb, nbb = applicable_r2[b]
            fb = set(tplb[0] + tplb[1])
            if fa & fb:
                continue
            flips = tuple(sorted(fa | fb))
            ns_idx = frozenset(st.symmetric_difference(flips))
            ns = {ep[i] for i in ns_idx}
            _, nm = state_metrics(rids, ns, reward)
            rows.append({
                "family": "r2_pair", "templates": [tia, tib],
                "k": 4, "state": ns, "metrics": nm,
                "flips": flips,
            })
    return rows


def annotate_rank_lag(row, rids, ep, cost, incoming, outgoing,
                      state, current_ranks, op):
    stale = build_abqm(ep, cost, incoming, outgoing, current_ranks, op)
    e0_stale = energy(stale, ep, state)
    e1_stale = energy(stale, ep, row["state"])
    e0_self = self_projected_energy(
        rids, ep, cost, incoming, outgoing, state, op
    )
    e1_self = self_projected_energy(
        rids, ep, cost, incoming, outgoing, row["state"], op
    )
    row["stale_rank_delta"] = e1_stale - e0_stale
    row["self_projected_delta"] = e1_self - e0_self
    return row


def clean(row):
    return {
        k: v for k, v in row.items()
        if k not in ("state", "flips")
    } | {
        "flip_count": len(row.get("flips", ())),
        "flip_edges": None,
    }


def main():
    (
        rids, ep, reward, cost, incoming, outgoing,
        selected, ranks, op,
    ) = replay_stuck_checkpoint()
    state = set(selected)
    r2 = fc.build_reconnect_templates(ep)
    r3 = build_r3_templates(ep)

    trace = []
    diagnostics = {}

    # Stage A: non-oracle endpoint repair.
    current_ranks = project_rank(rids, state)
    opens = open_candidates(rids, ep, reward, state)
    for row in opens:
        annotate_rank_lag(
            row, rids, ep, cost, incoming, outgoing,
            state, current_ranks, op,
        )
    diagnostics["open_candidate_count"] = len(opens)
    diagnostics["open_valid_path_count"] = sum(
        r["metrics"]["valid_path"] for r in opens
    )

    # Non-oracle ranking: valid path > fewer cycles > lower intrinsic cost.
    # Among valid paths, maximizing true overlap score is exactly the intended
    # weighted path objective, not target leakage.
    valid = [r for r in opens if r["metrics"]["valid_path"]]
    if valid:
        chosen = max(
            valid,
            key=lambda r: (
                r["metrics"]["path_score"],
                -r["k"],
                -r["self_projected_delta"],
            ),
        )
    else:
        chosen = min(
            opens,
            key=lambda r: (
                r["metrics"]["cycle_count"],
                r["metrics"]["degree_conflicts"],
                r["self_projected_delta"],
                r["k"],
            ),
        )
    trace.append(clean(chosen))
    state = set(chosen["state"])

    # Stage B: within the repaired fiber, greedily improve valid HP score using
    # structured closed moves.  Reproject rank after every accepted macro move.
    for step in range(8):
        cranks, cmet = state_metrics(rids, state, reward)
        if not cmet["valid_path"]:
            break
        cscore = cmet["path_score"]
        cands = closed_candidates(rids, ep, reward, state, r2, r3)
        for row in cands:
            annotate_rank_lag(
                row, rids, ep, cost, incoming, outgoing,
                state, cranks, op,
            )
        improving = [
            r for r in cands
            if r["metrics"]["valid_path"]
            and r["metrics"]["path_score"] > cscore
        ]
        if not improving:
            diagnostics["closed_stop_reason"] = "no improving feasible structured move"
            diagnostics["closed_candidates_last"] = len(cands)
            break
        chosen = max(
            improving,
            key=lambda r: (
                r["metrics"]["path_score"],
                -r["k"],
                -r["self_projected_delta"],
            ),
        )
        trace.append(clean(chosen))
        state = set(chosen["state"])
        if chosen["metrics"]["path_score"] == CERT:
            diagnostics["closed_stop_reason"] = "certified optimum reached"
            break

    final_ranks, final_metrics = state_metrics(rids, state, reward)
    result = {
        "method": "non-oracle residual open-trail + feasible R2/R3/R2-pair greedy refinement",
        "order_penalty_used_for_rank_lag_diagnostic": op,
        "r2_templates": len(r2),
        "r3_templates": len(r3),
        "diagnostics": diagnostics,
        "trace": trace,
        "final_metrics": final_metrics,
        "ground_hit": final_metrics["path_score"] == CERT,
        "interpretation": (
            "The optimizer never uses target edges or the certified path. "
            "CERT is used only to label whether the final score equals the "
            "independently certified optimum."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
