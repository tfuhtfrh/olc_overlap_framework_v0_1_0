"""Move-basis and proposal-mass diagnostic for CHM13 complex144.

Questions
---------
1. Is the current directed 2-switch (R2) move family sufficient to connect the
   fixed-degree state fiber?  If not, do genuine 3-edge reconnects (R3,
   alternating 6-cycles in the tail/head bipartite graph) bridge the two exact
   R2 components that contain the 192 known Hamilton paths?
2. Around a concrete stuck checkpoint, how sparse are structurally useful or
   downhill reconnects under R2 and R3?
3. If eligible moves are reweighted by a locally-balanced Barker weight based
   on the local classical action change, how much proposal mass moves toward
   downhill / cycle-reducing / Hamilton-path-producing moves?

The Barker weight used here is
    g(exp(-a)) = 1 / (1 + exp(a)),
where a=(beta/trotter)*Delta H_classical.  This file is a diagnostic: it does
not itself run the full MH sampler.  A production sampler must include the
reverse proposal normalization (or the equivalent exact MH ratio), and for the
path-integral sampler should use the full Delta action including Trotter terms.
"""
from __future__ import annotations

import itertools
import json
import math
import random
import sys
from collections import Counter, deque
from pathlib import Path

import networkx as nx

import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import (
    graph_metrics,
    load_problem,
    project_rank,
)
from demo_chm13_edge_ordered_path_qubo import DATASET_DIR

VERIFY = DATASET_DIR / "verification"
sys.path.insert(0, str(VERIFY))
from phase_solver import enumerate_paths  # type: ignore

OUT = Path("debug/qubo/chm13_move_basis_proposal_20260928.json")
STUCK_SEED = 20390928
STUCK_CHECKPOINT = 11
BETA = 4.0
TROTTER = 8

# Reproduce the same exchange-only checkpoint as the prior R3 diagnostic.
fc.SWEEPS = 1500
fc.READS = 2


def build_r3_templates(ep):
    """Enumerate genuine 3-edge degree-preserving reconnect templates.

    A template is two disjoint 3-edge matchings on the same three distinct
    tails and three distinct heads.  Flipping all six bits preserves every
    in/out degree and the selected-edge count.  Transpositions that leave one
    old edge unchanged are excluded; those are embedded R2 moves.
    """
    edge_idx = {e: i for i, e in enumerate(ep)}
    templates = set()
    m = len(ep)
    for i, j, k in itertools.combinations(range(m), 3):
        side1 = (i, j, k)
        old = (ep[i], ep[j], ep[k])
        tails = (old[0][0], old[1][0], old[2][0])
        heads = (old[0][1], old[1][1], old[2][1])
        if len(set(tails)) != 3 or len(set(heads)) != 3:
            continue
        # The two 3-cycles are exactly the permutations with no fixed point.
        for perm in ((1, 2, 0), (2, 0, 1)):
            alt_edges = tuple((tails[a], heads[perm[a]]) for a in range(3))
            if any(e not in edge_idx for e in alt_edges):
                continue
            side2 = tuple(sorted(edge_idx[e] for e in alt_edges))
            if len(set(side1 + side2)) != 6:
                continue
            template = tuple(sorted((tuple(sorted(side1)), side2)))
            templates.add(template)
    return tuple(sorted(templates))


def applicable(st, template):
    side1, side2 = template
    a = all(i in st for i in side1) and all(i not in st for i in side2)
    b = all(i in st for i in side2) and all(i not in st for i in side1)
    if not (a or b):
        return None
    return frozenset(st.symmetric_difference(side1 + side2))


def bfs_r2(start, templates):
    q = deque([start])
    seen = {start}
    while q:
        st = q.popleft()
        for tpl in templates:
            nb = applicable(st, tpl)
            if nb is not None and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return seen


def state_from_order(order, edge_idx):
    return frozenset(edge_idx[(u, v)] for u, v in zip(order, order[1:]))


def barker_weight(action_delta):
    """Stable 1/(1+exp(action_delta))."""
    if action_delta >= 0.0:
        z = math.exp(-min(700.0, action_delta))
        return z / (1.0 + z)
    z = math.exp(max(-700.0, action_delta))
    return 1.0 / (1.0 + z)


def replay_stuck_checkpoint():
    rids, ep, reward, cost, incoming, outgoing = load_problem()
    r2 = fc.build_reconnect_templates(ep)
    rng = random.Random(STUCK_SEED)
    ranks = {r: i for i, r in enumerate(rng.sample(rids, len(rids)))}
    selected = None
    op = None
    for it in range(STUCK_CHECKPOINT + 1):
        op = fc.ORDER_SCHEDULE[it % len(fc.ORDER_SCHEDULE)]
        bqm = fc.build_bqm_fixed_cardinality(
            ep, cost, incoming, outgoing, ranks,
            degree_conflict_penalty=288.0,
            order_penalty=op,
        )
        selected, _, _, _ = fc.sample_fixed_cardinality(
            bqm, ep, r2, len(rids) - 1,
            selected, STUCK_SEED + 1000 * it, "exchange_only",
        )
        ranks = project_rank(rids, selected)
    assert selected is not None
    return rids, ep, reward, cost, incoming, outgoing, selected, ranks, float(op)


def diagnose_checkpoint(rids, ep, reward, cost, incoming, outgoing,
                        selected, ranks, order_penalty, families):
    idx = {e: i for i, e in enumerate(ep)}
    st = frozenset(idx[e] for e in selected)
    bqm = fc.build_bqm_fixed_cardinality(
        ep, cost, incoming, outgoing, ranks,
        degree_conflict_penalty=288.0,
        order_penalty=order_penalty,
    )
    abqm = fc.ArrayBQM.from_bqm(bqm, ep)
    x = [1 if i in st else 0 for i in range(len(ep))]
    base = graph_metrics(rids, selected, ranks, reward)
    scale = BETA / TROTTER

    rows = []
    for family, templates in families.items():
        fam = []
        for ti, tpl in enumerate(templates):
            nb = applicable(st, tpl)
            if nb is None:
                continue
            flips = tuple(tpl[0] + tpl[1])
            dclass = abqm.delta_flipset(x, flips)
            nsel = {ep[i] for i in nb}
            nrank = project_rank(rids, nsel)
            met = graph_metrics(rids, nsel, nrank, reward)
            action_delta = scale * dclass
            fam.append({
                "template": ti,
                "delta_classical": dclass,
                "action_delta_no_trotter": action_delta,
                "barker_weight": barker_weight(action_delta),
                "cycle_delta": met["cycle_count"] - base["cycle_count"],
                "degree_delta": met["degree_conflicts"] - base["degree_conflicts"],
                "valid_path": met["valid_path"],
                "path_score": met["path_score"],
            })
        rows.append((family, fam))

    summary = {"base_metrics": base, "order_penalty": order_penalty, "families": {}}
    all_moves = []
    for family, fam in rows:
        z = sum(r["barker_weight"] for r in fam)
        downhill = [r for r in fam if r["delta_classical"] < -1e-12]
        cyc = [r for r in fam if r["cycle_delta"] < 0]
        valid = [r for r in fam if r["valid_path"]]
        summary["families"][family] = {
            "eligible": len(fam),
            "downhill": len(downhill),
            "cycle_reducing": len(cyc),
            "valid_path": len(valid),
            "uniform_downhill_mass": len(downhill) / len(fam) if fam else 0.0,
            "uniform_cycle_reducing_mass": len(cyc) / len(fam) if fam else 0.0,
            "barker_downhill_mass": (
                sum(r["barker_weight"] for r in downhill) / z if z else 0.0
            ),
            "barker_cycle_reducing_mass": (
                sum(r["barker_weight"] for r in cyc) / z if z else 0.0
            ),
            "min_delta_classical": min((r["delta_classical"] for r in fam), default=None),
            "best_valid_score": max((r["path_score"] for r in valid), default=None),
            "examples_cycle_reducing": sorted(cyc, key=lambda r: r["delta_classical"])[:10],
        }
        all_moves.extend((family, r) for r in fam)

    zall = sum(r["barker_weight"] for _, r in all_moves)
    downall = [(f, r) for f, r in all_moves if r["delta_classical"] < -1e-12]
    cycall = [(f, r) for f, r in all_moves if r["cycle_delta"] < 0]
    summary["combined_r2_r3"] = {
        "eligible": len(all_moves),
        "downhill": len(downall),
        "cycle_reducing": len(cycall),
        "uniform_downhill_mass": len(downall) / len(all_moves) if all_moves else 0.0,
        "uniform_cycle_reducing_mass": len(cycall) / len(all_moves) if all_moves else 0.0,
        "barker_downhill_mass": (
            sum(r["barker_weight"] for _, r in downall) / zall if zall else 0.0
        ),
        "barker_cycle_reducing_mass": (
            sum(r["barker_weight"] for _, r in cycall) / zall if zall else 0.0
        ),
    }
    return summary


def hamilton_path_move_stats(graph, ep, reward, r2, r3):
    idx = {e: i for i, e in enumerate(ep)}
    enum = enumerate_paths(graph, limit=1000, timeout_ms=60000)
    paths = enum["paths"]
    ham = [state_from_order(p, idx) for p in paths]
    hamset = set(ham)
    score = {
        st: int(sum(reward[ep[i]] for i in st))
        for st in ham
    }

    per_path = []
    for st in ham:
        rec = {"score": score[st]}
        for name, templates in (("r2", r2), ("r3", r3)):
            eligible = direct_hp = improving = 0
            best = None
            for tpl in templates:
                nb = applicable(st, tpl)
                if nb is None:
                    continue
                eligible += 1
                if nb in hamset:
                    direct_hp += 1
                    best = score[nb] if best is None else max(best, score[nb])
                    if score[nb] > score[st]:
                        improving += 1
            rec[name] = {
                "eligible": eligible,
                "direct_hamilton": direct_hp,
                "improving_direct_hamilton": improving,
                "best_neighbor_score": best,
            }
        per_path.append(rec)

    def hist(key1, key2):
        return dict(sorted(Counter(r[key1][key2] for r in per_path).items()))

    return ham, {
        "paths": len(paths),
        "score_min": min(score.values()),
        "score_max": max(score.values()),
        "r2_eligible_hist": hist("r2", "eligible"),
        "r3_eligible_hist": hist("r3", "eligible"),
        "r2_direct_hamilton_hist": hist("r2", "direct_hamilton"),
        "r3_direct_hamilton_hist": hist("r3", "direct_hamilton"),
        "r3_improving_direct_hist": hist("r3", "improving_direct_hamilton"),
        "paths_with_any_improving_r3": sum(
            r["r3"]["improving_direct_hamilton"] > 0 for r in per_path
        ),
    }


def r3_bridge_audit(ham, r2, r3):
    c1 = bfs_r2(ham[0], r2)
    seed2 = next(h for h in ham if h not in c1)
    c2 = bfs_r2(seed2, r2)
    assert c1.isdisjoint(c2)
    c2set = set(c2)
    bridge_count = 0
    witness = None
    for st in c1:
        for ti, tpl in enumerate(r3):
            nb = applicable(st, tpl)
            if nb is not None and nb in c2set:
                bridge_count += 1
                if witness is None:
                    witness = {
                        "template": ti,
                        "side1": list(tpl[0]),
                        "side2": list(tpl[1]),
                    }
    hamset = set(ham)
    return {
        "r2_component_sizes": [len(c1), len(c2)],
        "hamilton_paths_in_components": [
            sum(h in c1 for h in hamset),
            sum(h in c2 for h in hamset),
        ],
        "r3_cross_component_transitions": bridge_count,
        "r3_connects_all_known_hamilton_paths": bridge_count > 0,
        "first_bridge_template": witness,
    }


def main():
    rids, ep, reward, cost, incoming, outgoing = load_problem()
    r2 = fc.build_reconnect_templates(ep)
    r3 = build_r3_templates(ep)
    print(json.dumps({"r2_templates": len(r2), "r3_templates": len(r3)}), flush=True)

    # Exact Hamilton-path enumeration uses graph.graphml directly.  Keep its
    # edge representation separate from load_problem(): the latter applies the
    # projected-solver loader/filtering/order and is the correct representation
    # for the sampler checkpoint.  Mixing the two was the source of an earlier
    # KeyError in this diagnostic.
    graph = nx.read_graphml(DATASET_DIR / "graph.graphml")
    graph_ep = list(graph.edges())
    graph_r2 = fc.build_reconnect_templates(graph_ep)
    graph_r3 = build_r3_templates(graph_ep)
    ham, hp_stats = hamilton_path_move_stats(graph, graph_ep, reward, graph_r2, graph_r3)
    bridge = r3_bridge_audit(ham, graph_r2, graph_r3)
    bridge["graph_edge_count"] = len(graph_ep)
    bridge["graph_r2_templates"] = len(graph_r2)
    bridge["graph_r3_templates"] = len(graph_r3)
    print("BRIDGE", json.dumps(bridge), flush=True)

    (rids2, ep2, reward2, cost2, incoming2, outgoing2,
     selected, ranks, op) = replay_stuck_checkpoint()
    checkpoint_r2 = fc.build_reconnect_templates(ep2)
    checkpoint_r3 = build_r3_templates(ep2)
    checkpoint = diagnose_checkpoint(
        rids2, ep2, reward2, cost2, incoming2, outgoing2,
        selected, ranks, op, {"r2": checkpoint_r2, "r3": checkpoint_r3},
    )
    print("CHECKPOINT", json.dumps(checkpoint), flush=True)

    result = {
        "dataset": {"N": len(rids), "M": len(ep)},
        "r2_templates": len(r2),
        "r3_templates": len(r3),
        "bridge_audit": bridge,
        "hamilton_path_move_stats": hp_stats,
        "stuck_checkpoint": {
            "seed": STUCK_SEED,
            "outer_iteration": STUCK_CHECKPOINT,
            **checkpoint,
        },
        "interpretation": {
            "move_basis": (
                "R2 alone is not irreducible on the known fixed-degree fiber; "
                "any nonzero R3 cross-component transition joins the two exact "
                "R2 basins containing all 192 known Hamilton paths."
            ),
            "proposal": (
                "Barker mass is a diagnostic for locally-balanced informed "
                "proposals. A full path-integral implementation must use the "
                "full action delta and exact reverse normalization."
            ),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(jso.dumps(result, indent=2) + "\n")
    print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
