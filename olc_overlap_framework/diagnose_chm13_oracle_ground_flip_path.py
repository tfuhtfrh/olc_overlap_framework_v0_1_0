"""Oracle diagnostic: construct an alternating-component flip path from the
CHM13 stuck checkpoint to the certified weighted Hamilton-path optimum.

This is not a production solver.  The optimum is used only to reveal the exact
geometry of the required transition.

Both states are degree-correct N-1-edge partial permutations.  In the bipartite
tail/head representation their symmetric difference decomposes into vertex-
disjoint alternating paths and cycles.  Flipping one whole component preserves
cardinality and degree constraints except that an open component transfers an
endpoint defect.

The script:
1) reproduces the stuck checkpoint;
2) enumerates the 192 exact Hamilton paths and selects the certified optimum;
3) decomposes stuck XOR optimum into alternating components;
4) searches all component orders (small factorial if possible) for the route
   with minimum maximum fixed-BQM energy barrier;
5) reports topology, endpoints, objective score, and energy after every flip.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path

import networkx as nx

import benchmark_chm13_fixed_cardinality_kernels as fc
from benchmark_chm13_projected_edge_hybrid import (
    graph_metrics,
    project_rank,
    selected_degrees,
)
from demo_chm13_edge_ordered_path_qubo import DATASET_DIR
from diagnose_chm13_move_basis_proposal import replay_stuck_checkpoint

VERIFY = DATASET_DIR / "verification"
sys.path.insert(0, str(VERIFY))
from phase_solver import enumerate_paths  # type: ignore

OUT = Path("debug/qubo/chm13_oracle_ground_flip_path_20260928.json")
CERT = 1_343_093


def normalized_exact_paths():
    graph = nx.read_graphml(DATASET_DIR / "graph.graphml")
    with (DATASET_DIR / "nodes.tsv").open(newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h, delimiter="\t"))
    mp = {row["node"]: row["read_id"] for row in rows}
    raw = enumerate_paths(graph, limit=1000, timeout_ms=60000)["paths"]
    return [[mp.get(v, v.rstrip("+-")) for v in p] for p in raw]


def selected_score(order, reward):
    edges = list(zip(order, order[1:]))
    if not all(e in reward for e in edges):
        return None
    return int(sum(reward[e] for e in edges))


def componentize(stuck, target):
    """Connected components of XOR in tail/head bipartite graph.

    Each edge-bit becomes an undirected bipartite edge connecting ("out",u) and
    ("in",v), labelled old/remove or target/add.
    """
    xor = (stuck - target) | (target - stuck)
    adj = defaultdict(list)
    for e in xor:
        u, v = e
        a, b = ("out", u), ("in", v)
        lab = "remove" if e in stuck else "add"
        adj[a].append((b, e, lab))
        adj[b].append((a, e, lab))

    seen_nodes = set()
    comps = []
    for start in list(adj):
        if start in seen_nodes:
            continue
        q = [start]
        nodes = set()
        edges = set()
        while q:
            a = q.pop()
            if a in nodes:
                continue
            nodes.add(a)
            seen_nodes.add(a)
            for b, e, lab in adj[a]:
                edges.add((e, lab))
                if b not in nodes:
                    q.append(b)

        deg1 = [n for n in nodes if len(adj[n]) == 1]
        ctype = "cycle" if not deg1 else "open_path"
        # Traverse in order for readable alternating sequence.
        s = deg1[0] if deg1 else next(iter(nodes))
        seq = []
        prev = None
        cur = s
        used = set()
        while True:
            options = [(b,e,lab) for b,e,lab in adj[cur] if e not in used]
            if not options:
                break
            b,e,lab = options[0]
            used.add(e)
            seq.append({"edge": list(e), "action": lab,
                        "from": list(cur), "to": list(b)})
            prev, cur = cur, b
            if cur == s and len(used) == len(edges):
                break

        adds = [e for e,lab in edges if lab == "add"]
        rems = [e for e,lab in edges if lab == "remove"]
        comps.append({
            "type": ctype,
            "nodes": [list(n) for n in sorted(nodes)],
            "n_add": len(adds),
            "n_remove": len(rems),
            "k": max(len(adds), len(rems)),
            "add_edges": [list(e) for e in sorted(adds)],
            "remove_edges": [list(e) for e in sorted(rems)],
            "sequence": seq,
            "_flip_edges": set(adds + rems),
        })
    comps.sort(key=lambda c: (c["type"] != "open_path", c["k"], c["add_edges"]))
    return comps


def summarize_state(rids, state, ranks, reward):
    met = graph_metrics(rids, state, ranks, reward)
    indeg, outdeg = selected_degrees(rids, state)
    met["sources"] = [r for r in rids if indeg[r] == 0]
    met["sinks"] = [r for r in rids if outdeg[r] == 0]
    return met


def build_energy_context(ep, cost, incoming, outgoing, ranks, op):
    bqm = fc.build_bqm_fixed_cardinality(
        ep, cost, incoming, outgoing, ranks,
        degree_conflict_penalty=288.0,
        order_penalty=op,
    )
    return fc.ArrayBQM.from_bqm(bqm, ep), {e:i for i,e in enumerate(ep)}


def energy_of(state, abqm, ep):
    ss = set(state)
    x = [1 if e in ss else 0 for e in ep]
    return abqm.energy(x)


def apply_component(state, comp):
    ns = set(state)
    for e in comp["_flip_edges"]:
        if e in ns:
            ns.remove(e)
        else:
            ns.add(e)
    return ns


def route_eval(order, comps, stuck, rids, ranks, reward, abqm, ep):
    state = set(stuck)
    e0 = energy_of(state, abqm, ep)
    rows = []
    max_bar = 0.0
    sum_positive = 0.0
    last_e = e0
    for step, ci in enumerate(order, 1):
        state = apply_component(state, comps[ci])
        en = energy_of(state, abqm, ep)
        de = en - last_e
        max_bar = max(max_bar, en - e0)
        sum_positive += max(0.0, de)
        met = summarize_state(rids, state, ranks, reward)
        rows.append({
            "step": step,
            "component": ci,
            "component_type": comps[ci]["type"],
            "k": comps[ci]["k"],
            "delta_energy": de,
            "energy_from_start": en - e0,
            **met,
        })
        last_e = en
    return {
        "order": list(order),
        "max_energy_barrier_from_start": max_bar,
        "sum_positive_delta": sum_positive,
        "final_energy_delta": last_e - e0,
        "steps": rows,
    }


def main():
    rids, ep, reward, cost, incoming, outgoing, stuck, ranks, op = replay_stuck_checkpoint()

    exact = normalized_exact_paths()
    candidates = []
    for p in exact:
        sc = selected_score(p, reward)
        if sc is None:
            continue
        edges = set(zip(p, p[1:]))
        if not edges.issubset(set(ep)):
            continue
        candidates.append((sc, p, edges))
    candidates.sort(reverse=True, key=lambda z: z[0])
    assert candidates and candidates[0][0] == CERT
    opt_score, opt_order, target = candidates[0]

    comps = componentize(set(stuck), target)
    clean_comps = []
    for c in comps:
        cc = {k:v for k,v in c.items() if not k.startswith("_")}
        clean_comps.append(cc)

    abqm, idx = build_energy_context(ep, cost, incoming, outgoing, ranks, op)
    base = summarize_state(rids, set(stuck), ranks, reward)
    target_metrics = summarize_state(rids, target, ranks, reward)
    start_e = energy_of(stuck, abqm, ep)
    target_e = energy_of(target, abqm, ep)

    n = len(comps)
    if n <= 9:
        routes = [
            route_eval(perm, comps, stuck, rids, ranks, reward, abqm, ep)
            for perm in itertools.permutations(range(n))
        ]
        routes.sort(key=lambda r: (
            r["max_energy_barrier_from_start"],
            r["sum_positive_delta"],
        ))
        best = routes[0]
        route_count = len(routes)
    else:
        # Fallback greedy: prefer open path first, then smallest immediate dE.
        remaining = set(range(n))
        state = set(stuck)
        order = []
        while remaining:
            if not order:
                opens = [i for i in remaining if comps[i]["type"] == "open_path"]
            else:
                opens = []
            pool = opens or list(remaining)
            en0 = energy_of(state, abqm, ep)
            scored = []
            for i in pool:
                ns = apply_component(state, comps[i])
                scored.append((energy_of(ns, abqm, ep) - en0, i))
            _, i = min(scored)
            state = apply_component(state, comps[i])
            order.append(i)
            remaining.remove(i)
        best = route_eval(order, comps, stuck, rids, ranks, reward, abqm, ep)
        route_count = None

    # Also report the direct optimum symmetric-difference flip size.
    result = {
        "certified_optimum_score": opt_score,
        "exact_paths_considered": len(candidates),
        "stuck_metrics": base,
        "optimum_metrics_under_stuck_rank": target_metrics,
        "stuck_fixed_bqm_energy": start_e,
        "optimum_fixed_bqm_energy_under_stuck_rank": target_e,
        "direct_symmetric_difference_edges": len(set(stuck) ^ target),
        "component_count": len(comps),
        "components": clean_comps,
        "component_order_routes_enumerated": route_count,
        "best_component_route": best,
        "interpretation": (
            "Oracle geometry only: target optimum is used to expose the exact "
            "alternating-component path. A production sampler must discover "
            "open/closed alternating moves without target information."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
