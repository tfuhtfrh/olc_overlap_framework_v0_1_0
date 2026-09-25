"""Experimental latent vertex-rank comparator QUBO for CHM13.

Design:
- x_e is the only edge-selection / edge-weight variable.
- Each vertex has K latent rank bits p[v,k].
- Candidate edges do NOT force ranks to zero when unselected.
- For every candidate edge e=(u,v), K borrow bits compute the unsigned
  comparison P_v - P_u - 1 using a ripple-borrow comparator.
- The borrow recurrence is enforced unconditionally; for any P_u,P_v there is
  a unique zero-penalty borrow chain, so unselected edges do not constrain ranks.
- Only a selected edge is required to have final borrow 0:
      x_e * b[e,K] = 0
  which means P_v >= P_u + 1, i.e. P_v > P_u.

This is enough to exclude directed subtours once degree/source/sink constraints
hold.  Unlike the +1 formulation, ranks need only be strictly increasing and
may contain arbitrary gaps.

The borrow relation for subtracting u + borrow_in from v is
    borrow_out = majority(u, borrow_in, 1-v)
and has the exact quadratic zero-penalty representation
    R = u + b + c + u*b - u*v - b*v - 2*u*c - 2*b*c + 2*v*c
where b=borrow_in and c=borrow_out.
"""
from __future__ import annotations
import json, math
from dataclasses import dataclass

from demo_chm13_edge_ordered_path_qubo import DATASET_DIR, load_chm13_graph
from olc_pipeline.layout_solver import QUBOModel


@dataclass(frozen=True)
class LatentRankComparatorConfig:
    degree_penalty: float = 144.0
    void_penalty: float = 144.0
    comparator_penalty: float = 288.0
    order_gate_penalty: float = 144.0
    edge_cost_scale: float = 1.0


class LatentRankComparatorModel(QUBOModel):
    void_id = "__void__"

    def __init__(self, read_ids, edge_pairs, k):
        self.edge_pairs = list(edge_pairs)
        self.k = k
        labels = []
        self.edge_index = {}
        self.source_index = {}
        self.sink_index = {}
        self.rank_index = {}
        self.borrow_index = {}

        for e in edge_pairs:
            self.edge_index[e] = len(labels)
            labels.append(f"x[{e[0]},{e[1]}]")

        for r in read_ids:
            self.source_index[r] = len(labels)
            labels.append(f"s[{r}]")
        for r in read_ids:
            self.sink_index[r] = len(labels)
            labels.append(f"t[{r}]")

        for r in read_ids:
            for bit in range(k):
                self.rank_index[(r, bit)] = len(labels)
                labels.append(f"p[{r},{bit}]")

        # borrow_index[(e,j)] for j=1..K.  b_0 is the fixed input 1
        # corresponding to subtracting P_u + 1 from P_v.
        for e in edge_pairs:
            for j in range(1, k + 1):
                self.borrow_index[(e, j)] = len(labels)
                labels.append(f"b[{e[0]},{e[1]},{j}]")

        super().__init__(read_ids=list(read_ids), linear=[0.0] * len(labels))
        self._labels = labels

    def variable_label(self, index):
        return self._labels[index]


def add_square(model, coeffs, constant, penalty):
    items = [(i, float(a)) for i, a in coeffs.items() if a]
    model.add_constant(penalty * constant * constant)
    for i, a in items:
        model.add_linear(i, penalty * (a * a + 2.0 * constant * a))
    for oi, (i, a) in enumerate(items):
        for j, b in items[oi + 1:]:
            model.add_quadratic(i, j, 2.0 * penalty * a * b)


def add_general_borrow_penalty(model, u, v, b, c, penalty):
    # R = u + b + c + u*b - u*v - b*v - 2*u*c - 2*b*c + 2*v*c
    model.add_linear(u, penalty)
    model.add_linear(b, penalty)
    model.add_linear(c, penalty)
    model.add_quadratic(u, b, penalty)
    model.add_quadratic(u, v, -penalty)
    model.add_quadratic(b, v, -penalty)
    model.add_quadratic(u, c, -2.0 * penalty)
    model.add_quadratic(b, c, -2.0 * penalty)
    model.add_quadratic(v, c, 2.0 * penalty)


def add_first_borrow_penalty(model, u, v, c, penalty):
    # General relation with fixed borrow-in b_0 = 1:
    # R = 1 + 2u - v - c - u*v - 2u*c + 2v*c.
    model.add_constant(penalty)
    model.add_linear(u, 2.0 * penalty)
    model.add_linear(v, -penalty)
    model.add_linear(c, -penalty)
    model.add_quadratic(u, v, -penalty)
    model.add_quadratic(u, c, -2.0 * penalty)
    model.add_quadratic(v, c, 2.0 * penalty)


def build_model(cfg=LatentRankComparatorConfig()):
    reads, edges, reward = load_chm13_graph(DATASET_DIR)
    rids = [r.rid for r in reads]
    rank = {r: i for i, r in enumerate(rids)}
    ep = sorted(reward, key=lambda e: (rank[e[0]], rank[e[1]]))

    n = len(rids)
    k = max(1, math.ceil(math.log2(max(2, n))))
    m = LatentRankComparatorModel(rids, ep, k)

    shift = max(reward.values(), default=0.0)
    raw_cost = {e: shift - reward[e] for e in ep}
    normalizer = max(max(raw_cost.values(), default=1.0), 1.0)

    incoming = {r: [] for r in rids}
    outgoing = {r: [] for r in rids}
    for e in ep:
        u, v = e
        incoming[v].append(e)
        outgoing[u].append(e)
        m.add_linear(
            m.edge_index[e],
            cfg.edge_cost_scale * raw_cost[e] / normalizer,
        )

    # Path-cover / virtual-node degree constraints.
    for r in rids:
        add_square(
            m,
            {
                m.source_index[r]: -1.0,
                **{m.edge_index[e]: -1.0 for e in incoming[r]},
            },
            1.0,
            cfg.degree_penalty,
        )
        add_square(
            m,
            {
                m.sink_index[r]: -1.0,
                **{m.edge_index[e]: -1.0 for e in outgoing[r]},
            },
            1.0,
            cfg.degree_penalty,
        )

    add_square(
        m,
        {m.source_index[r]: -1.0 for r in rids},
        1.0,
        cfg.void_penalty,
    )
    add_square(
        m,
        {m.sink_index[r]: -1.0 for r in rids},
        1.0,
        cfg.void_penalty,
    )

    # Comparator circuit for every candidate edge.  This never constrains
    # P_u/P_v by itself; it only computes the correct final borrow.
    for e in ep:
        u, v = e
        for bit in range(k):
            ub = m.rank_index[(u, bit)]
            vb = m.rank_index[(v, bit)]
            bout = m.borrow_index[(e, bit + 1)]
            if bit == 0:
                add_first_borrow_penalty(
                    m, ub, vb, bout, cfg.comparator_penalty
                )
            else:
                bin_ = m.borrow_index[(e, bit)]
                add_general_borrow_penalty(
                    m, ub, vb, bin_, bout, cfg.comparator_penalty
                )

        # Final borrow=1 iff P_v < P_u+1, i.e. P_v <= P_u.
        # Only selected edges are forbidden from having that relation.
        m.add_quadratic(
            m.edge_index[e],
            m.borrow_index[(e, k)],
            cfg.order_gate_penalty,
        )

    m.quadratic = {
        pair: value
        for pair, value in m.quadratic.items()
        if abs(value) > 1e-12
    }
    return m, reward, cfg


def encode_order(model: LatentRankComparatorModel, order, ranks=None):
    if ranks is None:
        ranks = {r: i for i, r in enumerate(order)}
    if set(ranks) != set(order):
        raise ValueError("ranks must cover exactly the order vertices")
    maxv = (1 << model.k) - 1
    if any(v < 0 or v > maxv for v in ranks.values()):
        raise ValueError("rank does not fit")

    s = [0] * model.num_variables
    s[model.source_index[order[0]]] = 1
    s[model.sink_index[order[-1]]] = 1

    for r, value in ranks.items():
        for bit in range(model.k):
            if (value >> bit) & 1:
                s[model.rank_index[(r, bit)]] = 1

    for e in zip(order, order[1:]):
        s[model.edge_index[e]] = 1

    # Set borrow chains for ALL candidate edges because comparator relations
    # are unconditional definitions of the latent comparison state.
    for e in model.edge_pairs:
        u, v = e
        U = ranks[u]
        V = ranks[v]
        borrow = 1
        for bit in range(model.k):
            ub = (U >> bit) & 1
            vb = (V >> bit) & 1
            borrow = int(ub + borrow > vb)
            if borrow:
                s[model.borrow_index[(e, bit + 1)]] = 1

    return s


def audit(s, model, reward, cfg):
    selected = [e for e in model.edge_pairs if s[model.edge_index[e]]]
    sources = [r for r in model.read_ids if s[model.source_index[r]]]
    sinks = [r for r in model.read_ids if s[model.sink_index[r]]]

    indeg = {r: 0 for r in model.read_ids}
    outdeg = {r: 0 for r in model.read_ids}
    nxt = {}
    for u, v in selected:
        indeg[v] += 1
        outdeg[u] += 1
        if u not in nxt:
            nxt[u] = v

    dsq = sum(
        (1 - int(r in sources) - indeg[r]) ** 2
        + (1 - int(r in sinks) - outdeg[r]) ** 2
        for r in model.read_ids
    )
    vsq = (1 - len(sources)) ** 2 + (1 - len(sinks)) ** 2

    ranks = {}
    for r in model.read_ids:
        ranks[r] = sum(
            (1 << bit) * int(s[model.rank_index[(r, bit)]])
            for bit in range(model.k)
        )

    comparator_violations = 0
    selected_order_violations = 0
    final_borrow_mismatch = 0
    for e in model.edge_pairs:
        u, v = e
        borrow = 1
        for bit in range(model.k):
            ub = (ranks[u] >> bit) & 1
            vb = (ranks[v] >> bit) & 1
            expected = int(ub + borrow > vb)
            actual = int(s[model.borrow_index[(e, bit + 1)]])
            comparator_violations += int(actual != expected)
            borrow = expected
        actual_final = int(s[model.borrow_index[(e, model.k)]])
        expected_final = int(ranks[v] <= ranks[u])
        final_borrow_mismatch += int(actual_final != expected_final)
        if s[model.edge_index[e]] and ranks[v] <= ranks[u]:
            selected_order_violations += 1

    structural = False
    order = []
    if dsq == 0 and vsq == 0 and len(sources) == 1 and len(sinks) == 1:
        seen = set()
        cur = sources[0]
        while cur not in seen:
            seen.add(cur)
            order.append(cur)
            if cur == sinks[0]:
                break
            if cur not in nxt:
                break
            cur = nxt[cur]
        structural = (
            len(order) == len(model.read_ids)
            and order[-1] == sinks[0]
            and len(seen) == len(model.read_ids)
        )

    valid = (
        structural
        and comparator_violations == 0
        and selected_order_violations == 0
    )
    score = (
        int(sum(reward[e] for e in zip(order, order[1:])))
        if valid
        else None
    )

    return {
        "energy": float(model.energy(s)),
        "valid_path": valid,
        "path_score": score,
        "selected_edges": len(selected),
        "sources": len(sources),
        "sinks": len(sinks),
        "degree_residual_sq": int(dsq),
        "void_residual_sq": int(vsq),
        "comparator_violations": int(comparator_violations),
        "final_borrow_mismatch": int(final_borrow_mismatch),
        "selected_order_violations": int(selected_order_violations),
        "distinct_ranks": len(set(ranks.values())),
        "hamming_weight": int(sum(s)),
    }


if __name__ == "__main__":
    m, reward, cfg = build_model()
    ref = json.loads(
        (DATASET_DIR / "reference_path.json").read_text(encoding="utf-8")
    )["normalized_nodes"]

    rank_sets = {
        "consecutive_0": {r: i for i, r in enumerate(ref)},
        "consecutive_10": {r: 10 + i for i, r in enumerate(ref)},
        # Nonuniform gaps while preserving strict order and fitting in 8 bits.
        "gapped": {
            r: int(round(i * 255 / (len(ref) - 1)))
            for i, r in enumerate(ref)
        },
    }
    for name, ranks in rank_sets.items():
        if len(set(ranks.values())) != len(ranks):
            continue
        s = encode_order(m, ref, ranks)
        print(json.dumps({
            "rank_scheme": name,
            "variables": m.num_variables,
            "quadratic_terms": len(m.quadratic),
            "max_abs_linear": max(abs(x) for x in m.linear),
            "max_abs_quadratic": max(abs(x) for x in m.quadratic.values()),
            "audit": audit(s, m, reward, cfg),
        }, indent=2))
