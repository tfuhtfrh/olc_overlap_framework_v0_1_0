"""Experimental bounded-coefficient vertex-order QUBO for CHM13 complex144.

Purpose
-------
This is an isolated Hamiltonian experiment.  It does not modify the production
layout solver.

Compared with the current edge-position formulation:

* real-edge variables x_e are selection / objective variables, but are no longer
  part of the numeric position value;
* each read has K binary position bits p[v,k];
* source/sink variables are used only as virtual path endpoints in degree
  constraints; they carry no objective and no order-position penalty;
* a selected real edge u->v enforces p_v = p_u + 1 through a local binary
  ripple-carry relation with coefficients in {1, 2, 3}, rather than powers of 2
  in one global squared integer residual.

The implication is quadratized with Rosenberg product auxiliaries:
    a[e,k] = x_e * p[u,k]
    b[e,k] = x_e * p[v,k]

and local carry bits c[e,k].  For each selected edge and bit k:
    a[e,k] + c[e,k] = b[e,k] + 2*c[e,k+1]
with c[e,0] = x_e and the final carry fixed to 0.

When x_e=0 all auxiliaries can be zero, so the endpoint position bits are not
constrained by that candidate edge.

The construction trades more variables for bounded coefficients and sparser
coupling.  It is intended to test Hamiltonian conditioning, not to claim a
better asymptotic encoding.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from demo_chm13_edge_ordered_path_qubo import (
    DATASET_DIR,
    load_chm13_graph,
)
from olc_pipeline.layout_solver import QUBOModel


@dataclass(frozen=True)
class BoundedVertexOrderConfig:
    degree_penalty: float = 144.0
    void_degree_penalty: float = 144.0
    successor_penalty: float = 144.0
    product_penalty: float = 144.0
    edge_cost_scale: float = 1.0


class BoundedVertexOrderModel(QUBOModel):
    def __init__(
        self,
        read_ids: list[str],
        edge_pairs: list[tuple[str, str]],
        position_bits: int,
    ) -> None:
        self.edge_pairs = list(edge_pairs)
        self.position_bits = position_bits
        self._read_index = {rid: i for i, rid in enumerate(read_ids)}

        labels: list[str] = []
        self.edge_index: dict[tuple[str, str], int] = {}
        for pair in edge_pairs:
            self.edge_index[pair] = len(labels)
            labels.append(f"x[{pair[0]},{pair[1]}]")

        self.source_index: dict[str, int] = {}
        for rid in read_ids:
            self.source_index[rid] = len(labels)
            labels.append(f"s[void,{rid}]")

        self.sink_index: dict[str, int] = {}
        for rid in read_ids:
            self.sink_index[rid] = len(labels)
            labels.append(f"t[{rid},void]")

        self.position_index: dict[tuple[str, int], int] = {}
        for rid in read_ids:
            for bit in range(position_bits):
                self.position_index[(rid, bit)] = len(labels)
                labels.append(f"p[{rid},{bit}]")

        self.left_copy_index: dict[tuple[tuple[str, str], int], int] = {}
        self.right_copy_index: dict[tuple[tuple[str, str], int], int] = {}
        for pair in edge_pairs:
            for bit in range(position_bits):
                self.left_copy_index[(pair, bit)] = len(labels)
                labels.append(f"a[{pair[0]},{pair[1]},{bit}]")
            for bit in range(position_bits):
                self.right_copy_index[(pair, bit)] = len(labels)
                labels.append(f"b[{pair[0]},{pair[1]},{bit}]")

        # carry_index[(edge, bit)] is the carry INTO bit k, for k=1..K-1.
        # carry into bit 0 is x_e itself; carry out of the top bit is fixed 0.
        self.carry_index: dict[tuple[tuple[str, str], int], int] = {}
        for pair in edge_pairs:
            for bit in range(1, position_bits):
                self.carry_index[(pair, bit)] = len(labels)
                labels.append(f"c[{pair[0]},{pair[1]},{bit}]")

        super().__init__(read_ids=read_ids, linear=[0.0] * len(labels))
        self._labels = labels

    def variable_label(self, index: int) -> str:
        return self._labels[index]


def add_square(
    model: QUBOModel,
    coefficients: dict[int, float],
    *,
    constant: float,
    penalty: float,
) -> None:
    items = [(i, float(a)) for i, a in coefficients.items() if a != 0.0]
    model.add_constant(penalty * constant * constant)
    for i, a in items:
        model.add_linear(i, penalty * (a * a + 2.0 * constant * a))
    for offset, (i, a) in enumerate(items):
        for j, b in items[offset + 1 :]:
            model.add_quadratic(i, j, 2.0 * penalty * a * b)


def add_product_equality(
    model: QUBOModel,
    *,
    product: int,
    left: int,
    right: int,
    penalty: float,
) -> None:
    """Rosenberg penalty for product = left * right."""
    # left*right - 2*left*product - 2*right*product + 3*product
    model.add_quadratic(left, right, penalty)
    model.add_quadratic(left, product, -2.0 * penalty)
    model.add_quadratic(right, product, -2.0 * penalty)
    model.add_linear(product, 3.0 * penalty)


def build_model(config: BoundedVertexOrderConfig):
    reads, edges, reward_by_pair = load_chm13_graph(DATASET_DIR)
    read_ids = [read.rid for read in reads]
    rank = {rid: i for i, rid in enumerate(read_ids)}

    # One directed edge per pair in this dataset.
    edge_pairs = sorted(
        reward_by_pair,
        key=lambda pair: (rank[pair[0]], rank[pair[1]]),
    )
    n = len(read_ids)
    k = max(1, math.ceil(math.log2(max(2, n))))
    model = BoundedVertexOrderModel(read_ids, edge_pairs, k)

    reward_shift = max(reward_by_pair.values(), default=0.0)
    raw_cost = {
        pair: reward_shift - reward
        for pair, reward in reward_by_pair.items()
    }
    normalizer = max(max(raw_cost.values(), default=1.0), 1.0)

    incoming = {rid: [] for rid in read_ids}
    outgoing = {rid: [] for rid in read_ids}
    for pair in edge_pairs:
        u, v = pair
        x = model.edge_index[pair]
        incoming[v].append(x)
        outgoing[u].append(x)
        model.add_linear(
            x,
            config.edge_cost_scale * raw_cost[pair] / normalizer,
        )

    # Read degrees.  Source/sink variables behave as virtual edges to/from one
    # void node.  They have no objective and no position-order term.
    for rid in read_ids:
        add_square(
            model,
            {
                model.source_index[rid]: -1.0,
                **{x: -1.0 for x in incoming[rid]},
            },
            constant=1.0,
            penalty=config.degree_penalty,
        )
        add_square(
            model,
            {
                model.sink_index[rid]: -1.0,
                **{x: -1.0 for x in outgoing[rid]},
            },
            constant=1.0,
            penalty=config.degree_penalty,
        )

    add_square(
        model,
        {model.source_index[rid]: -1.0 for rid in read_ids},
        constant=1.0,
        penalty=config.void_degree_penalty,
    )
    add_square(
        model,
        {model.sink_index[rid]: -1.0 for rid in read_ids},
        constant=1.0,
        penalty=config.void_degree_penalty,
    )

    # Local selected-edge successor constraints.
    for pair in edge_pairs:
        u, v = pair
        x = model.edge_index[pair]
        for bit in range(k):
            pu = model.position_index[(u, bit)]
            pv = model.position_index[(v, bit)]
            a = model.left_copy_index[(pair, bit)]
            b = model.right_copy_index[(pair, bit)]

            add_product_equality(
                model,
                product=a,
                left=x,
                right=pu,
                penalty=config.product_penalty,
            )
            add_product_equality(
                model,
                product=b,
                left=x,
                right=pv,
                penalty=config.product_penalty,
            )

            # a + carry_in = b + 2*carry_out
            coeffs = {a: 1.0, b: -1.0}
            if bit == 0:
                coeffs[x] = 1.0
            else:
                coeffs[model.carry_index[(pair, bit)]] = 1.0

            if bit < k - 1:
                coeffs[model.carry_index[(pair, bit + 1)]] = -2.0

            add_square(
                model,
                coeffs,
                constant=0.0,
                penalty=config.successor_penalty,
            )

    model.quadratic = {
        pair: value
        for pair, value in model.quadratic.items()
        if abs(value) > 1e-12
    }
    return model, reward_by_pair, reward_shift, normalizer


def encode_order(
    model: BoundedVertexOrderModel,
    order: list[str],
) -> list[int]:
    if len(order) != len(model.read_ids) or set(order) != set(model.read_ids):
        raise ValueError("order must contain every read exactly once")

    sample = [0] * model.num_variables
    position = {rid: i for i, rid in enumerate(order)}

    sample[model.source_index[order[0]]] = 1
    sample[model.sink_index[order[-1]]] = 1

    for rid, pos in position.items():
        for bit in range(model.position_bits):
            if (pos >> bit) & 1:
                sample[model.position_index[(rid, bit)]] = 1

    for u, v in zip(order, order[1:]):
        pair = (u, v)
        if pair not in model.edge_index:
            raise ValueError(f"missing selected edge: {u} -> {v}")
        sample[model.edge_index[pair]] = 1

        pu = position[u]
        pv = position[v]
        if pv != pu + 1:
            raise ValueError("encoded order positions must be consecutive")

        for bit in range(model.position_bits):
            ubit = (pu >> bit) & 1
            vbit = (pv >> bit) & 1
            if ubit:
                sample[model.left_copy_index[(pair, bit)]] = 1
            if vbit:
                sample[model.right_copy_index[(pair, bit)]] = 1

        carry = 1
        for bit in range(1, model.position_bits):
            carry &= (pu >> (bit - 1)) & 1
            if carry:
                sample[model.carry_index[(pair, bit)]] = 1

    return sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree-penalty", type=float, default=144.0)
    parser.add_argument("--void-degree-penalty", type=float, default=144.0)
    parser.add_argument("--successor-penalty", type=float, default=144.0)
    parser.add_argument("--product-penalty", type=float, default=144.0)
    parser.add_argument("--edge-cost-scale", type=float, default=1.0)
    args = parser.parse_args()

    config = BoundedVertexOrderConfig(
        degree_penalty=args.degree_penalty,
        void_degree_penalty=args.void_degree_penalty,
        successor_penalty=args.successor_penalty,
        product_penalty=args.product_penalty,
        edge_cost_scale=args.edge_cost_scale,
    )
    model, reward_by_pair, reward_shift, normalizer = build_model(config)

    reference = json.loads(
        (DATASET_DIR / "reference_path.json").read_text(encoding="utf-8")
    )
    reference_order = list(reference["normalized_nodes"])
    reference_sample = encode_order(model, reference_order)

    certificate = json.loads(
        (DATASET_DIR / "weighted_certificate.json").read_text(encoding="utf-8")
    )
    runner = next(
        entry
        for entry in certificate["top_paths"]
        if int(entry["score"]) == int(certificate["runner_up_score"])
    )
    runner_order = [node[:-1] for node in runner["nodes"]]
    runner_sample = encode_order(model, runner_order)

    report = {
        "config": config.__dict__,
        "reads": len(model.read_ids),
        "edges": len(model.edge_pairs),
        "position_bits_per_read": model.position_bits,
        "logical_variables": model.num_variables,
        "linear_terms": len(model.linear),
        "quadratic_terms": len(model.quadratic),
        "maximum_absolute_linear": max(abs(v) for v in model.linear),
        "maximum_absolute_quadratic": max(
            abs(v) for v in model.quadratic.values()
        ),
        "reward_shift": reward_shift,
        "cost_normalizer": normalizer,
        "reference_energy": model.energy(reference_sample),
        "reference_one_bits": sum(reference_sample),
        "runner_up_energy": model.energy(runner_sample),
        "runner_up_hamming_from_reference": sum(
            a != b for a, b in zip(reference_sample, runner_sample)
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
