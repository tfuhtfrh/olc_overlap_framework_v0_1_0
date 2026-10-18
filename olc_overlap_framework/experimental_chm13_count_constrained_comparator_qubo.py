"""Endpoint-free latent-rank comparator with an exact N-1 edge-count square.

This variant replaces the linear cardinality reward of
experimental_chm13_endpoint_free_comparator_qubo.py with

    A_count * (sum_e x_e - (N-1))^2.

It keeps:
- at-most-one incoming/outgoing selected edge per vertex,
- latent vertex ranks,
- exact ripple-borrow comparators,
- selected-edge strict-rank gate.

The purpose is to compare a global exact-cardinality constraint against both:
1) the 3,501-variable linear-reward endpoint-free model, and
2) the 3,789-variable source/sink comparator model.
"""
from __future__ import annotations
import json
from dataclasses import dataclass

from demo_chm13_edge_ordered_path_qubo import DATASET_DIR
from experimental_chm13_endpoint_free_comparator_qubo import (
    EndpointFreeComparatorConfig,
    build_model as build_endpoint_free,
    encode_order,
    audit as base_audit,
)


@dataclass(frozen=True)
class CountConstrainedComparatorConfig:
    count_penalty: float = 72.0
    degree_conflict_penalty: float = 288.0
    comparator_penalty: float = 576.0
    order_gate_penalty: float = 288.0
    edge_cost_scale: float = 1.0


def add_count_square(model, target: int, penalty: float) -> None:
    """Add penalty * (sum_i x_i - target)^2 over real-edge variables."""
    edge_vars = [model.edge_index[e] for e in model.edge_pairs]
    model.add_constant(penalty * target * target)

    # x_i^2 = x_i
    linear_coeff = penalty * (1.0 - 2.0 * target)
    for i in edge_vars:
        model.add_linear(i, linear_coeff)

    pair_coeff = 2.0 * penalty
    for oi, i in enumerate(edge_vars):
        for j in edge_vars[oi + 1:]:
            model.add_quadratic(i, j, pair_coeff)


def build_model(cfg=CountConstrainedComparatorConfig()):
    base_cfg = EndpointFreeComparatorConfig(
        cardinality_reward=0.0,
        degree_conflict_penalty=cfg.degree_conflict_penalty,
        comparator_penalty=cfg.comparator_penalty,
        order_gate_penalty=cfg.order_gate_penalty,
        edge_cost_scale=cfg.edge_cost_scale,
    )
    model, reward, _ = build_endpoint_free(base_cfg)
    target = len(model.read_ids) - 1
    add_count_square(model, target, cfg.count_penalty)
    model.quadratic = {
        pair: value
        for pair, value in model.quadratic.items()
        if abs(value) > 1e-12
    }
    return model, reward, cfg


def audit(sample, model, reward, cfg):
    out = base_audit(sample, model, reward, cfg)
    target = len(model.read_ids) - 1
    count = out["selected_edges"]
    out["target_edges"] = target
    out["edge_count_residual"] = count - target
    out["edge_count_penalty_energy"] = (
        cfg.count_penalty * (count - target) ** 2
    )
    return out


if __name__ == "__main__":
    ref = json.loads(
        (DATASET_DIR / "reference_path.json").read_text(encoding="utf-8")
    )["normalized_nodes"]

    for count_penalty in (8.0, 32.0, 72.0, 144.0):
        cfg = CountConstrainedComparatorConfig(count_penalty=count_penalty)
        model, reward, cfg = build_model(cfg)
        sample = encode_order(model, ref)
        result = {
            "count_penalty": count_penalty,
            "variables": model.num_variables,
            "quadratic_terms": len(model.quadratic),
            "max_abs_linear": max(abs(x) for x in model.linear),
            "max_abs_quadratic": max(
                abs(x) for x in model.quadratic.values()
            ),
            "audit": audit(sample, model, reward, cfg),
        }
        print(json.dumps(result, indent=2))
