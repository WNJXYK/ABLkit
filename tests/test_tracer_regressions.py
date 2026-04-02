"""Focused regression tests for tracer guardrails."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ablkit.reasoning.ifw_dp import make_chain
from ablkit.reasoning.tracer import (
    _extract_static_graph,
    _verify_decomposition,
    discover_decomposition,
)


def test_verify_decomposition_requires_exact_recovery():
    """Verification must reject decompositions that only recover y."""

    def logic_forward(z):
        return int(bool(z[0] or z[1]))

    bad = make_chain(
        L=2,
        var_groups=[[0], [1]],
        transition_fn=lambda h_prev, z_vals, step, y: (
            ()
            if step == 0
            else ("__target__", y) if z_vals[0] == y else None
        ),
        h_init=(),
        h_final_fn=lambda y: ("__target__", y),
        n=2,
        H=0,
    )

    assert not _verify_decomposition(bad, logic_forward, n=2, K=2, num_checks=50, seed=0)


def test_discover_decomposition_returns_bruteforce_when_selected():
    """If cost estimation selects brute-force, auto-discovery should stop there."""

    def kb_sum2(z):
        return z[0] + z[1]

    decomp, info = discover_decomposition(
        kb_sum2, n=2, K=2, num_samples=100, seed=0, min_compression=1.0
    )

    assert info["decomposition_type"] == "brute-force"
    assert info["estimated_brute-force_cost"] <= info["estimated_chain_cost"]


def test_lazy_precompute_reduces_warm_start_calls():
    """Lazy mode should defer most internal transition filling to runtime."""

    def kb_add_bits(z):
        s0 = z[0] + z[2]
        carry = s0 // 2
        out0 = s0 % 2
        s1 = z[1] + z[3] + carry
        out1 = s1 % 2
        out2 = s1 // 2
        return out0 + 2 * out1 + 4 * out2

    _, eager_info = discover_decomposition(
        kb_add_bits, n=4, K=2, num_samples=200, seed=0, max_precompute=1000
    )
    _, lazy_info = discover_decomposition(
        kb_add_bits,
        n=4,
        K=2,
        num_samples=200,
        seed=0,
        max_precompute=1000,
        lazy_precompute=True,
    )

    assert eager_info["decomposition_type"] == "chain"
    assert lazy_info["decomposition_type"] == "chain"
    assert lazy_info["precompute_mode"] == "lazy"
    assert lazy_info["precompute_kb_calls"] < eager_info["precompute_kb_calls"]


def test_minfill_reports_conservative_bag_baseline():
    """Min-fill should report both CSS and bag-style baseline costs."""

    def kb_add_bits(z):
        s0 = z[0] + z[2]
        carry = s0 // 2
        out0 = s0 % 2
        s1 = z[1] + z[3] + carry
        out1 = s1 % 2
        out2 = s1 // 2
        return out0 + 2 * out1 + 4 * out2

    _, info = discover_decomposition(
        kb_add_bits, n=4, K=2, num_samples=200, seed=0, max_precompute=1000
    )

    assert "estimated_minfill_css_cost" in info
    assert "estimated_minfill_bag_cost" in info
    assert info["estimated_minfill_cost"] >= info["estimated_minfill_css_cost"]
    assert info["estimated_minfill_cost"] >= info["estimated_minfill_bag_cost"]
    assert any(item["name"] == "minfill" for item in info["candidate_topologies"])
    minfill_item = next(item for item in info["candidate_topologies"] if item["name"] == "minfill")
    assert minfill_item["treewidth"] == 1
    assert minfill_item["bag_cost"] == info["estimated_minfill_bag_cost"]


def test_static_graph_extraction_tracks_fixed_positions():
    """Static graph extraction should preserve position/source-set schema."""

    def kb_add_bits(z):
        s0 = z[0] + z[2]
        carry = s0 // 2
        out0 = s0 % 2
        s1 = z[1] + z[3] + carry
        out1 = s1 % 2
        out2 = s1 // 2
        return out0 + 2 * out1 + 4 * out2

    graph = _extract_static_graph(kb_add_bits, n=4, K=2, num_samples=50, seed=0)

    assert graph["is_static"]
    assert graph["schema_mismatches"] == 0
    assert graph["num_nodes"] > 4
    pos_to_sources = {node["pos"]: node["sources"] for node in graph["nodes"]}
    assert frozenset({0, 2}) in pos_to_sources.values()
    assert frozenset({0, 1, 2, 3}) in pos_to_sources.values()


def test_y_pruning_reduces_estimated_graph_cost():
    """y-pruning should reduce the effective graph cost estimate."""

    def kb_add_bits(z):
        s0 = z[0] + z[2]
        carry = s0 // 2
        out0 = s0 % 2
        s1 = z[1] + z[3] + carry
        out1 = s1 % 2
        out2 = s1 // 2
        return out0 + 2 * out1 + 4 * out2

    def y_decompose_fn(y):
        return [y % 2, (y // 2) % 2, y // 4]

    def constraint_fn(z, y_parts):
        s0 = z[0] + z[2]
        carry = s0 // 2
        digit0 = s0 % 2
        s1 = z[1] + z[3] + carry
        digit1 = s1 % 2
        carry1 = s1 // 2
        return digit0 == y_parts[0] and digit1 == y_parts[1] and carry1 == y_parts[2]

    _, info = discover_decomposition(
        kb_add_bits,
        n=4,
        K=2,
        num_samples=200,
        seed=0,
        constraint_fn=constraint_fn,
        y_size=3,
        Y_domains=[2, 2, 2],
        y_decompose_fn=y_decompose_fn,
    )

    assert info["estimated_chain_y_pruned_cost"] < info["estimated_chain_css_cost"]
    assert info["y_node_assignment"]
