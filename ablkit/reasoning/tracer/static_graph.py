"""
静态图提取与瓶颈发现 -- 通过多次追踪提取 CSS 候选。
"""

import random
from typing import Callable, List, Optional

from .traced_value import TracedValue


def _trace_graph_once(logic_forward: Callable, z_values: List[int]) -> Optional[list]:
    """Run one traced execution and return the resulting static graph nodes."""
    graph = []
    inputs = [TracedValue(z_values[i], frozenset({i}), graph) for i in range(len(z_values))]
    try:
        logic_forward(inputs)
    except Exception:
        return None
    return graph


def _extract_static_graph(
    logic_forward: Callable,
    n: int,
    K: int,
    num_samples: int = 500,
    seed: int = 0,
) -> dict:
    """
    Extract a static execution graph by tracing multiple random executions.

    Under the fixed-graph assumption, every successful trace should yield the
    same node count and the same source-set schema at each position. We keep
    the first successful schema and aggregate observed node values over later
    traces that match it exactly.
    """
    rng = random.Random(seed)
    schema = None
    node_by_pos = {}
    samples_used = 0
    schema_mismatches = 0

    for _ in range(num_samples):
        z = [rng.randint(0, K - 1) for _ in range(n)]
        graph = _trace_graph_once(logic_forward, z)
        if graph is None:
            continue

        sample_schema = tuple(node.sources for node in graph)
        if schema is None:
            schema = sample_schema
            node_by_pos = {
                pos: {"pos": pos, "sources": src, "values": set()}
                for pos, src in enumerate(schema)
            }
        elif sample_schema != schema:
            schema_mismatches += 1
            continue

        samples_used += 1
        for pos, node in enumerate(graph):
            node_by_pos[pos]["values"].add(node.value)

    nodes = []
    for pos in sorted(node_by_pos):
        entry = node_by_pos[pos]
        nodes.append({
            "pos": pos,
            "sources": entry["sources"],
            "domain_size": len(entry["values"]),
            "values": entry["values"],
        })

    return {
        "is_static": schema is not None and schema_mismatches == 0,
        "num_traces": samples_used,
        "schema_mismatches": schema_mismatches,
        "num_nodes": len(schema) if schema is not None else 0,
        "node_by_pos": node_by_pos,
        "nodes": nodes,
    }


def find_bottlenecks(
    logic_forward: Callable,
    n: int,
    K: int,
    num_samples: int = 500,
    seed: int = 0,
    constraint_fn: Optional[Callable] = None,
    y_size: int = 0,
    Y_domains: Optional[List[int]] = None,
    y_decompose_fn: Optional[Callable] = None,
) -> tuple:
    """
    Discover information bottlenecks in KB's logic_forward.

    Runs logic_forward with TracedValue inputs many times, records
    each intermediate node's (source set, observed values).
    Nodes with high compression ratio (K^coverage / domain_size)
    are CSS candidates.

    Args:
        logic_forward: Takes list of values, returns value.
        n: Number of input variables.
        K: Domain size per variable.
        num_samples: Number of random executions.

    Returns:
        bottlenecks:
            List of dicts sorted by compression ratio (highest first).
        static_graph:
            Static graph summary with node-by-position source sets and domains.
        y_check_info:
            Optional mapping from y-part index to required z-sources.
    """
    static_graph = _extract_static_graph(logic_forward, n, K, num_samples=num_samples, seed=seed)
    node_by_pos = static_graph["node_by_pos"]

    # CSS discovery: use smallest-domain node per source set for cost estimation.
    # Correctness is ensured by multi-value CSS in transition building + verification.
    best_domain = {}  # sources -> smallest observed value set
    best_pos = {}
    for pos, entry in node_by_pos.items():
        src = entry["sources"]
        if src is None or not (0 < len(src) < n):
            continue
        vals = entry["values"]
        if src not in best_domain or len(vals) < len(best_domain[src]):
            best_domain[src] = vals
            best_pos[src] = pos

    bottlenecks = []
    for sources, vals in best_domain.items():
        css_domain = len(vals)
        coverage = len(sources)
        compression = K ** coverage / max(css_domain, 1)
        bottlenecks.append({
            "sources": sources,
            "pos": best_pos[sources],
            "domain_size": css_domain,
            "coverage": coverage,
            "compression": compression,
            "min_domain": css_domain,
        })

    bottlenecks.sort(key=lambda x: x["compression"], reverse=True)

    y_check_info = {}
    if constraint_fn is not None and y_size > 0:
        y_check_info = _discover_y_checks(
            constraint_fn, n, K, y_size, Y_domains, num_samples, seed,
            logic_forward=logic_forward, y_decompose_fn=y_decompose_fn,
        )

    static_graph["css_candidates"] = [
        {
            "pos": b["pos"],
            "sources": b["sources"],
            "domain_size": b["domain_size"],
            "compression": b["compression"],
        }
        for b in bottlenecks
    ]
    return bottlenecks, static_graph, y_check_info


def _discover_y_checks(
    constraint_fn: Callable,
    n: int,
    K: int,
    y_size: int,
    Y_domains: Optional[List[int]],
    num_samples: int,
    seed: int,
    logic_forward: Optional[Callable] = None,
    y_decompose_fn: Optional[Callable] = None,
) -> dict:
    """
    Trace constraint_fn(z, y_parts) to discover y-check structure.

    For each y-part index j (0..y_size-1), finds the minimal z-source set
    required before y_parts[j] can be checked.

    When logic_forward and y_decompose_fn are provided, uses matched (z, y)
    pairs so that constraint_fn's early-return checks always pass, ensuring
    all y-parts are reached and discovered.

    Returns:
        {y_part_index: frozenset of z-variable indices}
    """
    rng = random.Random(seed)
    if Y_domains is None:
        Y_domains = [K] * y_size

    use_matched = logic_forward is not None and y_decompose_fn is not None

    y_z_sources = {}  # y_part_index -> frozenset of z indices

    for _ in range(num_samples):
        z_vals = [rng.randint(0, K - 1) for _ in range(n)]

        if use_matched:
            try:
                y_full = logic_forward(z_vals)
                if isinstance(y_full, TracedValue):
                    y_full = y_full.value
                y_vals = list(y_decompose_fn(y_full))
            except Exception:
                y_vals = [rng.randint(0, Y_domains[j] - 1) for j in range(y_size)]
        else:
            y_vals = [rng.randint(0, Y_domains[j] - 1) for j in range(y_size)]

        graph = []
        z_traced = [TracedValue(z_vals[i], frozenset({i}), graph) for i in range(n)]
        y_traced = [TracedValue(y_vals[j], frozenset({n + j}), graph) for j in range(y_size)]

        try:
            constraint_fn(z_traced, y_traced)
        except Exception:
            continue

        # For this sample, find first computation node where each y-index
        # appears alongside z-indices (skip pure y-input nodes)
        sample_y_z_src = {}
        for node in graph:
            y_in_node = {s - n for s in node.sources if s >= n}
            z_in_node = frozenset(s for s in node.sources if s < n)
            if not z_in_node:
                continue  # skip pure-y nodes (y inputs themselves)
            for yi in y_in_node:
                if yi not in sample_y_z_src:
                    sample_y_z_src[yi] = z_in_node

        # Union of z-sources across samples (handles branching code paths)
        for yi, z_src in sample_y_z_src.items():
            if yi not in y_z_sources:
                y_z_sources[yi] = z_src
            else:
                y_z_sources[yi] = y_z_sources[yi] | z_src

    if y_z_sources:
        print(f"  [Tracer] y-check discovery (y_size={y_size}):")
        for yi in sorted(y_z_sources):
            print(f"    y[{yi}] depends on z-sources {sorted(y_z_sources[yi])}")

    return y_z_sources
