"""
Automatic static-graph extraction and CSS-guided IFW decomposition.

This module assumes the KB follows a fixed, branch-free execution graph when
run on `TracedValue` inputs. Under that assumption it:

1. traces repeated executions to recover a static node-position/source-set graph;
2. scores CSS candidates by observed domain compression on that graph;
3. proposes chain/tree/minfill topologies from the recovered source-set structure;
4. optionally incorporates y-pruning into graph-level cost estimation; and
5. builds a verified IFW decomposition, falling back when a heuristic choice
   fails to recover the exact original z.

Supported operations on TracedValue:
  Arithmetic: +, -, *, //, %, **, abs, neg
  Comparison: ==, !=, <, >, <=, >= (return TracedValue, not bool)
  Boolean:    |, &, ~  (Python short-circuit boolean operators are not traced)
  Conversion: int(), float(), bool()
"""

import random
from collections import defaultdict
from typing import Callable, List, Optional, Tuple, Any, FrozenSet

from .ifw_dp import Decomposition, make_chain, make_tree


# ============================================================
# TracedValue
# ============================================================


def _safe_div(a, b):
    return a / b if b else 0


def _safe_floordiv(a, b):
    return a // b if b else 0


class TracedValue:
    """A value that tracks which input variables it depends on.

    Also records operations for compiled-evaluator support: each node stores
    its graph position (_pos) and the operation that created it (_op_record).
    """

    __slots__ = ("value", "sources", "_graph", "_pos", "_op_record")

    def __init__(self, value, sources: FrozenSet[int], graph: Optional[list] = None):
        self.value = value
        self.sources = sources
        self._graph = graph
        self._op_record = None
        if graph is not None:
            self._pos = len(graph)
            graph.append(self)
        else:
            self._pos = -1

    def _binop(self, other, fn):
        """Apply binary operation, merging source sets."""
        ov = other.value if isinstance(other, TracedValue) else other
        os = other.sources if isinstance(other, TracedValue) else frozenset()
        result = TracedValue(fn(self.value, ov), self.sources | os, self._graph)
        # Record: (fn, left_pos, right_pos_or_None, right_const_or_None)
        other_pos = other._pos if isinstance(other, TracedValue) else None
        result._op_record = (fn, self._pos, other_pos, ov if other_pos is None else None)
        return result

    def _rbinop(self, other, fn):
        """Apply reverse binary operation (other OP self)."""
        ov = other.value if isinstance(other, TracedValue) else other
        os = other.sources if isinstance(other, TracedValue) else frozenset()
        result = TracedValue(fn(ov, self.value), self.sources | os, self._graph)
        other_pos = other._pos if isinstance(other, TracedValue) else None
        result._op_record = (fn, other_pos, self._pos, ov if other_pos is None else None)
        return result

    def _unaryop(self, fn):
        result = TracedValue(fn(self.value), self.sources, self._graph)
        result._op_record = (fn, self._pos, None, None)
        return result

    # ── Arithmetic ──
    def __add__(self, o):      return self._binop(o, lambda a, b: a + b)
    def __radd__(self, o):     return self._rbinop(o, lambda a, b: a + b)
    def __sub__(self, o):      return self._binop(o, lambda a, b: a - b)
    def __rsub__(self, o):     return self._rbinop(o, lambda a, b: a - b)
    def __mul__(self, o):      return self._binop(o, lambda a, b: a * b)
    def __rmul__(self, o):     return self._rbinop(o, lambda a, b: a * b)
    def __neg__(self):         return self._unaryop(lambda a: -a)
    def __abs__(self):         return self._unaryop(abs)
    def __floordiv__(self, o): return self._binop(o, _safe_floordiv)
    def __rfloordiv__(self, o):return self._rbinop(o, _safe_floordiv)
    def __mod__(self, o):      return self._binop(o, lambda a, b: a % b if b else 0)
    def __pow__(self, o):      return self._binop(o, lambda a, b: a ** b)
    def __truediv__(self, o):  return self._binop(o, _safe_div)
    def __rtruediv__(self, o): return self._rbinop(o, _safe_div)

    # ── Comparison (return TracedValue to keep tracking) ──
    def __eq__(self, o):  return self._binop(o, lambda a, b: int(a == b))
    def __ne__(self, o):  return self._binop(o, lambda a, b: int(a != b))
    def __lt__(self, o):  return self._binop(o, lambda a, b: int(a < b))
    def __le__(self, o):  return self._binop(o, lambda a, b: int(a <= b))
    def __gt__(self, o):  return self._binop(o, lambda a, b: int(a > b))
    def __ge__(self, o):  return self._binop(o, lambda a, b: int(a >= b))

    # ── Boolean (for `or`, `and`, `not` in KB logic) ──
    def __or__(self, o):  return self._binop(o, lambda a, b: int(bool(a) or bool(b)))
    def __ror__(self, o): return self._rbinop(o, lambda a, b: int(bool(a) or bool(b)))
    def __and__(self, o): return self._binop(o, lambda a, b: int(bool(a) and bool(b)))
    def __rand__(self, o):return self._rbinop(o, lambda a, b: int(bool(a) and bool(b)))
    def __invert__(self): return self._unaryop(lambda a: int(not bool(a)))

    # ── Conversion ──
    def __int__(self):   return int(self.value)
    def __float__(self): return float(self.value)
    def __bool__(self):  return bool(self.value)
    def __hash__(self):  return hash(self.value)
    def __round__(self, ndigits=None): return self._unaryop(lambda a: round(a, ndigits))
    def __repr__(self):  return f"T({self.value}, {set(self.sources)})"


# ============================================================
# Compiled Evaluator
# ============================================================


def compile_graph(graph, n, target_positions):
    """Compile a TracedValue graph into a fast replay function.

    After one full TracedValue trace, this builds a function that replays
    the same computation with plain Python values — no TracedValue objects,
    no frozenset tracking. ~100x faster than re-tracing.

    Args:
        graph: list of TracedValue nodes from a traced execution.
        n: number of input variables (first n graph nodes are inputs).
        target_positions: list of graph positions to extract values from.

    Returns:
        A function: inputs (list of n values) -> tuple of values at target_positions.
        Returns None if the graph cannot be compiled (missing op records).
    """
    # Build instruction list from operation records.
    # Each record is (fn, a1, a2, const) where:
    #   a1, a2 are graph positions (or None if the operand is a constant)
    #   const holds the constant value when a1 or a2 is None
    instructions = []
    for i in range(n, len(graph)):
        rec = graph[i]._op_record
        if rec is None:
            return None
        fn, a1, a2, const = rec
        if a1 is not None and a2 is not None:
            # Both operands are graph nodes: fn(vals[a1], vals[a2])
            instructions.append((0, fn, a1, a2))
        elif a1 is not None and a2 is None and const is not None:
            # Right operand is constant: fn(vals[a1], const)
            instructions.append((1, fn, a1, const))
        elif a1 is None and a2 is not None and const is not None:
            # Left operand is constant: fn(const, vals[a2])
            instructions.append((2, fn, a2, const))
        elif a2 is None and const is None:
            # Unary: fn(vals[a1])
            instructions.append((3, fn, a1, None))
        else:
            return None  # unexpected pattern

    _instrs = instructions
    _targets = target_positions

    def evaluate(inputs):
        vals = list(inputs)
        for op, fn, a, b in _instrs:
            if op == 0:
                vals.append(fn(vals[a], vals[b]))
            elif op == 1:
                vals.append(fn(vals[a], b))
            elif op == 2:
                vals.append(fn(b, vals[a]))
            else:
                vals.append(fn(vals[a]))
        return tuple(vals[p] for p in _targets)

    return evaluate


# ============================================================
# Bottleneck Discovery
# ============================================================


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
) -> dict:
    """
    Trace constraint_fn(z, y_parts) to discover y-check structure.

    For each y-part index j (0..y_size-1), finds the minimal z-source set
    required before y_parts[j] can be checked.

    Returns:
        {y_part_index: frozenset of z-variable indices}
    """
    rng = random.Random(seed)
    if Y_domains is None:
        Y_domains = [K] * y_size

    y_z_sources = {}  # y_part_index -> frozenset of z indices

    for _ in range(num_samples):
        z_vals = [rng.randint(0, K - 1) for _ in range(n)]
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


# ============================================================
# Topology Helpers
# ============================================================


def _verify_coverage(var_groups, n):
    """Check that var_groups partitions {0, ..., n-1} exactly."""
    all_vars = sorted(v for vg in var_groups for v in vg)
    return all_vars == list(range(n))


def _greedy_chain_cover(bottlenecks, n, min_compression=1.5):
    """
    Build a chain cover from bottleneck source sets via greedy nested selection.

    Selects bottlenecks whose source sets form a nested chain:
    {} < S_1 < S_2 < ... < {0,...,n-1}, picking smallest-coverage first
    (with highest compression as tiebreaker).

    Returns:
        List of bottleneck dicts forming the chain cuts, or [].
    """
    chain_cuts = []
    covered = frozenset()
    for b in sorted(bottlenecks, key=lambda x: (x["coverage"], -x["compression"])):
        if b["domain_size"] <= 1:
            continue  # constant values carry zero information
        if b["sources"] > covered and b["compression"] >= min_compression:
            if b["sources"] != frozenset(range(n)):
                chain_cuts.append(b)
                covered = b["sources"]
    return chain_cuts


def _chain_topology(chain_cuts, n):
    """
    Convert chain cuts to tree topology (var_groups, children, root).

    Chain topology: children = [[]] + [[i] for i in range(L-1)], root = L-1.

    Returns:
        (var_groups, children, root) or None if chain_cuts is empty or invalid.
    """
    if not chain_cuts:
        return None

    all_sources = [frozenset()] + [c["sources"] for c in chain_cuts] + [frozenset(range(n))]
    var_groups = []
    for i in range(len(all_sources) - 1):
        group = sorted(all_sources[i + 1] - all_sources[i])
        if group:
            var_groups.append(group)

    if not var_groups or not _verify_coverage(var_groups, n):
        return None

    L = len(var_groups)
    children = [[]] + [[i] for i in range(L - 1)]
    root = L - 1
    return var_groups, children, root


# ============================================================
# Unified Cost Estimation
# ============================================================


def _compute_subtree_vars(var_groups, children, root):
    """Compute subtree_vars[i] = frozenset of all variables in subtree rooted at i."""
    num_nodes = len(var_groups)
    subtree_vars = [None] * num_nodes

    def _recurse(i):
        result = set(var_groups[i])
        for ch in children[i]:
            _recurse(ch)
            result |= subtree_vars[ch]
        subtree_vars[i] = frozenset(result)

    _recurse(root)
    return subtree_vars


def _assign_y_parts(subtree_vars, root, y_check_info):
    """
    Assign y-parts to tree nodes. For each y_part with z_sources S,
    find the smallest non-root node whose subtree covers S.

    Returns:
        {node_idx: [y_part_indices]}
    """
    if not y_check_info:
        return {}

    result = {}
    num_nodes = len(subtree_vars)

    for yi, z_src in y_check_info.items():
        best_node = None
        best_size = len(subtree_vars[root]) + 1
        for node_idx in range(num_nodes):
            if node_idx == root:
                continue
            if z_src <= subtree_vars[node_idx] and len(subtree_vars[node_idx]) < best_size:
                best_node = node_idx
                best_size = len(subtree_vars[node_idx])
        if best_node is not None:
            result.setdefault(best_node, []).append(yi)

    return result


def _estimate_cost(var_groups, children, root, bn_domain, K, y_check_info=None, y_domains=None):
    """
    Estimate topology cost from the traced static graph and optional y-pruning.

    `css_cost` captures the usual DP cost proxy:
        sum_i K^(num new vars_i) * product(child CSS domains)

    If y-pruning is available, we also estimate a reduced outgoing domain at
    each node by dividing its CSS domain by the product of assigned y-part
    domains. This is heuristic, but it makes the topology scorer reflect that
    early y-checks shrink the graph-level state space seen by ancestors.
    """
    num_nodes = len(var_groups)
    subtree_vars = _compute_subtree_vars(var_groups, children, root)

    # Look up CSS domain for each node from bn_domain
    css_domain_est = {}
    for i in range(num_nodes):
        src = subtree_vars[i]
        css_domain_est[i] = float(bn_domain.get(src, K ** len(var_groups[i])))

    y_node_asgn = {}
    if y_check_info:
        y_node_asgn = _assign_y_parts(subtree_vars, root, y_check_info)
        assigned = set()
        for yis in y_node_asgn.values():
            assigned.update(yis)
        unassigned = [yi for yi in y_check_info if yi not in assigned]
        if unassigned:
            y_node_asgn.setdefault(root, []).extend(unassigned)

    effective_domain_est = dict(css_domain_est)
    if y_node_asgn and y_domains:
        for node, yis in y_node_asgn.items():
            prune_factor = 1.0
            for yi in yis:
                if yi < len(y_domains) and y_domains[yi] > 0:
                    prune_factor *= float(y_domains[yi])
            if prune_factor > 1:
                effective_domain_est[node] = max(1.0, css_domain_est[node] / prune_factor)

    total = 0.0
    pruned_total = 0.0
    for i in range(num_nodes):
        r = len(var_groups[i])
        child_product = 1
        pruned_child_product = 1
        for ch in children[i]:
            child_product *= css_domain_est.get(ch, K)
            pruned_child_product *= effective_domain_est.get(ch, K)
        node_cost = child_product * (K ** r)
        pruned_node_cost = pruned_child_product * (K ** r)
        if y_node_asgn.get(i):
            prune_factor = css_domain_est[i] / max(effective_domain_est[i], 1.0)
            pruned_node_cost = max(1.0, pruned_node_cost / max(prune_factor, 1.0))
        total += node_cost
        pruned_total += pruned_node_cost

    return {
        "css_cost": total,
        "effective_cost": pruned_total if y_node_asgn and y_domains else total,
        "y_node_assignment": y_node_asgn or None,
        "effective_css_domains": effective_domain_est,
    }


# ============================================================
# Unified Decomposition Builder
# ============================================================


def _build_decomposition(
    logic_forward, n, K, var_groups, children, root, max_precompute,
    y_check_info=None, y_decompose_fn=None, min_css_pos=None,
    lazy_precompute: bool = False, max_states: int = 0,
    compiled_cache: dict = None,
):
    """
    Build a tree Decomposition with precomputed transition tables.

    Two modes based on whether y_decompose_fn + min_css_pos are provided:

    **Minimal CSS mode** (y_decompose_fn available):
      CSS = single minimal-domain value per source set (e.g., carry for addition, H=2).
      Every step (including root) uses y-conditioning to verify output digits.
      Root returns ("__target__", y) after y-checks — no KB(z)==y needed.

    **Full CSS mode** (no y_decompose_fn):
      CSS = tuple of all TracedValue nodes at source set (original behavior).
      Root checks KB(z)==y directly.
    """
    import itertools

    num_nodes = len(var_groups)
    subtree_vars = _compute_subtree_vars(var_groups, children, root)
    _use_min_css = bool(min_css_pos)
    _use_y_cond = bool(y_decompose_fn and min_css_pos)

    # ── Helpers ──

    def _run_kb(z_values):
        res = logic_forward(z_values)
        return res.value if isinstance(res, TracedValue) else res

    # Cache for compiled evaluators: target_sources -> evaluate_fn
    # Shared across multiple _build_decomposition calls via compiled_cache param.
    _compiled = compiled_cache if compiled_cache is not None else {}

    def _extract_css(z_values, target_sources):
        """Extract CSS for a non-root node from TracedValue graph.

        First call per target_sources: full TracedValue trace + compile
        an evaluator for ALL positions at that source set.
        Subsequent calls: use compiled evaluator (~100x faster).
        """
        # Fast path: compiled evaluator returns all values at target positions
        entry = _compiled.get(target_sources)
        if entry is not None:
            all_vals, min_pos = entry
            vals = all_vals(z_values)
            if _use_min_css and min_pos is not None and min_pos < len(vals):
                return vals[min_pos]
            return vals

        # Slow path: full TracedValue trace (once per target_sources)
        graph = []
        inputs = [TracedValue(z_values[i], frozenset({i}), graph)
                  for i in range(len(z_values))]
        try:
            logic_forward(inputs)
        except Exception:
            return _run_kb(z_values)

        # Find ALL positions matching target_sources and compile once
        positions = [i for i, nd in enumerate(graph) if nd.sources == target_sources]
        if not positions:
            return _run_kb(z_values)

        evaluator = compile_graph(graph, n, positions)
        # Determine which position is the min_css one (if applicable)
        min_pos_in_list = None
        if min_css_pos:
            global_pos = min_css_pos.get(target_sources)
            if global_pos is not None and global_pos in positions:
                min_pos_in_list = positions.index(global_pos)
        _compiled[target_sources] = (evaluator, min_pos_in_list) if evaluator else None

        # Return value for this first call
        if _use_min_css and min_pos_in_list is not None:
            return graph[positions[min_pos_in_list]].value
        return tuple(graph[p].value for p in positions)

    # ── Y-part assignment ──

    y_node_asgn = {}
    if y_check_info and y_decompose_fn:
        y_node_asgn = _assign_y_parts(subtree_vars, root, y_check_info)
        # In minimal CSS mode, assign remaining y-parts to root
        # (original _assign_y_parts skips root; we need full coverage)
        if _use_y_cond:
            assigned = set()
            for yis in y_node_asgn.values():
                assigned.update(yis)
            unassigned = [yi for yi in y_check_info if yi not in assigned]
            if unassigned:
                y_node_asgn.setdefault(root, []).extend(unassigned)

    # ── Post-order traversal ──

    order = []

    def _postorder(i):
        for ch in children[i]:
            _postorder(ch)
        order.append(i)

    _postorder(root)

    # ── Precompute transition tables ──

    trans_table = [{} for _ in range(num_nodes)]      # (h, z) → css
    partial_table = [{} for _ in range(num_nodes)]     # (h, z) → _run_kb(z)
    css_states = [set() for _ in range(num_nodes)]
    css_representative = [{} for _ in range(num_nodes)]
    total_kb_calls = 0
    budget_exceeded = False
    precompute_skipped = False

    for node in order:
        if budget_exceeded:
            break

        ch = children[node]
        step_vars = var_groups[node]
        r = len(step_vars)

        # Lazy mode only warms leaf states eagerly; internal transitions are
        # populated on demand during DP and cached thereafter.
        if lazy_precompute and ch:
            precompute_skipped = True
            continue

        if ch:
            child_h_lists = [sorted(css_states[c]) for c in ch]
            if any(len(hl) == 0 for hl in child_h_lists):
                child_combos = []
            else:
                # Beam-aware budget: use min(child_count, max_states)
                estimated = K ** r
                for hl in child_h_lists:
                    h_count = min(len(hl), max_states) if max_states > 0 else len(hl)
                    estimated *= h_count
                if estimated > max_precompute - total_kb_calls:
                    budget_exceeded = True
                    break
                child_combos = list(itertools.product(*child_h_lists))
        else:
            child_combos = [()]

        need_partial = node in y_node_asgn

        for h_combo in child_combos:
            if budget_exceeded:
                break

            rep = {}
            for ci, c in enumerate(ch):
                child_rep = css_representative[c].get(h_combo[ci], {})
                rep.update(child_rep)

            for z_vals in itertools.product(range(K), repeat=r):
                total_kb_calls += 1
                if total_kb_calls > max_precompute:
                    budget_exceeded = True
                    break

                z = [0] * n
                for vid, val in rep.items():
                    z[vid] = val
                for j, vid in enumerate(step_vars):
                    z[vid] = z_vals[j]

                if node == root and not _use_min_css:
                    css_next = _run_kb(z)
                else:
                    css_next = _extract_css(z, subtree_vars[node])

                trans_table[node][(h_combo, z_vals)] = css_next
                css_states[node].add(css_next)

                if need_partial:
                    partial_table[node][(h_combo, z_vals)] = _run_kb(z)

                if css_next not in css_representative[node]:
                    new_rep = dict(rep)
                    for j, vid in enumerate(step_vars):
                        new_rep[vid] = z_vals[j]
                    css_representative[node][css_next] = new_rep

        # Beam-prune: keep only top max_states CSS states by frequency
        if max_states > 0 and len(css_states[node]) > max_states:
            freq = defaultdict(int)
            for css in trans_table[node].values():
                freq[css] += 1
            top = set(s for s, _ in sorted(freq.items(), key=lambda x: -x[1])[:max_states])
            css_states[node] = top
            trans_table[node] = {k: v for k, v in trans_table[node].items() if v in top}
            partial_table[node] = {k: v for k, v in partial_table[node].items() if k in trans_table[node]}
            css_representative[node] = {k: v for k, v in css_representative[node].items() if k in top}

    # Representative lookup for on-the-fly fallback
    _reps = {}
    for node_idx in range(num_nodes):
        for css_val, rep in css_representative[node_idx].items():
            _reps[(node_idx, css_val)] = rep

    _tt = trans_table
    _pt = partial_table
    _vg = var_groups
    _ch = children
    _n = n

    def _reconstruct_z(h_children, z_vals, node):
        rep = {}
        for ci, c in enumerate(_ch[node]):
            child_rep = _reps.get((c, h_children[ci]))
            if child_rep is None:
                return None
            rep.update(child_rep)
        z = [0] * _n
        for vid, val in rep.items():
            z[vid] = val
        for j, vid in enumerate(_vg[node]):
            z[vid] = z_vals[j]
        return z

    # ── Transition function ──

    _y_cache = {}

    def _get_y_parts(y):
        if y not in _y_cache:
            _y_cache[y] = y_decompose_fn(y)
        return _y_cache[y]

    def _y_check(node, key, h_children, z_vals, y):
        """Check assigned y-parts against partial KB output. Returns False if mismatch."""
        if node not in y_node_asgn:
            return True
        partial = _pt[node].get(key)
        if partial is None:
            z = _reconstruct_z(h_children, z_vals, node)
            if z is None:
                return True
            partial = _run_kb(z)
            _pt[node][key] = partial
        try:
            h_parts = y_decompose_fn(partial)
        except Exception:
            return True
        y_parts = _get_y_parts(y)
        for j in y_node_asgn[node]:
            if j < len(h_parts) and j < len(y_parts):
                if h_parts[j] != y_parts[j]:
                    return False
        return True

    def transition_fn(h_children, z_vals, node, y):
        is_root = (node == root)
        key = (h_children, z_vals)

        # 1. Get CSS
        css = _tt[node].get(key)
        if css is None:
            z = _reconstruct_z(h_children, z_vals, node)
            if z is None:
                return None
            if is_root and not _use_min_css:
                css = _run_kb(z)
            else:
                css = _extract_css(z, subtree_vars[node])
            _tt[node][key] = css
            if (node, css) not in _reps:
                new_rep = {}
                for ci, c in enumerate(_ch[node]):
                    cr = _reps.get((c, h_children[ci]))
                    if cr:
                        new_rep.update(cr)
                for j, vid in enumerate(_vg[node]):
                    new_rep[vid] = z_vals[j]
                _reps[(node, css)] = new_rep

        # 2. Y-check (all nodes including root when min_css)
        if y_node_asgn and not _y_check(node, key, h_children, z_vals, y):
            return None

        # 3. Return
        if is_root:
            if _use_y_cond:
                # All y-parts verified component-wise → accept
                return ("__target__", y)
            elif _use_min_css:
                # Min CSS without y-conditioning: CSS is not KB output,
                # so verify via representative KB call
                z = _reconstruct_z(h_children, z_vals, node)
                if z is not None and _run_kb(z) == y:
                    return ("__target__", y)
                return None
            else:
                # Full mode: CSS at root IS the KB output
                if css == y:
                    return ("__target__", y)
                return None

        return css

    if y_node_asgn:
        print(f"  [Tracer] y-pruning enabled: {y_node_asgn}")

    decomp = make_tree(
        var_groups=var_groups,
        children=children,
        root=root,
        transition_fn=transition_fn,
        h_final_fn=lambda y: ("__target__", y),
        n=n, H=0,
    )

    _css_domain_sizes = [
        len(css_states[node]) for node in range(num_nodes) if node != root
    ]

    info = {
        "precompute_complete": not (budget_exceeded or precompute_skipped),
        "precompute_kb_calls": total_kb_calls,
        "precompute_mode": "lazy" if lazy_precompute else "budgeted",
        "css_domain_sizes": _css_domain_sizes,
        "var_groups": var_groups,
        "children": children,
        "root": root,
        "y_node_assignment": y_node_asgn if y_node_asgn else None,
    }
    return decomp, info


# ============================================================
# Main Orchestrator
# ============================================================


def discover_decomposition(
    logic_forward: Callable,
    n: int,
    K: int,
    num_samples: int = 500,
    seed: int = 0,
    min_compression: float = 1.5,
    max_precompute: int = 100_000,
    max_states: int = 0,
    lazy_precompute: bool = False,
    constraint_fn: Optional[Callable] = None,
    y_size: int = 0,
    Y_domains: Optional[List[int]] = None,
    y_decompose_fn: Optional[Callable] = None,
) -> Tuple[Decomposition, list]:
    """
    Automatically discover an IFW decomposition from a traced static graph.

    Generates candidate topologies (chain, tree, minfill), scores them using
    CSS compression and optional y-pruning, builds the cheapest verified
    decomposition, and falls back as needed.
    """
    # 1. Find bottlenecks + y-checks
    bottlenecks, static_graph, y_check_info = find_bottlenecks(
        logic_forward, n, K, num_samples, seed,
        constraint_fn=constraint_fn, y_size=y_size, Y_domains=Y_domains,
    )
    if not static_graph["is_static"]:
        print(
            "  [Tracer] WARNING: trace schema mismatches detected "
            f"({static_graph['schema_mismatches']} mismatches over {static_graph['num_traces']} accepted traces)"
        )
    y_info = y_check_info if y_decompose_fn else None
    bn_domain = {b["sources"]: b["domain_size"] for b in bottlenecks}

    # Build CSS candidate positions per source set, sorted by domain size.
    # Deduplicate by domain: keep only the first position at each distinct domain.
    _css_by_src = defaultdict(list)
    for pos, entry in static_graph["node_by_pos"].items():
        src = entry["sources"]
        if src is not None:
            _css_by_src[src].append((len(entry["values"]), pos))
    for src in _css_by_src:
        _css_by_src[src].sort()
        # Deduplicate: keep first position per distinct domain
        deduped = []
        seen_doms = set()
        for dom, pos in _css_by_src[src]:
            if dom not in seen_doms:
                seen_doms.add(dom)
                deduped.append((dom, pos))
        _css_by_src[src] = deduped

    def _css_candidates(max_tries=8):
        """Yield min_css_pos dicts with progressively larger domains, then None."""
        for idx in range(max_tries):
            css_pos = {}
            for src, cands in _css_by_src.items():
                i = min(idx, len(cands) - 1)
                css_pos[src] = cands[i][1]
            yield css_pos
        yield None  # full tuple fallback

    # 2. Generate candidate topologies
    _topology_builders = [
        ("chain",   lambda: _chain_topology(_greedy_chain_cover(bottlenecks, n, min_compression), n)),
        ("tree",    lambda: _build_hasse_tree(bottlenecks, n, min_compression)),
        ("minfill", lambda: _css_aware_minfill(n, K, bottlenecks, min_compression)),
    ]

    candidates = []
    for name, builder in _topology_builders:
        topo = builder()
        if topo:
            if len(topo) == 3:
                vg, ch, rt = topo
                meta = {}
            else:
                vg, ch, rt, meta = topo
            candidates.append((name, (vg, ch, rt), meta))

    # 3. Estimate costs and select cheapest
    brute_cost = K ** n
    cost_by_name = {"brute-force": brute_cost}
    scored = [(brute_cost, "brute-force", None)]
    candidate_meta = {}
    candidate_summaries = []
    for name, (vg, ch, rt), meta in candidates:
        cost_info = _estimate_cost(vg, ch, rt, bn_domain, K, y_info, Y_domains)
        css_cost = cost_info["css_cost"]
        effective_cost = cost_info["effective_cost"]
        candidate_meta[name] = dict(meta)
        if cost_info.get("y_node_assignment"):
            candidate_meta[name]["y_node_assignment"] = cost_info["y_node_assignment"]
            candidate_meta[name]["y_pruned_cost"] = cost_info["effective_cost"]
            candidate_meta[name]["effective_css_domains"] = cost_info["effective_css_domains"]

        if cost_info["effective_cost"] != css_cost:
            cost_by_name[f"{name}_css"] = css_cost
            cost_by_name[f"{name}_y_pruned"] = cost_info["effective_cost"]

        if meta.get("bag_cost") is not None:
            # Treat CSS as an optional compression overlay on top of a
            # treewidth-style baseline, not as a free optimistic estimate.
            cost_by_name[f"{name}_bag"] = meta["bag_cost"]
            effective_cost = max(cost_info["effective_cost"], meta["bag_cost"])

        cost_by_name[name] = effective_cost
        scored.append((effective_cost, name, (vg, ch, rt)))
        summary = {"name": name, "effective_cost": effective_cost, "css_cost": css_cost}
        if cost_info["effective_cost"] != css_cost:
            summary["y_pruned_cost"] = cost_info["effective_cost"]
        summary.update({
            k: v for k, v in candidate_meta[name].items()
            if k not in {"bags", "effective_css_domains"}
        })
        candidate_summaries.append(summary)
    scored.sort(key=lambda x: x[0])

    # 4. Try cheapest first: build + verify + fallback
    def _make_info(decomp_type, extra=None):
        info = {
            "decomposition_type": decomp_type,
            "bottlenecks": bottlenecks[:10],
            "static_graph": {
                "is_static": static_graph["is_static"],
                "num_traces": static_graph["num_traces"],
                "schema_mismatches": static_graph["schema_mismatches"],
                "num_nodes": static_graph["num_nodes"],
                "css_candidates": static_graph.get("css_candidates", []),
            },
            "candidate_topologies": candidate_summaries,
        }
        info.update({f"estimated_{cn}_cost": cc for cn, cc in cost_by_name.items()})
        if decomp_type in candidate_meta:
            info.update(candidate_meta[decomp_type])
        if extra:
            info.update(extra)
        return info

    chosen_name = scored[0][1]
    _print_cost_report(cost_by_name, candidate_meta, chosen_name, n, K)

    # Heuristic CSS discovery should not override a cheaper brute-force plan.
    if chosen_name == "brute-force":
        return _build_brute_chain(logic_forward, n, K), _make_info("brute-force")

    for cost, name, topo_data in scored:
        if name == "brute-force":
            continue
        vg, ch, rt = topo_data
        meta = candidate_meta.get(name, {})

        _shared_compiled = {}
        for css_pos in _css_candidates():
            try:
                decomp, build_info = _build_decomposition(
                    logic_forward, n, K, vg, ch, rt, max_precompute,
                    y_check_info=y_info, y_decompose_fn=y_decompose_fn,
                    min_css_pos=css_pos,
                    lazy_precompute=lazy_precompute,
                    max_states=max_states,
                    compiled_cache=_shared_compiled,
                )
            except Exception:
                continue
            build_info.update(meta)
            build_info.setdefault(
                "state_mode",
                "minimal-css" if (y_decompose_fn and css_pos is not None) else "css",
            )
            build_info.update(_make_info(name))
            if _verify_decomposition(decomp, logic_forward, n, K, max_states=max_states):
                return decomp, build_info

    # 5. Brute-force fallback
    return _build_brute_chain(logic_forward, n, K), _make_info("brute-force")


# ============================================================
# CSS-Aware Min-Fill Elimination
# ============================================================


def _build_interaction_graph(bottlenecks, n, min_compression=1.5):
    """
    Build variable interaction graph from MINIMAL bottleneck source sets.

    Only uses source sets that are not proper supersets of other bottleneck
    source sets. This captures local variable interactions without creating
    spurious edges from composite bottlenecks (e.g., carry accumulation).
    """
    # Collect all useful source sets
    useful = set()
    for b in bottlenecks:
        if b["compression"] >= min_compression:
            useful.add(b["sources"])

    # Keep only minimal sets (no proper subset is also a useful bottleneck)
    minimal = []
    for s in sorted(useful, key=len):
        if not any(s2 < s for s2 in minimal):
            minimal.append(s)

    # Build adjacency from minimal source sets (local interactions)
    adj = [set() for _ in range(n)]
    for src in minimal:
        src_list = sorted(src)
        for i in range(len(src_list)):
            for j in range(i + 1, len(src_list)):
                adj[src_list[i]].add(src_list[j])
                adj[src_list[j]].add(src_list[i])

    # Add CSS bridge edges: if two minimal groups appear in the same larger
    # bottleneck, they interact through CSS. Connect them with a single
    # bridge edge (one representative from each group).
    for b in bottlenecks:
        if b["compression"] < min_compression:
            continue
        src = b["sources"]
        contained = [m for m in minimal if m < src]
        if len(contained) >= 2:
            for gi in range(len(contained)):
                for gj in range(gi + 1, len(contained)):
                    u = max(contained[gi])
                    v = min(contained[gj])
                    adj[u].add(v)
                    adj[v].add(u)

    # Connect isolated variables to their smallest containing bottleneck
    covered = set()
    for src in minimal:
        covered |= src
    for v in set(range(n)) - covered:
        best = None
        for b in bottlenecks:
            if v in b["sources"]:
                if best is None or len(b["sources"]) < len(best):
                    best = b["sources"]
        if best is not None:
            for u in best:
                if u != v:
                    adj[v].add(u)
                    adj[u].add(v)

    return adj


def _css_aware_minfill(n, K, bottlenecks, min_compression=1.5):
    """
    CSS-aware min-fill variable elimination ordering.

    Like standard min-fill, but scores each candidate elimination by
    estimated DP cost K^|new_vars| * H (using CSS domain) instead of
    just fill edge count.
    """
    # Build CSS domain lookup: frozenset -> domain_size
    css_domains = {}
    for b in bottlenecks:
        src = b["sources"]
        if src not in css_domains or b["domain_size"] < css_domains[src]:
            css_domains[src] = b["domain_size"]

    # Build interaction graph
    adj = _build_interaction_graph(bottlenecks, n, min_compression)

    # If graph is complete (all connected), min-fill can't help
    remaining = set(range(n))
    if all(len(adj[v] & remaining) == n - 1 for v in range(n)):
        return None

    # Elimination loop
    elim_order = []  # (variable, bag_frozenset)
    adj_mut = [set(s) for s in adj]

    for _ in range(n):
        best_v = None
        best_score = float("inf")
        best_fill = float("inf")

        for v in remaining:
            neighbors = adj_mut[v] & remaining - {v}
            bag = frozenset({v}) | neighbors

            # Count fill edges needed
            fill = 0
            nb_list = sorted(neighbors)
            for i in range(len(nb_list)):
                for j in range(i + 1, len(nb_list)):
                    if nb_list[j] not in adj_mut[nb_list[i]]:
                        fill += 1

            # CSS-aware cost estimation
            separator = frozenset(neighbors)
            if separator:
                h_sep = _lookup_css_domain(separator, css_domains, K)
                h_bag = _lookup_css_domain(bag, css_domains, K)
                h = min(h_sep, h_bag)
            else:
                h = 1
            score = K * h

            if score < best_score or (score == best_score and fill < best_fill):
                best_v = v
                best_score = score
                best_fill = fill

        # Eliminate best_v
        neighbors = adj_mut[best_v] & remaining - {best_v}
        bag = frozenset({best_v}) | neighbors
        elim_order.append((best_v, bag))

        # Add fill edges
        nb_list = sorted(neighbors)
        for i in range(len(nb_list)):
            for j in range(i + 1, len(nb_list)):
                adj_mut[nb_list[i]].add(nb_list[j])
                adj_mut[nb_list[j]].add(nb_list[i])

        remaining.remove(best_v)

    topo = _elimination_to_tree(elim_order, n)
    if topo is None:
        return None

    vg, ch, rt, meta = topo
    meta = dict(meta)
    meta["bag_cost"] = _estimate_bag_dp_cost(meta["bag_sizes"], K)
    return vg, ch, rt, meta


def _lookup_css_domain(bag, css_domains, K):
    """
    Look up CSS domain size for a bag of variables.

    Tries exact match first, then smallest superset with known CSS.
    Falls back to K^|bag| (no compression) if nothing found.
    """
    if bag in css_domains:
        return css_domains[bag]

    # Find smallest known superset
    best_size = None
    best_h = None
    for src, h in css_domains.items():
        if bag <= src:
            if best_size is None or len(src) < best_size:
                best_size = len(src)
                best_h = h

    if best_h is not None:
        return best_h

    return K ** len(bag)


def _estimate_bag_dp_cost(bag_sizes, K):
    """
    Conservative treewidth-style baseline cost for a bag decomposition.

    Standard bag-based DP scales with K^|bag| per bag up to constant factors,
    so summing K^|bag| gives a simple topology-dependent baseline that does
    not assume any semantic CSS compression.
    """
    return sum(K ** size for size in bag_sizes)


def _elimination_to_tree(elim_order, n):
    """
    Convert elimination ordering to tree decomposition.

    Standard algorithm: each elimination step produces a bag.
    Parent of bag_i = first later bag that shares a variable.
    Then remove redundant bags and derive var_groups.
    """
    num_steps = len(elim_order)

    # Step 1: Build raw tree from elimination ordering
    parent = [None] * num_steps
    for i in range(num_steps):
        _, bag_i = elim_order[i]
        for j in range(i + 1, num_steps):
            _, bag_j = elim_order[j]
            if bag_i & bag_j:
                parent[i] = j
                break

    # Step 2: Remove redundant bags (where bag_i ⊆ bag_parent)
    keep = [True] * num_steps
    for i in range(num_steps):
        if parent[i] is not None:
            _, bag_i = elim_order[i]
            _, bag_p = elim_order[parent[i]]
            if bag_i <= bag_p:
                keep[i] = False

    # Reparent: chase up to first kept ancestor
    def _kept_ancestor(i):
        p = parent[i]
        while p is not None and not keep[p]:
            p = parent[p]
        return p

    # Step 3: Build compacted tree
    kept_indices = [i for i in range(num_steps) if keep[i]]
    if not kept_indices:
        bag_sizes = [n]
        return (
            [[i for i in range(n)]],
            [[]],
            0,
            {"bag_sizes": bag_sizes, "treewidth": max(n - 1, 0)},
        )

    old_to_new = {old: new for new, old in enumerate(kept_indices)}
    num_nodes = len(kept_indices)
    children = [[] for _ in range(num_nodes)]
    root = None

    for new_idx, old_idx in enumerate(kept_indices):
        p = _kept_ancestor(old_idx)
        if p is not None and p in old_to_new:
            children[old_to_new[p]].append(new_idx)
        elif root is None:
            root = new_idx
        else:
            children[root].append(new_idx)

    if root is None:
        root = num_nodes - 1

    # Step 4: Derive var_groups
    bags = [elim_order[kept_indices[i]][1] for i in range(num_nodes)]
    var_groups = []
    for i in range(num_nodes):
        child_vars = frozenset().union(*(bags[ch] for ch in children[i])) if children[i] else frozenset()
        var_groups.append(sorted(bags[i] - child_vars))

    if not _verify_coverage(var_groups, n):
        return None

    bag_sizes = [len(bag) for bag in bags]
    return (
        [list(vg) for vg in var_groups],
        children,
        root,
        {
            "bag_sizes": bag_sizes,
            "bags": [sorted(bag) for bag in bags],
            "treewidth": max(bag_sizes) - 1 if bag_sizes else 0,
        },
    )


# ============================================================
# Tree Decomposition Discovery (Hasse-based)
# ============================================================


def _build_hasse_tree(bottlenecks, n, min_compression=1.5):
    """
    Build a tree decomposition from bottleneck source sets using greedy
    Hasse diagram construction.

    Inserts bottlenecks (sorted by compression, highest first) into a tree
    based on set inclusion. The root covers all n variables.
    """
    full = frozenset(range(n))

    # Filter and deduplicate bottlenecks by sources
    seen_sources = set()
    candidates = []
    for b in bottlenecks:
        src = b["sources"]
        if src in seen_sources or src == full or len(src) == 0:
            continue
        if b["compression"] < min_compression:
            continue
        seen_sources.add(src)
        candidates.append(src)

    if not candidates:
        return None

    # Tree nodes: start with root (full set), insert candidates by inclusion
    nodes = [full]
    children_list = [[]]
    root_idx = 0

    for src in candidates:
        # Find the smallest existing node that strictly contains src
        best_parent = None
        
        best_parent_size = n + 1
        for i, existing in enumerate(nodes):
            if src < existing and len(existing) < best_parent_size:
                best_parent = i
                best_parent_size = len(existing)

        if best_parent is None:
            continue

        new_idx = len(nodes)
        nodes.append(src)
        children_list.append([])

        # Steal children: any child of parent that is a subset of new node
        stolen = []
        remaining = []
        for ch_idx in children_list[best_parent]:
            if nodes[ch_idx] < src:
                stolen.append(ch_idx)
            else:
                remaining.append(ch_idx)

        children_list[best_parent] = remaining + [new_idx]
        children_list[new_idx] = stolen

    # Derive var_groups: node's NEW variables = sources - union(children sources)
    var_groups = []
    for i, src in enumerate(nodes):
        child_vars = frozenset().union(*(nodes[ch] for ch in children_list[i])) if children_list[i] else frozenset()
        var_groups.append(sorted(src - child_vars))

    if not _verify_coverage(var_groups, n):
        return None

    # Only useful if tree has actual branching (otherwise chain is equivalent)
    if not any(len(ch) >= 2 for ch in children_list):
        return None

    return var_groups, children_list, root_idx


# ============================================================
# Cost Report
# ============================================================


def _print_cost_report(cost_by_name, candidate_meta, chosen, n, K):
    """Print a summary of estimated DP costs and speedup ratios."""
    brute = cost_by_name["brute-force"]
    chain = cost_by_name.get("chain")

    print(f"  [Tracer] Cost estimation (n={n}, K={K}):")
    print(f"    Brute-force : {brute:>12,}")

    for name, label, ref_name in [
        ("chain",   "Chain      ", None),
        ("tree",    "Tree       ", "chain"),
        ("minfill", "MinFill    ", "chain"),
    ]:
        cost = cost_by_name.get(name)
        if cost is None:
            print(f"    {label}: {'N/A':>12}")
            continue
        parts = [f"speedup vs brute: {brute / max(cost, 1):,.1f}x"]
        ref = cost_by_name.get(ref_name)
        if ref is not None and ref_name:
            parts.append(f"vs {ref_name}: {ref / max(cost, 1):,.1f}x")
        print(f"    {label}: {cost:>12,.0f}  ({', '.join(parts)})")
        meta = candidate_meta.get(name, {})
        details = []
        if cost_by_name.get(f"{name}_css") is not None:
            details.append(f"css={cost_by_name[f'{name}_css']:,.0f}")
        elif cost != brute:
            details.append(f"css={cost:,.0f}")
        if cost_by_name.get(f"{name}_y_pruned") is not None:
            details.append(f"y-pruned={cost_by_name[f'{name}_y_pruned']:,.0f}")
        if meta.get("bag_cost") is not None:
            details.append(f"bag={meta['bag_cost']:,.0f}")
            details.append(f"treewidth={meta.get('treewidth', 'N/A')}")
        if details:
            print(" " * 18 + ", ".join(details))

    print(f"    Selected    : {chosen}")


# ============================================================
# Verification & Fallback
# ============================================================


def _verify_decomposition(decomp, logic_forward, n, K, num_checks=20, seed=12345, max_states=0):
    """
    Verify a decomposition by checking random inputs against dp_map.

    Runs logic_forward on random valid inputs, then checks if dp_map
    with perfect log-probs (true z gets 0, others get -inf) recovers
    the exact original z. Returns True if all checks pass.
    """
    from .ifw_dp import dp_map

    rng = random.Random(seed)
    for _ in range(num_checks):
        z = [rng.randint(0, K - 1) for _ in range(n)]
        res = logic_forward(z)
        if isinstance(res, TracedValue):
            res = res.value
        if res is None:
            continue

        # Perfect log-probs: true z gets 0, others get -inf
        log_p = [[-100.0] * K for _ in range(n)]
        for i in range(n):
            log_p[i][z[i]] = 0.0

        z_hat, score = dp_map(decomp, K, res, log_p, max_states=max_states)
        if score <= float("-inf") or z_hat != z:
            return False
        y_hat = logic_forward(z_hat)
        if isinstance(y_hat, TracedValue):
            y_hat = y_hat.value
        if y_hat != res:
            return False
    return True


def _build_brute_chain(logic_forward, n, K):
    """Fallback: one variable per step, no compression."""
    def transition_fn(h_prev, z_vals, step, y):
        prev = dict(h_prev) if h_prev else {}
        prev[step] = z_vals[0]
        if step == n - 1:
            z = [prev.get(i, 0) for i in range(n)]
            res = logic_forward(z)
            if isinstance(res, TracedValue):
                res = res.value
            if res == y:
                return ("__target__", y)
            return None
        return tuple(sorted(prev.items()))

    return make_chain(
        L=n, var_groups=[[i] for i in range(n)],
        transition_fn=transition_fn, h_init=(),
        h_final_fn=lambda y: ("__target__", y), n=n, H=0,
    )
