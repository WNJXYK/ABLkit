"""
Automatic CSS Discovery via Operator Overloading + Functional Fingerprinting.

Traces a KB's logic_forward execution to discover information bottlenecks,
then builds an IFW Decomposition (tree or chain) with precomputed transition
tables.

Tree decomposition is preferred when the KB has independent sub-constraints
(e.g., f(z) = g(z[0:3]) + h(z[3:6])), allowing parallel sub-problem solving.
Chain decomposition is used when dependencies are sequential (e.g., addition
carry propagation). The algorithm automatically selects the cheaper option.

CSS correctness is guaranteed by functional fingerprinting: for each source
set S, CSS(z_S) = tuple(KB(z_S, z_rest_j) for j=1..m) using random probes.
This ensures losslessness (same fingerprint ⟺ same constraint behavior).

Supported operations on TracedValue:
  Arithmetic: +, -, *, //, %, **, abs, neg
  Comparison: ==, !=, <, >, <=, >= (return TracedValue, not bool)
  Boolean:    or, and, not (via __or__, __and__, __invert__)
  Conversion: int(), float(), bool()

Unsupported (requires KB rewrite):
  eval(), exec(), external solvers (Z3, Prolog), C extensions

Usage:
    from ablkit.reasoning.tracer import find_bottlenecks, discover_decomposition

    # Step 1: Analyze bottlenecks
    bottlenecks = find_bottlenecks(kb.logic_forward, n=4, K=10)
    for b in bottlenecks[:5]:
        print(f"sources={b['sources']} domain={b['domain_size']} compression={b['compression']:.0f}x")

    # Step 2: Build decomposition (auto-selects tree or chain)
    decomp, info = discover_decomposition(kb.logic_forward, n=4, K=10)
    print(info["decomposition_type"])  # "tree" or "chain"
"""

import random
from collections import defaultdict
from typing import Callable, List, Optional, Tuple, Any, FrozenSet

from .ifw_dp import Decomposition, make_chain, make_tree


# ============================================================
# TracedValue
# ============================================================

class TracedValue:
    """A value that tracks which input variables it depends on."""

    __slots__ = ("value", "sources", "_graph")

    def __init__(self, value, sources: FrozenSet[int], graph: Optional[list] = None):
        self.value = value
        self.sources = sources
        self._graph = graph
        if graph is not None:
            graph.append(self)

    def _new(self, value, other_sources=frozenset()):
        return TracedValue(value, self.sources | other_sources, self._graph)

    def _other_src(self, other):
        return other.sources if isinstance(other, TracedValue) else frozenset()

    def _other_val(self, other):
        return other.value if isinstance(other, TracedValue) else other

    # ── Arithmetic ──
    def __add__(self, o):  return self._new(self.value + self._other_val(o), self._other_src(o))
    def __radd__(self, o): return self._new(self._other_val(o) + self.value, self._other_src(o))
    def __sub__(self, o):  return self._new(self.value - self._other_val(o), self._other_src(o))
    def __rsub__(self, o): return self._new(self._other_val(o) - self.value, self._other_src(o))
    def __mul__(self, o):  return self._new(self.value * self._other_val(o), self._other_src(o))
    def __rmul__(self, o): return self._new(self._other_val(o) * self.value, self._other_src(o))
    def __neg__(self):     return self._new(-self.value)
    def __abs__(self):     return self._new(abs(self.value))

    def __floordiv__(self, o):
        d = self._other_val(o)
        return self._new(self.value // d if d else 0, self._other_src(o))

    def __rfloordiv__(self, o):
        return self._new(self._other_val(o) // self.value if self.value else 0, self._other_src(o))

    def __mod__(self, o):
        d = self._other_val(o)
        return self._new(self.value % d if d else 0, self._other_src(o))

    def __pow__(self, o):
        return self._new(self.value ** self._other_val(o), self._other_src(o))

    def __truediv__(self, o):
        d = self._other_val(o)
        return self._new(self.value / d if d else 0, self._other_src(o))

    def __rtruediv__(self, o):
        return self._new(self._other_val(o) / self.value if self.value else 0, self._other_src(o))

    # ── Comparison (return TracedValue to keep tracking) ──
    def __eq__(self, o):  return self._new(int(self.value == self._other_val(o)), self._other_src(o))
    def __ne__(self, o):  return self._new(int(self.value != self._other_val(o)), self._other_src(o))
    def __lt__(self, o):  return self._new(int(self.value < self._other_val(o)), self._other_src(o))
    def __le__(self, o):  return self._new(int(self.value <= self._other_val(o)), self._other_src(o))
    def __gt__(self, o):  return self._new(int(self.value > self._other_val(o)), self._other_src(o))
    def __ge__(self, o):  return self._new(int(self.value >= self._other_val(o)), self._other_src(o))

    # ── Boolean (for `or`, `and`, `not` in KB logic) ──
    def __or__(self, o):
        return self._new(int(bool(self.value) or bool(self._other_val(o))), self._other_src(o))

    def __ror__(self, o):
        return self._new(int(bool(self._other_val(o)) or bool(self.value)), self._other_src(o))

    def __and__(self, o):
        return self._new(int(bool(self.value) and bool(self._other_val(o))), self._other_src(o))

    def __rand__(self, o):
        return self._new(int(bool(self._other_val(o)) and bool(self.value)), self._other_src(o))

    def __invert__(self):
        return self._new(int(not bool(self.value)))

    # ── Conversion ──
    def __int__(self):   return int(self.value)
    def __float__(self): return float(self.value)
    def __bool__(self):  return bool(self.value)
    def __hash__(self):  return hash(self.value)
    def __repr__(self):  return f"T({self.value}, {set(self.sources)})"


# ============================================================
# Bottleneck Discovery
# ============================================================

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
        List of dicts sorted by compression ratio (highest first):
        [{"sources": frozenset, "domain_size": int, "compression": float}, ...]
    """
    rng = random.Random(seed)

    # Track each node POSITION separately (not merged by sources)
    # node_id = position in graph (stable across executions for same code path)
    node_by_pos = defaultdict(lambda: {"sources": None, "values": set()})

    # Track first-seen graph position per source set (lossless CSS candidate)
    first_pos_for_src = {}

    for _ in range(num_samples):
        z = [rng.randint(0, K - 1) for _ in range(n)]
        graph = []
        inputs = [TracedValue(z[i], frozenset({i}), graph) for i in range(n)]
        try:
            result = logic_forward(inputs)
        except Exception:
            continue
        for pos, node in enumerate(graph):
            src = node.sources
            if 0 < len(src) < n:
                entry = node_by_pos[pos]
                entry["sources"] = src
                entry["values"].add(node.value)
                if src not in first_pos_for_src:
                    first_pos_for_src[src] = pos

    # For CSS discovery: use the FIRST node per source set (lossless).
    # Later nodes (like carry) may lose information and aren't valid CSS.
    # Also track smallest-domain node for compression reporting.
    css_values = {}      # sources -> set of values at first-seen position
    css_pos = {}         # sources -> first-seen graph position
    report_values = {}   # sources -> smallest observed domain (for reporting)

    for pos, entry in sorted(node_by_pos.items()):
        src = entry["sources"]
        if src is None: continue
        vals = entry["values"]
        # CSS: use first-seen position (guaranteed lossless)
        if src not in css_values and pos == first_pos_for_src.get(src):
            css_values[src] = vals
            css_pos[src] = pos
        # Reporting: track smallest domain for compression analysis
        if src not in report_values or len(vals) < len(report_values[src]):
            report_values[src] = vals

    bottlenecks = []
    for sources in css_values:
        css_domain = len(css_values[sources])
        report_domain = len(report_values.get(sources, css_values[sources]))
        coverage = len(sources)
        # Compression based on CSS domain (what DP actually uses)
        compression = K ** coverage / max(css_domain, 1)
        bottlenecks.append({
            "sources": sources,
            "domain_size": css_domain,
            "coverage": coverage,
            "compression": compression,
            "graph_pos": css_pos[sources],
            "min_domain": report_domain,  # for informational purposes
        })

    bottlenecks.sort(key=lambda x: x["compression"], reverse=True)

    y_check_info = {}
    if constraint_fn is not None and y_size > 0:
        y_check_info = _discover_y_checks(
            constraint_fn, n, K, y_size, Y_domains, num_samples, seed,
        )

    return bottlenecks, node_by_pos, y_check_info


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


def discover_decomposition(
    logic_forward: Callable,
    n: int,
    K: int,
    num_samples: int = 500,
    seed: int = 0,
    min_compression: float = 1.5,
    max_precompute: int = 100_000,
    constraint_fn: Optional[Callable] = None,
    y_size: int = 0,
    Y_domains: Optional[List[int]] = None,
    y_decompose_fn: Optional[Callable] = None,
) -> Tuple[Decomposition, list]:
    """
    Automatically discover an IFW decomposition (tree or chain) from KB code.

    Tries tree decomposition first (exploits parallel sub-constraints),
    falls back to chain if no branching structure is found or chain is cheaper.

    Steps:
      1. Trace execution to find bottleneck nodes
      2. Try tree decomposition via greedy Hasse construction
      3. Try chain decomposition via greedy nested cover
      4. Compare estimated costs, pick the cheaper one
      5. Build CSS lookup tables (with budget limit + on-the-fly fallback)

    Args:
        logic_forward: KB's forward function (must use supported ops).
        n: Number of input variables.
        K: Domain size.
        num_samples: Sampling budget.
        min_compression: Minimum compression ratio for a useful bottleneck.
        max_precompute: Max KB calls for precomputation. Beyond this,
            transitions are computed on-the-fly during DP and cached.

    Returns:
        (Decomposition, bottleneck_info)
    """
    rng = random.Random(seed)

    # Step 1: Find bottlenecks
    bottlenecks, node_by_pos, y_check_info = find_bottlenecks(
        logic_forward, n, K, num_samples, seed,
        constraint_fn=constraint_fn, y_size=y_size, Y_domains=Y_domains,
    )

    # Step 2: Try tree decomposition
    tree_result = _build_hasse_tree(bottlenecks, n, min_compression)

    # Step 3: Greedy chain cover
    chain_cuts = []
    covered = frozenset()
    for b in sorted(bottlenecks, key=lambda x: (x["coverage"], -x["compression"])):
        if b["sources"] > covered and b["compression"] >= min_compression:
            if b["sources"] != frozenset(range(n)):
                chain_cuts.append(b)
                covered = b["sources"]

    # Step 4: Estimate all costs and choose the best decomposition
    brute_cost = _estimate_brute_cost(n, K)
    chain_cost = _estimate_chain_cost(chain_cuts, n, K)
    tree_cost = None
    use_tree = False
    tree_var_groups = tree_children = tree_root = None

    if tree_result is not None:
        tree_var_groups, tree_children, tree_root = tree_result

        # Estimate tree cost using bottleneck domain sizes
        bn_domain = {b["sources"]: b["domain_size"] for b in bottlenecks}
        tree_nodes_sources = []
        for i in range(len(tree_var_groups)):
            src = frozenset()
            stack = [i]
            while stack:
                nd = stack.pop()
                src |= frozenset(tree_var_groups[nd])
                stack.extend(tree_children[nd])
            tree_nodes_sources.append(src)

        css_domain_est = {}
        for i, src in enumerate(tree_nodes_sources):
            css_domain_est[i] = bn_domain.get(src, K ** len(tree_var_groups[i]))

        tree_cost = _estimate_tree_cost(
            tree_var_groups, tree_children, css_domain_est, K
        )

        if tree_cost < chain_cost:
            use_tree = True

    # Determine chosen type for reporting
    if use_tree:
        chosen = "tree"
    elif chain_cuts:
        chosen = "chain"
    else:
        chosen = "brute-force"

    _print_cost_report(brute_cost, chain_cost, tree_cost, chosen, n, K)

    # Step 5: Build the chosen decomposition
    if use_tree:
        decomp, precompute_info = _build_tree_decomposition(
            logic_forward, n, K,
            tree_var_groups, tree_children, tree_root,
            max_precompute,
            y_check_info=y_check_info, y_decompose_fn=y_decompose_fn,
        )
        info = {
            "bottlenecks": bottlenecks[:10],
            "decomposition_type": "tree",
            "var_groups": tree_var_groups,
            "children": tree_children,
            "root": tree_root,
            "estimated_brute_cost": brute_cost,
            "estimated_chain_cost": chain_cost,
            "estimated_tree_cost": tree_cost,
            **precompute_info,
        }
        return decomp, info

    # Fall back to chain decomposition
    if not chain_cuts:
        info = {
            "bottlenecks": bottlenecks[:10],
            "decomposition_type": "brute-force",
            "estimated_brute_cost": brute_cost,
            "estimated_chain_cost": brute_cost,
            "estimated_tree_cost": tree_cost,
        }
        return _build_brute_chain(logic_forward, n, K), info

    # Derive var_groups from chain cuts
    all_sources = [frozenset()] + [c["sources"] for c in chain_cuts] + [frozenset(range(n))]
    var_groups = []
    for i in range(len(all_sources) - 1):
        group = sorted(all_sources[i + 1] - all_sources[i])
        if group:
            var_groups.append(group)

    if not var_groups or sum(len(g) for g in var_groups) != n:
        return _build_brute_chain(logic_forward, n, K), bottlenecks

    L = len(var_groups)

    # Precompute chain transitions (with budget limit)
    import itertools

    def _run_kb(z_values):
        """Run KB on plain values (not traced)."""
        res = logic_forward(z_values)
        return res.value if isinstance(res, TracedValue) else res

    trans_table = [{} for _ in range(L)]
    final_output = {}
    css_states = [set() for _ in range(L + 1)]
    css_states[0].add(None)
    css_representative = [{} for _ in range(L + 1)]
    css_representative[0][None] = {}

    total_kb_calls = 0
    budget_exceeded = False

    for step in range(L - 1):
        if budget_exceeded:
            break
        step_vars = var_groups[step]
        r = len(step_vars)

        for css_prev in list(css_states[step]):
            if budget_exceeded:
                break
            rep = css_representative[step].get(css_prev, {})
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

                css_next = _run_kb(z)
                trans_table[step][(css_prev, z_vals)] = css_next
                css_states[step + 1].add(css_next)

                if css_next not in css_representative[step + 1]:
                    new_rep = dict(rep)
                    for j, vid in enumerate(step_vars):
                        new_rep[vid] = z_vals[j]
                    css_representative[step + 1][css_next] = new_rep

    # Precompute final step (if within budget)
    if not budget_exceeded:
        final_vars = var_groups[L - 1]
        r_final = len(final_vars)
        for css_prev in list(css_states[L - 1]):
            if budget_exceeded:
                break
            rep = css_representative[L - 1].get(css_prev, {})
            for z_vals in itertools.product(range(K), repeat=r_final):
                total_kb_calls += 1
                if total_kb_calls > max_precompute:
                    budget_exceeded = True
                    break

                z = [0] * n
                for vid, val in rep.items():
                    z[vid] = val
                for j, vid in enumerate(final_vars):
                    z[vid] = z_vals[j]
                final_output[(css_prev, z_vals)] = _run_kb(z)

    # Build chain transition function (hybrid: table + on-the-fly)
    _tt = trans_table
    _fo = final_output
    _vg = var_groups
    _L = L
    _n = n

    _reps = {}
    for step_idx in range(L + 1):
        for css_val, rep in css_representative[step_idx].items():
            _reps[(step_idx, css_val)] = rep

    def transition_fn(h_prev, z_vals, step, y):
        """
        Hybrid transition: precomputed table lookup + on-the-fly fallback.
        State = KB partial output (compact scalar).
        """
        if step == _L - 1:
            key = (h_prev, z_vals)
            y_pred = _fo.get(key)
            if y_pred is None:
                rep = _reps.get((step, h_prev))
                if rep is None:
                    return None
                z = [0] * _n
                for vid, val in rep.items():
                    z[vid] = val
                for j, vid in enumerate(_vg[step]):
                    z[vid] = z_vals[j]
                y_pred = _run_kb(z)
                _fo[key] = y_pred
            if y_pred is not None and y_pred == y:
                return ("__target__", y)
            return None

        key = (h_prev, z_vals)
        result = _tt[step].get(key)
        if result is None:
            rep = _reps.get((step, h_prev))
            if rep is None:
                return None
            z = [0] * _n
            for vid, val in rep.items():
                z[vid] = val
            for j, vid in enumerate(_vg[step]):
                z[vid] = z_vals[j]
            result = _run_kb(z)
            _tt[step][key] = result
            if (step + 1, result) not in _reps:
                new_rep = dict(rep)
                for j, vid in enumerate(_vg[step]):
                    new_rep[vid] = z_vals[j]
                _reps[(step + 1, result)] = new_rep
        return result

    h_init = None

    def h_final_fn(y):
        return ("__target__", y)

    # ── Y-conditioned pruning ──
    y_step_assignment = {}
    if y_check_info and y_decompose_fn:
        assigned = set()
        cumulative = frozenset()
        for step_idx in range(L):
            cumulative = cumulative | frozenset(var_groups[step_idx])
            for yi, z_src in y_check_info.items():
                if yi not in assigned and z_src <= cumulative:
                    y_step_assignment.setdefault(step_idx, []).append(yi)
                    assigned.add(yi)

        if y_step_assignment:
            # Only wrap if some y-parts are assigned before the final step
            early_steps = {s for s in y_step_assignment if s < L - 1}
            if early_steps:
                _base_tfn = transition_fn
                _y_dec = y_decompose_fn
                _y_asgn = y_step_assignment
                _y_cache = {}

                def _get_y_parts(y):
                    if y not in _y_cache:
                        _y_cache[y] = _y_dec(y)
                    return _y_cache[y]

                def transition_fn(h_prev, z_vals, step, y):
                    h_next = _base_tfn(h_prev, z_vals, step, y)
                    if h_next is None:
                        return None
                    # Y-pruning at intermediate steps
                    if step < _L - 1 and step in _y_asgn:
                        y_parts = _get_y_parts(y)
                        try:
                            h_parts = _y_dec(h_next)
                        except Exception:
                            return h_next
                        for j in _y_asgn[step]:
                            if j < len(h_parts) and j < len(y_parts):
                                if h_parts[j] != y_parts[j]:
                                    return None
                    return h_next

                print(f"  [Tracer] y-pruning enabled: {y_step_assignment}")

    decomp = make_chain(
        L=L,
        var_groups=var_groups,
        transition_fn=transition_fn,
        h_init=h_init,
        h_final_fn=h_final_fn,
        n=n, H=0,
    )

    info = {
        "bottlenecks": bottlenecks[:10],
        "decomposition_type": "chain",
        "chain_cuts": chain_cuts,
        "var_groups": var_groups,
        "css_domain_sizes": [c["min_domain"] for c in chain_cuts],
        "estimated_brute_cost": brute_cost,
        "estimated_chain_cost": chain_cost,
        "estimated_tree_cost": tree_cost,
        "precompute_complete": not budget_exceeded,
        "precompute_kb_calls": total_kb_calls,
        "y_step_assignment": y_step_assignment if y_step_assignment else None,
    }
    return decomp, info


# ============================================================
# Tree Decomposition Discovery
# ============================================================


def _build_hasse_tree(bottlenecks, n, min_compression=1.5):
    """
    Build a tree decomposition from bottleneck source sets using greedy
    Hasse diagram construction.

    Inserts bottlenecks (sorted by compression, highest first) into a tree
    based on set inclusion. The root covers all n variables.

    Returns:
        (var_groups, children_list, root_idx) or None if no useful tree found.
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
    nodes = [full]  # index 0 = root
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
            continue  # src not a strict subset of any node, skip

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
        child_vars = frozenset()
        for ch_idx in children_list[i]:
            child_vars |= nodes[ch_idx]
        var_groups.append(sorted(src - child_vars))

    # Verify complete coverage
    all_vars = []
    for vg in var_groups:
        all_vars.extend(vg)
    if sorted(all_vars) != list(range(n)):
        return None

    # Only useful if tree has actual branching (otherwise chain is equivalent)
    has_branch = any(len(ch) >= 2 for ch in children_list)
    if not has_branch:
        return None

    return var_groups, children_list, root_idx


def _estimate_brute_cost(n, K):
    """Estimate total DP transitions for brute-force (one variable per step)."""
    return K ** n


def _estimate_tree_cost(var_groups, children_list, css_domain_est, K):
    """
    Estimate total DP transitions for a tree decomposition.

    For each node: cost = product(child CSS domain sizes) × K^(num new vars).
    """
    total = 0
    for i in range(len(var_groups)):
        r = len(var_groups[i])
        child_product = 1
        for ch in children_list[i]:
            child_product *= css_domain_est.get(ch, K)
        total += child_product * (K ** r)
    return total


def _estimate_chain_cost(chain_cuts, n, K):
    """
    Estimate total DP transitions for a chain decomposition with bottlenecks.

    For each step: cost = prev_css_size × K^(num new vars in this step).
    """
    if not chain_cuts:
        return _estimate_brute_cost(n, K)

    all_sources = [frozenset()] + [c["sources"] for c in chain_cuts] + [frozenset(range(n))]
    total = 0
    prev_css_size = 1
    for i in range(len(all_sources) - 1):
        r = len(all_sources[i + 1] - all_sources[i])
        total += prev_css_size * (K ** r)
        if i < len(chain_cuts):
            prev_css_size = chain_cuts[i]["domain_size"]
        else:
            prev_css_size = K ** len(all_sources[i + 1])
    return total


def _print_cost_report(brute_cost, chain_cost, tree_cost, chosen, n, K):
    """Print a summary of estimated DP costs and speedup ratios."""
    print(f"  [Tracer] Cost estimation (n={n}, K={K}):")
    print(f"    Brute-force : {brute_cost:>12,}")
    print(f"    Chain       : {chain_cost:>12,}"
          f"  (speedup vs brute: {brute_cost / max(chain_cost, 1):,.1f}x)")
    if tree_cost is not None:
        print(f"    Tree        : {tree_cost:>12,}"
              f"  (speedup vs brute: {brute_cost / max(tree_cost, 1):,.1f}x,"
              f"  vs chain: {chain_cost / max(tree_cost, 1):,.1f}x)")
    else:
        print(f"    Tree        :          N/A  (no branching structure found)")
    print(f"    Selected    : {chosen}")


def _build_tree_decomposition(
    logic_forward, n, K, var_groups, children_list, root_idx, max_precompute,
    y_check_info=None, y_decompose_fn=None,
):
    """
    Build a tree Decomposition with precomputed transition tables.

    Precomputes transitions via post-order traversal: leaves first, root last.
    Each node's CSS = KB output with only that subtree's vars set, rest = 0.
    Uses budget control + on-the-fly fallback (same strategy as chain).

    Returns:
        (Decomposition, precompute_info_dict)
    """
    import itertools

    num_nodes = len(var_groups)

    def _run_kb(z_values):
        res = logic_forward(z_values)
        return res.value if isinstance(res, TracedValue) else res

    # Compute subtree_vars[i] = all variables in subtree rooted at i
    subtree_vars = [None] * num_nodes

    def _compute_subtree(i):
        result = set(var_groups[i])
        for ch in children_list[i]:
            _compute_subtree(ch)
            result |= subtree_vars[ch]
        subtree_vars[i] = frozenset(result)

    _compute_subtree(root_idx)

    # Post-order traversal
    order = []

    def _postorder(i):
        for ch in children_list[i]:
            _postorder(ch)
        order.append(i)

    _postorder(root_idx)

    # Precompute transition tables per node
    # trans_table[node][(h_children_tuple, z_vals)] = css_state
    trans_table = [{} for _ in range(num_nodes)]
    css_states = [set() for _ in range(num_nodes)]
    css_representative = [{} for _ in range(num_nodes)]

    # Leaves start with a single "no children" state
    for i in range(num_nodes):
        if not children_list[i]:
            css_states[i] = set()  # will be populated during precompute

    total_kb_calls = 0
    budget_exceeded = False

    for node in order:
        if budget_exceeded:
            break

        ch = children_list[node]
        step_vars = var_groups[node]
        r = len(step_vars)

        # Enumerate all combinations of children CSS states
        if ch:
            child_h_lists = [sorted(css_states[c]) for c in ch]
            # If any child has no states yet (shouldn't happen in post-order,
            # but guard against it), skip
            if any(len(hl) == 0 for hl in child_h_lists):
                child_combos = []
            else:
                child_combos = list(itertools.product(*child_h_lists))
        else:
            child_combos = [()]  # leaves: single empty tuple

        for h_combo in child_combos:
            if budget_exceeded:
                break

            # Build representative assignment from children
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

                css_next = _run_kb(z)
                trans_table[node][(h_combo, z_vals)] = css_next
                css_states[node].add(css_next)

                if css_next not in css_representative[node]:
                    new_rep = dict(rep)
                    for j, vid in enumerate(step_vars):
                        new_rep[vid] = z_vals[j]
                    css_representative[node][css_next] = new_rep

    # Build representative lookup for on-the-fly fallback
    _reps = {}
    for node_idx in range(num_nodes):
        for css_val, rep in css_representative[node_idx].items():
            _reps[(node_idx, css_val)] = rep

    # Also store child representatives for reconstructing parent reps
    _child_reps = {}
    for node_idx in range(num_nodes):
        for css_val, rep in css_representative[node_idx].items():
            _child_reps[(node_idx, css_val)] = rep

    _tt = trans_table
    _vg = var_groups
    _ch = children_list
    _n = n

    def transition_fn(h_children, z_vals, node, y):
        """
        Tree transition: precomputed table lookup + on-the-fly fallback.
        h_children = tuple of child CSS states (empty for leaves).
        """
        is_root = (node == root_idx)

        key = (h_children, z_vals)
        result = _tt[node].get(key)

        if result is None:
            # On-the-fly: reconstruct representative and run KB
            rep = {}
            for ci, c in enumerate(_ch[node]):
                child_rep = _child_reps.get((c, h_children[ci]))
                if child_rep is None:
                    return None
                rep.update(child_rep)

            z = [0] * _n
            for vid, val in rep.items():
                z[vid] = val
            for j, vid in enumerate(_vg[node]):
                z[vid] = z_vals[j]
            result = _run_kb(z)
            _tt[node][key] = result

            # Cache representative for future use
            if (node, result) not in _child_reps:
                new_rep = dict(rep)
                for j, vid in enumerate(_vg[node]):
                    new_rep[vid] = z_vals[j]
                _child_reps[(node, result)] = new_rep

        if is_root:
            if result == y:
                return ("__target__", y)
            return None

        return result

    def h_final_fn(y):
        return ("__target__", y)

    # ── Y-conditioned pruning for tree ──
    y_node_assignment = {}
    if y_check_info and y_decompose_fn:
        assigned = set()
        # Assign y-parts to the smallest subtree node that covers their z-sources
        for yi, z_src in y_check_info.items():
            best_node = None
            best_size = n + 1
            for node_idx in range(num_nodes):
                if z_src <= subtree_vars[node_idx] and len(subtree_vars[node_idx]) < best_size:
                    best_node = node_idx
                    best_size = len(subtree_vars[node_idx])
            if best_node is not None and best_node != root_idx:
                y_node_assignment.setdefault(best_node, []).append(yi)
                assigned.add(yi)

        if y_node_assignment:
            _base_tfn = transition_fn
            _y_dec = y_decompose_fn
            _y_asgn = y_node_assignment
            _y_cache = {}

            def _get_y_parts(y):
                if y not in _y_cache:
                    _y_cache[y] = _y_dec(y)
                return _y_cache[y]

            def transition_fn(h_children, z_vals, node, y):
                h_next = _base_tfn(h_children, z_vals, node, y)
                if h_next is None:
                    return None
                if node != root_idx and node in _y_asgn:
                    y_parts = _get_y_parts(y)
                    try:
                        h_parts = _y_dec(h_next)
                    except Exception:
                        return h_next
                    for j in _y_asgn[node]:
                        if j < len(h_parts) and j < len(y_parts):
                            if h_parts[j] != y_parts[j]:
                                return None
                return h_next

            print(f"  [Tracer] tree y-pruning enabled: {y_node_assignment}")

    decomp = make_tree(
        var_groups=var_groups,
        children=children_list,
        root=root_idx,
        transition_fn=transition_fn,
        h_final_fn=h_final_fn,
        n=n, H=0,
    )

    precompute_info = {
        "precompute_complete": not budget_exceeded,
        "precompute_kb_calls": total_kb_calls,
        "y_node_assignment": y_node_assignment if y_node_assignment else None,
    }
    return decomp, precompute_info


def _build_brute_chain(logic_forward, n, K):
    """Fallback: one variable per step, no compression."""
    def transition_fn(h_prev, z_vals, step, y):
        # h_prev is a tuple of (var_idx, val) pairs (hashable)
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
        return tuple(sorted(prev.items()))  # hashable

    return make_chain(
        L=n, var_groups=[[i] for i in range(n)],
        transition_fn=transition_fn, h_init=(),
        h_final_fn=lambda y: ("__target__", y), n=n, H=0,
    )
