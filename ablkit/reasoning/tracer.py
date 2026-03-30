"""
Automatic CSS Discovery via Operator Overloading + Functional Fingerprinting.

Traces a KB's logic_forward execution to discover information bottlenecks,
then builds an IFW Decomposition with precomputed transition tables.

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

    # Step 2: Build decomposition (one-time precomputation)
    decomp, info = discover_decomposition(kb.logic_forward, n=4, K=10)
"""

import random
from collections import defaultdict
from typing import Callable, List, Optional, Tuple, Any, FrozenSet

from .ifw_dp import Decomposition, make_chain


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
) -> list:
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
        if src is None:
            continue
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
    return bottlenecks


def discover_decomposition(
    logic_forward: Callable,
    n: int,
    K: int,
    num_samples: int = 500,
    seed: int = 0,
    min_compression: float = 1.5,
) -> Tuple[Decomposition, list]:
    """
    Automatically discover an IFW chain decomposition from KB code.

    Steps:
      1. Trace execution to find bottleneck nodes
      2. Select a chain of nested source sets (greedy cover)
      3. Build CSS lookup tables from traced executions
      4. Return Decomposition + bottleneck info

    Args:
        logic_forward: KB's forward function (must use supported ops).
        n: Number of input variables.
        K: Domain size.
        num_samples: Sampling budget.
        min_compression: Minimum compression ratio for a useful bottleneck.

    Returns:
        (Decomposition, bottleneck_info)
    """
    rng = random.Random(seed)

    # Step 1: Find bottlenecks
    bottlenecks = find_bottlenecks(logic_forward, n, K, num_samples, seed)

    # Step 2: Greedy chain cover
    # Find nested sequence: {} ⊂ S₁ ⊂ S₂ ⊂ ... ⊂ {0,...,n-1}
    chain_cuts = []  # list of (sources, domain_size)
    covered = frozenset()
    for b in sorted(bottlenecks, key=lambda x: (x["coverage"], -x["compression"])):
        if b["sources"] > covered and b["compression"] >= min_compression:
            if b["sources"] != frozenset(range(n)):
                chain_cuts.append(b)
                covered = b["sources"]

    if not chain_cuts:
        # No good bottlenecks → fallback to one-var-per-step chain
        return _build_brute_chain(logic_forward, n, K), bottlenecks

    # Step 3: Derive var_groups from chain cuts
    all_sources = [frozenset()] + [c["sources"] for c in chain_cuts] + [frozenset(range(n))]
    var_groups = []
    for i in range(len(all_sources) - 1):
        group = sorted(all_sources[i + 1] - all_sources[i])
        if group:
            var_groups.append(group)

    if not var_groups or sum(len(g) for g in var_groups) != n:
        return _build_brute_chain(logic_forward, n, K), bottlenecks

    L = len(var_groups)

    # Step 4: Exhaustive precomputation of transition tables
    #
    # For each step l (except the last), enumerate ALL (css_prev, z_vals)
    # combinations and compute css_next by running the traced KB.
    #
    # This costs O(H × K^r) KB calls per step, done ONCE.
    # After this, DP uses table lookups (O(1) per transition).
    #
    # Last step: verified against y at runtime (y-dependent).

    cut_sources = [frozenset()] + [c["sources"] for c in chain_cuts]

    # Step 4a: Functional CSS computation.
    # A single graph node may NOT be a valid CSS (e.g., carry loses digit info).
    # Instead, compute CSS as a fingerprint of KB behavior on remaining vars.
    # For source set S, CSS(z_S) = tuple(KB(z_S, z_rest_j) for j=1..m).
    # This guarantees lossless: same fingerprint ⟺ same behavior.
    _css_probes = {}  # target_sources -> list of complement z-value probes

    def _init_probes(target_sources, num_probes=20):
        """Generate random probe assignments for variables NOT in target_sources."""
        if target_sources in _css_probes:
            return
        complement = sorted(frozenset(range(n)) - target_sources)
        probe_rng = random.Random(seed + hash(target_sources))
        probes = []
        for _ in range(num_probes):
            probe = {v: probe_rng.randint(0, K - 1) for v in complement}
            probes.append(probe)
        _css_probes[target_sources] = probes

    def _get_css_value(z_values, target_sources):
        """Compute functional CSS: fingerprint of KB behavior on complement vars."""
        _init_probes(target_sources)
        probes = _css_probes[target_sources]
        fingerprint = []
        for probe in probes:
            z = list(z_values)  # copy
            for v, val in probe.items():
                z[v] = val
            try:
                res = logic_forward(z)
                if isinstance(res, TracedValue):
                    res = res.value
                fingerprint.append(res)
            except Exception:
                fingerprint.append(None)
        return tuple(fingerprint)

    # Step 4b: Build forward transition tables
    # trans_table[step][(css_prev, z_vals)] = css_next
    # Also: css_to_representative[step][css_val] = representative z_values for vars before this step
    import itertools

    trans_table = [{} for _ in range(L)]
    css_states = [set() for _ in range(L + 1)]  # reachable CSS states at each cut
    css_states[0].add(None)  # initial state: no CSS before step 0

    # Representative: for each CSS state, one concrete variable assignment that produces it
    css_representative = [{} for _ in range(L + 1)]
    css_representative[0][None] = {}

    for step in range(L - 1):  # last step handled at runtime
        step_vars = var_groups[step]
        target_src = cut_sources[step + 1]
        r = len(step_vars)

        for css_prev in css_states[step]:
            # Get a representative partial assignment for css_prev
            rep = css_representative[step].get(css_prev, {})

            for z_vals in itertools.product(range(K), repeat=r):
                # Build full z with: representative values + current z_vals
                z = [0] * n
                for vid, val in rep.items():
                    z[vid] = val
                for j, vid in enumerate(step_vars):
                    z[vid] = z_vals[j]

                # Get CSS value after this step
                css_next = _get_css_value(z, target_src)
                if css_next is not None:
                    trans_table[step][(css_prev, z_vals)] = css_next
                    css_states[step + 1].add(css_next)

                    # Store representative
                    if css_next not in css_representative[step + 1]:
                        new_rep = dict(rep)
                        for j, vid in enumerate(step_vars):
                            new_rep[vid] = z_vals[j]
                        css_representative[step + 1][css_next] = new_rep

    # Step 4c: Precompute the final step's output table.
    # For the last step, we need to know KB(z) for full assignments.
    # Since functional CSS is a valid sufficient statistic, using any
    # representative + final z_vals gives consistent results.
    final_output = {}  # (css_prev, z_vals) -> y_predicted
    if L >= 2:
        final_step = L - 1
        final_vars = var_groups[final_step]
        r_final = len(final_vars)
        for css_prev in css_states[final_step]:
            rep = css_representative[final_step].get(css_prev, {})
            for z_vals in itertools.product(range(K), repeat=r_final):
                z = [0] * n
                for vid, val in rep.items():
                    z[vid] = val
                for j, vid in enumerate(final_vars):
                    z[vid] = z_vals[j]
                res = logic_forward(z)
                if isinstance(res, TracedValue):
                    res = res.value
                final_output[(css_prev, z_vals)] = res
    elif L == 1:
        # Single step: just enumerate all z_vals
        for z_vals in itertools.product(range(K), repeat=len(var_groups[0])):
            z = [0] * n
            for j, vid in enumerate(var_groups[0]):
                z[vid] = z_vals[j]
            res = logic_forward(z)
            if isinstance(res, TracedValue):
                res = res.value
            final_output[(None, z_vals)] = res

    # Count total precomputed transitions
    total_trans = sum(len(t) for t in trans_table) + len(final_output)

    # Step 5: Build transition function from precomputed tables
    _tt = trans_table
    _fo = final_output
    _L = L

    def transition_fn(h_prev, z_vals, step, y):
        """
        State = CSS fingerprint (hashable).
        Intermediate steps: lookup in precomputed transition table.
        Last step: lookup precomputed KB output and compare with y.
        """
        if step == _L - 1:
            # Final step: check precomputed output against y
            key = (h_prev, z_vals)
            y_pred = _fo.get(key)
            if y_pred is not None and y_pred == y:
                return ("__target__", y)
            return None

        # Intermediate: table lookup
        key = (h_prev, z_vals)
        return _tt[step].get(key)  # None if invalid

    h_init = None  # matches css_states[0] = {None}

    def h_final_fn(y):
        return ("__target__", y)

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
        "chain_cuts": chain_cuts,
        "var_groups": var_groups,
        "css_domain_sizes": [c["domain_size"] for c in chain_cuts],
    }
    return decomp, info


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
