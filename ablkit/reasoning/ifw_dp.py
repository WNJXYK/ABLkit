"""
IFW-DP: Unified Tree/Chain Dynamic Programming for ABL Abduction.

Supports both chain decompositions (e.g., addition carry propagation)
and tree decompositions (e.g., BDD-OIA independent branches).
A chain is simply a tree where each node has at most one child.

Usage:
    # Chain (addition)
    decomp = make_chain(L=2, var_groups=[[0,2],[1,3]],
                        transition_fn=fn, h_init=0, h_final_fn=final, n=4)

    # Tree (BDD-OIA)
    decomp = make_tree(var_groups=[[0..8],[9..14],[15..20],[]],
                       children=[[],[],[],[0,1,2]], root=3,
                       transition_fn=fn, h_final_fn=final, n=21)

    # Same DP functions for both:
    z_hat, score = dp_map(decomp, K=10, y=57, log_p=log_p)
    q, Z = dp_marginal(decomp, K=10, y=57, p=p)
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple, Any
import itertools


# ============================================================
# Data Structure
# ============================================================


@dataclass
class Decomposition:
    """
    Unified tree/chain decomposition of KB constraint KB(z) = y.

    The constraint factors over a tree of nodes. Each node i processes
    variables var_groups[i] and receives CSS states from its children.

    A chain is a special case: node 0 is a leaf, node l's child is l-1.

    Attributes:
        n:              Total number of input variables.
        var_groups:     var_groups[i] = list of variable indices at node i.
        children:       children[i] = list of child node indices.
        root:           Root node index.
        transition_fn:  (h_children: tuple, z_vals: tuple, node: int, y: Any)
                        -> h (CSS state) or None (invalid).
                        h_children is a tuple of child states (empty for leaves).
        h_final_fn:     y -> required CSS state at root.
        H:              CSS state space size hint (0 = sparse/unbounded).
    """
    n: int
    var_groups: List[List[int]]
    children: List[List[int]]
    root: int
    transition_fn: Callable
    h_final_fn: Callable
    H: int = 0


# ============================================================
# Constructors
# ============================================================


def make_chain(
    L: int,
    var_groups: List[List[int]],
    transition_fn: Callable,
    h_init: Any,
    h_final_fn: Callable,
    n: int,
    H: int = 0,
) -> Decomposition:
    """
    Create a chain (linear) decomposition.

    Wraps old-style chain transition_fn(h_prev, z_vals, step, y) into
    unified tree transition_fn(h_children, z_vals, node, y).

    Args:
        L: Number of steps.
        var_groups: var_groups[l] = variables at step l.
        transition_fn: Old-style (h_prev, z_vals, step, y) -> h_next | None.
        h_init: Initial CSS state (before step 0).
        h_final_fn: y -> required terminal state.
        n: Total number of variables.
        H: CSS state space size (0 = sparse).
    """
    # Chain topology: node 0 is leaf, node l's child is l-1
    children = [[]] + [[l] for l in range(L - 1)]
    root = L - 1

    _orig_fn = transition_fn

    def unified_fn(h_children, z_vals, node, y):
        h_prev = h_children[0] if h_children else h_init
        return _orig_fn(h_prev, z_vals, node, y)

    return Decomposition(
        n=n, var_groups=var_groups, children=children, root=root,
        transition_fn=unified_fn, h_final_fn=h_final_fn, H=H,
    )


def make_tree(
    var_groups: List[List[int]],
    children: List[List[int]],
    root: int,
    transition_fn: Callable,
    h_final_fn: Callable,
    n: int,
    H: int = 0,
) -> Decomposition:
    """
    Create a tree decomposition directly.

    Args:
        var_groups: var_groups[i] = variables at node i.
        children: children[i] = child node indices.
        root: Root node index.
        transition_fn: (h_children: tuple, z_vals: tuple, node: int, y) -> h | None.
        h_final_fn: y -> required state at root.
        n: Total number of variables.
        H: CSS state space size hint (0 = sparse).
    """
    return Decomposition(
        n=n, var_groups=var_groups, children=children, root=root,
        transition_fn=transition_fn, h_final_fn=h_final_fn, H=H,
    )


# ============================================================
# Helpers
# ============================================================


def _postorder(decomp: Decomposition) -> List[int]:
    """Compute post-order traversal (leaves first, root last)."""
    order = []
    visited = set()

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for c in decomp.children[node]:
            dfs(c)
        order.append(node)

    dfs(decomp.root)
    return order


def _iter_z_vals(var_ids, K, var_domains):
    """Iterate over all z_vals combinations for a node's variables."""
    if not var_ids:
        yield ()  # single empty tuple for nodes with no variables
        return
    if var_domains is not None:
        domains = [var_domains[v] for v in var_ids]
    else:
        domains = [range(K)] * len(var_ids)
    yield from itertools.product(*domains)


# ============================================================
# MAP DP (Viterbi-style)
# ============================================================


def dp_map(
    decomp: Decomposition,
    K: int,
    y: Any,
    log_p: List[List[float]],
    var_domains: Optional[List[List[int]]] = None,
    max_states: int = 0,
) -> Tuple[List[int], float]:
    """
    MAP abduction on tree/chain decomposition.

    Solves:  argmax_{z in Omega_y} sum_i log p_i(z_i)

    Works for both chain and tree decompositions.

    Args:
        decomp: Tree or chain decomposition.
        K: Domain size per variable.
        y: Target output value.
        log_p: log_p[i][k] = log P(z_i = k | x_i).
        var_domains: Optional per-variable restricted domains.
        max_states: Beam width (0 = unlimited).

    Returns:
        best_z: Optimal assignment, length n.
        best_score: Log-probability of best assignment.
    """
    NEG_INF = float("-inf")
    order = _postorder(decomp)

    # node_best[node] = {h: (score, h_children_tuple, z_vals)}
    node_best = [None] * len(decomp.var_groups)

    for node in order:
        ch = decomp.children[node]
        var_ids = decomp.var_groups[node]
        
        new_states = {}

        # Children state combinations
        if ch:
            child_h_lists = [list(node_best[c].keys()) for c in ch]
            child_combos = itertools.product(*child_h_lists)
        else:
            child_combos = [()]

        for h_combo in child_combos:
            # Sum children scores
            ch_score = sum(node_best[c][h_c][0]
                          for c, h_c in zip(ch, h_combo)) if ch else 0.0

            for z_vals in _iter_z_vals(var_ids, K, var_domains):
                h = decomp.transition_fn(h_combo, z_vals, node, y)
                if h is None:
                    continue
                score = ch_score
                for j, vid in enumerate(var_ids):
                    score += log_p[vid][z_vals[j]]

                if h not in new_states or score > new_states[h][0]:
                    new_states[h] = (score, h_combo, z_vals)

        if max_states > 0 and len(new_states) > max_states:
            sorted_items = sorted(new_states.items(),
                                  key=lambda x: x[1][0], reverse=True)
            new_states = dict(sorted_items[:max_states])

        node_best[node] = new_states

    # Check root
    h_final = decomp.h_final_fn(y)
    root = decomp.root
    if h_final not in node_best[root]:
        return [0] * decomp.n, NEG_INF

    # Traceback: root to leaves
    best_z = [0] * decomp.n
    best_score = node_best[root][h_final][0]

    def traceback(nd, h):
        _, h_combo, z_vals = node_best[nd][h]
        for j, vid in enumerate(decomp.var_groups[nd]):
            best_z[vid] = z_vals[j]
        for c, h_c in zip(decomp.children[nd], h_combo):
            traceback(c, h_c)

    traceback(root, h_final)
    return best_z, best_score


# ============================================================
# Marginal DP (Belief Propagation on Tree)
# ============================================================


def dp_marginal(
    decomp: Decomposition,
    K: int,
    y: Any,
    p: List[List[float]],
    var_domains: Optional[List[List[int]]] = None,
    max_states: int = 0,
) -> Tuple[List[List[float]], float]:
    """
    Marginal abduction on tree/chain decomposition.

    Computes:  q_i(k) = P(z_i = k | y)  for all i, k.

    Uses upward-downward message passing (generalizes forward-backward).

    Args:
        decomp: Tree or chain decomposition.
        K: Domain size per variable.
        y: Target output value.
        p: p[i][k] = P(z_i = k | x_i).
        var_domains: Optional per-variable restricted domains.
        max_states: Beam width (0 = unlimited).

    Returns:
        q: q[i][k] posterior marginals.
        Z: Partition function.
    """
    n = decomp.n
    order = _postorder(decomp)

    # ── Upward pass (leaves → root): compute alpha ──
    # alpha[node] = {h: probability mass from subtree}
    alpha = [None] * len(decomp.var_groups)

    for node in order:
        ch = decomp.children[node]
        var_ids = decomp.var_groups[node]
        
        new_alpha = {}

        if ch:
            child_h_lists = [list(alpha[c].keys()) for c in ch]
            child_combos = itertools.product(*child_h_lists)
        else:
            child_combos = [()]

        for h_combo in child_combos:
            ch_prob = 1.0
            for c, h_c in zip(ch, h_combo):
                ch_prob *= alpha[c][h_c]
            if ch_prob == 0.0:
                continue

            for z_vals in _iter_z_vals(var_ids, K, var_domains):
                h = decomp.transition_fn(h_combo, z_vals, node, y)
                if h is None:
                    continue
                prob = ch_prob
                for j, vid in enumerate(var_ids):
                    prob *= p[vid][z_vals[j]]
                new_alpha[h] = new_alpha.get(h, 0.0) + prob

        if max_states > 0 and len(new_alpha) > max_states:
            sorted_items = sorted(new_alpha.items(),
                                  key=lambda x: x[1], reverse=True)
            new_alpha = dict(sorted_items[:max_states])

        alpha[node] = new_alpha

    # Partition function
    root = decomp.root
    h_final = decomp.h_final_fn(y)
    Z = alpha[root].get(h_final, 0.0)
    if Z < 1e-300:
        return [[1.0 / K] * K for _ in range(n)], 0.0

    # ── Downward pass (root → leaves): compute beta + marginals ──
    # beta[node] = {h: external probability}
    beta = [None] * len(decomp.var_groups)
    beta[root] = {h_final: 1.0}

    q = [[0.0] * K for _ in range(n)]

    for node in reversed(order):
        if beta[node] is None:
            continue
        ch = decomp.children[node]
        var_ids = decomp.var_groups[node]
        

        if ch:
            child_h_lists = [list(alpha[c].keys()) for c in ch]
            child_combos = list(itertools.product(*child_h_lists))
        else:
            child_combos = [()]

        for h_combo in child_combos:
            # Precompute product of all children's alpha
            ch_alpha_all = 1.0
            for c, h_c in zip(ch, h_combo):
                ch_alpha_all *= alpha[c][h_c]
            if ch_alpha_all == 0.0:
                continue

            for z_vals in _iter_z_vals(var_ids, K, var_domains):
                h = decomp.transition_fn(h_combo, z_vals, node, y)
                if h is None or h not in beta[node]:
                    continue

                local_prob = 1.0
                for j, vid in enumerate(var_ids):
                    local_prob *= p[vid][z_vals[j]]

                w = beta[node][h] * ch_alpha_all * local_prob / Z

                # Accumulate marginals for this node's variables
                for j, vid in enumerate(var_ids):
                    q[vid][z_vals[j]] += w

                # Propagate beta to children
                for ci, c in enumerate(ch):
                    h_c = h_combo[ci]
                    # beta[c][h_c] += beta[node][h] * local_prob
                    #                 * prod(alpha[other][h_other] for other != c)
                    other_alpha = local_prob
                    for cj, c2 in enumerate(ch):
                        if cj != ci:
                            other_alpha *= alpha[c2][h_combo[cj]]

                    if beta[c] is None:
                        beta[c] = {}
                    beta[c][h_c] = beta[c].get(h_c, 0.0) + \
                        beta[node][h] * other_alpha

    return q, Z


# ============================================================
# Backward-compatible aliases
# ============================================================

# These call the unified dp_map/dp_marginal directly.
# Old code using precompute_transitions + dp_map(decomp, trans, log_p)
# should migrate to dp_map(decomp, K, y, log_p).

sparse_dp_map = dp_map
sparse_dp_marginal = dp_marginal


def precompute_transitions(decomp, K, y, var_domains=None):
    """Deprecated: unified dp_map/dp_marginal no longer need this."""
    return None  # kept for import compatibility


