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
    h_final_accept: Callable = None,
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
    root = decomp.root
    if h_final_accept is not None:
        # Custom acceptance: find best accepted state at root.
        # Sort by h_final_accept's returned priority (lower = better),
        # then by score (higher = better).
        h_final = None
        best_key = None
        for h, (sc, _, _) in node_best[root].items():
            priority = h_final_accept(h)
            if not priority:
                continue
            # priority: int = revision_count+1 (lower = fewer revisions = better)
            #           True = accept without ordering
            if isinstance(priority, bool):
                key = (0, sc)
            else:
                key = (-int(priority), sc)  # negate: fewer revisions → higher key
            if best_key is None or key > best_key:
                best_key = key
                h_final = h
        if h_final is None:
            return [0] * decomp.n, NEG_INF
    else:
        h_final = decomp.h_final_fn(y)
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
    h_final_accept: Callable = None,
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
    if h_final_accept is not None:
        # Sum over all accepted root states
        Z = 0.0
        accepted_states = {}
        for h, a in alpha[root].items():
            if h_final_accept(h):
                Z += a
                accepted_states[h] = a
        if Z < 1e-300:
            return [[1.0 / K] * K for _ in range(n)], 0.0
        beta_root = {h: a / Z for h, a in accepted_states.items()}
    else:
        h_final = decomp.h_final_fn(y)
        Z = alpha[root].get(h_final, 0.0)
        if Z < 1e-300:
            return [[1.0 / K] * K for _ in range(n)], 0.0
        beta_root = {h_final: 1.0}

    # ── Downward pass (root → leaves): compute beta + marginals ──
    # beta[node] = {h: external probability}
    beta = [None] * len(decomp.var_groups)
    beta[root] = beta_root

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
# Revision-aware DP
# ============================================================


def _iter_z_exact_delta(var_ids, pseudo_idx, K, delta, var_domains):
    """Iterate z_vals with exactly `delta` positions differing from pseudo."""
    r_step = len(var_ids)
    if delta < 0 or delta > r_step:
        return
    if r_step == 0:
        if delta == 0:
            yield ()
        return
    for revised in itertools.combinations(range(r_step), delta):
        revised_set = set(revised)
        options = []
        valid = True
        for j in range(r_step):
            vid = var_ids[j]
            pv = pseudo_idx[vid]
            if j in revised_set:
                dom = var_domains[vid] if var_domains else range(K)
                alt = [k for k in dom if k != pv]
                if not alt:
                    valid = False
                    break
                options.append(alt)
            else:
                if var_domains and pv not in var_domains[vid]:
                    valid = False
                    break
                options.append((pv,))
        if not valid:
            continue
        yield from itertools.product(*options)


def _split_budget(children, node_dp, budget):
    """Yield (h_tuple, score, r_tuple) splitting exact budget among children.

    For MAP: score = sum of children scores.
    node_dp[c][r] = {h: (score, h_combo, z_vals)}.
    """
    if not children:
        if budget == 0:
            yield (), 0.0, ()
        return
    if len(children) == 1:
        c = children[0]
        if budget < len(node_dp[c]):
            for h, (sc, _, _) in node_dp[c][budget].items():
                yield (h,), sc, (budget,)
        return
    # Multiple children: enumerate first child's budget, recurse rest
    first, rest = children[0], children[1:]
    for r0 in range(budget + 1):
        if r0 >= len(node_dp[first]):
            continue
        for h0, (s0, _, _) in node_dp[first][r0].items():
            for h_rest, s_rest, rs_rest in _split_budget(rest, node_dp, budget - r0):
                yield (h0,) + h_rest, s0 + s_rest, (r0,) + rs_rest


def _split_budget_marginal(children, node_dp, budget):
    """Like _split_budget but for marginal (scores are probabilities, summed)."""
    if not children:
        if budget == 0:
            yield (), 1.0, ()
        return
    if len(children) == 1:
        c = children[0]
        if budget < len(node_dp[c]):
            for h, prob in node_dp[c][budget].items():
                yield (h,), prob, (budget,)
        return
    first, rest = children[0], children[1:]
    for r0 in range(budget + 1):
        if r0 >= len(node_dp[first]):
            continue
        for h0, p0 in node_dp[first][r0].items():
            for h_rest, p_rest, rs_rest in _split_budget_marginal(rest, node_dp, budget - r0):
                yield (h0,) + h_rest, p0 * p_rest, (r0,) + rs_rest


def dp_map_revision(
    decomp: Decomposition,
    K: int,
    y: Any,
    log_p: List[List[float]],
    pseudo_idx: List[int],
    max_revision: int,
    require_more_revision: int = 0,
    var_domains: Optional[List[List[int]]] = None,
    max_states: int = 0,
) -> Tuple[List[int], float]:
    """MAP abduction with incremental revision budget.

    Maintains dp[node][r][h] = best score with exactly r revisions in subtree.
    Iterates budget from 0 upward; for budget r=0, only pseudo values are
    tried (O(L*H) per step). Early termination once solutions are found
    and require_more_revision additional levels have been explored.

    Returns (best_z, best_score).
    """
    NEG_INF = float("-inf")
    n = decomp.n
    order = _postorder(decomp)
    vg = decomp.var_groups
    ch = decomp.children

    # node_dp[node][r] = {h: (score, h_combo, z_vals)}
    node_dp = {nd: [] for nd in range(len(vg))}

    h_final = decomp.h_final_fn(y)
    min_rev_found = None
    effective_max = max_revision
    root = decomp.root

    for budget in range(max_revision + 1):
        # Compute exact-budget table for each node
        for nd in order:
            children_n = ch[nd]
            var_ids = vg[nd]
            r_step = len(var_ids)
            max_delta = min(budget, r_step)

            dp_r = {}  # h -> (score, h_combo, z_vals)

            for delta in range(max_delta + 1):
                remaining = budget - delta
                for z_vals in _iter_z_exact_delta(var_ids, pseudo_idx, K, delta, var_domains):
                    local = 0.0
                    for j, vid in enumerate(var_ids):
                        local += log_p[vid][z_vals[j]]

                    for h_combo, ch_score, ch_rs in _split_budget(children_n, node_dp, remaining):
                        h = decomp.transition_fn(h_combo, z_vals, nd, y)
                        if h is None:
                            continue
                        total = ch_score + local
                        if h not in dp_r or total > dp_r[h][0]:
                            dp_r[h] = (total, h_combo, z_vals)

            if max_states > 0 and len(dp_r) > max_states:
                top = sorted(dp_r.items(), key=lambda x: x[1][0], reverse=True)[:max_states]
                dp_r = dict(top)

            node_dp[nd].append(dp_r)

        # Check root for valid solution at this budget
        dp_root = node_dp[root][budget]
        if h_final in dp_root:
            if min_rev_found is None:
                min_rev_found = budget
                effective_max = min(max_revision, min_rev_found + require_more_revision)

        if min_rev_found is not None and budget >= effective_max:
            break

    # Find best across all computed budgets
    best_score = NEG_INF
    best_budget = -1
    for r in range(len(node_dp[root])):
        dp_r = node_dp[root][r]
        if h_final in dp_r and dp_r[h_final][0] > best_score:
            best_score = dp_r[h_final][0]
            best_budget = r

    if best_budget < 0:
        return [0] * n, NEG_INF

    # Traceback
    best_z = [0] * n

    def traceback(nd, r, h):
        sc, h_combo, z_vals = node_dp[nd][r][h]
        for j, vid in enumerate(vg[nd]):
            best_z[vid] = z_vals[j]
        # Determine child budgets: need to recover the split
        # Re-derive from stored h_combo
        for ci, c in enumerate(ch[nd]):
            h_c = h_combo[ci]
            # Find which r_c was used for this child
            for r_c in range(r + 1):
                if r_c < len(node_dp[c]) and h_c in node_dp[c][r_c]:
                    # Check consistency: total child budget + local delta = r
                    traceback(c, r_c, h_c)
                    break

    traceback(root, best_budget, h_final)
    return best_z, best_score


def dp_marginal_revision(
    decomp: Decomposition,
    K: int,
    y: Any,
    p: List[List[float]],
    pseudo_idx: List[int],
    max_revision: int,
    require_more_revision: int = 0,
    var_domains: Optional[List[List[int]]] = None,
    max_states: int = 0,
    return_alpha: bool = False,
) -> Tuple[List[List[float]], float]:
    """Marginal abduction with revision budget.

    Computes q_i(k) = P(z_i=k | y, hamming(z,pseudo) <= R).

    Uses upward-downward message passing with separate tables per revision
    count. Early termination: once solutions found at min_rev, continues
    up to min(max_revision, min_rev + require_more_revision).

    Returns (q, Z) or (q, Z, alpha, min_rev_found) if return_alpha=True.
    """
    n = decomp.n
    order = _postorder(decomp)
    vg = decomp.var_groups
    ch_list = decomp.children
    root = decomp.root

    # ── Upward pass: alpha[node][r] = {h: prob} ──
    alpha = {nd: [] for nd in range(len(vg))}

    h_final = decomp.h_final_fn(y)
    min_rev_found = None
    effective_max = max_revision

    for budget in range(max_revision + 1):
        for nd in order:
            children_n = ch_list[nd]
            var_ids = vg[nd]
            r_step = len(var_ids)
            max_delta = min(budget, r_step)

            a_r = {}  # h -> prob

            for delta in range(max_delta + 1):
                remaining = budget - delta
                for z_vals in _iter_z_exact_delta(var_ids, pseudo_idx, K, delta, var_domains):
                    local = 1.0
                    for j, vid in enumerate(var_ids):
                        local *= p[vid][z_vals[j]]

                    for h_combo, ch_prob, _ in _split_budget_marginal(children_n, alpha, remaining):
                        h = decomp.transition_fn(h_combo, z_vals, nd, y)
                        if h is None:
                            continue
                        a_r[h] = a_r.get(h, 0.0) + ch_prob * local

            if max_states > 0 and len(a_r) > max_states:
                top = sorted(a_r.items(), key=lambda x: x[1], reverse=True)[:max_states]
                a_r = dict(top)

            alpha[nd].append(a_r)

        # Check root
        a_root = alpha[root][budget]
        if h_final in a_root and a_root[h_final] > 0:
            if min_rev_found is None:
                min_rev_found = budget
                effective_max = min(max_revision, min_rev_found + require_more_revision)

        if min_rev_found is not None and budget >= effective_max:
            break

    # Partition function: sum alpha[root][r][h_final] for r = 0..computed
    Z = 0.0
    num_budgets = len(alpha[root])
    for r in range(num_budgets):
        Z += alpha[root][r].get(h_final, 0.0)

    if Z < 1e-300:
        return [[1.0 / K] * K for _ in range(n)], 0.0

    # ── Downward pass: compute marginals ──
    # beta[node][r] = {h: external prob}
    # beta[root][r][h_final] = alpha[root][r][h_final] / Z (normalized weight per budget)
    beta = {nd: [None] * num_budgets for nd in range(len(vg))}
    for r in range(num_budgets):
        a_val = alpha[root][r].get(h_final, 0.0)
        if a_val > 0:
            beta[root][r] = {h_final: 1.0}
        else:
            beta[root][r] = {}

    q = [[0.0] * K for _ in range(n)]

    for nd in reversed(order):
        children_n = ch_list[nd]
        var_ids = vg[nd]
        r_step = len(var_ids)

        for r_total in range(num_budgets):
            if beta[nd][r_total] is None or not beta[nd][r_total]:
                continue

            max_delta = min(r_total, r_step)

            for delta in range(max_delta + 1):
                remaining = r_total - delta
                for z_vals in _iter_z_exact_delta(var_ids, pseudo_idx, K, delta, var_domains):
                    local = 1.0
                    for j, vid in enumerate(var_ids):
                        local *= p[vid][z_vals[j]]

                    for h_combo, ch_prob, ch_rs in _split_budget_marginal(children_n, alpha, remaining):
                        if ch_prob == 0.0:
                            continue
                        h = decomp.transition_fn(h_combo, z_vals, nd, y)
                        if h is None or h not in beta[nd][r_total]:
                            continue

                        w = beta[nd][r_total][h] * ch_prob * local / Z

                        # Accumulate marginals
                        for j, vid in enumerate(var_ids):
                            q[vid][z_vals[j]] += w

                        # Propagate beta to children
                        for ci, c in enumerate(children_n):
                            h_c = h_combo[ci]
                            r_c = ch_rs[ci]
                            other_prob = local
                            for cj, c2 in enumerate(children_n):
                                if cj != ci:
                                    h_c2 = h_combo[cj]
                                    r_c2 = ch_rs[cj]
                                    other_prob *= alpha[c2][r_c2].get(h_c2, 0.0)
                            if beta[c][r_c] is None:
                                beta[c][r_c] = {}
                            beta[c][r_c][h_c] = beta[c][r_c].get(h_c, 0.0) + \
                                beta[nd][r_total][h] * other_prob

    if return_alpha:
        return q, Z, alpha, min_rev_found
    return q, Z


# ============================================================
# Top-K Candidate Enumeration from DP Alpha Tables
# ============================================================


def dp_enumerate_topk(
    decomp: Decomposition,
    K: int,
    y: Any,
    p: List[List[float]],
    pseudo_idx: List[int],
    alpha: dict,
    min_rev: int,
    require_more_revision: int = 0,
    max_revision: int = 0,
    topK: int = 16,
    var_domains: Optional[List[List[int]]] = None,
) -> List[Tuple[List[int], float]]:
    """Enumerate top-K candidates from DP alpha tables.

    Given the alpha tables from dp_marginal_revision's forward pass,
    enumerates valid complete assignments (z_1,...,z_n) at revision
    distances min_rev..min(max_revision, min_rev+require_more_revision),
    ranked by their DP weight (product of p[i][z_i]).

    Returns list of (candidate_z_indices, weight) sorted by weight descending.
    """
    import heapq

    if min_rev is None:
        return []

    effective_max = min(max_revision, min_rev + require_more_revision)
    vg = decomp.var_groups
    ch_list = decomp.children
    root = decomp.root
    n = decomp.n
    h_final = decomp.h_final_fn(y)

    # Collect valid total budgets
    valid_budgets = []
    for r in range(min_rev, effective_max + 1):
        if r < len(alpha[root]) and h_final in alpha[root][r]:
            valid_budgets.append(r)
    if not valid_budgets:
        return []

    # Min-heap of (weight, tie, z_list), keeps best topK candidates
    heap = []
    _tie = [0]

    def _add_candidate(z_list, weight):
        _tie[0] += 1
        if len(heap) < topK:
            heapq.heappush(heap, (weight, _tie[0], z_list))
        elif weight > heap[0][0]:
            heapq.heapreplace(heap, (weight, _tie[0], z_list))

    def _enum_subtree(node, h_target, r_budget):
        """Yield (assignment_dict, weight) for valid assignments in this subtree."""
        children_n = ch_list[node]
        var_ids = vg[node]
        r_step = len(var_ids)
        max_delta = min(r_budget, r_step)

        for delta in range(max_delta + 1):
            remaining = r_budget - delta
            for z_vals in _iter_z_exact_delta(var_ids, pseudo_idx, K, delta, var_domains):
                local_w = 1.0
                for j, vid in enumerate(var_ids):
                    local_w *= p[vid][z_vals[j]]

                node_assign = {vid: z_vals[j] for j, vid in enumerate(var_ids)}

                if not children_n:
                    # Leaf: remaining must be 0
                    if remaining != 0:
                        continue
                    h = decomp.transition_fn((), z_vals, node, y)
                    if h != h_target:
                        continue
                    yield node_assign, local_w
                else:
                    # Internal: enumerate child splits
                    for child_assigns, child_w in _enum_children_split(
                        children_n, 0, remaining, z_vals, node, h_target,
                        [], [],
                    ):
                        merged = dict(node_assign)
                        merged.update(child_assigns)
                        yield merged, local_w * child_w

    def _enum_children_split(children_n, ci, remaining, z_vals, parent_node, h_target,
                             h_combo_acc, child_acc):
        """Yield (merged_child_assigns, combined_weight) for children[ci:].

        h_combo_acc and child_acc are per-call accumulators (not shared globally).
        """
        if ci == len(children_n):
            # All children enumerated — check parent transition
            h_combo = tuple(h_combo_acc)
            h = decomp.transition_fn(h_combo, z_vals, parent_node, y)
            if h == h_target:
                merged = {}
                w = 1.0
                for ca, cw in child_acc:
                    merged.update(ca)
                    w *= cw
                yield merged, w
            return

        c = children_n[ci]
        for r_c in range(remaining + 1):
            if r_c >= len(alpha[c]):
                continue
            for h_c in alpha[c][r_c]:
                if alpha[c][r_c][h_c] <= 0:
                    continue
                h_combo_acc.append(h_c)
                for c_assign, c_w in _enum_subtree(c, h_c, r_c):
                    child_acc.append((c_assign, c_w))
                    yield from _enum_children_split(
                        children_n, ci + 1, remaining - r_c,
                        z_vals, parent_node, h_target,
                        h_combo_acc, child_acc,
                    )
                    child_acc.pop()
                h_combo_acc.pop()

    for total_budget in valid_budgets:
        for assign, weight in _enum_subtree(root, h_final, total_budget):
            z_list = [0] * n
            for vid, val in assign.items():
                z_list[vid] = val
            _add_candidate(z_list, weight)

    # Sort by weight descending
    heap.sort(key=lambda x: x[0], reverse=True)
    return [(z, w) for w, _t, z in heap]


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


