"""
Task-specific IFW decompositions and GPU engines.

Each function builds a Decomposition encoding how a particular KB constraint
factors into a chain of local steps.  The core DP algorithms in ifw_dp.py
and ifw_gpu.py are fully generic; only the code in THIS file is task-specific.

Provided decompositions:
    - make_addition_decomposition(d)  : d-digit base-10 addition (IFW=2)
    - make_addition_css(d)            : CSS function for addition
    - make_mod_decomposition(n, p)    : modular arithmetic (IFW=p)
    - make_hwf_decomposition(length)  : handwritten formula evaluation
    - make_bdd_oia_decomposition()    : BDD-OIA driving scene (IFW=4)
    - BatchedAdditionDP               : GPU-optimized batched DP for addition
"""

from typing import Callable, List, Optional, Tuple, Any

import itertools

from .ifw_dp import Decomposition, make_chain, make_tree, decomposition_from_kb


# ============================================================
# Addition (MNIST-Add, CIFAR-Add, SVHN-Add, etc.)
# ============================================================


def make_addition_decomposition(d: int) -> Decomposition:
    """
    Decomposition for d-digit base-10 addition: A + B = S.

    Variables (2d total):
        z = [a_0, a_1, ..., a_{d-1}, b_0, b_1, ..., b_{d-1}]
        a_i, b_i in {0,...,9}, digit 0 = least significant.

    CSS: carry in {0,1}, IFW = 2.
    Local constraint at step l:
        a_l + b_l + carry_in = s_l + 10 * carry_out
        where s_l = (y // 10^l) % 10.
    """

    def transition_fn(h_prev, z_vals, step_l, y):
        a, b = z_vals
        s_l = (y // (10**step_l)) % 10
        total = a + b + h_prev
        if total % 10 != s_l:
            return None
        return total // 10

    def h_final_fn(y):
        return (y // (10**d)) % 10

    return make_chain(
        L=d, var_groups=[[l, d + l] for l in range(d)],
        transition_fn=transition_fn, h_init=0,
        h_final_fn=h_final_fn, n=2 * d, H=2,
    )


def make_addition_css(d: int) -> Callable:
    """
    CSS function for d-digit base-10 addition.

    Returns a css_fn compatible with decomposition_from_kb().
    """

    def css_fn(h_prev, z_vals, step, y):
        if step == d:
            return (y // (10**d)) % 10
        a, b = z_vals
        s_l = (y // (10**step)) % 10
        total = a + b + h_prev
        if total % 10 != s_l:
            return None
        return total // 10

    return css_fn


# ============================================================
# Modular Arithmetic
# ============================================================


def make_mod_decomposition(n: int, p: int, op: str = "add") -> Decomposition:
    """
    Decomposition for modular arithmetic:
        f(z_0, ..., z_{n-1}) = r  (mod p)

    where f is iterated addition or multiplication.
    CSS: partial result mod p, IFW = p.
    """

    def transition_fn(h_prev, z_vals, step_l, y):
        (z_val,) = z_vals
        if op == "add":
            return (h_prev + z_val) % p
        elif op == "mul":
            return (h_prev * z_val) % p
        else:
            raise ValueError(f"Unknown op: {op}")

    identity = 0 if op == "add" else 1

    def h_final_fn(y):
        return y % p

    return make_chain(
        L=n, var_groups=[[i] for i in range(n)],
        transition_fn=transition_fn, h_init=identity,
        h_final_fn=h_final_fn, n=n, H=p,
    )


# ============================================================
# HWF (Handwritten Formula) — sparse CSS state
# ============================================================

# Label mapping: indices 0-8 = digits "1"-"9", indices 9-12 = ops "+","-","*","/"

# Number of decimal places for rounding float states (prevents state explosion)
_HWF_ROUND = 8


def _hwf_round(x: float) -> float:
    """Round float to fixed precision for hashable state key."""
    return round(x, _HWF_ROUND)


def make_hwf_decomposition(length: int, max_err: float = 1e-10) -> Decomposition:
    """
    Decomposition for HWF formula evaluation with operator precedence.

    Formula: d1 op1 d2 op2 d3 ... (length = 2k+1, k+1 digits, k operators)

    CSS state: (additive_acc, multiplicative_acc) as rounded floats.
      - additive_acc: sum of all completed additive terms
      - multiplicative_acc: current multiplicative chain value (includes sign)

    Processing:
      - Step 0: first digit d -> state = (0.0, float(d))
      - Step l (l>=1): processes (operator, digit) pair
        - +/- : finalize mult into add, start new mult chain
        - * / : extend multiplicative chain

    K = 13 (9 digits + 4 operators), structural constraints enforced.
    State space is unbounded -> use sparse_dp_map / sparse_dp_marginal.

    Args:
        length: Formula length (must be odd, >= 1).
        max_err: Tolerance for comparing final value to target y.
    """
    assert length >= 1 and length % 2 == 1, f"length must be odd >= 1, got {length}"

    n = length
    k = (length - 1) // 2  # number of operators

    # Step 0 processes position 0 (first digit)
    # Step l (l>=1) processes positions [2l-1, 2l] (operator, digit)
    L = k + 1
    var_groups = [[0]] + [[2 * l - 1, 2 * l] for l in range(1, L)]

    def transition_fn(h_prev, z_vals, step, y):
        if step == 0:
            (idx,) = z_vals
            if idx > 8:  # not a digit
                return None
            d = float(idx + 1)
            return (0.0, d)

        # step >= 1: process (operator_idx, digit_idx)
        op_idx, digit_idx = z_vals
        if digit_idx > 8 or op_idx < 9 or op_idx > 12:
            return None

        d = float(digit_idx + 1)
        add_acc, mult_acc = h_prev
        op_i = op_idx - 9  # 0=+, 1=-, 2=*, 3=/

        if op_i == 0:  # +
            new_state = (_hwf_round(add_acc + mult_acc), d)
        elif op_i == 1:  # -
            new_state = (_hwf_round(add_acc + mult_acc), -d)
        elif op_i == 2:  # *
            new_state = (add_acc, _hwf_round(mult_acc * d))
        elif op_i == 3:  # /
            new_state = (add_acc, _hwf_round(mult_acc / d))
        else:
            return None
        return new_state

    def h_final_fn(y):
        try:
            return ("__target__", _hwf_round(float(y)))
        except (ValueError, TypeError):
            return ("__target__", 0.0)

    original_transition = transition_fn

    def wrapped_transition(h_prev, z_vals, step, y):
        h_next = original_transition(h_prev, z_vals, step, y)
        if h_next is None:
            return None
        if step == L - 1:
            add_acc, mult_acc = h_next
            result = _hwf_round(add_acc + mult_acc)
            try:
                target = _hwf_round(float(y))
            except (ValueError, TypeError):
                return None
            if abs(result - target) <= max_err:
                return ("__target__", target)
            return None
        return h_next

    return make_chain(
        L=L, var_groups=var_groups,
        transition_fn=wrapped_transition, h_init=(0.0, 0.0),
        h_final_fn=h_final_fn, n=n, H=0,
    )


# ============================================================
# BDD-OIA (Autonomous Driving) — 3-step chain decomposition
# ============================================================

# 21 binary concepts → 4 binary actions via deterministic rules.
# Tree structure: 3 independent leaf groups + 1 root combiner.
#   Node 0 (concepts 0-8):   → (move_forward, stop)   H=4
#   Node 1 (concepts 9-14):  → turn_left              H=2
#   Node 2 (concepts 15-20): → turn_right             H=2
#   Node 3 (root, no vars):  combines children → final action tuple


def make_bdd_oia_decomposition() -> Decomposition:
    """
    Tree decomposition for BDD-OIA driving scene understanding.

    3 leaf nodes process independent concept groups, root combines results.
    Treewidth = 1 (star topology), much tighter than chain IFW = 16.

    Tree:
            Node 3 (root: combine)
           /        |        \\
        Node 0    Node 1    Node 2
        (0-8)     (9-14)    (15-20)

    K = 2 (binary), n = 21.
    Complexity: O(2^9*4 + 2^6*2 + 2^6*2 + 4*2*2) ≈ O(2.3K)
    vs chain O(3 * 2^9 * 16) ≈ O(24K)
    vs brute-force O(2^21) ≈ O(2M).
    """

    def transition_fn(h_children, z_vals, node, y):
        if node == 0:
            # Leaf: concepts 0-8 → (move_forward, stop)
            (green_light, follow, road_clear, red_light,
             traffic_sign, car, person, rider, other_obstacle) = z_vals

            if red_light == 1 and green_light == 1:
                return None
            obstacle = car or person or rider or other_obstacle
            if road_clear == obstacle:
                return None

            move_forward = int(green_light or follow or road_clear)
            stop = int(red_light or traffic_sign or obstacle)
            if stop:
                move_forward = 0
            return move_forward * 2 + stop  # 0-3

        elif node == 1:
            # Leaf: concepts 9-14 → turn_left
            (left_lane, left_green_light, left_follow,
             no_left_lane, left_obstacle, left_solid_line) = z_vals
            can = left_lane or left_green_light or left_follow
            cannot = no_left_lane or left_obstacle or left_solid_line
            return int(can and not cannot)  # 0 or 1

        elif node == 2:
            # Leaf: concepts 15-20 → turn_right
            (right_lane, right_green_light, right_follow,
             no_right_lane, right_obstacle, right_solid_line) = z_vals
            can = right_lane or right_green_light or right_follow
            cannot = no_right_lane or right_obstacle or right_solid_line
            return int(can and not cannot)  # 0 or 1

        elif node == 3:
            # Root: combine children → (mf, stop, tl, tr) as 4-bit int
            h_fwd_stop, h_left, h_right = h_children
            mf = (h_fwd_stop >> 1) & 1
            stop = h_fwd_stop & 1
            return mf * 8 + stop * 4 + h_left * 2 + h_right

        return None

    def h_final_fn(y):
        mf, stop, tl, tr = y
        return mf * 8 + stop * 4 + tl * 2 + tr

    return make_tree(
        var_groups=[
            list(range(0, 9)),     # node 0: forward/stop
            list(range(9, 15)),    # node 1: left turn
            list(range(15, 21)),   # node 2: right turn
            [],                    # node 3: root (no variables, just combines)
        ],
        children=[[], [], [], [0, 1, 2]],  # star topology
        root=3,
        transition_fn=transition_fn,
        h_final_fn=h_final_fn,
        n=21, H=16,
    )


# ============================================================
# BatchedAdditionDP — GPU-optimized for addition
# ============================================================

try:
    from .ifw_gpu import BatchedIFWDP

    class BatchedAdditionDP(BatchedIFWDP):
        """
        Optimized batched DP for d-digit addition.

        Overrides _build_transition_matrices with a fully vectorized version
        that avoids per-sample Python loops.
        """

        def __init__(self, d: int, K: int = 10, xp=None):
            decomp = make_addition_decomposition(d)
            super().__init__(decomp, K, xp)
            self.d = d

        def _build_transition_matrices(self, p, y_array):
            xp = self.xp
            B = p.shape[0]
            K = self.K
            d = self.d
            H = 2

            p_a = p[:, :d, :]
            p_b = p[:, d:, :]

            y_np = y_array
            s = xp.zeros((B, d + 1), dtype=xp.int32)
            for l in range(d + 1):
                s[:, l] = (y_np // (10**l)) % 10

            T_list = []
            for l in range(d):
                T = xp.zeros((B, H, H), dtype=p.dtype)
                for c_in in range(H):
                    for c_out in range(H):
                        target = s[:, l] + 10 * c_out - c_in
                        for a in range(K):
                            b_val = target - a
                            valid = (b_val >= 0) & (b_val < K)
                            b_clamped = xp.clip(b_val, 0, K - 1)
                            pb = p_b[xp.arange(B), l, b_clamped]
                            contrib = xp.where(valid, p_a[:, l, a] * pb, 0.0)
                            T[:, c_in, c_out] += contrib
                T_list.append(T)
            return T_list

        def marginal(self, p, y_array):
            xp = self.xp
            B, n, K = p.shape
            d = self.d
            H = 2

            y_array = xp.asarray(y_array)
            T_list = self._build_transition_matrices(p, y_array)

            h_final = xp.array(
                [(int(y_array[i]) // (10**d)) % 10 for i in range(B)],
                dtype=xp.int32,
            )

            alpha = self._forward(T_list, h_final)
            beta = self._backward(T_list, h_final)

            b_idx = xp.arange(B)
            Z = alpha[b_idx, self.L, h_final]
            Z_safe = xp.where(Z > 1e-300, Z, xp.ones_like(Z))

            s = xp.zeros((B, d), dtype=xp.int32)
            for l in range(d):
                s[:, l] = (y_array // (10**l)) % 10

            p_a = p[:, :d, :]
            p_b = p[:, d:, :]
            q = xp.zeros((B, n, K), dtype=p.dtype)

            for l in range(d):
                for c_in in range(H):
                    for c_out in range(H):
                        target = s[:, l] + 10 * c_out - c_in
                        for a in range(K):
                            b_val = target - a
                            valid = (b_val >= 0) & (b_val < K)
                            b_clamped = xp.clip(b_val, 0, K - 1)
                            pb = p_b[b_idx, l, b_clamped]
                            w = xp.where(
                                valid,
                                alpha[:, l, c_in] * p_a[:, l, a] * pb
                                * beta[:, l + 1, c_out] / Z_safe,
                                0.0,
                            )
                            q[:, l, a] += w
                            for bi in range(B):
                                if valid[bi]:
                                    q[bi, d + l, int(b_clamped[bi])] += float(w[bi])

            return q, Z

except ImportError:
    pass
