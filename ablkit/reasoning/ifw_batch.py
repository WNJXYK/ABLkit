"""
Batched GPU-accelerated IFW-DP for chain/tree decompositions.

Supports revision-aware DP: tracks per-sample revision count as a tensor
dimension, enabling batched ABL (MAP with min-revision) and A3BL (marginal
with revision budget).

The engine is y-independent: transition structure precomputed once,
y only determines which root state is accepted at runtime.

Usage:
    engine = BatchDPEngine(decomp, K, device='cpu', root_output_fn=fn)
    # Marginal (A3BL):
    q, Z = engine.batch_marginal(y_batch, p_batch, pseudo_idx_batch, max_rev)
    # MAP (ABL):
    z, scores = engine.batch_map(y_batch, log_p_batch, pseudo_idx_batch, max_rev)
"""

import itertools
from typing import Any, List, Optional, Tuple

import torch

from .ifw_dp import Decomposition


class BatchDPEngine:
    """Batched revision-aware DP engine for chain/tree decompositions.

    State: alpha (N, H, R) — per-sample, per-CSS-state, per-revision-count.
    Revision count tracked via tensor slice-shift, fully batched.
    """

    def __init__(self, decomp: Decomposition, K: int, device: str = "cpu",
                 root_output_fn=None):
        self.decomp = decomp
        self.K = K
        self.n = decomp.n
        self.device = torch.device(device)
        self._root_output_fn = root_output_fn

        order = []
        visited = set()
        def _po(i):
            if i in visited: return
            visited.add(i)
            for c in decomp.children[i]:
                _po(c)
            order.append(i)
        _po(decomp.root)
        self._order = order

        self._precompute()

    def _precompute(self):
        decomp = self.decomp
        K = self.K
        root = decomp.root

        self.h_states = {}
        self.h_to_idx = {}
        self.trans = {}  # node -> list of (ch_idx_tuple, z_vals_tuple, h_next_idx, var_ids_for_z)
        node_state_lists = {}

        for node in self._order:
            ch = decomp.children[node]
            var_ids = decomp.var_groups[node]
            r = len(var_ids)

            if ch:
                child_combos = list(itertools.product(
                    *[node_state_lists[c] for c in ch]))
            else:
                child_combos = [()]

            states = {}
            transitions = []

            for h_combo in child_combos:
                for z_vals in itertools.product(range(K), repeat=r):
                    if node == root and self._root_output_fn is not None:
                        h = self._root_output_fn(h_combo, z_vals, node)
                    else:
                        h = decomp.transition_fn(h_combo, z_vals, node, 0)
                    if h is None:
                        continue
                    if h not in states:
                        states[h] = len(states)
                    ch_idx = tuple(self.h_to_idx[c][h_c]
                                   for c, h_c in zip(ch, h_combo)) if ch else ()
                    transitions.append((ch_idx, z_vals, states[h]))

            self.h_states[node] = list(states.keys())
            self.h_to_idx[node] = states
            self.trans[node] = transitions
            node_state_lists[node] = list(states.keys())

    def H(self, node):
        return len(self.h_states[node])

    # ================================================================
    #  Marginal DP (A3BL)
    # ================================================================

    def batch_marginal(
        self,
        y_batch: List[Any],
        p_batch: torch.Tensor,
        pseudo_idx_batch: Optional[torch.Tensor] = None,
        max_rev: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched marginal with optional revision constraint.

        Args:
            y_batch: (N,) target values.
            p_batch: (N, n, K) perception probabilities.
            pseudo_idx_batch: (N, n) pseudo label indices. Required if max_rev >= 0.
            max_rev: max revisions allowed (-1 = unlimited).

        Returns:
            q: (N, n, K) posterior marginals.
            Z: (N,) partition functions.
        """
        N = p_batch.shape[0]
        K = self.K; n = self.n; dev = self.device
        decomp = self.decomp; root = decomp.root
        use_rev = max_rev >= 0 and pseudo_idx_batch is not None
        R = max_rev + 1 if use_rev else 1

        # ── Forward ──
        alpha = {}
        for node in self._order:
            H_n = self.H(node)
            var_ids = decomp.var_groups[node]
            ch = decomp.children[node]
            a = torch.zeros(N, H_n, R, device=dev)

            for ch_idx, z_vals, h_next_idx in self.trans[node]:
                # Perception weight
                w = torch.ones(N, device=dev)
                for j, vid in enumerate(var_ids):
                    w = w * p_batch[:, vid, z_vals[j]]
                # Children alpha (sum over revision dim via outer product)
                if ch:
                    # For each child, get alpha slice: (N, R)
                    # We need to convolve revision counts across children
                    # Start with first child, then convolve with each subsequent
                    ch_alpha_r = alpha[ch[0]][:, ch_idx[0], :]  # (N, R)
                    for ci in range(1, len(ch)):
                        # Convolve revision dims: r_total = r_child0 + r_child1 + ...
                        # For small R this is fine with a loop
                        c = ch[ci]
                        other = alpha[c][:, ch_idx[ci], :]  # (N, R)
                        conv = torch.zeros(N, R, device=dev)
                        for r1 in range(R):
                            for r2 in range(R - r1):
                                conv[:, r1 + r2] += ch_alpha_r[:, r1] * other[:, r2]
                        ch_alpha_r = conv
                else:
                    ch_alpha_r = torch.zeros(N, R, device=dev)
                    ch_alpha_r[:, 0] = 1.0  # no children = 0 revisions

                # Revision increment at this node
                if use_rev:
                    delta = torch.zeros(N, dtype=torch.long, device=dev)
                    for j, vid in enumerate(var_ids):
                        delta += (pseudo_idx_batch[:, vid] != z_vals[j]).long()
                    # Group by delta value and shift
                    for d in range(len(var_ids) + 1):
                        mask = (delta == d)
                        if not mask.any():
                            continue
                        if R - d <= 0:
                            continue
                        a[mask, h_next_idx, d:] += ch_alpha_r[mask, :R-d] * w[mask, None]
                else:
                    a[:, h_next_idx, 0] += ch_alpha_r[:, 0] * w

            alpha[node] = a

        # Z per sample
        Z = torch.zeros(N, device=dev)
        h_final_indices = []
        for i, y_val in enumerate(y_batch):
            h_final = decomp.h_final_fn(y_val)
            idx = self.h_to_idx[root].get(h_final, -1)
            h_final_indices.append(idx)
            if 0 <= idx < alpha[root].shape[1]:
                Z[i] = alpha[root][i, idx, :].sum()  # sum over all valid revisions
        Z = Z.clamp(min=1e-30)

        # ── Backward + marginals ──
        H_root = self.H(root)
        beta = {}
        b_root = torch.zeros(N, H_root, R, device=dev)
        for i, idx in enumerate(h_final_indices):
            if 0 <= idx < H_root:
                b_root[i, idx, :] = 1.0  # accept any revision count
        beta[root] = b_root

        q = torch.zeros(N, n, K, device=dev)

        for node in reversed(self._order):
            if node not in beta:
                continue
            var_ids = decomp.var_groups[node]
            ch = decomp.children[node]

            for ch_idx, z_vals, h_next_idx in self.trans[node]:
                # local prob
                local_p = torch.ones(N, device=dev)
                for j, vid in enumerate(var_ids):
                    local_p = local_p * p_batch[:, vid, z_vals[j]]

                # Children alpha product (with revision convolution)
                if ch:
                    ch_alpha_r = alpha[ch[0]][:, ch_idx[0], :]
                    for ci in range(1, len(ch)):
                        c = ch[ci]
                        other = alpha[c][:, ch_idx[ci], :]
                        conv = torch.zeros(N, R, device=dev)
                        for r1 in range(R):
                            for r2 in range(R - r1):
                                conv[:, r1 + r2] += ch_alpha_r[:, r1] * other[:, r2]
                        ch_alpha_r = conv
                else:
                    ch_alpha_r = torch.zeros(N, R, device=dev)
                    ch_alpha_r[:, 0] = 1.0

                # Beta * alpha * local_p / Z, summed over revision dim
                if use_rev:
                    delta = torch.zeros(N, dtype=torch.long, device=dev)
                    for j, vid in enumerate(var_ids):
                        delta += (pseudo_idx_batch[:, vid] != z_vals[j]).long()

                    # w_total = sum_r [beta[h_next, r+delta] * ch_alpha[r] * local_p] / Z
                    for d in range(len(var_ids) + 1):
                        mask = (delta == d)
                        if not mask.any() or R - d <= 0:
                            continue
                        # beta_shifted: beta[:, h_next, d:] corresponds to r_after = r_before + d
                        b_shifted = beta[node][mask, h_next_idx, d:]  # (Nm, R-d)
                        ca = ch_alpha_r[mask, :R-d]                   # (Nm, R-d)
                        lp = local_p[mask, None]                      # (Nm, 1)
                        z_inv = (1.0 / Z[mask, None])                 # (Nm, 1)
                        w_per_r = b_shifted * ca * lp * z_inv         # (Nm, R-d)
                        w_total = w_per_r.sum(dim=1)                  # (Nm,)

                        for j, vid in enumerate(var_ids):
                            q[mask, vid, z_vals[j]] += w_total

                        # Propagate beta to children
                        for ci, c in enumerate(ch):
                            if c not in beta:
                                beta[c] = torch.zeros(N, self.H(c), R, device=dev)
                            # other children's alpha (excluding child ci)
                            other_a = local_p[mask, None]  # start with local_p
                            for cj, c2 in enumerate(ch):
                                if cj != ci:
                                    other_a = other_a * alpha[c2][mask, ch_idx[cj], :R-d]
                            # beta[c][:, h_c, r_before] += beta[node][:, h_next, r_before + d] * other_a
                            beta[c][mask, ch_idx[ci], :R-d] += b_shifted * other_a
                else:
                    b_h = beta[node][:, h_next_idx, 0]
                    ca = ch_alpha_r[:, 0]
                    w = b_h * ca * local_p / Z
                    for j, vid in enumerate(var_ids):
                        q[:, vid, z_vals[j]] += w
                    for ci, c in enumerate(ch):
                        if c not in beta:
                            beta[c] = torch.zeros(N, self.H(c), R, device=dev)
                        other = local_p.clone()
                        for cj, c2 in enumerate(ch):
                            if cj != ci:
                                other = other * alpha[c2][:, ch_idx[cj], 0]
                        beta[c][:, ch_idx[ci], 0] += b_h * other

        return q, Z

    # ================================================================
    #  MAP DP (ABL)
    # ================================================================

    def batch_map(
        self,
        y_batch: List[Any],
        log_p_batch: torch.Tensor,
        pseudo_idx_batch: Optional[torch.Tensor] = None,
        max_rev: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched MAP with optional revision constraint.

        With revision: finds the solution with minimum revisions first,
        then highest perception score as tiebreak (matches base ABL).

        Args:
            y_batch: (N,) target values.
            log_p_batch: (N, n, K) log perception probabilities.
            pseudo_idx_batch: (N, n) pseudo label indices.
            max_rev: max revisions (-1 = unlimited).

        Returns:
            z_hat: (N, n) best assignments.
            scores: (N,) best scores.
        """
        N = log_p_batch.shape[0]
        n = self.n; dev = self.device
        decomp = self.decomp; root = decomp.root
        NEG_INF = -1e30
        use_rev = max_rev >= 0 and pseudo_idx_batch is not None
        R = max_rev + 1 if use_rev else 1

        # Forward: scores (N, H, R), backpointers
        scores_dict = {}
        bp_z = {}
        bp_ch = {}

        for node in self._order:
            H_n = self.H(node)
            var_ids = decomp.var_groups[node]
            r_step = len(var_ids)
            ch = decomp.children[node]

            sc = torch.full((N, H_n, R), NEG_INF, device=dev)
            bz = torch.zeros(N, H_n, R, max(r_step, 1), dtype=torch.long, device=dev)
            bh = {}  # h_next_idx -> {rev_level -> ch_idx}

            for ch_idx, z_vals, h_next_idx in self.trans[node]:
                # Log-perception score
                s_local = torch.zeros(N, device=dev)
                for j, vid in enumerate(var_ids):
                    s_local = s_local + log_p_batch[:, vid, z_vals[j]]

                # Children scores (with revision convolution for MAP = max over r splits)
                if ch:
                    # For MAP with revision: need to find best split of r across children
                    # Simple approach: iterate over r splits (small R)
                    ch_score_r = scores_dict[ch[0]][:, ch_idx[0], :]  # (N, R)
                    for ci in range(1, len(ch)):
                        c = ch[ci]
                        other = scores_dict[c][:, ch_idx[ci], :]
                        conv = torch.full((N, R), NEG_INF, device=dev)
                        for r1 in range(R):
                            for r2 in range(R - r1):
                                cand = ch_score_r[:, r1] + other[:, r2]
                                conv[:, r1 + r2] = torch.max(conv[:, r1 + r2], cand)
                        ch_score_r = conv
                else:
                    ch_score_r = torch.full((N, R), NEG_INF, device=dev)
                    ch_score_r[:, 0] = 0.0

                if use_rev:
                    delta = torch.zeros(N, dtype=torch.long, device=dev)
                    for j, vid in enumerate(var_ids):
                        delta += (pseudo_idx_batch[:, vid] != z_vals[j]).long()

                    for d in range(r_step + 1):
                        mask = (delta == d)
                        if not mask.any() or R - d <= 0:
                            continue
                        # score at revision r = ch_score[r - d] + s_local
                        cand = ch_score_r[mask, :R-d] + s_local[mask, None]  # (Nm, R-d)
                        old = sc[mask, h_next_idx, d:]
                        better = cand > old
                        sc[mask, h_next_idx, d:] = torch.where(better, cand, old)
                        for j in range(r_step):
                            bz[mask, h_next_idx, d:, j] = torch.where(
                                better,
                                torch.tensor(z_vals[j], device=dev, dtype=torch.long).expand_as(better),
                                bz[mask, h_next_idx, d:, j])
                        if better.any():
                            bh[(h_next_idx, d)] = ch_idx
                else:
                    cand = ch_score_r[:, 0] + s_local
                    better = cand > sc[:, h_next_idx, 0]
                    sc[:, h_next_idx, 0] = torch.where(better, cand, sc[:, h_next_idx, 0])
                    for j in range(r_step):
                        bz[:, h_next_idx, 0, j] = torch.where(
                            better,
                            torch.tensor(z_vals[j], device=dev, dtype=torch.long),
                            bz[:, h_next_idx, 0, j])
                    bh[(h_next_idx, 0)] = ch_idx

            scores_dict[node] = sc
            bp_z[node] = bz
            bp_ch[node] = bh

        # Root: per sample, find h_final, pick min revision with best score
        z_hat = torch.zeros(N, n, dtype=torch.long, device=dev)
        best_scores = torch.full((N,), NEG_INF, device=dev)

        for i, y_val in enumerate(y_batch):
            h_final = decomp.h_final_fn(y_val)
            if h_final not in self.h_to_idx[root]:
                continue
            fidx = self.h_to_idx[root][h_final]
            # Pick minimum revision with valid score
            for r in range(R):
                if scores_dict[root][i, fidx, r] > NEG_INF + 1:
                    best_scores[i] = scores_dict[root][i, fidx, r]
                    # Traceback from (root, fidx, r)
                    self._traceback_map(i, root, fidx, r, z_hat, bp_z, bp_ch, scores_dict)
                    break

        return z_hat, best_scores

    def _traceback_map(self, sample_idx, node, h_idx, rev, z_hat, bp_z, bp_ch, scores_dict):
        """Traceback for a single sample."""
        decomp = self.decomp
        var_ids = decomp.var_groups[node]
        for j, vid in enumerate(var_ids):
            z_hat[sample_idx, vid] = bp_z[node][sample_idx, h_idx, rev, j]

        # Find which delta and ch_idx were used
        # Try each possible delta
        ch = decomp.children[node]
        if not ch:
            return
        for d in range(len(var_ids) + 1):
            key = (h_idx, d)
            if key in bp_ch[node] and rev >= d:
                ch_idx = bp_ch[node][key]
                # Distribute remaining revision (rev - d) across children
                # Simple: give all to first child for now (TODO: proper split tracking)
                r_remaining = rev - d
                for ci, c in enumerate(ch):
                    # Find best r for this child
                    for rc in range(r_remaining + 1):
                        if scores_dict[c][sample_idx, ch_idx[ci], rc] > -1e30 + 1:
                            self._traceback_map(sample_idx, c, ch_idx[ci], rc, z_hat, bp_z, bp_ch, scores_dict)
                            r_remaining -= rc
                            break
                return
