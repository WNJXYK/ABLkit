"""
Batched GPU-accelerated IFW-DP for chain/tree decompositions.

Replaces per-sample Python DP with batched tensor operations.
All N samples in a batch share the same decomposition structure;
only perception probabilities and y values differ.

The engine is y-independent: transition structure is precomputed once,
and y only determines which root states are accepted (h_final).

Usage:
    engine = BatchDPEngine(decomp, K, device='cpu')
    q_batch, Z_batch = engine.batch_marginal(y_batch, p_batch)
    z_batch, scores = engine.batch_map(y_batch, log_p_batch)
"""

import itertools
from typing import Any, List, Optional, Tuple

import torch

from .ifw_dp import Decomposition


class BatchDPEngine:
    """Batched DP engine for chain/tree decompositions.

    Precomputes transition structure ONCE (y-independent).
    At runtime, y only affects which root state is accepted.
    """

    def __init__(self, decomp: Decomposition, K: int, device: str = "cpu",
                 root_output_fn=None):
        """
        Args:
            decomp: IFW Decomposition.
            K: Domain size per variable.
            device: 'cpu' or 'cuda'.
            root_output_fn: Optional callable(h_children, z_vals, node) -> output.
                If provided, used for root instead of transition_fn (which does y-check).
                Should return the KB output value (without y filtering).
                If None, uses transition_fn with probe y values.
        """
        self.decomp = decomp
        self.K = K
        self.n = decomp.n
        self.device = torch.device(device)
        self._root_output_fn = root_output_fn

        # Post-order traversal
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
        """Enumerate all valid transitions per node (y-independent).

        Non-root nodes: transition_fn doesn't depend on y, use y=0 as probe.
        Root node: if root_output_fn is provided, use it to get KB output
        directly (bypassing y-check). Otherwise use transition_fn with y=0.
        Root states are KB output values; h_final matching done at runtime.
        """
        decomp = self.decomp
        K = self.K
        root = decomp.root

        self.h_states = {}
        self.h_to_idx = {}
        self.trans = {}
        node_state_lists = {}

        for node in self._order:
            ch = decomp.children[node]
            var_ids = decomp.var_groups[node]
            r = len(var_ids)

            if ch:
                child_combos = list(itertools.product(
                    *[node_state_lists[c] for c in ch]
                ))
            else:
                child_combos = [()]

            states = {}
            transitions = []

            for h_combo in child_combos:
                for z_vals in itertools.product(range(K), repeat=r):
                    if node == root and self._root_output_fn is not None:
                        # Use root_output_fn: returns KB output without y-check
                        h = self._root_output_fn(h_combo, z_vals, node)
                    else:
                        h = decomp.transition_fn(h_combo, z_vals, node, 0)
                    if h is None:
                        continue
                    if h not in states:
                        states[h] = len(states)
                    ch_idx = tuple(self.h_to_idx[c][h_c] for c, h_c in zip(ch, h_combo)) if ch else ()
                    transitions.append((ch_idx, z_vals, states[h]))

            self.h_states[node] = list(states.keys())
            self.h_to_idx[node] = states
            self.trans[node] = transitions
            node_state_lists[node] = list(states.keys())

    def H(self, node):
        return len(self.h_states[node])

    def batch_marginal(
        self, y_batch: List[Any], p_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched marginal computation.

        Args:
            y_batch: list of N target values (can differ per sample).
            p_batch: (N, n, K) perception probabilities.

        Returns:
            q: (N, n, K) posterior marginals.
            Z: (N,) partition functions.
        """
        N = p_batch.shape[0]
        K = self.K
        n = self.n
        dev = self.device
        decomp = self.decomp
        root = decomp.root

        # ── Upward (forward) ──
        alpha = {}
        for node in self._order:
            H_n = self.H(node)
            var_ids = decomp.var_groups[node]
            ch = decomp.children[node]

            a = torch.zeros(N, H_n, device=dev)
            for ch_idx, z_vals, h_next_idx in self.trans[node]:
                w = torch.ones(N, device=dev)
                for j, vid in enumerate(var_ids):
                    w = w * p_batch[:, vid, z_vals[j]]
                for ci, c in enumerate(ch):
                    w = w * alpha[c][:, ch_idx[ci]]
                a[:, h_next_idx] += w
            alpha[node] = a

        # Z: per-sample based on y
        Z = torch.zeros(N, device=dev)
        h_final_indices = []
        for i, y_val in enumerate(y_batch):
            h_final = decomp.h_final_fn(y_val)
            if h_final in self.h_to_idx[root]:
                idx = self.h_to_idx[root][h_final]
                if idx < alpha[root].shape[1]:
                    Z[i] = alpha[root][i, idx]
            h_final_indices.append(self.h_to_idx[root].get(h_final, -1))
        Z = Z.clamp(min=1e-30)

        # ── Downward (backward) + marginals ──
        H_root = alpha[root].shape[1]
        beta_root = torch.zeros(N, H_root, device=dev)
        for i, idx in enumerate(h_final_indices):
            if 0 <= idx < H_root:
                beta_root[i, idx] = 1.0

        beta = {root: beta_root}
        q = torch.zeros(N, n, K, device=dev)

        for node in reversed(self._order):
            if node not in beta:
                continue
            var_ids = decomp.var_groups[node]
            ch = decomp.children[node]

            for ch_idx, z_vals, h_next_idx in self.trans[node]:
                if h_next_idx >= beta[node].shape[1]:
                    continue
                b_h = beta[node][:, h_next_idx]

                local_p = torch.ones(N, device=dev)
                for j, vid in enumerate(var_ids):
                    local_p = local_p * p_batch[:, vid, z_vals[j]]

                ch_alpha_all = torch.ones(N, device=dev)
                for ci, c in enumerate(ch):
                    ch_alpha_all = ch_alpha_all * alpha[c][:, ch_idx[ci]]

                w = b_h * ch_alpha_all * local_p / Z

                for j, vid in enumerate(var_ids):
                    q[:, vid, z_vals[j]] += w

                for ci, c in enumerate(ch):
                    if c not in beta:
                        beta[c] = torch.zeros(N, self.H(c), device=dev)
                    other = local_p.clone()
                    for cj, c2 in enumerate(ch):
                        if cj != ci:
                            other = other * alpha[c2][:, ch_idx[cj]]
                    beta[c][:, ch_idx[ci]] += b_h * other

        return q, Z

    def batch_map(
        self, y_batch: List[Any], log_p_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched MAP computation.

        Args:
            y_batch: list of N target values.
            log_p_batch: (N, n, K) log perception probabilities.

        Returns:
            z_hat: (N, n) best assignments (indices).
            scores: (N,) best scores.
        """
        N = log_p_batch.shape[0]
        n = self.n
        dev = self.device
        decomp = self.decomp
        root = decomp.root
        NEG_INF = -1e30

        # Forward
        scores_dict = {}
        bp_z = {}
        bp_ch = {}

        for node in self._order:
            H_n = self.H(node)
            var_ids = decomp.var_groups[node]
            r = len(var_ids)
            ch = decomp.children[node]

            sc = torch.full((N, H_n), NEG_INF, device=dev)
            bz = torch.zeros(N, H_n, max(r, 1), dtype=torch.long, device=dev)
            bh = {}

            for ch_idx, z_vals, h_next_idx in self.trans[node]:
                s = torch.zeros(N, device=dev)
                for j, vid in enumerate(var_ids):
                    s = s + log_p_batch[:, vid, z_vals[j]]
                for ci, c in enumerate(ch):
                    s = s + scores_dict[c][:, ch_idx[ci]]

                better = s > sc[:, h_next_idx]
                sc[:, h_next_idx] = torch.where(better, s, sc[:, h_next_idx])
                for j in range(r):
                    bz[:, h_next_idx, j] = torch.where(
                        better, torch.tensor(z_vals[j], device=dev, dtype=torch.long),
                        bz[:, h_next_idx, j])
                bh[h_next_idx] = ch_idx

            scores_dict[node] = sc
            bp_z[node] = bz
            bp_ch[node] = bh

        # Traceback per sample
        z_hat = torch.zeros(N, n, dtype=torch.long, device=dev)
        best_scores = torch.full((N,), NEG_INF, device=dev)

        for i, y_val in enumerate(y_batch):
            h_final = decomp.h_final_fn(y_val)
            if h_final not in self.h_to_idx[root]:
                continue
            fidx = self.h_to_idx[root][h_final]
            if fidx >= scores_dict[root].shape[1]:
                continue
            best_scores[i] = scores_dict[root][i, fidx]

            def _tb(nd, h_idx):
                var_ids = decomp.var_groups[nd]
                for j, vid in enumerate(var_ids):
                    z_hat[i, vid] = bp_z[nd][i, h_idx, j]
                ch_idx_t = bp_ch[nd].get(h_idx, ())
                for ci, c in enumerate(decomp.children[nd]):
                    if ci < len(ch_idx_t):
                        _tb(c, ch_idx_t[ci])

            _tb(root, fidx)

        return z_hat, best_scores
