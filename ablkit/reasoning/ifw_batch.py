"""
Batched GPU-accelerated IFW-DP for chain/tree decompositions.

Replaces per-sample Python DP with batched tensor operations.
All N samples in a batch share the same decomposition structure;
only perception probabilities differ.

Usage:
    engine = BatchDPEngine(decomp, K, y, device='cuda')
    q_batch, Z_batch = engine.batch_marginal(p_batch)
    z_batch, scores = engine.batch_map(log_p_batch)
"""

import itertools
from typing import Any, List, Optional, Tuple

import torch

from .ifw_dp import Decomposition


class BatchDPEngine:
    """Batched DP engine for chain/tree decompositions.

    Precomputes transition structure once for a given (decomp, y),
    then runs batched forward/backward for any number of samples.
    """

    def __init__(self, decomp: Decomposition, K: int, y: Any, device: str = "cpu"):
        self.decomp = decomp
        self.K = K
        self.y = y
        self.n = decomp.n
        self.device = torch.device(device)

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

        # Precompute transitions
        self._precompute(y)

    def _precompute(self, y):
        decomp = self.decomp
        K = self.K

        # Per-node: states and transitions
        # Indexed by node id (not order position)
        self.h_states = {}   # node -> [h0, h1, ...]
        self.h_to_idx = {}   # node -> {h: idx}
        self.trans = {}       # node -> list of (child_h_indices_tuple, z_vals_tuple, h_next_idx)

        node_state_lists = {}  # node -> list of states

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
                    h = decomp.transition_fn(h_combo, z_vals, node, y)
                    if h is None:
                        continue
                    if h not in states:
                        states[h] = len(states)

                    # Child state indices
                    if ch:
                        ch_idx = tuple(self.h_to_idx[c][h_c] for c, h_c in zip(ch, h_combo))
                    else:
                        ch_idx = ()

                    transitions.append((ch_idx, z_vals, states[h]))

            self.h_states[node] = list(states.keys())
            self.h_to_idx[node] = states
            self.trans[node] = transitions
            node_state_lists[node] = list(states.keys())

        # Root final state
        root = decomp.root
        h_final = decomp.h_final_fn(y)
        self.h_final_idx = self.h_to_idx[root].get(h_final, -1)

    def H(self, node):
        return len(self.h_states[node])

    def batch_marginal(self, p_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched marginal computation.

        Args:
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
        alpha = {}  # node -> (N, H)
        for node in self._order:
            H_n = self.H(node)
            a = torch.zeros(N, H_n, device=dev)
            var_ids = decomp.var_groups[node]
            ch = decomp.children[node]

            for ch_idx, z_vals, h_next_idx in self.trans[node]:
                w = torch.ones(N, device=dev)
                for j, vid in enumerate(var_ids):
                    w = w * p_batch[:, vid, z_vals[j]]
                for ci, c in enumerate(ch):
                    w = w * alpha[c][:, ch_idx[ci]]
                a[:, h_next_idx] += w

            alpha[node] = a

        # Z
        if self.h_final_idx < 0:
            return torch.zeros(N, n, K, device=dev), torch.zeros(N, device=dev)
        Z = alpha[root][:, self.h_final_idx].clamp(min=1e-30)

        # ── Downward (backward) + marginals ──
        beta = {}
        H_root = self.H(root)
        b = torch.zeros(N, H_root, device=dev)
        b[:, self.h_final_idx] = 1.0
        beta[root] = b

        q = torch.zeros(N, n, K, device=dev)

        for node in reversed(self._order):
            if node not in beta:
                continue
            var_ids = decomp.var_groups[node]
            ch = decomp.children[node]

            for ch_idx, z_vals, h_next_idx in self.trans[node]:
                b_h = beta[node][:, h_next_idx]

                local_p = torch.ones(N, device=dev)
                for j, vid in enumerate(var_ids):
                    local_p = local_p * p_batch[:, vid, z_vals[j]]

                ch_alpha_all = torch.ones(N, device=dev)
                for ci, c in enumerate(ch):
                    ch_alpha_all = ch_alpha_all * alpha[c][:, ch_idx[ci]]

                w = b_h * ch_alpha_all * local_p / Z

                # Marginals
                for j, vid in enumerate(var_ids):
                    q[:, vid, z_vals[j]] += w

                # Beta to children
                for ci, c in enumerate(ch):
                    if c not in beta:
                        beta[c] = torch.zeros(N, self.H(c), device=dev)
                    other = local_p.clone()
                    for cj, c2 in enumerate(ch):
                        if cj != ci:
                            other = other * alpha[c2][:, ch_idx[cj]]
                    beta[c][:, ch_idx[ci]] += b_h * other

        return q, Z

    def batch_map(self, log_p_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched MAP computation.

        Args:
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
        scores_dict = {}     # node -> (N, H)
        bp_z = {}            # node -> (N, H, r) best z_vals
        bp_ch = {}           # node -> {h_idx: ch_idx_tuple}  (simplified)

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
                        bz[:, h_next_idx, j],
                    )
                bh[h_next_idx] = ch_idx  # last winner per state

            scores_dict[node] = sc
            bp_z[node] = bz
            bp_ch[node] = bh

        # Traceback
        z_hat = torch.zeros(N, n, dtype=torch.long, device=dev)
        if self.h_final_idx < 0:
            return z_hat, torch.full((N,), NEG_INF, device=dev)

        best_scores = scores_dict[root][:, self.h_final_idx]

        def _tb(nd, h_idx):
            var_ids = decomp.var_groups[nd]
            for j, vid in enumerate(var_ids):
                z_hat[:, vid] = bp_z[nd][:, h_idx, j]
            ch_idx = bp_ch[nd].get(h_idx, ())
            for ci, c in enumerate(decomp.children[nd]):
                if ci < len(ch_idx):
                    _tb(c, ch_idx[ci])

        _tb(root, self.h_final_idx)
        return z_hat, best_scores
