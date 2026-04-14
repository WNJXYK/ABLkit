"""
Batched GPU/CPU DP for dense IFW decompositions.

Accelerates dp_marginal_revision by processing an entire batch of samples
simultaneously using torch tensor operations.  Only works for *dense*
decompositions where every node has a bounded CSS domain.

Sparse decompositions (H=0, unbounded states) fall back to the sequential
Python implementation in ifw_dp.py.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from .ifw_dp import Decomposition

# ------------------------------------------------------------------ #
#  Precomputed batch data                                             #
# ------------------------------------------------------------------ #


@dataclass
class BatchDecompData:
    """Precomputed transition tables for batched DP."""

    node_order: List[int]
    num_nodes: int
    root: int
    var_ids: List[List[int]]
    children: List[List[int]]
    h_sizes: List[int]         # number of CSS states per node
    h_maps: List[Dict]         # CSS object -> int, per node
    z_combos: List[torch.Tensor]  # [n_combos, m] per node
    n_combos: List[int]
    # trans[node]: [n_y, n_hch, n_c] -> h_idx or -1
    trans: List[torch.Tensor]
    unique_ys: List[Any]
    y_to_idx: Dict[Any, int]
    h_final_idx: torch.Tensor  # [n_unique_y]
    h_children_flat_size: List[int]


def _postorder(decomp: Decomposition) -> List[int]:
    order, visited = [], set()

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for c in decomp.children[node]:
            dfs(c)
        order.append(node)

    dfs(decomp.root)
    return order


def precompute_batch_tables(
    decomp: Decomposition,
    K: int,
    unique_ys: List[Any],
) -> BatchDecompData:
    """Build dense transition tables by calling transition_fn exhaustively."""
    num_nodes = len(decomp.var_groups)
    node_order = _postorder(decomp)
    y_to_idx = {y: i for i, y in enumerate(unique_ys)}
    n_y = len(unique_ys)

    h_maps = [None] * num_nodes
    h_sizes = [0] * num_nodes
    z_combos_list = [None] * num_nodes
    n_combos_list = [0] * num_nodes
    trans_list = [None] * num_nodes
    h_children_flat_size = [0] * num_nodes

    for node in node_order:
        var_ids = decomp.var_groups[node]
        ch = decomp.children[node]
        m = len(var_ids)

        # z-combos
        z_combos_py = list(itertools.product(*[range(K)] * m)) if m > 0 else [()]
        n_c = len(z_combos_py)
        n_combos_list[node] = n_c
        z_combos_list[node] = torch.tensor(z_combos_py, dtype=torch.long) if m > 0 \
            else torch.zeros(1, 0, dtype=torch.long)

        # h_children combos — use integer indices, build inverse maps
        if ch:
            ch_h_ranges = [range(h_sizes[c]) for c in ch]
            ch_combos_int = list(itertools.product(*ch_h_ranges))
            ch_inv_maps = [{v: k for k, v in h_maps[c].items()} for c in ch]
        else:
            ch_combos_int = [()]
            ch_inv_maps = []

        n_hch = len(ch_combos_int)
        h_children_flat_size[node] = n_hch

        # Discover CSS states and build transition tensor
        node_h_map: Dict[Any, int] = {}
        trans_raw = {}  # (yi, hch_idx, ci) -> h_idx

        for yi, y in enumerate(unique_ys):
            for hch_idx, hch_ints in enumerate(ch_combos_int):
                h_combo_objs = tuple(ch_inv_maps[ci][hch_ints[ci]]
                                     for ci in range(len(ch))) if ch else ()
                for ci, z_vals in enumerate(z_combos_py):
                    h = decomp.transition_fn(h_combo_objs, z_vals, node, y)
                    if h is None:
                        continue
                    if h not in node_h_map:
                        node_h_map[h] = len(node_h_map)
                    trans_raw[(yi, hch_idx, ci)] = node_h_map[h]

        h_maps[node] = node_h_map
        h_sizes[node] = max(len(node_h_map), 1)

        trans = torch.full((n_y, n_hch, n_c), -1, dtype=torch.long)
        for (yi, hch_idx, ci), h_idx in trans_raw.items():
            trans[yi, hch_idx, ci] = h_idx
        trans_list[node] = trans

    # h_final per unique y
    root = decomp.root
    h_final_idx = torch.full((n_y,), -1, dtype=torch.long)
    for yi, y in enumerate(unique_ys):
        hf = decomp.h_final_fn(y)
        if hf in h_maps[root]:
            h_final_idx[yi] = h_maps[root][hf]

    return BatchDecompData(
        node_order=node_order, num_nodes=num_nodes, root=root,
        var_ids=decomp.var_groups, children=decomp.children,
        h_sizes=h_sizes, h_maps=h_maps,
        z_combos=z_combos_list, n_combos=n_combos_list,
        trans=trans_list, unique_ys=unique_ys, y_to_idx=y_to_idx,
        h_final_idx=h_final_idx, h_children_flat_size=h_children_flat_size,
    )


# ------------------------------------------------------------------ #
#  Core batch DP                                                      #
# ------------------------------------------------------------------ #


def batch_dp_marginal_revision(
    bd: BatchDecompData,
    K: int,
    y_batch: List[Any],
    p_batch: torch.Tensor,       # [B, n, K]
    pseudo_idx_batch: torch.Tensor,  # [B, n] int
    max_revision: int,
    require_more_revision: int = 0,
    return_alpha: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched dp_marginal_revision for dense decompositions.

    Returns:
        q: [B, n, K] posterior marginals.
        Z: [B] partition functions.
    """
    B, n = p_batch.shape[0], p_batch.shape[1]
    R = max_revision
    device = p_batch.device
    dtype = p_batch.dtype
    b_arange = torch.arange(B, device=device)

    y_idx = torch.tensor([bd.y_to_idx[y] for y in y_batch],
                         dtype=torch.long, device=device)

    # ── Upward pass ──
    alpha = [None] * bd.num_nodes

    for node in bd.node_order:
        alpha[node] = _forward_node(
            bd, node, alpha, p_batch, pseudo_idx_batch, y_idx,
            B, R, K, device, dtype, b_arange
        )

    # ── Partition function with early termination ──
    root = bd.root
    h_final_per_sample = bd.h_final_idx.to(device)[y_idx]  # [B]
    valid_hf = h_final_per_sample >= 0
    hf_safe = h_final_per_sample.clamp(min=0)

    # root_alpha_at_hf[b, r] = alpha[root][b, r, h_final[b]]
    root_alpha_at_hf = alpha[root][b_arange, :, hf_safe]  # [B, R+1]

    # Per-sample early termination: find min budget with valid solution
    has_solution = root_alpha_at_hf > 0  # [B, R+1]
    # min_rev_found[b] = first r where has_solution[b, r], or R+1 if none
    r_grid = torch.arange(R + 1, device=device).unsqueeze(0)  # [1, R+1]
    # Set non-solution positions to R+1 so argmin finds only solution positions
    r_masked = torch.where(has_solution, r_grid.expand(B, -1),
                           torch.full_like(r_grid.expand(B, -1), R + 1))
    min_rev_found = r_masked.min(dim=1).values  # [B]
    effective_R = torch.clamp(min_rev_found + require_more_revision, max=R)  # [B]

    # Budget mask: include budgets 0..effective_R[b]
    budget_mask = (r_grid <= effective_R.unsqueeze(1)) & valid_hf.unsqueeze(1)  # [B, R+1]

    Z = (root_alpha_at_hf * budget_mask.to(dtype)).sum(dim=1)

    # ── Downward pass ──
    beta = [None] * bd.num_nodes
    beta_root = torch.zeros(B, R + 1, bd.h_sizes[root], device=device, dtype=dtype)
    # beta[root][b, r, h_final[b]] = 1.0 for active budgets
    if valid_hf.any():
        beta_root[b_arange, :, hf_safe] = budget_mask.to(dtype)
    beta[root] = beta_root

    q = torch.zeros(B, n, K, device=device, dtype=dtype)

    for node in reversed(bd.node_order):
        if beta[node] is None:
            continue
        _backward_node(
            bd, node, alpha, beta, q, p_batch, pseudo_idx_batch, y_idx,
            Z, B, R, K, device, dtype, b_arange
        )

    if not return_alpha:
        return q, Z

    # Convert batch alpha tensors to per-sample CPU dict format
    # for use with dp_enumerate_topk.
    alpha_cpu = _batch_alpha_to_dicts(bd, alpha, min_rev_found, effective_R, B)
    min_rev_np = min_rev_found.cpu().numpy()
    return q, Z, alpha_cpu, min_rev_np


def _batch_alpha_to_dicts(bd, alpha, min_rev_found, effective_R, B):
    """Convert GPU batch alpha tensors to per-sample CPU dict format.

    Returns: list of B items, each is alpha_dict[node][r] = {h_obj: prob}.
    Only includes budgets from 0 to effective_R[b] for each sample.
    """
    import numpy as np

    num_nodes = bd.num_nodes
    # Inverse maps: int -> CSS object
    h_inv_maps = [{v: k for k, v in bd.h_maps[nd].items()} for nd in range(num_nodes)]

    # Transfer to CPU
    alpha_np = [alpha[nd].cpu().numpy() for nd in range(num_nodes)]
    eff_R_np = effective_R.cpu().numpy()

    result = []
    for b in range(B):
        max_r = int(eff_R_np[b])
        sample_alpha = {}
        for nd in range(num_nodes):
            node_alpha = []
            a_nd = alpha_np[nd]  # [B, R+1, H]
            inv_map = h_inv_maps[nd]
            for r in range(max_r + 1):
                d = {}
                for h_idx in range(a_nd.shape[2]):
                    val = float(a_nd[b, r, h_idx])
                    if val > 0 and h_idx in inv_map:
                        d[inv_map[h_idx]] = val
                node_alpha.append(d)
            sample_alpha[nd] = node_alpha
        result.append(sample_alpha)
    return result


# ------------------------------------------------------------------ #
#  Forward node                                                       #
# ------------------------------------------------------------------ #


def _forward_node(bd, node, alpha, p_batch, pseudo_idx_batch, y_idx,
                  B, R, K, device, dtype, b_arange):
    """Compute alpha[node]: [B, R+1, H_node]."""
    var_ids = bd.var_ids[node]
    ch = bd.children[node]
    H_node = bd.h_sizes[node]
    n_c = bd.n_combos[node]
    n_hch = bd.h_children_flat_size[node]
    z_combos = bd.z_combos[node].to(device)
    trans_batch = bd.trans[node].to(device)[y_idx]  # [B, n_hch, n_c]

    # Children r-convolution
    combined = _convolve_children(alpha, ch, bd.h_sizes, R, B, device, dtype)
    # combined: [B, R+1, n_hch]

    # Local perception and delta
    local_prob, delta = _compute_local_and_delta(
        p_batch, pseudo_idx_batch, var_ids, z_combos, B, K, device
    )

    # Fully vectorized accumulation
    alpha_node = torch.zeros(B, R + 1, H_node, device=device, dtype=dtype)

    # w[b, hch, c, r] = combined[b, r, hch] * local_prob[b, c]
    # combined: [B, R+1, n_hch] → permute → [B, n_hch, R+1] → [B, n_hch, 1, R+1]
    # local_prob: [B, n_c] → [B, 1, n_c, 1]
    w = (combined.permute(0, 2, 1).unsqueeze(2)
         * local_prob.unsqueeze(1).unsqueeze(3))  # [B, n_hch, n_c, R+1]

    # Target revision: r + delta[b, c]
    r_grid = torch.arange(R + 1, device=device)  # [R+1]
    # delta: [B, n_c] → [B, 1, n_c, 1]
    target_r = r_grid + delta.unsqueeze(1).unsqueeze(3)  # [B, 1, n_c, R+1]

    # Valid mask: transition exists AND target_r <= R
    trans_valid = trans_batch >= 0  # [B, n_hch, n_c]
    mask = trans_valid.unsqueeze(3) & (target_r <= R)  # [B, n_hch, n_c, R+1]

    # Gather flat indices where mask is True
    h_next_safe = trans_batch.clamp(min=0)  # [B, n_hch, n_c]

    # Expand to 4D for indexing
    B_grid = torch.arange(B, device=device).view(B, 1, 1, 1)
    h_4d = h_next_safe.unsqueeze(3).expand_as(w)  # [B, n_hch, n_c, R+1]
    tr_4d = target_r.expand_as(w).long()  # [B, n_hch, n_c, R+1]
    b_4d = B_grid.expand_as(w)  # [B, n_hch, n_c, R+1]

    sel = mask.reshape(-1).nonzero(as_tuple=True)[0]
    if sel.numel() > 0:
        alpha_node.index_put_(
            (b_4d.reshape(-1)[sel], tr_4d.reshape(-1)[sel], h_4d.reshape(-1)[sel]),
            w.reshape(-1)[sel],
            accumulate=True,
        )

    return alpha_node


# ------------------------------------------------------------------ #
#  Backward node                                                      #
# ------------------------------------------------------------------ #


def _backward_node(bd, node, alpha, beta, q, p_batch, pseudo_idx_batch,
                   y_idx, Z, B, R, K, device, dtype, b_arange):
    """Backward pass for one node: compute marginals and propagate beta."""
    var_ids = bd.var_ids[node]
    ch = bd.children[node]
    H_node = bd.h_sizes[node]
    n_c = bd.n_combos[node]
    n_hch = bd.h_children_flat_size[node]
    z_combos = bd.z_combos[node].to(device)
    trans_batch = bd.trans[node].to(device)[y_idx]

    local_prob, delta = _compute_local_and_delta(
        p_batch, pseudo_idx_batch, var_ids, z_combos, B, K, device
    )

    combined = _convolve_children(alpha, ch, bd.h_sizes, R, B, device, dtype)

    # Initialize children beta
    for c_node in ch:
        if beta[c_node] is None:
            beta[c_node] = torch.zeros(B, R + 1, bd.h_sizes[c_node], device=device, dtype=dtype)

    Z_safe = Z.clamp(min=1e-30)
    m = len(var_ids)

    # Vectorized backward: compute all (b, hch, c, r_src) at once
    # r_src is the source revision for children, r_total = r_src + delta
    r_src = torch.arange(R + 1, device=device)  # [R+1]
    # delta: [B, n_c] → [B, 1, n_c, 1]
    r_total = r_src + delta.unsqueeze(1).unsqueeze(3)  # [B, 1, n_c, R+1]

    # Validity mask
    trans_valid = trans_batch >= 0  # [B, n_hch, n_c]
    mask = trans_valid.unsqueeze(3) & (r_total <= R)  # [B, n_hch, n_c, R+1]

    h_next_safe = trans_batch.clamp(min=0)  # [B, n_hch, n_c]

    # Gather beta[node][b, r_total, h_next] for all positions
    # beta[node]: [B, R+1, H_node]
    # We need: [B, n_hch, n_c, R+1]
    B_grid = torch.arange(B, device=device).view(B, 1, 1, 1)
    rt_4d = r_total.expand(B, n_hch, n_c, R + 1).clamp(max=R).long()
    h_4d = h_next_safe.unsqueeze(3).expand(B, n_hch, n_c, R + 1)
    b_4d = B_grid.expand(B, n_hch, n_c, R + 1)

    beta_vals = beta[node][b_4d.reshape(-1), rt_4d.reshape(-1), h_4d.reshape(-1)]
    beta_vals = beta_vals.reshape(B, n_hch, n_c, R + 1) * mask.to(dtype)

    # combined: [B, R+1, n_hch] → [B, n_hch, 1, R+1]
    comb_4d = combined.permute(0, 2, 1).unsqueeze(2)
    # local_prob: [B, n_c] → [B, 1, n_c, 1]
    lp_4d = local_prob.unsqueeze(1).unsqueeze(3)

    # w[b, hch, c, r_src] = beta_vals * combined * local_prob / Z
    w = beta_vals * comb_4d * lp_4d / Z_safe.view(B, 1, 1, 1)

    # ── Accumulate marginals ──
    if m > 0:
        # w_sum[b, hch, c] = sum_{r_src} w[b, hch, c, r_src]
        # → marginal_w[b, c] = sum_{hch} w_sum[b, hch, c]
        marginal_w = w.sum(dim=(1, 3))  # [B, n_c]

        # q[b, var_ids[j], z_combos[c, j]] += marginal_w[b, c]
        for j in range(m):
            vid = var_ids[j]
            z_k = z_combos[:, j]  # [n_c]
            # scatter_add: q[:, vid, :] at index z_k
            q[:, vid, :].scatter_add_(1, z_k.unsqueeze(0).expand(B, -1), marginal_w)

    # ── Propagate beta to children ──
    if ch:
        ch_h_sizes = [bd.h_sizes[c] for c in ch]
        hch_to_child_h = _decode_flat_indices(n_hch, ch_h_sizes)

        # beta_contrib[b, hch, c, r_src] = beta_vals * local_prob (no combined, no /Z)
        beta_contrib = beta_vals * lp_4d  # [B, n_hch, n_c, R+1]
        # Sum over z-combos: [B, n_hch, R+1]
        bc_sum = beta_contrib.sum(dim=2)

        if len(ch) == 1:
            # Chain case: beta[child][b, r_src, h_child] += bc_sum
            c_node = ch[0]
            h_child_per_hch = torch.tensor([hch_to_child_h[hch][0] for hch in range(n_hch)],
                                           device=device, dtype=torch.long)
            for hch in range(n_hch):
                h_c = h_child_per_hch[hch].item()
                beta[c_node][:, :R + 1, h_c] += bc_sum[:, hch, :]
        else:
            # General tree case: propagate to each child ci, weighting by
            # the OTHER children's alpha (r-deconvolution).
            #
            # Correct formula for child ci at (r_ci, h_ci):
            #   beta[ci][b, r_ci, h_ci] += sum_{h_other, r_other}
            #       bc_sum[b, hch(h_ci, h_other), r_ci+r_other]
            #       * alpha_other[b, r_other, h_other]
            #
            # where alpha_other = convolution of alpha for all children != ci.

            for ci, c_node in enumerate(ch):
                H_ci = bd.h_sizes[c_node]
                other_ch = [ch[j] for j in range(len(ch)) if j != ci]
                other_alpha = _convolve_children(alpha, other_ch, bd.h_sizes,
                                                 R, B, device, dtype)
                # other_alpha: [B, R+1, H_other_flat]
                H_other = other_alpha.shape[2]

                # Build hch lookup: for each (h_ci, h_other_flat) -> hch
                # Use _decode_flat_indices to get per-child h values from hch,
                # then re-encode the "other" children into a flat index.
                other_h_sizes = [bd.h_sizes[c] for c in other_ch]
                hch_lookup = torch.full((H_ci, H_other), -1, dtype=torch.long,
                                        device=device)
                for hch_idx in range(n_hch):
                    decoded = hch_to_child_h[hch_idx]
                    h_ci_val = decoded[ci]
                    # Encode other children's h values as flat index
                    other_flat = 0
                    for j in range(len(ch)):
                        if j != ci:
                            other_flat = other_flat * bd.h_sizes[ch[j]] + decoded[j]
                    hch_lookup[h_ci_val, other_flat] = hch_idx

                # Reorganize bc_sum: [B, H_ci, H_other, R+1]
                bc_reorg = torch.zeros(B, H_ci, H_other, R + 1,
                                       device=device, dtype=dtype)
                valid_pairs = hch_lookup >= 0
                for h_ci_val in range(H_ci):
                    for h_oth in range(H_other):
                        hch_idx = hch_lookup[h_ci_val, h_oth].item()
                        if hch_idx >= 0:
                            bc_reorg[:, h_ci_val, h_oth, :] = bc_sum[:, hch_idx, :]

                # R-deconvolution: beta[ci][b, r_ci, h_ci] +=
                #   sum_{r_other, h_other} bc_reorg[b, h_ci, h_other, r_ci+r_other]
                #                          * other_alpha[b, r_other, h_other]
                beta_new = torch.zeros(B, R + 1, H_ci, device=device, dtype=dtype)
                for r_ci in range(R + 1):
                    for r_other in range(R + 1 - r_ci):
                        r_src = r_ci + r_other
                        # bc_reorg[:, :, :, r_src]: [B, H_ci, H_other]
                        # other_alpha[:, r_other, :]: [B, H_other]
                        # Sum over H_other, get [B, H_ci]
                        beta_new[:, r_ci, :] += (
                            bc_reorg[:, :, :, r_src] *
                            other_alpha[:, r_other, :].unsqueeze(1)
                        ).sum(dim=2)

                beta[c_node] += beta_new


# ------------------------------------------------------------------ #
#  Utilities                                                          #
# ------------------------------------------------------------------ #


def _convolve_children(alpha_list, children, h_sizes, R, B, device, dtype):
    """Convolve children alpha tables over revision dimension.

    Returns [B, R+1, prod(H_children)] or [B, 1, 1] for leaves.
    """
    if not children:
        out = torch.zeros(B, R + 1, 1, device=device, dtype=dtype)
        out[:, 0, 0] = 1.0
        return out

    result = alpha_list[children[0]]  # [B, R+1, H0]

    for ci in range(1, len(children)):
        c = children[ci]
        child_alpha = alpha_list[c]
        H_prev, H_ci = result.shape[2], child_alpha.shape[2]
        new = torch.zeros(B, R + 1, H_prev * H_ci, device=device, dtype=dtype)
        for r1 in range(R + 1):
            a1 = result[:, r1, :]  # [B, H_prev]
            for r2 in range(R + 1 - r1):
                a2 = child_alpha[:, r2, :]  # [B, H_ci]
                outer = a1.unsqueeze(2) * a2.unsqueeze(1)  # [B, H_prev, H_ci]
                new[:, r1 + r2, :] += outer.reshape(B, -1)
        result = new

    return result


def _compute_local_and_delta(p_batch, pseudo_idx_batch, var_ids, z_combos,
                             B, K, device):
    """Compute local perception probs and hamming deltas for all z-combos.

    Returns:
        local_prob: [B, n_c]
        delta: [B, n_c] long
    """
    m = len(var_ids)
    n_c = z_combos.shape[0]

    if m == 0:
        return (torch.ones(B, 1, device=device),
                torch.zeros(B, 1, dtype=torch.long, device=device))

    var_t = torch.tensor(var_ids, dtype=torch.long, device=device)

    # local_prob[b, c] = prod_j p[b, var_ids[j], z_combos[c, j]]
    p_local = p_batch[:, var_t, :]  # [B, m, K]
    # Gather: p_local[b, j, z_combos[c, j]] for all (b, c, j)
    z_exp = z_combos.unsqueeze(0).expand(B, n_c, m)  # [B, n_c, m]
    p_exp = p_local.unsqueeze(1).expand(B, n_c, m, K)  # [B, n_c, m, K]
    p_gathered = torch.gather(p_exp, 3, z_exp.unsqueeze(3)).squeeze(3)  # [B, n_c, m]
    local_prob = p_gathered.prod(dim=2)  # [B, n_c]

    # delta[b, c] = sum_j (z_combos[c, j] != pseudo[b, var_ids[j]])
    pseudo_local = pseudo_idx_batch[:, var_t]  # [B, m]
    delta = (z_combos.unsqueeze(0) != pseudo_local.unsqueeze(1)).long().sum(dim=2)

    return local_prob, delta



def _decode_flat_indices(n_hch, ch_h_sizes):
    """Decode flat hch index to per-child h indices.

    Returns list of n_hch tuples, each of length n_children.
    """
    if not ch_h_sizes:
        return [() for _ in range(n_hch)]
    result = []
    for flat in range(n_hch):
        indices = []
        remaining = flat
        for H_ci in reversed(ch_h_sizes):
            indices.insert(0, remaining % H_ci)
            remaining //= H_ci
        result.append(tuple(indices))
    return result


def is_dense_decomp(decomp: Decomposition) -> bool:
    """Check if decomposition has bounded CSS domains (suitable for batch DP)."""
    info = getattr(decomp, '_batch_info', None)
    if info is None:
        return False
    css_sizes = info.get('css_domain_sizes', [])
    if not css_sizes:
        return False
    return all(s > 0 for s in css_sizes) and max(css_sizes) < 1000
