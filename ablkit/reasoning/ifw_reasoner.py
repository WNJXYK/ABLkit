"""
IFW Reasoners for ABL and A3BL pipelines.

  IFWReasoner       ->  ABL  (MAP DP)
  IFWA3BLReasoner   ->  A3BL (marginal DP, soft labels)

Supports chain and tree decompositions via unified dp_map/dp_marginal.
"""

import math
import statistics
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch


# ================================================================
#  Perception Health Monitor
# ================================================================

class PerceptionMonitor:
    """Monitors perception quality via var_domain statistics.

    Tracks per-batch statistics and exposes early-restart / early-stop signals.

    The causal chain being monitored:
        perception degrades → pred_prob flattens → var_domains expand
        → DP search space explodes → abduction quality drops → death spiral

    Signals:
        should_restart: avg_domain_size spiked vs historical best → perception degraded
                        OR domain stagnated above threshold for too long → never learned
        should_stop: avg_domain_size ≈ 1 and validity ≈ 1 sustained → converged
    """

    def __init__(
        self,
        enabled: bool = False,
        warmup: int = 3,
        restart_threshold: float = 3.0,
        stagnation_window: int = 10,
        stagnation_domain_min: float = 1.5,
        converge_window: int = 5,
        converge_domain_max: float = 1.5,
        converge_validity_min: float = 0.95,
    ):
        self.enabled = enabled
        self.warmup = warmup
        self.restart_threshold = restart_threshold
        self.stagnation_window = stagnation_window
        self.stagnation_domain_min = stagnation_domain_min
        self.converge_window = converge_window
        self.converge_domain_max = converge_domain_max
        self.converge_validity_min = converge_validity_min

        # Per-example accumulator within a batch
        self._batch_domain_sizes: List[float] = []
        self._batch_max_probs: List[float] = []
        self._batch_n_abduce = 0
        self._batch_n_valid = 0

        # Per-batch history
        self._history: List[dict] = []

    def record_abduce(self, var_domains, pred_prob, n, K, valid: bool):
        """Called after each abduce() to record per-example stats."""
        if not self.enabled:
            return
        # Domain size: how many candidates per variable after pruning
        if var_domains is not None:
            avg_ds = sum(len(d) for d in var_domains) / max(n, 1)
        else:
            avg_ds = float(K)  # no pruning = full domain
        self._batch_domain_sizes.append(avg_ds)

        # Max probability: how confident is perception
        avg_mp = sum(max(pred_prob[i]) for i in range(n)) / max(n, 1)
        self._batch_max_probs.append(avg_mp)

        self._batch_n_abduce += 1
        if valid:
            self._batch_n_valid += 1

    def end_batch(self):
        """Called after batch_abduce() to commit batch stats to history."""
        if not self.enabled or not self._batch_domain_sizes:
            self._batch_domain_sizes.clear()
            self._batch_max_probs.clear()
            self._batch_n_abduce = 0
            self._batch_n_valid = 0
            return

        entry = {
            "avg_domain_size": statistics.mean(self._batch_domain_sizes),
            "avg_max_prob": statistics.mean(self._batch_max_probs),
            "validity_ratio": self._batch_n_valid / max(self._batch_n_abduce, 1),
            "n_abduce": self._batch_n_abduce,
        }
        self._history.append(entry)

        self._batch_domain_sizes.clear()
        self._batch_max_probs.clear()
        self._batch_n_abduce = 0
        self._batch_n_valid = 0

    @property
    def should_restart(self) -> bool:
        """Perception degraded (spike) or stagnated (never learned).

        Spike: current domain > baseline * restart_threshold.
        Stagnation: domain stayed above stagnation_domain_min for
                    stagnation_window consecutive batches after warmup.
        """
        if not self.enabled or len(self._history) <= self.warmup:
            return False
        post_warmup = self._history[self.warmup:]

        # Spike detection: compare current vs median of recent history
        # (median is robust to lucky outliers, unlike min)
        recent_domains = [s["avg_domain_size"] for s in post_warmup[-self.stagnation_window:]]
        baseline = statistics.median(recent_domains)
        current = self._history[-1]["avg_domain_size"]
        if current > baseline * self.restart_threshold:
            return True

        # Stagnation detection
        if len(post_warmup) >= self.stagnation_window:
            recent = post_warmup[-self.stagnation_window:]
            if all(s["avg_domain_size"] > self.stagnation_domain_min for s in recent):
                return True

        return False

    @property
    def should_stop(self) -> bool:
        """Converged: domain ≈ 1 and validity ≈ 1 sustained over window."""
        if not self.enabled or len(self._history) < self.warmup + self.converge_window:
            return False
        recent = self._history[-self.converge_window:]
        return (
            all(s["avg_domain_size"] <= self.converge_domain_max for s in recent)
            and all(s["validity_ratio"] >= self.converge_validity_min for s in recent)
        )

    @property
    def last_stats(self) -> Optional[dict]:
        return self._history[-1] if self._history else None

    def reset(self):
        """Reset after a restart (keep config, clear history)."""
        self._history.clear()
        self._batch_domain_sizes.clear()
        self._batch_max_probs.clear()
        self._batch_n_abduce = 0
        self._batch_n_valid = 0

from ..data.structures import ListData
from ..reasoning import KBBase
from .reasoner import Reasoner
from .ifw_dp import Decomposition, dp_map, dp_marginal


# ================================================================
#  IFWKB — IFW as KB acceleration backend
# ================================================================

class IFWKB(KBBase):
    """KB wrapper that uses IFW-DP for fast candidate enumeration.

    Wraps an existing traceable KB and overrides ``abduce_candidates``
    to use dp_map instead of brute-force search. Returns a single MAP
    candidate, compatible with both Reasoner (ABL) and A3BLReasoner.

    Usage::

        inner_kb = MyTraceableKB()
        kb = IFWKB(inner_kb, perception_threshold=0.05)
        reasoner = IFWABLReasoner(kb)  # or Reasoner(kb) with manual pred_prob bridging

    Args:
        inner_kb: A KBBase subclass whose logic_forward supports TracedValue.
        perception_threshold: Pruning threshold for var_domains.
        max_states: Beam width for sparse DP.
        monitor: Enable PerceptionMonitor for warmup / restart signals.
        auto_decompose_num_samples: Samples for bottleneck discovery.
        auto_decompose_max_precompute: Budget for precomputation.
    """

    def __init__(
        self,
        inner_kb: KBBase,
        perception_threshold: float = 0.0,
        max_states: int = 100,
        revision_prior: float = 100.0,
        monitor: bool = False,
        auto_decompose_num_samples: int = 500,
        auto_decompose_max_precompute: int = 100_000,
    ):
        super().__init__(inner_kb.pseudo_label_list, inner_kb.max_err, use_cache=False)
        self._inner_kb = inner_kb
        self.perception_threshold = perception_threshold
        self.max_states = max_states
        self.revision_prior = revision_prior
        self.auto_decompose_num_samples = auto_decompose_num_samples
        self.auto_decompose_max_precompute = auto_decompose_max_precompute
        self._decomp_cache = {}
        self._rev_decomp_cache = {}
        self._rev_pseudo_idx = None  # set per-sample in abduce_candidates
        self._rev_max = 0
        self._pred_prob = None  # set by IFWABLReasoner before abduce_candidates
        self.monitor = PerceptionMonitor(enabled=monitor)

        # Label ↔ index mappings
        self.idx_to_label = {i: lbl for i, lbl in enumerate(self.pseudo_label_list)}
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.pseudo_label_list)}

    def logic_forward(self, z, x=None):
        if self._inner_kb._num_args == 2:
            return self._inner_kb.logic_forward(z, x)
        return self._inner_kb.logic_forward(z)

    def _get_decomp(self, n: int) -> Decomposition:
        if n in self._decomp_cache:
            return self._decomp_cache[n]
        from .tracer import discover_decomposition
        from ..utils import print_log

        extra_kw = {}
        kb = self._inner_kb
        if hasattr(kb, 'constraint_fn') and hasattr(kb, 'y_decompose_fn'):
            extra_kw['constraint_fn'] = kb.constraint_fn
            extra_kw['y_decompose_fn'] = kb.y_decompose_fn
            extra_kw['y_size'] = getattr(kb, 'y_size', 0)
            extra_kw['Y_domains'] = getattr(kb, 'Y_domains', None)

        decomp, info = discover_decomposition(
            kb.logic_forward,
            n=n,
            K=len(self.pseudo_label_list),
            num_samples=self.auto_decompose_num_samples,
            max_precompute=self.auto_decompose_max_precompute,
            max_states=self.max_states,
            **extra_kw,
        )
        if isinstance(info, dict):
            print_log(
                f"[IFWKB] Auto-decomposed n={n}: "
                f"{len(info.get('var_groups', []))} steps, "
                f"CSS domains={info.get('css_domain_sizes', [])}, "
                f"type={info.get('decomposition_type', '?')}",
                logger="current",
            )
        self._decomp_cache[n] = decomp
        return decomp

    @property
    def _in_warmup(self) -> bool:
        return self.monitor.enabled and len(self.monitor._history) < self.monitor.warmup

    def _build_revision_decomp(self, decomp):
        """Wrap decomp with revision-count tracking in CSS state.

        Augmented state: (h_original, n_revisions).
        Reads pseudo_idx and max_revision from self._rev_pseudo_idx and
        self._rev_max at call time — so the SAME decomp object can be
        reused across samples (just update self._rev_pseudo_idx).

        Complexity: H * (max_revision+1) states per node.
        """
        orig_tfn = decomp.transition_fn
        orig_hfinal = decomp.h_final_fn
        vg = decomp.var_groups
        _self = self  # capture IFWKB instance

        def transition_fn(h_children, z_vals, node, y):
            if h_children:
                h_orig = tuple(h[0] for h in h_children)
                n_rev = sum(h[1] for h in h_children)
            else:
                h_orig = ()
                n_rev = 0

            _pseudo = _self._rev_pseudo_idx
            _max_rev = _self._rev_max
            for j, vid in enumerate(vg[node]):
                if z_vals[j] != _pseudo[vid]:
                    n_rev += 1
                    if n_rev > _max_rev:
                        return None

            h_next = orig_tfn(h_orig, z_vals, node, y)
            if h_next is None:
                return None

            return (h_next, n_rev)

        _expected_final = [None]

        def h_final_accept(h):
            if not isinstance(h, tuple) or len(h) != 2:
                return False
            h_orig, n_rev = h
            if _expected_final[0] is not None and h_orig != _expected_final[0]:
                return False
            if n_rev > _self._rev_max:
                return False
            return n_rev + 1

        def h_final_fn(y):
            _expected_final[0] = orig_hfinal(y)
            return (_expected_final[0], 0)

        rev_decomp = Decomposition(
            n=decomp.n, var_groups=decomp.var_groups, children=decomp.children,
            root=decomp.root, transition_fn=transition_fn,
            h_final_fn=h_final_fn, H=0,
        )
        return rev_decomp, h_final_accept

    def abduce_candidates(self, pseudo_label, y, x, max_revision_num, require_more_revision):
        """IFW-accelerated candidate enumeration with revision-aware DP.

        Augments the DP state with a revision counter: (h, n_rev).
        Each step checks if z_vals differs from pseudo_label and increments.
        Paths exceeding max_revision are pruned. This gives exact base-ABL
        behavior (prefer fewest revisions) in a single O(L·K^r·H·R) dp_map
        call, where R = max_revision+1.

        Falls back to inner KB when pred_prob is unavailable.
        """
        if self._pred_prob is None:
            return self._inner_kb.abduce_candidates(
                pseudo_label, y, x, max_revision_num, require_more_revision,
            )

        n = len(pseudo_label)
        K = len(self.pseudo_label_list)
        decomp = self._get_decomp(n)
        pred_prob = _normalize_pred_prob(self._pred_prob, K)

        if self._in_warmup:
            var_domains = None
        else:
            var_domains = _compute_var_domains(
                pred_prob, n, K, 0, self.perception_threshold,
            )

        log_p = [
            [math.log(max(float(pred_prob[i][k]), 1e-30)) for k in range(K)]
            for i in range(n)
        ]

        pseudo_idx = [self.label_to_idx.get(lbl, 0) for lbl in pseudo_label]

        # Set per-sample revision state (read by cached transition_fn)
        self._rev_pseudo_idx = pseudo_idx
        self._rev_max = max_revision_num

        # Build revision-aware decomposition (cached per n)
        cache_key = n
        if cache_key not in self._rev_decomp_cache:
            self._rev_decomp_cache[cache_key] = self._build_revision_decomp(decomp)
        rev_decomp, h_accept = self._rev_decomp_cache[cache_key]

        rev_decomp.h_final_fn(y)

        z_hat, score = dp_map(
            rev_decomp, K, y, log_p,
            var_domains=var_domains,
            max_states=self.max_states,
            h_final_accept=h_accept,
        )

        valid = score > float("-inf")
        self.monitor.record_abduce(var_domains, pred_prob, n, K, valid)

        if not valid:
            return [], []

        candidate = [self.idx_to_label[z_hat[i]] for i in range(n)]
        res = self.logic_forward(candidate, x)

        if self._check_equal(res, y):
            return [candidate], [res]
        return [], []


class IFWABLReasoner(Reasoner):
    """Base ABL Reasoner with IFWKB acceleration.

    Per-sample: injects pred_prob into IFWKB, uses revision-aware dp_map
    via abduce_candidates. Inherits all selection logic from base Reasoner.

    Note: batch_abduce uses per-sample path (not BatchDPEngine) because
    revision-aware DP requires per-sample pseudo_label.
    """

    def abduce(self, data_example: ListData) -> List[Any]:
        self.kb._pred_prob = data_example.pred_prob
        result = super().abduce(data_example)
        self.kb._pred_prob = None
        return result

    def batch_abduce(self, data_examples: ListData) -> List[List[Any]]:
        """Per-sample MAP with revision-aware DP.

        Uses per-sample path because MAP traceback with batched revision
        split is complex. The abduce call itself is fast (cached decomp).
        """
        abduced = [self.abduce(ex) for ex in data_examples]
        data_examples.abduced_pseudo_label = abduced
        return abduced

    def _batch_abduce_TODO(self, data_examples: ListData) -> List[List[Any]]:
        """(TODO) Batched MAP with revision — needs proper traceback."""
        from .ifw_batch import BatchDPEngine
        from collections import defaultdict

        kb = self.kb
        K = len(kb.pseudo_label_list)
        n_examples = len(data_examples)
        if n_examples == 0:
            data_examples.abduced_pseudo_label = []
            return []

        all_log_p = []; all_y = []; all_n = []; all_pseudo = []
        for ex in data_examples:
            pred_prob = _normalize_pred_prob(ex.pred_prob, K)
            n = len(pred_prob)
            log_p = [[math.log(max(float(pred_prob[i][k]), 1e-30)) for k in range(K)]
                      for i in range(n)]
            all_log_p.append(log_p)
            all_y.append(ex.Y)
            all_n.append(n)
            all_pseudo.append([kb.label_to_idx.get(lbl, 0) for lbl in ex.pred_pseudo_label])

        groups = defaultdict(list)
        for i, n_i in enumerate(all_n):
            groups[n_i].append(i)

        results = [[] for _ in range(n_examples)]

        for n_i, indices in groups.items():
            decomp = kb._get_decomp(n_i)
            if not hasattr(kb, '_batch_engine_cache'):
                kb._batch_engine_cache = {}
            if n_i not in kb._batch_engine_cache:
                _kb = kb; _d = decomp; _reps = getattr(decomp, '_reps', {})
                def _root_output(h_combo, z_vals, node,
                                 __kb=_kb, __d=_d, __reps=_reps):
                    ch = __d.children[node]; z = [0] * __d.n
                    for ci, c in enumerate(ch):
                        rep = __reps.get((c, h_combo[ci]))
                        if rep:
                            for vid, val in rep.items(): z[vid] = val
                    for j, vid in enumerate(__d.var_groups[node]): z[vid] = z_vals[j]
                    labels = [__kb.idx_to_label[z[v]] for v in range(len(z))]
                    kb_out = __kb.logic_forward(labels)
                    return ("__target__", kb_out) if kb_out is not None else None
                kb._batch_engine_cache[n_i] = BatchDPEngine(
                    decomp, K, device='cpu', root_output_fn=_root_output)
            engine = kb._batch_engine_cache[n_i]

            log_p_batch = torch.tensor([all_log_p[i] for i in indices], dtype=torch.float32)
            y_list = [all_y[i] for i in indices]
            pseudo_idx_batch = torch.tensor([all_pseudo[i] for i in indices], dtype=torch.long)
            max_rev = self._get_max_revision_num(self.max_revision, n_i)

            z_batch, score_batch = engine.batch_map(
                y_list, log_p_batch, pseudo_idx_batch, max_rev)

            for gi, idx in enumerate(indices):
                score = score_batch[gi].item()
                if score <= -1e29:
                    results[idx] = []
                else:
                    z_hat = z_batch[gi].tolist()
                    results[idx] = [kb.idx_to_label[z_hat[v]] for v in range(n_i)]
                pred_prob = _normalize_pred_prob(data_examples[idx].pred_prob, K)
                nn = len(pred_prob)
                vd = None if kb._in_warmup else _compute_var_domains(
                    [[max(float(pred_prob[i][k]), 1e-30) for k in range(K)] for i in range(nn)],
                    nn, K, 0, kb.perception_threshold)
                kb.monitor.record_abduce(vd, pred_prob, nn, K, score > -1e29)

        data_examples.abduced_pseudo_label = results
        return results

    def end_loop(self):
        """Commit monitor stats for this loop."""
        if hasattr(self.kb, 'monitor'):
            self.kb.monitor.end_batch()


class IFWA3BLReasoner_v2(Reasoner):
    """A3BL-compatible Reasoner using IFWKB's revision-aware dp_marginal.

    Computes posterior marginals via dp_marginal with revision-augmented
    state, matching base A3BL's behavior: marginals are computed only over
    solutions within the revision budget.

    Returns soft labels (marginal probabilities) and hard labels (argmax).
    """

    def __init__(self, kb, temperature: float = 1.0, **kwargs):
        super().__init__(kb, dist_func="confidence", **kwargs)
        self.temperature = temperature
        self.K = len(kb.pseudo_label_list)
        self.class_num = self.K
        self._n_abduce = 0
        self._n_valid = 0

    def abduce(self, data_example: ListData) -> Tuple[List[Any], List[Any]]:
        pred_prob = _normalize_pred_prob(data_example.pred_prob, self.K)
        pseudo_label = data_example.pred_pseudo_label
        y = data_example.Y
        n = len(pred_prob)
        K = self.K
        T = self.temperature

        # Temperature-scaled probabilities
        p = []
        for i in range(n):
            row_raw = [max(float(pred_prob[i][k]), 1e-30) for k in range(K)]
            if T != 1.0:
                row_pow = [r ** (1.0 / T) for r in row_raw]
                total = sum(row_pow)
                row = [r / total for r in row_pow]
            else:
                row = row_raw
            p.append(row)

        kb = self.kb
        decomp = kb._get_decomp(n)

        if kb._in_warmup:
            var_domains = None
        else:
            var_domains = _compute_var_domains(
                p, n, K, 0, kb.perception_threshold,
            )

        # Build revision-aware decomposition (cached, per-sample state via kb._rev_*)
        pseudo_idx = [kb.label_to_idx.get(lbl, 0) for lbl in pseudo_label]
        symbol_num = n
        max_revision_num = self._get_max_revision_num(self.max_revision, symbol_num)
        kb._rev_pseudo_idx = pseudo_idx
        kb._rev_max = max_revision_num
        cache_key = n
        if cache_key not in kb._rev_decomp_cache:
            kb._rev_decomp_cache[cache_key] = kb._build_revision_decomp(decomp)
        rev_decomp, h_accept = kb._rev_decomp_cache[cache_key]
        rev_decomp.h_final_fn(y)

        q, Z = dp_marginal(
            rev_decomp, K, y, p,
            var_domains=var_domains,
            max_states=kb.max_states,
            h_final_accept=h_accept,
        )

        self._n_abduce += 1
        valid = Z > 0
        if valid:
            self._n_valid += 1

        kb.monitor.record_abduce(var_domains, pred_prob, n, K, valid)

        if not valid:
            return [], []

        soft_labels = [torch.tensor(q[i], dtype=torch.float32) for i in range(n)]
        hard_labels = [
            kb.idx_to_label[max(range(K), key=lambda k: q[i][k])]
            for i in range(n)
        ]
        return soft_labels, hard_labels

    def batch_abduce(self, data_examples: ListData) -> List[List[Any]]:
        from .ifw_batch import BatchDPEngine
        from collections import defaultdict

        kb = self.kb
        K = self.K
        T = self.temperature
        n_examples = len(data_examples)
        if n_examples == 0:
            data_examples.abduced_soft_label = []
            data_examples.abduced_pseudo_label = []
            return []

        # Collect per-example data
        all_p = []       # list of (n_i, K) probs
        all_y = []
        all_n = []
        for ex in data_examples:
            pred_prob = _normalize_pred_prob(ex.pred_prob, K)
            n = len(pred_prob)
            p = []
            for i in range(n):
                row = [max(float(pred_prob[i][k]), 1e-30) for k in range(K)]
                if T != 1.0:
                    row = [r ** (1.0 / T) for r in row]
                    s = sum(row)
                    row = [r / s for r in row]
                p.append(row)
            all_p.append(p)
            all_y.append(ex.Y)
            all_n.append(n)

        # Group by n for batching (engine is y-independent)
        groups = defaultdict(list)  # n -> [indices]
        for i, n_i in enumerate(all_n):
            groups[n_i].append(i)

        # Results storage
        soft_results = [None] * n_examples
        hard_results = [None] * n_examples
        valid_flags = [False] * n_examples

        for n_i, indices in groups.items():
            decomp = kb._get_decomp(n_i)

            # Build or retrieve batch engine (y-independent, keyed by n only)
            if not hasattr(kb, '_batch_engine_cache'):
                kb._batch_engine_cache = {}
            if n_i not in kb._batch_engine_cache:
                # Root output fn: reconstruct z from representatives + z_vals,
                # compute KB output, return h_final(kb_output). No y-check.
                _kb = kb
                _d = decomp
                _reps = getattr(decomp, '_reps', {})

                def _root_output(h_combo, z_vals, node,
                                 __kb=_kb, __d=_d, __reps=_reps):
                    ch = __d.children[node]
                    z = [0] * __d.n
                    for ci, c in enumerate(ch):
                        rep = __reps.get((c, h_combo[ci]))
                        if rep:
                            for vid, val in rep.items():
                                z[vid] = val
                    for j, vid in enumerate(__d.var_groups[node]):
                        z[vid] = z_vals[j]
                    labels = [__kb.idx_to_label[z[v]] for v in range(len(z))]
                    kb_out = __kb.logic_forward(labels)
                    if kb_out is None:
                        return None
                    return ("__target__", kb_out)

                kb._batch_engine_cache[n_i] = BatchDPEngine(
                    decomp, K, device='cpu', root_output_fn=_root_output,
                )
            engine = kb._batch_engine_cache[n_i]

            p_batch = torch.tensor([all_p[i] for i in indices], dtype=torch.float32)
            y_list = [all_y[i] for i in indices]

            # Build pseudo_idx_batch for revision tracking
            pseudo_idx_batch = None
            max_revision_num = -1
            if hasattr(self, 'max_revision'):
                n_i_sym = n_i
                max_revision_num = self._get_max_revision_num(self.max_revision, n_i_sym)
                pseudo_idx_batch = torch.tensor(
                    [[kb.label_to_idx.get(lbl, 0) for lbl in data_examples[i].pred_pseudo_label]
                     for i in indices],
                    dtype=torch.long,
                )

            q_batch, Z_batch = engine.batch_marginal(
                y_list, p_batch, pseudo_idx_batch, max_revision_num,
            )
            # q_batch: (N_group, n_i, K), Z_batch: (N_group,)

            for gi, idx in enumerate(indices):
                Z = Z_batch[gi].item()
                self._n_abduce += 1
                if Z > 0:
                    self._n_valid += 1
                    valid_flags[idx] = True
                    soft = [q_batch[gi, v, :] for v in range(n_i)]
                    hard = [kb.idx_to_label[q_batch[gi, v, :].argmax().item()] for v in range(n_i)]
                    soft_results[idx] = soft
                    hard_results[idx] = hard
                else:
                    soft_results[idx] = []
                    hard_results[idx] = []

                # Monitor
                pred_prob = _normalize_pred_prob(data_examples[idx].pred_prob, K)
                n = len(pred_prob)
                if kb._in_warmup:
                    vd = None
                else:
                    vd = _compute_var_domains(all_p[idx], n, K, 0, kb.perception_threshold)
                kb.monitor.record_abduce(vd, pred_prob, n, K, valid_flags[idx])

        data_examples.abduced_soft_label = soft_results
        data_examples.abduced_pseudo_label = hard_results
        return soft_results

    def end_loop(self):
        if hasattr(self.kb, 'monitor'):
            self.kb.monitor.end_batch()

    def __call__(self, data_examples: ListData) -> List[List[Any]]:
        return self.batch_abduce(data_examples)


def _normalize_pred_prob(pred_prob, K: int):
    """
    Normalize pred_prob to a list of K-element distributions.

    Handles multiple input formats:
      - Standard: list of n arrays each of length K  → pass through
      - Multi-label binary: ndarray of shape (1, n) or (n,) with sigmoid probs
        → expand to [[1-p, p], ...] for K=2
    """
    import numpy as _np
    if isinstance(pred_prob, _np.ndarray):
        arr = pred_prob.squeeze()  # (1, n) -> (n,)
        if arr.ndim == 1 and K == 2:
            # Binary sigmoid: expand to [1-p, p]
            return [[1.0 - float(arr[i]), float(arr[i])] for i in range(len(arr))]
        elif arr.ndim == 2 and arr.shape[1] == K:
            return arr.tolist()
        elif arr.ndim == 1 and len(arr) == K:
            # Single variable with K probs
            return [arr.tolist()]
    # Default: assume it's already a list of lists
    if hasattr(pred_prob, '__len__') and len(pred_prob) > 0:
        first = pred_prob[0]
        if hasattr(first, '__len__') and len(first) == K:
            return pred_prob  # already correct format
        # Maybe list of scalars (binary case)
        if K == 2 and not hasattr(first, '__len__'):
            return [[1.0 - float(pred_prob[i]), float(pred_prob[i])] for i in range(len(pred_prob))]
    return pred_prob


def _compute_var_domains(probs, n, K, top_k=0, threshold=0.0):
    """
    Compute per-variable restricted domains from perception probabilities.

    Args:
        probs: List of n lists, each of K values (probabilities or log-probs).
        n: Number of variables.
        K: Domain size.
        top_k: If > 0, keep only top-k values per variable.
        threshold: If > 0, keep values with prob >= threshold * max_prob.
                   At least 1 value (argmax) is always kept.

    Returns:
        List of n lists of kept value indices, or None if no pruning.
    """
    if top_k <= 0 and threshold <= 0:
        return None
    if top_k > 0 and top_k >= K:
        return None

    var_domains = []
    for i in range(n):
        row = probs[i]
        if threshold > 0:
            max_val = max(row)
            cutoff = threshold * max_val
            kept = [k for k in range(K) if row[k] >= cutoff]
            if not kept:
                kept = [max(range(K), key=lambda k: row[k])]
            if top_k > 0 and len(kept) > top_k:
                kept = sorted(kept, key=lambda k: row[k], reverse=True)[:top_k]
            var_domains.append(kept)
        else:
            top = sorted(range(K), key=lambda k: row[k], reverse=True)[:top_k]
            var_domains.append(top)
    return var_domains


# ================================================================
#  IFWReasoner -> ABL  (drop-in for Reasoner + SimpleBridge)
# ================================================================
class IFWReasoner(Reasoner):
    """
    MAP abduction via IFW-DP.  Drop-in replacement for ``Reasoner``.

    Mathematically equivalent to confidence_dist selection over all valid
    candidates, but runs in O(L*K^r*H) instead of O(|Omega_y|*n).

    Args:
        kb: Knowledge base.
        decomp: Decomposition for fixed-length inputs, or None if using decomp_fn.
        K: Domain size per variable (default: len(kb.pseudo_label_list)).
        decomp_fn: Callable (n: int) -> Decomposition for variable-length inputs.
        idx_to_label: Optional index-to-label mapping.
        max_states: Beam width for sparse DP (0=unlimited).
        perception_top_k: If > 0, only consider the top-k most likely values
            per variable, reducing per-step complexity from O(K^r) to O(k^r).
            Set to 0 to disable. Mutually exclusive with perception_threshold.
        perception_threshold: If > 0, only consider values whose probability
            >= threshold * max_prob for that variable. Adaptive per-variable
            pruning: confident variables get pruned more aggressively.
            Set to 0 to disable. At least 1 value is always kept (the argmax).
    """

    def __init__(
        self,
        kb: KBBase,
        K: Optional[int] = None,
        idx_to_label: Optional[dict] = None,
        max_states: int = 0,
        perception_top_k: int = 0,
        perception_threshold: float = 0.0,
        auto_decompose: bool = True,
        auto_decompose_n: Optional[int] = None,
        auto_decompose_num_samples: int = 500,
        auto_decompose_max_precompute: int = 100_000,
        auto_decompose_lazy_precompute: bool = False,
        monitor: bool = False,
        **kwargs,
    ):
        super().__init__(kb, dist_func="confidence", idx_to_label=idx_to_label, **kwargs)
        self.K = K if K is not None else len(kb.pseudo_label_list)
        self.max_states = max_states
        self.perception_top_k = perception_top_k
        self.perception_threshold = perception_threshold
        self.auto_decompose_num_samples = auto_decompose_num_samples
        self.auto_decompose_max_precompute = auto_decompose_max_precompute
        self.auto_decompose_lazy_precompute = auto_decompose_lazy_precompute
        self._n_abduce = 0
        self._n_valid = 0
        self._decomp_cache = {}
        self.monitor = PerceptionMonitor(enabled=monitor)

        if auto_decompose_n is not None:
            # Fixed n: discover once at init
            self._decomp_cache[auto_decompose_n] = self._discover(auto_decompose_n)

    def _discover(self, n: int) -> Decomposition:
        """Auto-discover decomposition for n variables (cached)."""
        if n in self._decomp_cache:
            return self._decomp_cache[n]
        from .tracer import discover_decomposition
        from ..utils import print_log

        # Check KB for y-conditioned CSS support
        extra_kw = {}
        if hasattr(self.kb, 'constraint_fn') and hasattr(self.kb, 'y_decompose_fn'):
            extra_kw['constraint_fn'] = self.kb.constraint_fn
            extra_kw['y_decompose_fn'] = self.kb.y_decompose_fn
            extra_kw['y_size'] = getattr(self.kb, 'y_size', 0)
            extra_kw['Y_domains'] = getattr(self.kb, 'Y_domains', None)

        decomp, info = discover_decomposition(
            self.kb.logic_forward,
            n=n,
            K=self.K,
            num_samples=self.auto_decompose_num_samples,
            max_precompute=self.auto_decompose_max_precompute,
            max_states=self.max_states,
            lazy_precompute=self.auto_decompose_lazy_precompute,
            **extra_kw,
        )
        if isinstance(info, dict):
            cuts = info.get("chain_cuts", [])
            top_c = f"{cuts[0]['compression']:.0f}x" if cuts else "N/A"
            y_prune = info.get("y_step_assignment") or info.get("y_node_assignment")
            y_tag = f", y-pruning={y_prune}" if y_prune else ""
            precompute_tag = ""
            if "precompute_mode" in info:
                precompute_tag = (
                    f", precompute={info.get('precompute_mode')}"
                    f"/{info.get('precompute_kb_calls')}"
                )
            state_tag = f", state={info.get('state_mode')}" if info.get("state_mode") else ""
            tw_tag = ""
            if "treewidth" in info:
                tw_tag = f", treewidth={info.get('treewidth')}"
            print_log(
                f"[{self.__class__.__name__}] Auto-decomposed n={n}: "
                f"{len(info.get('var_groups', []))} steps, "
                f"CSS domains={info.get('css_domain_sizes', [])}, "
                f"top compression={top_c}{y_tag}{precompute_tag}{state_tag}{tw_tag}",
                logger="current",
            )
        else:
            print_log(f"[{self.__class__.__name__}] Auto-decomposed n={n} (fallback)", logger="current")
        self._decomp_cache[n] = decomp
        return decomp

    def _get_decomp(self, n: int) -> Decomposition:
        return self._discover(n)

    @property
    def _in_warmup(self) -> bool:
        """True during monitor warmup — perception pruning is disabled."""
        return self.monitor.enabled and len(self.monitor._history) < self.monitor.warmup

    def abduce(self, data_example: ListData) -> List[Any]:
        pred_prob = _normalize_pred_prob(data_example.pred_prob, self.K)
        y = data_example.Y
        n = len(pred_prob)
        K = self.K

        decomp = self._get_decomp(n)

        # Disable perception pruning during warmup to avoid poisoning early training
        if self._in_warmup:
            var_domains = None
        else:
            var_domains = _compute_var_domains(
                pred_prob, n, K, self.perception_top_k, self.perception_threshold,
            )

        log_p = [
            [math.log(max(float(pred_prob[i][k]), 1e-30)) for k in range(K)]
            for i in range(n)
        ]

        z_hat, score = dp_map(decomp, K, y, log_p,
                              var_domains=var_domains,
                              max_states=self.max_states)

        self._n_abduce += 1
        valid = score > float("-inf")
        if valid:
            self._n_valid += 1

        self.monitor.record_abduce(var_domains, pred_prob, n, K, valid)

        if not valid:
            return []
        return [self.idx_to_label[z_hat[i]] for i in range(n)]

    def batch_abduce(self, data_examples: ListData) -> List[List[Any]]:
        abduced = [self.abduce(ex) for ex in data_examples]
        data_examples.abduced_pseudo_label = abduced
        return abduced

    def end_loop(self):
        """Commit accumulated per-example stats as one history entry (call after each ABL loop)."""
        self.monitor.end_batch()

    def __call__(self, data_examples: ListData) -> List[List[Any]]:
        return self.batch_abduce(data_examples)

    def log_stats(self):
        from ..utils import print_log
        rate = self._n_valid / max(self._n_abduce, 1) * 100
        h_str = "sparse" if (self.decomp and _is_sparse(self.decomp)) else str(self.decomp.H if self.decomp else "dynamic")
        print_log(
            f"[IFWReasoner] MAP  abduced={self._n_abduce} "
            f"valid={self._n_valid}({rate:.1f}%) IFW={h_str}",
            logger="current",
        )

    def reset_stats(self):
        self._n_abduce = 0
        self._n_valid = 0


# ================================================================
#  IFWA3BLReasoner -> A3BL  (drop-in for A3BLReasoner + A3BLBridge)
# ================================================================
class IFWA3BLReasoner(Reasoner):
    """
    Marginal abduction via IFW forward-backward DP.
    Drop-in replacement for ``A3BLReasoner``.

    Returns exact posterior marginals as soft labels, with optional
    temperature scaling to match A3BLReasoner's sharpness control.

    Args:
        kb: Knowledge base.
        decomp: Decomposition for fixed-length inputs, or None if using decomp_fn.
        K: Domain size per variable (default: len(kb.pseudo_label_list)).
        temperature: Temperature scaling for probabilities.
        decomp_fn: Callable (n: int) -> Decomposition for variable-length inputs.
        idx_to_label: Optional index-to-label mapping.
        max_states: Beam width for sparse DP (0=unlimited).
        perception_top_k: If > 0, only consider top-k values per variable.
        perception_threshold: If > 0, keep values with prob >= threshold * max_prob.
        monitor: If True, enable PerceptionMonitor for early restart/stop signals.
    """

    def __init__(
        self,
        kb: KBBase,
        K: Optional[int] = None,
        temperature: float = 1.0,
        idx_to_label: Optional[dict] = None,
        max_states: int = 0,
        perception_top_k: int = 0,
        perception_threshold: float = 0.0,
        auto_decompose: bool = True,
        auto_decompose_n: Optional[int] = None,
        auto_decompose_num_samples: int = 500,
        auto_decompose_max_precompute: int = 100_000,
        auto_decompose_lazy_precompute: bool = False,
        monitor: bool = False,
        **kwargs,
    ):
        super().__init__(kb, dist_func="confidence", idx_to_label=idx_to_label, **kwargs)
        self.K = K if K is not None else len(kb.pseudo_label_list)
        self.class_num = self.K
        self.temperature = temperature
        self.max_states = max_states
        self.perception_top_k = perception_top_k
        self.perception_threshold = perception_threshold
        self.auto_decompose_num_samples = auto_decompose_num_samples
        self.auto_decompose_max_precompute = auto_decompose_max_precompute
        self.auto_decompose_lazy_precompute = auto_decompose_lazy_precompute
        self._decomp_cache = {}
        self._n_abduce = 0
        self._n_valid = 0
        self.monitor = PerceptionMonitor(enabled=monitor)

        if auto_decompose_n is not None:
            self._decomp_cache[auto_decompose_n] = self._discover(auto_decompose_n)

    def _discover(self, n: int) -> Decomposition:
        """Auto-discover decomposition for n variables (cached)."""
        if n in self._decomp_cache:
            return self._decomp_cache[n]
        from .tracer import discover_decomposition
        from ..utils import print_log

        extra_kw = {}
        if hasattr(self.kb, 'constraint_fn') and hasattr(self.kb, 'y_decompose_fn'):
            extra_kw['constraint_fn'] = self.kb.constraint_fn
            extra_kw['y_decompose_fn'] = self.kb.y_decompose_fn
            extra_kw['y_size'] = getattr(self.kb, 'y_size', 0)
            extra_kw['Y_domains'] = getattr(self.kb, 'Y_domains', None)

        decomp, info = discover_decomposition(
            self.kb.logic_forward,
            n=n,
            K=self.K,
            num_samples=self.auto_decompose_num_samples,
            max_precompute=self.auto_decompose_max_precompute,
            max_states=self.max_states,
            lazy_precompute=self.auto_decompose_lazy_precompute,
            **extra_kw,
        )
        if isinstance(info, dict):
            cuts = info.get("chain_cuts", [])
            top_c = f"{cuts[0]['compression']:.0f}x" if cuts else "N/A"
            y_prune = info.get("y_step_assignment") or info.get("y_node_assignment")
            y_tag = f", y-pruning={y_prune}" if y_prune else ""
            precompute_tag = ""
            if "precompute_mode" in info:
                precompute_tag = (
                    f", precompute={info.get('precompute_mode')}"
                    f"/{info.get('precompute_kb_calls')}"
                )
            state_tag = f", state={info.get('state_mode')}" if info.get("state_mode") else ""
            tw_tag = ""
            if "treewidth" in info:
                tw_tag = f", treewidth={info.get('treewidth')}"
            print_log(
                f"[{self.__class__.__name__}] Auto-decomposed n={n}: "
                f"{len(info.get('var_groups', []))} steps, "
                f"CSS domains={info.get('css_domain_sizes', [])}, "
                f"top compression={top_c}{y_tag}{precompute_tag}{state_tag}{tw_tag}",
                logger="current",
            )
        else:
            print_log(f"[{self.__class__.__name__}] Auto-decomposed n={n} (fallback)", logger="current")
        self._decomp_cache[n] = decomp
        return decomp

    def _get_decomp(self, n: int) -> Decomposition:
        return self._discover(n)

    @property
    def _in_warmup(self) -> bool:
        return self.monitor.enabled and len(self.monitor._history) < self.monitor.warmup

    def abduce(self, data_example: ListData) -> Tuple[List[Any], List[Any]]:
        pred_prob = _normalize_pred_prob(data_example.pred_prob, self.K)
        y = data_example.Y
        n = len(pred_prob)
        K = self.K
        T = self.temperature

        decomp = self._get_decomp(n)

        p = []
        for i in range(n):
            row_raw = [max(float(pred_prob[i][k]), 1e-30) for k in range(K)]
            if T != 1.0:
                row_pow = [r ** (1.0 / T) for r in row_raw]
                total = sum(row_pow)
                row = [r / total for r in row_pow]
            else:
                row = row_raw
            p.append(row)

        if self._in_warmup:
            var_domains = None
        else:
            var_domains = _compute_var_domains(
                p, n, K, self.perception_top_k, self.perception_threshold,
            )

        q, Z = dp_marginal(decomp, K, y, p,
                           var_domains=var_domains,
                           max_states=self.max_states)

        self._n_abduce += 1
        valid = Z > 0
        if valid:
            self._n_valid += 1

        self.monitor.record_abduce(var_domains, pred_prob, n, K, valid)

        if not valid:
            return [], []

        soft_labels = [torch.tensor(q[i], dtype=torch.float32) for i in range(n)]
        hard_labels = [
            self.idx_to_label[max(range(K), key=lambda k: q[i][k])]
            for i in range(n)
        ]
        return soft_labels, hard_labels

    def batch_abduce(self, data_examples: ListData) -> List[List[Any]]:
        results = [self.abduce(ex) for ex in data_examples]
        if not results:
            data_examples.abduced_soft_label = []
            data_examples.abduced_pseudo_label = []
            return []
        soft, hard = zip(*results)
        data_examples.abduced_soft_label = list(soft)
        data_examples.abduced_pseudo_label = list(hard)
        return list(soft)

    def end_loop(self):
        """Commit accumulated per-example stats as one history entry (call after each ABL loop)."""
        self.monitor.end_batch()

    def __call__(self, data_examples: ListData) -> List[List[Any]]:
        return self.batch_abduce(data_examples)

    def log_stats(self):
        from ..utils import print_log
        rate = self._n_valid / max(self._n_abduce, 1) * 100
        h_str = "sparse" if (self.decomp and _is_sparse(self.decomp)) else str(self.decomp.H if self.decomp else "dynamic")
        print_log(
            f"[IFWA3BLReasoner] marginal  temp={self.temperature}  "
            f"abduced={self._n_abduce} valid={self._n_valid}({rate:.1f}%) "
            f"IFW={h_str}",
            logger="current",
        )

    def reset_stats(self):
        self._n_abduce = 0
        self._n_valid = 0
