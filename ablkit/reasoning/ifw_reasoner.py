"""
IFW Reasoners for ABL and A3BL pipelines.

  IFWReasoner       ->  ABL  (MAP DP)
  IFWA3BLReasoner   ->  A3BL (marginal DP, soft labels)

Supports chain and tree decompositions via unified dp_map/dp_marginal.
"""

import math
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from ..data.structures import ListData
from ..reasoning import KBBase
from .reasoner import Reasoner
from .ifw_dp import Decomposition, dp_map, dp_marginal


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
        **kwargs,
    ):
        super().__init__(kb, dist_func="confidence", idx_to_label=idx_to_label, **kwargs)
        self.K = K if K is not None else len(kb.pseudo_label_list)
        self.max_states = max_states
        self.perception_top_k = perception_top_k
        self.perception_threshold = perception_threshold
        self._n_abduce = 0
        self._n_valid = 0
        self._decomp_cache = {}

        if auto_decompose_n is not None:
            # Fixed n: discover once at init
            self._decomp_cache[auto_decompose_n] = self._discover(auto_decompose_n)

    def _discover(self, n: int) -> Decomposition:
        """Auto-discover decomposition for n variables (cached)."""
        if n in self._decomp_cache:
            return self._decomp_cache[n]
        from .tracer import discover_decomposition
        from ..utils import print_log
        decomp, info = discover_decomposition(
            self.kb.logic_forward, n=n, K=self.K, num_samples=500,
        )
        if isinstance(info, dict):
            cuts = info.get("chain_cuts", [])
            top_c = f"{cuts[0]['compression']:.0f}x" if cuts else "N/A"
            print_log(
                f"[{self.__class__.__name__}] Auto-decomposed n={n}: "
                f"{len(info.get('var_groups', []))} steps, "
                f"CSS domains={info.get('css_domain_sizes', [])}, "
                f"top compression={top_c}",
                logger="current",
            )
        else:
            print_log(f"[{self.__class__.__name__}] Auto-decomposed n={n} (fallback)", logger="current")
        self._decomp_cache[n] = decomp
        return decomp

    def _get_decomp(self, n: int) -> Decomposition:
        return self._discover(n)

    def abduce(self, data_example: ListData) -> List[Any]:
        pred_prob = _normalize_pred_prob(data_example.pred_prob, self.K)
        y = data_example.Y
        n = len(pred_prob)
        K = self.K

        decomp = self._get_decomp(n)

        # Compute var_domains from raw probabilities (before log transform)
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
        if score <= float("-inf"):
            return []
        self._n_valid += 1

        return [self.idx_to_label[z_hat[i]] for i in range(n)]

    def batch_abduce(self, data_examples: ListData) -> List[List[Any]]:
        abduced = [self.abduce(ex) for ex in data_examples]
        data_examples.abduced_pseudo_label = abduced
        return abduced

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
        auto_decompose_n: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(kb, dist_func="confidence", idx_to_label=idx_to_label, **kwargs)
        self.K = K if K is not None else len(kb.pseudo_label_list)
        self.class_num = self.K
        self.temperature = temperature
        self.max_states = max_states
        self.perception_top_k = perception_top_k
        self.perception_threshold = perception_threshold
        self._decomp_cache = {}
        self._n_abduce = 0
        self._n_valid = 0

        if auto_decompose_n is not None:
            self._decomp_cache[auto_decompose_n] = self._discover(auto_decompose_n)

    def _discover(self, n: int) -> Decomposition:
        """Auto-discover decomposition for n variables (cached)."""
        if n in self._decomp_cache:
            return self._decomp_cache[n]
        from .tracer import discover_decomposition
        from ..utils import print_log
        decomp, info = discover_decomposition(
            self.kb.logic_forward, n=n, K=self.K, num_samples=500,
        )
        if isinstance(info, dict):
            cuts = info.get("chain_cuts", [])
            top_c = f"{cuts[0]['compression']:.0f}x" if cuts else "N/A"
            print_log(
                f"[{self.__class__.__name__}] Auto-decomposed n={n}: "
                f"{len(info.get('var_groups', []))} steps, "
                f"CSS domains={info.get('css_domain_sizes', [])}, "
                f"top compression={top_c}",
                logger="current",
            )
        else:
            print_log(f"[{self.__class__.__name__}] Auto-decomposed n={n} (fallback)", logger="current")
        self._decomp_cache[n] = decomp
        return decomp

    def _get_decomp(self, n: int) -> Decomposition:
        return self._discover(n)

    def abduce(self, data_example: ListData) -> Tuple[List[Any], List[Any]]:
        pred_prob = _normalize_pred_prob(data_example.pred_prob, self.K)
        y = data_example.Y
        n = len(pred_prob)
        K = self.K
        T = self.temperature

        decomp = self._get_decomp(n)

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

        var_domains = _compute_var_domains(
            p, n, K, self.perception_top_k, self.perception_threshold,
        )

        q, Z = dp_marginal(decomp, K, y, p,
                           var_domains=var_domains,
                           max_states=self.max_states)

        self._n_abduce += 1
        if Z <= 0:
            return [], []
        self._n_valid += 1

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
