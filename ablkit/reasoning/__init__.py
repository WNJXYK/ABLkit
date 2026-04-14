from .kb import GroundKB, KBBase, PrologKB
from .reasoner import Reasoner
from .a3bl_reasoner import A3BLReasoner, confidence_dist
from .cached_kb import CachedKB

from .ifw_reasoner import IFWReasoner, IFWA3BLReasoner, IFWKB, IFWABLReasoner

# Core DP infrastructure (generic, unified tree/chain)
from .ifw_dp import (
    Decomposition, make_chain, make_tree,
    dp_map, dp_marginal,
    # Backward-compatible aliases
    sparse_dp_map, sparse_dp_marginal, precompute_transitions,
)

__all__ = [
    # Knowledge bases
    "KBBase", "GroundKB", "PrologKB", "CachedKB",
    # Reasoners
    "Reasoner", "A3BLReasoner", "confidence_dist",
    "IFWReasoner", "IFWA3BLReasoner", "IFWKB", "IFWABLReasoner",
    # IFW core
    "Decomposition", "make_chain", "make_tree",
    "dp_map", "dp_marginal",
]
