from .kb import GroundKB, KBBase, PrologKB
from .reasoner import Reasoner
from .a3bl_reasoner import A3BLReasoner, confidence_dist
from .cached_kb import CachedKB
from .dabb_reasoner import DABBReasoner, DABBTopKReasoner
from .ifw_reasoner import IFWReasoner, IFWA3BLReasoner

# Core DP infrastructure (generic, unified tree/chain)
from .ifw_dp import (
    Decomposition, make_chain, make_tree,
    dp_map, dp_marginal,
    decomposition_from_kb,
    # Backward-compatible aliases
    sparse_dp_map, sparse_dp_marginal, precompute_transitions,
)

# Task-specific decompositions
from .ifw_tasks import (
    make_addition_decomposition, make_addition_css, make_mod_decomposition,
    make_hwf_decomposition, make_bdd_oia_decomposition,
)

__all__ = [
    # Knowledge bases
    "KBBase", "GroundKB", "PrologKB", "CachedKB",
    # Reasoners
    "Reasoner", "A3BLReasoner", "confidence_dist",
    "DABBReasoner", "DABBTopKReasoner",
    "IFWReasoner", "IFWA3BLReasoner",
    # IFW core
    "Decomposition", "make_chain", "make_tree",
    "dp_map", "dp_marginal",
    "decomposition_from_kb",
    # IFW task-specific
    "make_addition_decomposition", "make_addition_css", "make_mod_decomposition",
    "make_hwf_decomposition", "make_bdd_oia_decomposition",
]
