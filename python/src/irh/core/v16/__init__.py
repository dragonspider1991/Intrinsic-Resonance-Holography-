"""
IRH v16.0 Core Axiomatic Layer

This module implements the foundational axioms of Intrinsic Resonance Holography v16.0
as specified in docs/manuscripts/IRHv16.md and companion volumes.

THEORETICAL COMPLIANCE:
    All implementations reference and validate against docs/manuscripts/IRHv16.md
    
Modules:
    ahs: Algorithmic Holonomic States (Axiom 0) - IRHv16.md §1
    acw: Algorithmic Coherence Weights (Axiom 1) - IRHv16.md §1  
    crn: Cymatic Resonance Network (Axiom 2) - IRHv16.md §1
    
Coming in subsequent phases:
    evolution: Algorithmic Coherent Evolution (Axiom 4)
    harmony: Universal Harmony Functional with C_H constant

References:
    docs/manuscripts/IRHv16.md - Complete v16.0 theoretical framework (2763 lines)
    [IRH-MATH-2025-01] - The Algebra of Algorithmic Holonomic States
    [IRH-COMP-2025-02] - Exascale HarmonyOptimizer Architecture
"""

from .ahs import AlgorithmicHolonomicState, create_ahs_network
from .acw import (
    AlgorithmicCoherenceWeight,
    compute_ncd_magnitude,
    compute_phase_shift,
    compute_acw
)
from .crn import CymaticResonanceNetworkV16, create_crn_from_states

__version__ = "16.0.0-dev"
__all__ = [
    # Axiom 0: Algorithmic Holonomic States
    "AlgorithmicHolonomicState",
    "create_ahs_network",
    
    # Axiom 1: Algorithmic Coherence Weights
    "AlgorithmicCoherenceWeight",
    "compute_ncd_magnitude",
    "compute_phase_shift",
    "compute_acw",
    
    # Axiom 2: Network Emergence
    "CymaticResonanceNetworkV16",
    "create_crn_from_states",
]
