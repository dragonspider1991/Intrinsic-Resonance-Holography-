"""
IRH v16.0 Core Axiomatic Layer

This module implements the foundational axioms of Intrinsic Resonance Holography v16.0
as specified in the companion volumes and main manuscript (docs/manuscripts/IRHv16.md).

Modules:
    ahs: Algorithmic Holonomic States (Axiom 0)
    acw: Algorithmic Coherence Weights (Axiom 1)
    crn: Cymatic Resonance Network (Axiom 2)
    holographic: Combinatorial Holographic Principle (Axiom 3)
    dynamics: Coherent Evolution (Axiom 4)

Implementation Status:
    - Axiom 0 (AHS): COMPLETE
    - Axiom 1 (ACW): COMPLETE
    - Axiom 2 (CRN): COMPLETE
    - Axiom 3 (Holographic): COMPLETE
    - Axiom 4 (Evolution): COMPLETE (basic version)

References:
    IRHv16.md: Main manuscript
    [IRH-MATH-2025-01] - The Algebra of Algorithmic Holonomic States
    [IRH-COMP-2025-02] - Exascale HarmonyOptimizer Architecture
"""

__version__ = "16.0.0-dev"

# Axiom 0: Algorithmic Holonomic States
from .ahs import (
    AlgorithmicHolonomicState,
    AHSAlgebra,
    create_ahs_network,
)

# Axiom 1: Algorithmic Coherence Weights
from .acw import (
    AlgorithmicCoherenceWeight,
    compute_ncd_magnitude,
    compute_phase_shift,
    build_acw_matrix,
    MultiFidelityNCDEvaluator,
)

# Axiom 2: Network Emergence Principle
from .crn import (
    CymaticResonanceNetwork,
    EPSILON_THRESHOLD,
    EPSILON_THRESHOLD_ERROR,
    derive_epsilon_threshold,
)

# Axiom 3: Combinatorial Holographic Principle
from .holographic import (
    Subnetwork,
    HolographicAnalyzer,
    verify_holographic_principle,
    HOLOGRAPHIC_CONSTANT_K,
    HOLOGRAPHIC_CONSTANT_K_ERROR,
)

# Axiom 4: Coherent Evolution
from .dynamics import (
    EvolutionState,
    CoherentEvolution,
    AdaptiveResonanceOptimization,
)

__all__ = [
    # Axiom 0
    "AlgorithmicHolonomicState",
    "AHSAlgebra",
    "create_ahs_network",
    # Axiom 1
    "AlgorithmicCoherenceWeight",
    "compute_ncd_magnitude",
    "compute_phase_shift",
    "build_acw_matrix",
    "MultiFidelityNCDEvaluator",
    # Axiom 2
    "CymaticResonanceNetwork",
    "EPSILON_THRESHOLD",
    "EPSILON_THRESHOLD_ERROR",
    "derive_epsilon_threshold",
    # Axiom 3
    "Subnetwork",
    "HolographicAnalyzer",
    "verify_holographic_principle",
    "HOLOGRAPHIC_CONSTANT_K",
    "HOLOGRAPHIC_CONSTANT_K_ERROR",
    # Axiom 4
    "EvolutionState",
    "CoherentEvolution",
    "AdaptiveResonanceOptimization",
]
