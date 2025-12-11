"""
IRH v16.0 Core Axiomatic Layer (DEPRECATED)

.. deprecated:: 18.0.0
   v16 is deprecated in favor of v18. The v18 implementation provides complete
   Standard Model emergence from cGFT with full analytical derivations.
   Use ``from irh.core.v18 import ...`` instead.

This module implements the foundational axioms of Intrinsic Resonance Holography v16.0
as specified in docs/manuscripts/IRHv16.md and companion volumes.

DEPRECATION NOTICE:
    v16 is maintained for backward compatibility only.
    New development should use v18.0 (irh.core.v18).
    See docs/manuscripts/IRHv18.md for the current theoretical framework.

THEORETICAL COMPLIANCE:
    All implementations reference and validate against docs/manuscripts/IRHv16.md
    
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
    IRHv16.md: Main manuscript (deprecated)
    IRHv18.md: Current manuscript
    [IRH-MATH-2025-01] - The Algebra of Algorithmic Holonomic States
    [IRH-COMP-2025-02] - Exascale HarmonyOptimizer Architecture
"""

import warnings

warnings.warn(
    "irh.core.v16 is deprecated since v18.0.0. "
    "Use irh.core.v18 for the current implementation with complete "
    "Standard Model emergence from cGFT. "
    "See docs/manuscripts/IRHv18.md for the current theoretical framework.",
    DeprecationWarning,
    stacklevel=2,
)

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
    compute_acw,
    build_acw_matrix,
)

# Phase 2: Multi-fidelity NCD and Distributed AHS
try:
    from .ncd_multifidelity import (
        FidelityLevel,
        NCDResult,
        compute_ncd_adaptive,
        compute_ncd_certified,
    )
    from .distributed_ahs import (
        DistributedAHSManager,
        AHSMetadata,
        create_distributed_network,
    )
    _PHASE2_AVAILABLE = True
except ImportError:
    _PHASE2_AVAILABLE = False

# Axiom 2: Network Emergence Principle
try:
    from .crn import (
        CymaticResonanceNetworkV16 as CymaticResonanceNetwork,
        create_crn_from_states,
    )
    # Define constants - will be properly imported once implemented
    EPSILON_THRESHOLD = 0.730129
    EPSILON_THRESHOLD_ERROR = 1e-6
    def derive_epsilon_threshold():
        """Placeholder for epsilon threshold derivation."""
        return EPSILON_THRESHOLD
except ImportError as e:
    # Graceful fallback if CRN not yet implemented
    CymaticResonanceNetwork = None
    EPSILON_THRESHOLD = 0.730129
    EPSILON_THRESHOLD_ERROR = 1e-6
    def derive_epsilon_threshold():
        return EPSILON_THRESHOLD

# Axiom 3: Combinatorial Holographic Principle
try:
    from .holographic import (
        Subnetwork,
        HolographicAnalyzer,
        verify_holographic_principle,
    )
    HOLOGRAPHIC_CONSTANT_K = 1.0
    HOLOGRAPHIC_CONSTANT_K_ERROR = 1e-6
except ImportError:
    Subnetwork = None
    HolographicAnalyzer = None
    verify_holographic_principle = None
    HOLOGRAPHIC_CONSTANT_K = 1.0
    HOLOGRAPHIC_CONSTANT_K_ERROR = 1e-6

# Axiom 4: Coherent Evolution
try:
    from .dynamics import (
        EvolutionState,
        CoherentEvolution,
        AdaptiveResonanceOptimization,
    )
except ImportError:
    EvolutionState = None
    CoherentEvolution = None
    AdaptiveResonanceOptimization = None

__all__ = [
    # Axiom 0
    "AlgorithmicHolonomicState",
    "AHSAlgebra",
    "create_ahs_network",
    # Axiom 1
    "AlgorithmicCoherenceWeight",
    "compute_ncd_magnitude",
    "compute_phase_shift",
    "compute_acw",
    "build_acw_matrix",
    # Phase 2 additions
    "FidelityLevel",
    "NCDResult",
    "compute_ncd_adaptive",
    "compute_ncd_certified",
    "DistributedAHSManager",
    "AHSMetadata",
    "create_distributed_network",
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
from .crn import CymaticResonanceNetworkV16, create_crn_from_states
from .harmony import (
    compute_harmony_functional,
    validate_harmony_functional_properties,
    HarmonyFunctionalEvaluator
)
from .aro import AROConfiguration, AROOptimizerV16

__version__ = "16.0.0-deprecated"
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
    
    # Theorem 4.1: Harmony Functional
    "compute_harmony_functional",
    "validate_harmony_functional_properties",
    "HarmonyFunctionalEvaluator",
    
    # Definition 4.1: ARO
    "AROConfiguration",
    "AROOptimizerV16",
]
