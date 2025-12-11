"""
IRH v18.0 Core Implementation
=============================

Complex-weighted Group Field Theory (cGFT) framework achieving
full ontological and mathematical closure.

Modules:
    - group_manifold: G_inf = SU(2) × U(1)_φ implementation
    - cgft_field: Fundamental field φ(g₁,g₂,g₃,g₄) and bilocal Σ
    - cgft_action: S_kin + S_int + S_hol action terms
    - rg_flow: Beta functions and Cosmic Fixed Point

Key Classes:
    - GInfElement: Element of informational group manifold
    - cGFTFieldDiscrete: Discretized fundamental field
    - cGFTAction: Complete cGFT action
    - CosmicFixedPoint: The unique infrared attractor

References:
    docs/manuscripts/IRHv18.md: Complete theoretical framework
    docs/v18_IMPLEMENTATION_PLAN.md: Implementation roadmap
"""

from .group_manifold import (
    SU2Element,
    U1Element,
    GInfElement,
    compute_ncd,
    compute_ncd_distance,
    haar_integrate_su2,
    haar_integrate_ginf,
)

from .cgft_field import (
    cGFTField,
    cGFTFieldDiscrete,
    BiLocalField,
    CondensateState,
    compute_fluctuation_field,
)

from .cgft_action import (
    cGFTCouplings,
    InteractionKernel,
    cGFTAction,
    compute_effective_laplacian,
    compute_harmony_functional,
)

from .rg_flow import (
    BetaFunctions,
    CosmicFixedPoint,
    find_fixed_point,
    StabilityAnalysis,
    RGFlowSolution,
    integrate_rg_flow,
    compute_C_H_certified,
)

# Universal constant from Cosmic Fixed Point
C_H_V18 = 0.045935703598

__all__ = [
    # Group manifold
    'SU2Element',
    'U1Element',
    'GInfElement',
    'compute_ncd',
    'compute_ncd_distance',
    'haar_integrate_su2',
    'haar_integrate_ginf',
    
    # cGFT field
    'cGFTField',
    'cGFTFieldDiscrete',
    'BiLocalField',
    'CondensateState',
    'compute_fluctuation_field',
    
    # cGFT action
    'cGFTCouplings',
    'InteractionKernel',
    'cGFTAction',
    'compute_effective_laplacian',
    'compute_harmony_functional',
    
    # RG flow
    'BetaFunctions',
    'CosmicFixedPoint',
    'find_fixed_point',
    'StabilityAnalysis',
    'RGFlowSolution',
    'integrate_rg_flow',
    'compute_C_H_certified',
    
    # Constants
    'C_H_V18',
]
