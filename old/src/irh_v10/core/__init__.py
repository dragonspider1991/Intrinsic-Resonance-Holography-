"""
Core mathematical kernels for IRH v10.0

This module provides the fundamental building blocks:
- Cymatic Resonance Network (real-valued oscillator network)
- Interference Matrix (Graph Laplacian ℒ)
- Symplectic → U(N) theorem
- Harmony Functional ℋ_Harmony[K]
- Adaptive Resonance Optimization (ARO)
- Impedance Matching (derives ξ(N)=1/(N ln N))
"""

from .substrate import CymaticResonanceNetwork
from .interference_matrix import build_interference_matrix
from .symplectic_complex import symplectic_to_unitary
from .harmony_functional import harmony_functional
from .aro_optimizer import AdaptiveResonanceOptimizer
from .impedance_matching import impedance_coefficient

__all__ = [
    "CymaticResonanceNetwork",
    "build_interference_matrix",
    "symplectic_to_unitary",
    "harmony_functional",
    "AdaptiveResonanceOptimizer",
    "impedance_coefficient",
]
