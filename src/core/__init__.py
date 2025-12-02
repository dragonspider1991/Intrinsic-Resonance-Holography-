"""
IRH Core Mathematical Kernels - Formalism v9.5

This module provides explicit mathematical kernels for:
- GTEC (Graph Topological Emergent Complexity) entanglement energy
- NCGG (Non-Commutative Graph Geometry) covariant derivatives

Quantum Emergence Framework Classes:
- NCGG_Operator_Algebra: Position/Momentum operators and commutators
- GTEC_Functional: Entanglement entropy and dark energy cancellation

Zero Free Parameters: All constants are derived from graph structure.
"""

from .gtec import gtec_entanglement_energy, GTEC_Functional
from .ncgg import ncgg_covariant_derivative, NCGG_Operator_Algebra

__all__ = [
    "gtec_entanglement_energy",
    "GTEC_Functional",
    "ncgg_covariant_derivative",
    "NCGG_Operator_Algebra",
]
