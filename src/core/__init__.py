"""
IRH Core Mathematical Kernels - Formalism v9.4

This module provides explicit mathematical kernels for:
- GTEC (Graph Topological Emergent Complexity) entanglement energy
- NCGG (Non-Commutative Graph Geometry) covariant derivatives

Zero Free Parameters: All constants are derived from graph structure.
"""

from .gtec import gtec_entanglement_energy
from .ncgg import ncgg_covariant_derivative

__all__ = [
    "gtec_entanglement_energy",
    "ncgg_covariant_derivative",
]
