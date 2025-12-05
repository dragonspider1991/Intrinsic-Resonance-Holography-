"""
IRH Core Mathematical Kernels - Formalism v9.5

This module provides explicit mathematical kernels for:
- ARO (Graph Topological Emergent Complexity) entanglement energy
- NCGG (Non-Commutative Graph Geometry) covariant derivatives
- Spacetime emergence via Dimensional Bootstrap
- Matter genesis via Topological Defect Classification

Quantum Emergence Framework Classes:
- NCGG_Operator_Algebra: Position/Momentum operators and commutators
- ARO_Functional: Entanglement entropy and dark energy cancellation
- Dimensional_Bootstrap: Spectral and growth dimension computation
- Topological_Defect_Classifier: Cycle identification and gauge group verification

Zero Free Parameters: All constants are derived from graph structure.
"""

from .gtec import aro_entanglement_energy, ARO_Functional
from .ncgg import ncgg_covariant_derivative, NCGG_Operator_Algebra
from .spacetime import Dimensional_Bootstrap
from .matter import Topological_Defect_Classifier

__all__ = [
    "aro_entanglement_energy",
    "ARO_Functional",
    "ncgg_covariant_derivative",
    "NCGG_Operator_Algebra",
    "Dimensional_Bootstrap",
    "Topological_Defect_Classifier",
]
