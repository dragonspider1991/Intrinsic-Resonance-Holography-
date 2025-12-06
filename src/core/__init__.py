"""
IRH Core Mathematical Kernels

Legacy v9.5 implementations:
- ARO (Graph Topological Emergent Complexity) entanglement energy
- NCGG (Non-Commutative Graph Geometry) covariant derivatives
- Spacetime emergence via Dimensional Bootstrap
- Matter genesis via Topological Defect Classification

New v13.0 implementations:
- Spectral Zeta Regularized Harmony Functional (Theorem 4.1)
- Hybrid ARO Optimization Engine
- Topological Invariant Calculators

Zero Free Parameters: All constants are derived from graph structure.
"""

# Legacy v9.5 imports
from .gtec import aro_entanglement_energy, ARO_Functional
from .ncgg import ncgg_covariant_derivative, NCGG_Operator_Algebra
from .spacetime import Dimensional_Bootstrap
from .matter import Topological_Defect_Classifier

# v13.0 imports
from .harmony import (
    harmony_functional,
    compute_information_transfer_matrix,
    validate_harmony_properties
)
from .aro_optimizer import AROOptimizer

__all__ = [
    # Legacy v9.5
    "aro_entanglement_energy",
    "ARO_Functional",
    "ncgg_covariant_derivative",
    "NCGG_Operator_Algebra",
    "Dimensional_Bootstrap",
    "Topological_Defect_Classifier",
    # v13.0
    "harmony_functional",
    "compute_information_transfer_matrix",
    "validate_harmony_properties",
    "AROOptimizer",
]
