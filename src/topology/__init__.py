"""
IRH v13.0 Topology Module

This module contains implementations of topological invariants including
Betti numbers and Frustration density calculations.

Key Functions:
- calculate_frustration_density: Computes ρ_frust from phase holonomies
- derive_fine_structure_constant: α⁻¹ = 2π/ρ_frust (Theorem 1.2)
- calculate_betti_numbers: Homology groups (β₁ = 12 for SU(3)×SU(2)×U(1))
- validate_topological_predictions: Comprehensive validation suite

References: IRH v13.0 Theorems 1.2, 5.1
"""

from .invariants import (
    calculate_frustration_density,
    derive_fine_structure_constant,
    calculate_betti_numbers,
    validate_topological_predictions,
    TopologyAnalyzer
)

__all__ = [
    "calculate_frustration_density",
    "derive_fine_structure_constant",
    "calculate_betti_numbers",
    "validate_topological_predictions",
    "TopologyAnalyzer"
]

__version__ = "13.0.0"
