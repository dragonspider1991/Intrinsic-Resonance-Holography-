"""
Recovery Suite - Init module

This module provides physics recovery tests:
- Quantum Mechanics: Entanglement, GHZ states
- General Relativity: Einstein Field Equations
- Standard Model: Beta functions, gauge couplings
"""

from .quantum_mechanics import entanglement_test, ghz_state_test
from .general_relativity import efe_solver, ricci_match
from .standard_model import beta_functions, gauge_coupling_test

__all__ = [
    "entanglement_test",
    "ghz_state_test",
    "efe_solver",
    "ricci_match",
    "beta_functions",
    "gauge_coupling_test",
]
