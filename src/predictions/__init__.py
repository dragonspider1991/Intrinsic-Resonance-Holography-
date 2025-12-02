"""
IRH Prediction Modules - Formalism v9.5

This module provides prediction pipelines for:
- Dynamical Dark Energy: w(a) = -1 + 0.25 * (1 + a)^{-1.5}
- Fine structure constant error budget
- CPL parameterization fitting

Zero Free Parameters: All predictions are derived from graph structure.
"""

from .cosmology import dark_energy_eos, calculate_w0, calculate_wa, cpl_fit
from .fine_structure import calculate_alpha_error

__all__ = [
    "dark_energy_eos",
    "calculate_w0",
    "calculate_wa",
    "cpl_fit",
    "calculate_alpha_error",
]
