"""
Predictions Module - Init

This module provides prediction pipelines for:
- Fine structure constant α⁻¹ = 137.035999084
- Neutrino mass sum Σm_ν = 0.0583 eV
- CKM matrix elements
- Dark energy equation of state w_Λ = -0.75
"""

from .constants import (
    predict_alpha_inverse,
    predict_neutrino_masses,
    predict_ckm_matrix,
    predict_dark_energy,
)

__all__ = [
    "predict_alpha_inverse",
    "predict_neutrino_masses",
    "predict_ckm_matrix",
    "predict_dark_energy",
]
