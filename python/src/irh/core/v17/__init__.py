"""
IRH v17.0 Core Module

This module implements the Intrinsic Resonance Holography v17.0 theory,
featuring a local, analytically defined, complex-weighted Group Field Theory (cGFT)
with a unique non-Gaussian infrared fixed point: the Cosmic Fixed Point.

Key components:
- Beta functions for the cGFT couplings (Eq.1.13)
- Fixed-point solver (Eq.1.14)
- Universal constant C_H calculator (Eq.1.15-1.16)
- Spectral dimension flow (Eq.2.8-2.9)
- Physical constant derivations (α, w₀, fermion masses)

References:
    IRH v17.0 Manuscript: docs/manuscripts/IRHv17.md
"""

from .beta_functions import (
    beta_lambda,
    beta_gamma,
    beta_mu,
    compute_fixed_point,
    FIXED_POINT_LAMBDA,
    FIXED_POINT_GAMMA,
    FIXED_POINT_MU,
)
from .constants import (
    compute_C_H,
    compute_alpha_inverse,
    compute_w0,
    C_H_EXACT,
    ALPHA_INVERSE_EXACT,
    W0_EXACT,
)
from .spectral_dimension import (
    compute_spectral_dimension_flow,
    spectral_dimension_ode,
)
from .cgft_action import (
    CGFTAction,
    CGFTField,
    GroupElement,
    compute_ncd,
    create_fixed_point_action,
)

__all__ = [
    # Beta functions
    "beta_lambda",
    "beta_gamma",
    "beta_mu",
    "compute_fixed_point",
    "FIXED_POINT_LAMBDA",
    "FIXED_POINT_GAMMA",
    "FIXED_POINT_MU",
    # Constants
    "compute_C_H",
    "compute_alpha_inverse",
    "compute_w0",
    "C_H_EXACT",
    "ALPHA_INVERSE_EXACT",
    "W0_EXACT",
    # Spectral dimension
    "compute_spectral_dimension_flow",
    "spectral_dimension_ode",
    # cGFT Action
    "CGFTAction",
    "CGFTField",
    "GroupElement",
    "compute_ncd",
    "create_fixed_point_action",
]

__version__ = "17.0.0"
