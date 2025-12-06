"""Spacetime emergence - spectral dimension, Lorentzian signature, gravity."""

from .spectral_dimension import (
    compute_heat_kernel_trace,
    estimate_spectral_dimension,
    verify_4d_emergence,
)
from .lorentzian_signature import (
    count_negative_eigenvalues,
    verify_lorentzian_signature,
    arrow_of_time_test,
)

__all__ = [
    "compute_heat_kernel_trace",
    "estimate_spectral_dimension",
    "verify_4d_emergence",
    "count_negative_eigenvalues",
    "verify_lorentzian_signature",
    "arrow_of_time_test",
]
