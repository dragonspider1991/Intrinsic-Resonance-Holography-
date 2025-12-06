"""
IRH v13.0 Metrics Module

This module contains Dimensional Coherence Index and Holographic checks.

Key Functions:
- spectral_dimension: d_spec via heat kernel or eigenvalue scaling
- dimensional_coherence_index: χ_D = ℰ_H × ℰ_R × ℰ_C
- hausdorff_dimension: Box-counting fractal dimension
- validate_dimensional_predictions: Comprehensive validation suite

References: IRH v13.0 Section 6, Theorem 3.1
"""

from .dimensions import (
    spectral_dimension,
    dimensional_coherence_index,
    hausdorff_dimension,
    validate_dimensional_predictions
)

__all__ = [
    "spectral_dimension",
    "dimensional_coherence_index",
    "hausdorff_dimension",
    "validate_dimensional_predictions"
]

__version__ = "13.0.0"
