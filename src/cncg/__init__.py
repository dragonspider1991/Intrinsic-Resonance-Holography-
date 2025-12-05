"""
Computational Non-Commutative Geometry (cncg)

A professional-grade Python package for computational spectral geometry,
implementing the Spectral Action Principle and dynamical emergence of
spacetime dimensions and physical constants.

This package reproduces the results from:
"Spontaneous Emergence of Four Dimensions, the Fine-Structure Constant,
and Three Generations in Dynamical Finite Spectral Triples"
by Brandon D. McCrary (2025)

Main modules:
- spectral: Core FiniteSpectralTriple class
- action: Spectral action computation
- flow: Riemannian gradient descent
- analysis: Physical observables (spectral dimension, alpha)
- vis: Visualization tools
"""

__version__ = "14.0.0"
__author__ = "Brandon D. McCrary"

from .spectral import FiniteSpectralTriple
from .action import spectral_action, spectral_action_gradient
from .flow import riemannian_gradient_descent
from .analysis import compute_spectral_dimension, compute_fine_structure_constant

__all__ = [
    "FiniteSpectralTriple",
    "spectral_action",
    "spectral_action_gradient",
    "riemannian_gradient_descent",
    "compute_spectral_dimension",
    "compute_fine_structure_constant",
]
