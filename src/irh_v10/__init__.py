"""
Intrinsic Resonance Holography v10.0 - "Cymatic Resonance" Version

A complete, parameter-free, computationally verifiable Theory of Everything
derived from a classical network of real harmonic oscillators via Adaptive
Resonance Optimization.

Author: Brandon D. McCrary
Date: December 16, 2025
License: CC0-1.0 Universal (Public Domain)

This is the first theory to derive all fundamental constants from first
principles using only real-valued coupled harmonic oscillators with emergent
complex structure via symplectic geometry.
"""

__version__ = "10.0.0"
__author__ = "Brandon D. McCrary"
__license__ = "CC0-1.0"

from .core.substrate import CymaticResonanceNetwork
from .core.harmony_functional import harmony_functional
from .core.aro_optimizer import AdaptiveResonanceOptimizer
from .predictions.fine_structure_alpha import derive_alpha

__all__ = [
    "CymaticResonanceNetwork",
    "harmony_functional",
    "AdaptiveResonanceOptimizer",
    "derive_alpha",
]
