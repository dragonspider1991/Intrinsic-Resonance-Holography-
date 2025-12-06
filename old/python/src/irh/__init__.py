"""
IRH Suite v10.0 - Intrinsic Resonance Holography

A computational engine for discrete quantum spacetime through Cymatic Resonance Networks.

This package provides:
- Cymatic Resonance Network substrate formalization
- Spectral dimension computation
- Scaling flows and coarse-graining
- ARO (Adaptive Resonance Optimization) functional
- NCGG (Non-Commutative Graph Geometry) operators
- DHGA (Discrete Homotopy Group Analysis)
- Physical constant recovery and predictions

Example:
    >>> from irh import CymaticResonanceNetwork
    >>> from irh.spectral_dimension import SpectralDimension
    >>> G = CymaticResonanceNetwork(N=64, seed=42)
    >>> ds = SpectralDimension(G)
    >>> print(f"Spectral dimension: {ds.value:.2f} ± {ds.error:.2f}")
"""

__version__ = "10.0.0"
__author__ = "IRH Development Team"
__license__ = "CC0 1.0 Universal"

from .graph_state import CymaticResonanceNetwork, create_graph_state

# Backward compatibility
HyperGraph = CymaticResonanceNetwork

__all__ = [
    "CymaticResonanceNetwork",
    "HyperGraph",  # backward compatibility
    "create_graph_state",
]
