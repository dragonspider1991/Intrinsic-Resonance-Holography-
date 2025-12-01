"""
IRH Suite v9.2 - Intrinsic Resonance Holography

A computational engine for discrete quantum spacetime through graph-theoretic methods.

This package provides:
- HyperGraph substrate formalization
- Spectral dimension computation
- Scaling flows and coarse-graining
- GTEC (Graph Topological Emergent Complexity) functional
- NCGG (Non-Commutative Graph Geometry) operators
- DHGA (Discrete Homotopy Group Analysis)
- Physical constant recovery and predictions

Example:
    >>> from irh import HyperGraph
    >>> from irh.spectral_dimension import SpectralDimension
    >>> G = HyperGraph(N=64, seed=42)
    >>> ds = SpectralDimension(G)
    >>> print(f"Spectral dimension: {ds.value:.2f} ± {ds.error:.2f}")
"""

__version__ = "9.2.0"
__author__ = "IRH Development Team"
__license__ = "CC0 1.0 Universal"

from .graph_state import HyperGraph, create_graph_state

__all__ = [
    "HyperGraph",
    "create_graph_state",
]
