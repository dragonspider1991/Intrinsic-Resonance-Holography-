"""
SOTE Scaling Simulations for IRH Formalism v9.5

This package provides simulation tools for verifying the scaling arguments
used in the Self-Organizing Topological Entropy (SOTE) Principle derivation.
"""

from .sote_scaling_verification import (
    generate_random_geometric_graph,
    compute_holographic_action,
    verify_criticality,
)

__all__ = [
    "generate_random_geometric_graph",
    "compute_holographic_action",
    "verify_criticality",
]
