"""
SOTE Scaling Simulations for IRH Formalism v9.5

This package provides simulation tools for verifying the scaling arguments
used in the Self-Organizing Topological Entropy (SOTE) Principle derivation.

Includes:
- sote_scaling_verification: Random geometric graph analysis
- dimensional_bootstrap: Sparse grid graph analysis for d=1..4
"""

from .sote_scaling_verification import (
    generate_random_geometric_graph,
    compute_holographic_action,
    verify_criticality,
)

from .dimensional_bootstrap import (
    create_grid_graph,
    compute_sparse_laplacian,
    compute_holographic_entropy,
    analyze_dim_sparse,
    run_dimensional_scaling_analysis,
)

__all__ = [
    # SOTE scaling verification
    "generate_random_geometric_graph",
    "compute_holographic_action",
    "verify_criticality",
    # Dimensional bootstrap
    "create_grid_graph",
    "compute_sparse_laplacian",
    "compute_holographic_entropy",
    "analyze_dim_sparse",
    "run_dimensional_scaling_analysis",
]
