"""
Parallel computing infrastructure for IRH v15.0 (Phase 7)

This module provides MPI, GPU, and distributed computing capabilities
for scaling to exascale (N ≥ 10^10).
"""

__version__ = "15.0.0"

# Check for optional dependencies
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

__all__ = [
    'MPI_AVAILABLE',
    'GPU_AVAILABLE'
]
