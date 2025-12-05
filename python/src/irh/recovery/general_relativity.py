"""
general_relativity.py - General Relativity Recovery Tests

Tests for recovering GR from graph structure:
- Einstein Field Equations
- Ricci curvature matching
- Geodesic deviation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

if TYPE_CHECKING:
    from ..graph_state import HyperGraph


# Physical constants
G_N = 6.67430e-11  # Newton's constant
C = 299792458  # Speed of light


@dataclass
class EFEResult:
    """Result of EFE solver."""

    metric: NDArray[np.float64]
    ricci_tensor: NDArray[np.float64]
    einstein_tensor: NDArray[np.float64]
    stress_energy: NDArray[np.float64]
    residual: float
    passed: bool


@dataclass
class RicciMatchResult:
    """Result of Ricci curvature matching."""

    graph_ricci: NDArray[np.float64]
    target_ricci: NDArray[np.float64]
    match_score: float
    passed: bool


def efe_solver(
    graph: "HyperGraph", stress_energy: NDArray[np.float64] | None = None
) -> EFEResult:
    """
    Solve Einstein Field Equations on graph.

    G_μν = 8πG_N T_μν

    Args:
        graph: HyperGraph instance
        stress_energy: Optional stress-energy tensor

    Returns:
        EFEResult with solution
    """
    N = graph.N

    # Construct metric from graph (weight matrix + regularization)
    g = graph._weights_matrix + 0.1 * np.eye(N)

    # Discrete Ricci tensor
    ricci_tensor = compute_ollivier_ricci(graph)

    # Ricci scalar
    g_inv = np.linalg.pinv(g)
    R = np.trace(g_inv @ ricci_tensor)

    # Einstein tensor: G_μν = R_μν - (1/2) g_μν R
    einstein_tensor = ricci_tensor - 0.5 * g * R

    # Default stress-energy from degree distribution
    if stress_energy is None:
        degrees = np.sum(graph.adjacency_matrix, axis=1)
        stress_energy = np.diag(degrees / (np.sum(degrees) + 1e-10))

    # EFE residual
    efe_residual = np.linalg.norm(einstein_tensor - 8 * np.pi * G_N * stress_energy, "fro")

    # Normalize by typical scale
    typical_scale = np.linalg.norm(einstein_tensor, "fro") + 1e-10
    normalized_residual = efe_residual / typical_scale

    passed = normalized_residual < 0.1  # 10% tolerance

    return EFEResult(
        metric=g,
        ricci_tensor=ricci_tensor,
        einstein_tensor=einstein_tensor,
        stress_energy=stress_energy,
        residual=float(normalized_residual),
        passed=passed,
    )


def compute_ollivier_ricci(graph: "HyperGraph") -> NDArray[np.float64]:
    """
    Compute Ollivier-Ricci curvature on graph.

    κ(u,v) = 1 - W₁(μᵤ, μᵥ) / d(u,v)

    Args:
        graph: HyperGraph instance

    Returns:
        Ricci curvature matrix
    """
    N = graph.N
    G_nx = graph.to_networkx()
    ricci = np.zeros((N, N))

    for edge in graph.E:
        if len(edge) == 2:
            u, v = edge

            try:
                neighbors_u = list(G_nx.neighbors(u))
                neighbors_v = list(G_nx.neighbors(v))
            except Exception:
                continue

            if len(neighbors_u) == 0 or len(neighbors_v) == 0:
                continue

            # Simplified Wasserstein via overlap
            overlap = len(set(neighbors_u) & set(neighbors_v))
            total = len(set(neighbors_u) | set(neighbors_v))

            if total > 0:
                curvature = 2 * overlap / total - 1
            else:
                curvature = 0

            ricci[u, v] = curvature
            ricci[v, u] = curvature

    return ricci


def ricci_match(
    graph: "HyperGraph", target: str = "flat", tolerance: float = 0.01
) -> RicciMatchResult:
    """
    Match graph Ricci curvature to target geometry.

    Args:
        graph: HyperGraph instance
        target: Target geometry ("flat", "sphere", "hyperbolic")
        tolerance: Match tolerance

    Returns:
        RicciMatchResult with comparison
    """
    N = graph.N

    # Compute graph Ricci
    graph_ricci = compute_ollivier_ricci(graph)

    # Target Ricci for different geometries
    if target == "flat":
        target_ricci = np.zeros((N, N))
    elif target == "sphere":
        # Constant positive curvature
        target_ricci = 0.5 * (graph.adjacency_matrix > 0).astype(float)
    elif target == "hyperbolic":
        # Constant negative curvature
        target_ricci = -0.5 * (graph.adjacency_matrix > 0).astype(float)
    else:
        target_ricci = np.zeros((N, N))

    # Compute match score
    diff = graph_ricci - target_ricci
    match_score = 1.0 - np.linalg.norm(diff, "fro") / (
        np.linalg.norm(graph_ricci, "fro") + np.linalg.norm(target_ricci, "fro") + 1e-10
    )

    passed = match_score > (1 - tolerance)

    return RicciMatchResult(
        graph_ricci=graph_ricci,
        target_ricci=target_ricci,
        match_score=float(match_score),
        passed=passed,
    )


def geodesic_deviation(graph: "HyperGraph") -> dict:
    """
    Compute geodesic deviation to test for tidal forces.

    Args:
        graph: HyperGraph instance

    Returns:
        Geodesic deviation analysis
    """
    N = graph.N
    G_nx = graph.to_networkx()

    # Compute geodesic distances
    try:
        dist = nx.floyd_warshall_numpy(G_nx)
    except Exception:
        dist = np.ones((N, N))

    dist[~np.isfinite(dist)] = N

    # Deviation tensor (second derivative of distance)
    deviation = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                # Numerical second derivative
                neighbors_i = [k for k in range(N) if graph.adjacency_matrix[i, k] > 0]
                if len(neighbors_i) >= 2:
                    d_plus = dist[neighbors_i[0], j]
                    d_minus = dist[neighbors_i[-1], j]
                    d_center = dist[i, j]
                    deviation[i, j] = d_plus + d_minus - 2 * d_center

    return {
        "deviation_tensor": deviation,
        "max_deviation": float(np.max(np.abs(deviation))),
        "mean_deviation": float(np.mean(np.abs(deviation))),
    }


def gr_recovery_suite(graph: "HyperGraph") -> dict:
    """
    Run complete GR recovery test suite.

    Args:
        graph: HyperGraph instance

    Returns:
        Suite results
    """
    efe = efe_solver(graph)
    ricci = ricci_match(graph)
    geodesic = geodesic_deviation(graph)

    return {
        "efe": {
            "passed": efe.passed,
            "residual": efe.residual,
        },
        "ricci_match": {
            "passed": ricci.passed,
            "score": ricci.match_score,
        },
        "geodesic": geodesic,
        "all_passed": efe.passed and ricci.passed,
    }
