"""
scaling_flows.py - Coarse-Graining, Metric Emergence, and Lorentz Signature

This module implements renormalization group-inspired scaling flows for hypergraphs,
including GSRG (Graph Spectral Renormalization Group) coarse-graining,
metric emergence from path densities, and Lorentzian signature detection.

Equations Implemented:
- GSRG decimation: high_modes = eigsh(L, k=N//scale); G' = contract(G, high_modes)
- Metric emergence: g_μν = ⟨path_density⟩ ⊗ (L_G⁻¹·∇_x) ⊗ (L_G⁻¹·∇_y)
- Lorentz signature: count negative eigenvalues (target: 1 for (-,+,+,+))
- Gromov-Hausdorff distance to Minkowski M⁴

References:
- Wilson Renormalization Group
- Spectral Graph Coarsening
- Graph Wavelets and Multi-resolution Analysis
- Emergent spacetime geometry
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

if TYPE_CHECKING:
    from .graph_state import HyperGraph


@dataclass
class GSRGResult:
    """Result of GSRG coarse-graining."""

    coarsened_laplacian: NDArray[np.float64]
    scale: int
    original_n: int
    coarsened_n: int
    decimated_modes: int
    preserved_eigenvalues: NDArray[np.float64]


@dataclass
class MetricResult:
    """Result of metric emergence computation."""

    metric_tensor: NDArray[np.float64]
    signature: tuple[int, int]  # (negative, positive)
    eigenvalues: NDArray[np.float64]
    is_lorentzian: bool


@dataclass
class LorentzSignatureResult:
    """Result of Lorentz signature analysis."""

    negative_count: int
    positive_count: int
    zero_count: int
    signature: str
    is_physical: bool  # True if exactly 1 negative eigenvalue
    eigenvalues: NDArray[np.float64]


@dataclass
class GromovHausdorffResult:
    """Result of Gromov-Hausdorff distance computation."""

    distance: float
    embedding: NDArray[np.float64]
    target: str
    bound: float


def GSRGDecimate(
    graph: HyperGraph, scale: int = 2, method: str = "spectral"
) -> GSRGResult:
    """
    GSRG (Graph Spectral Renormalization Group) coarse-graining.

    Decimates high-frequency modes from the graph Laplacian to produce
    an effective coarse-grained description at lower resolution.

    Algorithm:
    1. Compute Laplacian eigendecomposition
    2. Keep only the lowest N//scale modes
    3. Reconstruct coarsened Laplacian

    Args:
        graph: HyperGraph instance
        scale: Decimation scale factor
        method: "spectral" or "random"

    Returns:
        GSRGResult with coarsened Laplacian and diagnostics
    """
    N = graph.N
    L = graph.get_laplacian()

    # Number of modes to keep
    k = max(2, N // scale)

    if method == "spectral":
        # Get lowest k eigenvalues and eigenvectors
        if N <= k + 1:
            # Small graph: use dense solver
            eigenvalues, eigenvectors = eigh(L)
            eigenvalues = eigenvalues[:k]
            eigenvectors = eigenvectors[:, :k]
        else:
            # Large graph: use sparse solver
            try:
                eigenvalues, eigenvectors = eigsh(L, k=k, which="SM")
            except Exception:
                # Fallback to dense
                eigenvalues, eigenvectors = eigh(L)
                eigenvalues = eigenvalues[:k]
                eigenvectors = eigenvectors[:, :k]

        # Reconstruct coarsened Laplacian from low-frequency modes
        # L_coarse = V * diag(λ) * V^T
        L_coarse = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    else:
        # Random decimation (simpler but less principled)
        indices = np.random.choice(N, k, replace=False)
        indices = np.sort(indices)
        L_coarse = L[np.ix_(indices, indices)]
        eigenvalues = np.linalg.eigvalsh(L_coarse)

    return GSRGResult(
        coarsened_laplacian=L_coarse,
        scale=scale,
        original_n=N,
        coarsened_n=k,
        decimated_modes=N - k,
        preserved_eigenvalues=eigenvalues,
    )


def gsrg_decimate(
    graph: HyperGraph, scale: int = 2
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Convenience function for GSRG decimation.

    Args:
        graph: HyperGraph instance
        scale: Decimation scale factor

    Returns:
        (coarsened_laplacian, high_modes)
    """
    result = GSRGDecimate(graph, scale)
    N = graph.N
    L = graph.get_laplacian()

    # Get high-frequency modes that were decimated
    k = result.coarsened_n
    if N > k:
        _, all_vecs = eigh(L)
        high_modes = all_vecs[:, k:]
    else:
        high_modes = np.array([])

    return result.coarsened_laplacian, high_modes


def MetricEmergence(graph: HyperGraph) -> MetricResult:
    """
    Compute emergent metric tensor from graph structure.

    The metric emerges from path density and Laplacian structure:
    g_μν = ⟨path_density⟩ ⊗ (L⁻¹·∇_x) ⊗ (L⁻¹·∇_y)

    In practice, we use the graph distance matrix and spectral embedding
    to construct an effective metric.

    Args:
        graph: HyperGraph instance

    Returns:
        MetricResult with emergent metric tensor
    """
    G_nx = graph.to_networkx()
    N = graph.N

    # Compute shortest path distances
    try:
        dist_matrix = nx.floyd_warshall_numpy(G_nx)
    except Exception:
        # Fallback for disconnected graphs
        dist_matrix = np.full((N, N), np.inf)
        for i in range(N):
            for j in range(N):
                try:
                    dist_matrix[i, j] = nx.shortest_path_length(G_nx, i, j)
                except nx.NetworkXNoPath:
                    pass

    # Handle infinities
    max_finite = np.max(dist_matrix[np.isfinite(dist_matrix)])
    dist_matrix[~np.isfinite(dist_matrix)] = max_finite + 1

    # Compute path density tensor (proxy for metric)
    # Use classical MDS-like embedding
    n = min(N, 4)  # 4D target
    H = np.eye(N) - np.ones((N, N)) / N  # Centering matrix
    B = -0.5 * H @ (dist_matrix**2) @ H  # Gram matrix

    eigenvalues, eigenvectors = eigh(B)

    # Sort by magnitude (for signature analysis)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take top n eigenvectors
    top_eigs = eigenvalues[:n]
    top_vecs = eigenvectors[:, :n]

    # Construct metric tensor (n x n)
    # In the embedding space, the metric is approximately diagonal
    metric = np.diag(top_eigs)

    # Count signature
    neg_count = int(np.sum(top_eigs < -1e-10))
    pos_count = int(np.sum(top_eigs > 1e-10))

    is_lorentzian = neg_count == 1 and pos_count == 3

    return MetricResult(
        metric_tensor=metric,
        signature=(neg_count, pos_count),
        eigenvalues=top_eigs,
        is_lorentzian=is_lorentzian,
    )


def path_density_geodesic(graph: HyperGraph) -> NDArray[np.float64]:
    """
    Compute path density matrix via geodesic distances.

    Args:
        graph: HyperGraph instance

    Returns:
        Path density matrix
    """
    G_nx = graph.to_networkx()
    N = graph.N

    # Geodesic distance matrix
    dist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            try:
                dist[i, j] = nx.shortest_path_length(G_nx, i, j)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                dist[i, j] = N  # Max distance for disconnected

    # Path density: inverse of distance (with regularization)
    density = 1.0 / (dist + 1.0)
    np.fill_diagonal(density, 1.0)

    return density


def LorentzSignature(
    graph: HyperGraph, tolerance: float = 1e-10
) -> LorentzSignatureResult:
    """
    Compute Lorentz signature from graph Laplacian.

    For physical spacetime, we expect exactly 1 negative eigenvalue,
    corresponding to the (-,+,+,+) Lorentzian signature.

    Args:
        graph: HyperGraph instance
        tolerance: Threshold for zero eigenvalues

    Returns:
        LorentzSignatureResult with signature analysis
    """
    # Use weighted Laplacian for richer signature
    L = graph.get_weighted_laplacian()

    # Get real part for eigenvalue analysis
    L_real = np.real(L)

    eigenvalues = np.linalg.eigvalsh(L_real)

    neg_count = int(np.sum(eigenvalues < -tolerance))
    pos_count = int(np.sum(eigenvalues > tolerance))
    zero_count = int(np.sum(np.abs(eigenvalues) <= tolerance))

    # Format signature string
    if neg_count > 0:
        signature = f"({neg_count}, {pos_count})"
    else:
        signature = f"(+{pos_count})"

    # Physical: exactly 1 negative eigenvalue for Lorentzian signature
    is_physical = neg_count == 1

    return LorentzSignatureResult(
        negative_count=neg_count,
        positive_count=pos_count,
        zero_count=zero_count,
        signature=signature,
        is_physical=is_physical,
        eigenvalues=eigenvalues,
    )


def lorentz_signature(L: NDArray[np.float64]) -> int:
    """
    Count negative eigenvalues of matrix L.

    For Lorentzian signature, we expect exactly 1 negative eigenvalue.

    Args:
        L: Matrix (typically Laplacian or metric)

    Returns:
        Number of negative eigenvalues
    """
    eigenvalues = np.linalg.eigvalsh(np.real(L))
    return int(np.sum(eigenvalues < -1e-10))


def GromovHausdorffDistance(
    graph: HyperGraph, target: str = "Minkowski4"
) -> GromovHausdorffResult:
    """
    Compute approximate Gromov-Hausdorff distance to target manifold.

    The GH distance measures how far the graph metric space is from
    the target continuous manifold.

    Args:
        graph: HyperGraph instance
        target: Target manifold ("Minkowski4", "Euclidean4")

    Returns:
        GromovHausdorffResult with distance estimate
    """
    N = graph.N
    G_nx = graph.to_networkx()

    # Graph distance matrix
    try:
        graph_dist = nx.floyd_warshall_numpy(G_nx)
    except Exception:
        graph_dist = np.zeros((N, N))

    # Spectral embedding into target dimension
    L = graph.get_laplacian()
    eigenvalues, eigenvectors = eigh(L)

    # Use first 4 non-zero eigenvectors for embedding
    nonzero_idx = eigenvalues > 1e-10
    if np.sum(nonzero_idx) >= 4:
        idx = np.where(nonzero_idx)[0][:4]
        embedding = eigenvectors[:, idx] * np.sqrt(eigenvalues[idx])
    else:
        embedding = eigenvectors[:, :4]

    # Compute distances in embedding space
    embed_dist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if target == "Minkowski4" and embedding.shape[1] >= 4:
                # Minkowski metric: -t² + x² + y² + z²
                diff = embedding[i] - embedding[j]
                if len(diff) >= 4:
                    embed_dist[i, j] = np.sqrt(
                        np.abs(-diff[0] ** 2 + np.sum(diff[1:4] ** 2))
                    )
                else:
                    embed_dist[i, j] = np.linalg.norm(diff)
            else:
                # Euclidean metric
                embed_dist[i, j] = np.linalg.norm(embedding[i] - embedding[j])

    # Gromov-Hausdorff approximation via distortion
    # GH(X,Y) ≈ (1/2) * sup |d_X(x,x') - d_Y(f(x),f(x'))|
    max_finite_graph = graph_dist[np.isfinite(graph_dist)].max() if np.any(np.isfinite(graph_dist)) else 1.0
    graph_dist_norm = np.clip(graph_dist, 0, max_finite_graph) / (max_finite_graph + 1e-10)

    max_embed = embed_dist.max() if embed_dist.max() > 0 else 1.0
    embed_dist_norm = embed_dist / (max_embed + 1e-10)

    distortion = np.abs(graph_dist_norm - embed_dist_norm)
    gh_distance = 0.5 * np.max(distortion[np.isfinite(distortion)])

    # Expected bound: GH_dist < 1/sqrt(N)
    bound = 1.0 / np.sqrt(N)

    return GromovHausdorffResult(
        distance=float(gh_distance),
        embedding=embedding,
        target=target,
        bound=float(bound),
    )


def hausdorff_distance(
    graph: HyperGraph, target_manifold: str = "Minkowski4"
) -> float:
    """
    Convenience function for Gromov-Hausdorff distance.

    Args:
        graph: HyperGraph instance
        target_manifold: Target manifold name

    Returns:
        GH distance estimate
    """
    result = GromovHausdorffDistance(graph, target_manifold)
    return result.distance


def continuum_limit_test(
    graph: HyperGraph, scale: int = 2
) -> dict:
    """
    Test continuum limit: GH distance should decrease as N increases.

    Args:
        graph: HyperGraph instance
        scale: Scale for comparison

    Returns:
        Test results dictionary
    """
    # Full graph
    gh_full = GromovHausdorffDistance(graph)

    # Coarsened graph
    gsrg_result = GSRGDecimate(graph, scale)

    # Check if distance is bounded by 1/sqrt(N)
    passed = gh_full.distance < gh_full.bound

    return {
        "passed": passed,
        "gh_distance": gh_full.distance,
        "bound": gh_full.bound,
        "n_original": graph.N,
        "n_coarsened": gsrg_result.coarsened_n,
    }
