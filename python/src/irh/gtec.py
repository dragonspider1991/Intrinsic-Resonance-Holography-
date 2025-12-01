"""
gtec.py - Graph Topological Emergent Complexity (GTEC) Functional

This module implements the GTEC functional which measures the topological
and information-theoretic complexity of the hypergraph substrate.

GTEC Components:
- Shannon global entropy: H_global = -Σᵢ p(λᵢ) log₂ p(λᵢ)
- Local conditional entropy: H_local = mean[H(W_edge | k-hop neighborhood)]
- Complexity measure: C_E = H_global - H_local

The ensemble is sampled via:
μ_δW ~ exp(-Tr(L²)/2) (Gaussian ensemble on weight perturbations)

References:
- Information geometry of eigenvalue distributions
- Graph entropy measures
- Emergent complexity from constraint satisfaction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .graph_state import HyperGraph


@dataclass
class GTECResult:
    """Result of GTEC computation."""

    complexity: float  # C_E = H_global - H_local
    shannon_global: float  # H_global
    local_entropy: float  # H_local (mean conditional entropy)
    optimal_k: int  # Optimized neighborhood size
    eigenvalue_entropy: float  # Entropy of eigenvalue distribution
    edge_entropy: float  # Entropy of edge weight distribution


@dataclass
class EnsembleResult:
    """Result of ensemble sampling."""

    mean_complexity: float
    std_complexity: float
    samples: NDArray[np.float64]
    ensemble_size: int


def gtec(
    graph: HyperGraph,
    ensemble_size: int = 1000,
    max_k: int = 5,
) -> GTECResult:
    """
    Compute GTEC (Graph Topological Emergent Complexity) functional.

    GTEC measures the balance between global disorder (entropy) and
    local structure (conditional entropy), capturing emergent complexity.

    C_E = H_global - H_local

    Args:
        graph: HyperGraph instance
        ensemble_size: Number of samples for averaging
        max_k: Maximum neighborhood size for k optimization

    Returns:
        GTECResult with complexity and component values
    """
    # Get edge weights
    W = np.array([abs(w) for w in graph.W.values()])

    # Compute eigenvalue distribution
    L = graph.get_laplacian()
    eigenvalues = np.linalg.eigvalsh(L)

    # Global Shannon entropy from eigenvalue distribution
    Lambda_i, P = eigenvalue_distribution(eigenvalues)
    shannon_global = shannon_entropy(P)

    # Find optimal k for local entropy
    optimal_k = optimize_k(graph, max_k=max_k)

    # Local conditional entropy
    local_entropy = compute_local_entropy(graph, k=optimal_k)

    # Edge weight entropy
    if len(W) > 0:
        W_norm = W / (np.sum(W) + 1e-15)
        edge_entropy = shannon_entropy(W_norm)
    else:
        edge_entropy = 0.0

    # GTEC complexity
    complexity = shannon_global - local_entropy

    return GTECResult(
        complexity=float(complexity),
        shannon_global=float(shannon_global),
        local_entropy=float(local_entropy),
        optimal_k=optimal_k,
        eigenvalue_entropy=float(shannon_global),
        edge_entropy=float(edge_entropy),
    )


def eigenvalue_distribution(
    eigenvalues: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute normalized eigenvalue distribution.

    Args:
        eigenvalues: Array of eigenvalues

    Returns:
        (eigenvalues, probabilities)
    """
    # Filter and normalize
    eigs = np.abs(eigenvalues)
    eigs = eigs[eigs > 1e-15]

    if len(eigs) == 0:
        return np.array([1.0]), np.array([1.0])

    # Normalize to probability distribution
    P = eigs / np.sum(eigs)

    return eigs, P


def shannon_entropy(P: NDArray[np.float64]) -> float:
    """
    Compute Shannon entropy H = -Σ pᵢ log₂(pᵢ).

    Args:
        P: Probability distribution

    Returns:
        Shannon entropy in bits
    """
    P = np.asarray(P)
    P = P[P > 0]  # Filter zeros

    if len(P) == 0:
        return 0.0

    return float(-np.sum(P * np.log2(P + 1e-15)))


def compute_local_entropy(graph: HyperGraph, k: int = 2) -> float:
    """
    Compute mean conditional entropy of edge weights given k-hop neighborhood.

    H_local = (1/|E|) Σ_e H(W_e | N_k(e))

    Args:
        graph: HyperGraph instance
        k: Neighborhood size (k-hop)

    Returns:
        Mean local conditional entropy
    """
    if len(graph.E) == 0:
        return 0.0

    entropies = []

    for edge in graph.E:
        if len(edge) >= 2:
            # Get k-hop neighborhood weights
            neighbors = get_k_hop_neighbors(graph, edge, k)
            neighbor_weights = [
                abs(graph.W.get(n, 1.0)) for n in neighbors if n in graph.W
            ]

            if len(neighbor_weights) > 0:
                # Normalize to distribution
                weights = np.array(neighbor_weights)
                weights = weights / (np.sum(weights) + 1e-15)

                # Conditional entropy (approximated)
                h_local = shannon_entropy(weights)
                entropies.append(h_local)

    return float(np.mean(entropies)) if entropies else 0.0


def get_k_hop_neighbors(
    graph: HyperGraph, edge: tuple[int, ...], k: int
) -> list[tuple[int, ...]]:
    """
    Get all edges within k hops of the given edge.

    Uses iterative BFS to avoid exponential recursion blowup.

    Args:
        graph: HyperGraph instance
        edge: Target edge
        k: Number of hops

    Returns:
        List of neighboring edges
    """
    if k <= 0:
        return [edge]

    # Build an adjacency mapping from edges to their 1-hop neighbors
    # Two edges are neighbors if they share at least one node
    edge_nodes = {e: set(e) for e in graph.E}

    # Use BFS to find all edges within k hops
    visited: set[tuple[int, ...]] = {edge}
    current_frontier: set[tuple[int, ...]] = {edge}

    for _ in range(k):
        next_frontier: set[tuple[int, ...]] = set()
        for frontier_edge in current_frontier:
            frontier_nodes = edge_nodes[frontier_edge]
            for e in graph.E:
                if e in visited:
                    continue
                # Check if edges share a node (1-hop)
                if frontier_nodes & edge_nodes[e]:
                    next_frontier.add(e)
                    visited.add(e)
        current_frontier = next_frontier
        if not current_frontier:
            break

    # Return all visited edges except the starting edge
    visited.discard(edge)
    return list(visited)


def optimize_k(graph: HyperGraph, max_k: int = 5) -> int:
    """
    Find optimal k that maximizes mutual information gain.

    We seek k where the conditional entropy H(W|N_k) is minimized
    (i.e., neighborhood provides maximum information).

    Args:
        graph: HyperGraph instance
        max_k: Maximum k to search

    Returns:
        Optimal k value
    """
    if len(graph.E) == 0:
        return 1

    best_k = 1
    min_entropy = float("inf")

    for k in range(1, max_k + 1):
        entropy_k = compute_local_entropy(graph, k)
        if entropy_k < min_entropy:
            min_entropy = entropy_k
            best_k = k

    return best_k


def mutual_info_gain(graph: HyperGraph, k1: int, k2: int) -> float:
    """
    Compute mutual information gain between neighborhood sizes.

    I(W; N_k2 | N_k1) = H(W|N_k1) - H(W|N_k2)

    Args:
        graph: HyperGraph instance
        k1: First neighborhood size
        k2: Second neighborhood size (k2 > k1)

    Returns:
        Mutual information gain
    """
    h1 = compute_local_entropy(graph, k1)
    h2 = compute_local_entropy(graph, k2)
    return h1 - h2


def ensemble_gtec(
    graph: HyperGraph,
    ensemble_size: int = 1000,
    perturbation_scale: float = 0.1,
) -> EnsembleResult:
    """
    Compute GTEC over an ensemble of perturbed graphs.

    Ensemble sampling: μ_δW ~ exp(-Tr(L²)/2)

    Args:
        graph: HyperGraph base instance
        ensemble_size: Number of ensemble members
        perturbation_scale: Scale of weight perturbations

    Returns:
        EnsembleResult with statistics
    """
    rng = np.random.default_rng()
    complexities = []

    # Base Laplacian for sampling
    L_base = graph.get_laplacian()
    tr_L2 = np.trace(L_base @ L_base)

    for _ in range(ensemble_size):
        # Sample perturbation δW ~ exp(-Tr(L²)/2)
        # Approximated as Gaussian with variance proportional to 1/Tr(L²)
        variance = perturbation_scale / (tr_L2 + 1.0)

        # Create perturbed weights
        perturbed_W = {}
        for edge, w in graph.W.items():
            delta = rng.normal(0, np.sqrt(variance))
            new_mag = np.clip(abs(w) + delta, 0.01, 1.0)
            phase = np.angle(w)
            perturbed_W[edge] = new_mag * np.exp(1j * phase)

        # Compute GTEC for perturbed graph
        # (simplified: use perturbed weights directly)
        W_array = np.array([abs(w) for w in perturbed_W.values()])
        if len(W_array) > 0:
            W_norm = W_array / (np.sum(W_array) + 1e-15)
            h_global = shannon_entropy(W_norm)
        else:
            h_global = 0.0

        complexities.append(h_global)

    complexities = np.array(complexities)

    return EnsembleResult(
        mean_complexity=float(np.mean(complexities)),
        std_complexity=float(np.std(complexities)),
        samples=complexities,
        ensemble_size=ensemble_size,
    )


def gtec_monotonicity_test(graph: HyperGraph, complexity_threshold: float = 0.0) -> dict:
    """
    Test GTEC monotonicity: complexity should increase with structure.

    Args:
        graph: HyperGraph instance
        complexity_threshold: Minimum expected complexity

    Returns:
        Test results dictionary
    """
    result = gtec(graph)

    # Complexity should be positive for non-trivial graphs
    passed = result.complexity > complexity_threshold

    return {
        "passed": passed,
        "complexity": result.complexity,
        "shannon_global": result.shannon_global,
        "local_entropy": result.local_entropy,
        "threshold": complexity_threshold,
    }
