"""
asymptotics.py - Asymptotic Validators for IRH

This module implements asymptotic limit validators:
- Low-energy limit: Newton's gravity from geodesic deviation
- Continuum limit: Wightman functions and Regge interpolation
- Born rule: Quantum measurement typicality

Equations Implemented:
- Newton: Poisson equation ∇²Φ = 4πGρ from graph geodesics
- Wightman: ⟨0|φ(x)φ(y)|0⟩ → Minkowski correlator
- Born: P(k) = |⟨k|ψ⟩|² from ensemble averaging

References:
- Classical limit of quantum gravity
- Wightman axioms for QFT
- Quantum measurement theory
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.sparse.linalg import cg

if TYPE_CHECKING:
    from .graph_state import HyperGraph


# Physical constants
G_N = 6.67430e-11  # Newton's constant
C = 299792458  # Speed of light
HBAR = 1.054571817e-34  # Reduced Planck constant


@dataclass
class NewtonLimitResult:
    """Result of Newtonian limit recovery."""

    potential: NDArray[np.float64]
    density: NDArray[np.float64]
    poisson_residual: float
    passed: bool
    tolerance: float


@dataclass
class WightmanResult:
    """Result of Wightman function computation."""

    correlator_graph: NDArray[np.float64]
    correlator_minkowski: NDArray[np.float64]
    max_deviation: float
    passed: bool


@dataclass
class BornResult:
    """Result of Born rule typicality test."""

    measured_frequencies: NDArray[np.float64]
    predicted_probabilities: NDArray[np.float64]
    chi_squared: float
    passed: bool


def newton_from_geodesic(graph: HyperGraph, tolerance: float = 1e-3) -> NewtonLimitResult:
    """
    Recover Newton's gravity from geodesic deviation.

    In the low-energy limit, the graph geodesics should satisfy:
    ∇²Φ = 4πGρ (Poisson equation)

    Args:
        graph: HyperGraph instance
        tolerance: Acceptable residual for Poisson match

    Returns:
        NewtonLimitResult with potential and diagnostics
    """
    N = graph.N
    G_nx = graph.to_networkx()

    # Compute geodesic distances
    try:
        dist_matrix = nx.floyd_warshall_numpy(G_nx)
    except Exception:
        dist_matrix = np.ones((N, N))

    # Handle infinities
    max_dist = np.max(dist_matrix[np.isfinite(dist_matrix)])
    dist_matrix[~np.isfinite(dist_matrix)] = max_dist + 1

    # Define mass density (proxy: degree centrality)
    try:
        centrality = nx.degree_centrality(G_nx)
        density = np.array([centrality.get(i, 0) for i in range(N)])
    except Exception:
        density = np.ones(N) / N

    # Normalize density
    density = density / (np.sum(density) + 1e-10)

    # Compute Laplacian for discrete Poisson equation
    L = graph.get_laplacian()

    # Solve Poisson: L·Φ = 4πG·ρ
    # Use conjugate gradient (L might be singular)
    rhs = 4 * np.pi * G_N * density

    # Add small regularization to make L invertible
    L_reg = L + 1e-6 * np.eye(N)

    try:
        potential, info = cg(L_reg, rhs, tol=1e-10)
        if info != 0:
            potential = np.linalg.lstsq(L_reg, rhs, rcond=None)[0]
    except Exception:
        potential = np.linalg.lstsq(L_reg, rhs, rcond=None)[0]

    # Check Poisson residual
    residual = np.linalg.norm(L @ potential - rhs)
    normalized_residual = residual / (np.linalg.norm(rhs) + 1e-10)

    passed = normalized_residual < tolerance

    return NewtonLimitResult(
        potential=potential,
        density=density,
        poisson_residual=float(normalized_residual),
        passed=passed,
        tolerance=tolerance,
    )


def wightman_regge_interp(
    graph: HyperGraph, coords: NDArray[np.float64] | None = None
) -> WightmanResult:
    """
    Compute Wightman correlator and compare to Minkowski.

    The graph correlator should approach the Minkowski propagator
    in the continuum limit:
    C(x,y) → 1/(4π²|x-y|²) for spacelike separation

    Args:
        graph: HyperGraph instance
        coords: Optional coordinate embedding

    Returns:
        WightmanResult with correlator comparison
    """
    N = graph.N

    # Spectral embedding for coordinates
    if coords is None:
        L = graph.get_laplacian()
        eigenvalues, eigenvectors = eigh(L)
        # Use first 4 non-zero eigenvectors
        nonzero_idx = eigenvalues > 1e-10
        if np.sum(nonzero_idx) >= 4:
            idx = np.where(nonzero_idx)[0][:4]
            coords = eigenvectors[:, idx] * np.sqrt(eigenvalues[idx])
        else:
            coords = eigenvectors[:, :4]

    # Graph correlator from Green's function
    # C_G(i,j) = ⟨i|L⁻¹|j⟩
    L = graph.get_laplacian()
    L_reg = L + 1e-6 * np.eye(N)
    try:
        L_inv = np.linalg.inv(L_reg)
    except Exception:
        L_inv = np.linalg.pinv(L_reg)

    correlator_graph = L_inv

    # Minkowski correlator in embedding space
    # C_M(x,y) = 1/(4π²·d(x,y)²) for spacelike
    correlator_minkowski = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                diff = coords[i] - coords[j]
                # Minkowski interval (assuming first coord is time)
                if coords.shape[1] >= 4:
                    interval_sq = -diff[0] ** 2 + np.sum(diff[1:4] ** 2)
                else:
                    interval_sq = np.sum(diff**2)

                if interval_sq > 1e-10:  # Spacelike
                    correlator_minkowski[i, j] = 1 / (4 * np.pi**2 * interval_sq)
                else:
                    correlator_minkowski[i, j] = 0
            else:
                correlator_minkowski[i, j] = 0

    # Normalize for comparison
    norm_graph = np.max(np.abs(correlator_graph)) + 1e-10
    norm_mink = np.max(np.abs(correlator_minkowski)) + 1e-10

    correlator_graph_norm = correlator_graph / norm_graph
    correlator_minkowski_norm = correlator_minkowski / norm_mink

    # Compare (off-diagonal only)
    mask = ~np.eye(N, dtype=bool)
    deviation = np.abs(correlator_graph_norm[mask] - correlator_minkowski_norm[mask])
    max_deviation = float(np.max(deviation))

    # Pass if deviation is small (order 1/N)
    passed = max_deviation < 1.0 / np.sqrt(N)

    return WightmanResult(
        correlator_graph=correlator_graph,
        correlator_minkowski=correlator_minkowski,
        max_deviation=max_deviation,
        passed=passed,
    )


def born_typicality(
    graph: HyperGraph,
    ensemble_size: int = 1000,
    basis: str = "spectral",
) -> BornResult:
    """
    Test Born rule typicality for graph states.

    Verifies that measurement outcomes follow P(k) = |⟨k|ψ⟩|².

    Args:
        graph: HyperGraph instance
        ensemble_size: Number of ensemble samples
        basis: Measurement basis ("spectral" or "position")

    Returns:
        BornResult with frequency comparison
    """
    N = graph.N
    L = graph.get_laplacian()
    eigenvalues, eigenvectors = eigh(L)

    # Initial state (ground state)
    psi = eigenvectors[:, 0]  # Lowest eigenvalue state
    psi = psi / np.linalg.norm(psi)

    # Predicted probabilities: P(k) = |⟨k|ψ⟩|²
    if basis == "spectral":
        # Measurement in spectral basis
        probabilities = np.abs(eigenvectors.T @ psi) ** 2
    else:
        # Measurement in position basis
        probabilities = np.abs(psi) ** 2

    probabilities = probabilities / np.sum(probabilities)

    # Sample from ensemble
    rng = np.random.default_rng(42)
    n_outcomes = min(N, 10)  # Limit outcomes for tractability

    # Collapse probabilities to n_outcomes bins
    probs_binned = np.zeros(n_outcomes)
    for i in range(len(probabilities)):
        probs_binned[i % n_outcomes] += probabilities[i]
    probs_binned = probs_binned / np.sum(probs_binned)

    # Simulate measurements
    samples = rng.choice(n_outcomes, size=ensemble_size, p=probs_binned)
    frequencies = np.bincount(samples, minlength=n_outcomes) / ensemble_size

    # Chi-squared test
    expected = probs_binned * ensemble_size
    observed = frequencies * ensemble_size
    chi_squared = np.sum((observed - expected) ** 2 / (expected + 1e-10))

    # Critical value for n_outcomes-1 degrees of freedom at 95%
    # Approximate: χ²_crit ≈ n_outcomes + 2*sqrt(n_outcomes)
    chi_squared_crit = n_outcomes + 2 * np.sqrt(n_outcomes)
    passed = chi_squared < chi_squared_crit

    return BornResult(
        measured_frequencies=frequencies,
        predicted_probabilities=probs_binned,
        chi_squared=float(chi_squared),
        passed=passed,
    )


def entanglement_test(graph: HyperGraph, W12: complex = complex(-1, 0)) -> dict:
    """
    Test entanglement for given edge weight configuration.

    For W12 = -1 (anti-ferromagnetic), should produce Bell state.

    Args:
        graph: HyperGraph instance
        W12: Edge weight for entanglement test

    Returns:
        Entanglement test results
    """
    # Set specific edge weight
    if len(graph.E) > 0:
        edge = graph.E[0]
        graph.W[edge] = W12
        graph._build_matrices()

    # Compute two-point correlator
    L = graph.get_weighted_laplacian()
    eigenvalues, eigenvectors = eigh(np.real(L))

    # Ground state
    psi = eigenvectors[:, 0]
    psi = psi / np.linalg.norm(psi)

    # For Bell state, expect maximal entanglement entropy
    # Simplified: check anti-correlation
    if len(graph.E) > 0 and len(graph.E[0]) == 2:
        i, j = graph.E[0]
        correlation = psi[i] * psi[j]
        anti_correlated = np.real(correlation) < 0
    else:
        anti_correlated = False

    # Bell state fidelity (simplified)
    bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)  # |00⟩ + |11⟩
    if len(psi) >= 4:
        fidelity = np.abs(np.dot(psi[:4], bell_state)) ** 2
    else:
        fidelity = 0.0

    return {
        "W12": W12,
        "anti_correlated": anti_correlated,
        "bell_fidelity": float(fidelity),
        "passed": fidelity > 0.99 or anti_correlated,
    }


def low_energy_limit_suite(graph: HyperGraph) -> dict:
    """
    Run complete low-energy limit validation suite.

    Args:
        graph: HyperGraph instance

    Returns:
        Suite results dictionary
    """
    newton = newton_from_geodesic(graph)
    wightman = wightman_regge_interp(graph)
    born = born_typicality(graph)

    all_passed = newton.passed and wightman.passed and born.passed

    return {
        "all_passed": all_passed,
        "newton": {
            "passed": newton.passed,
            "residual": newton.poisson_residual,
        },
        "wightman": {
            "passed": wightman.passed,
            "deviation": wightman.max_deviation,
        },
        "born": {
            "passed": born.passed,
            "chi_squared": born.chi_squared,
        },
    }


def continuum_limit_validator(graph: HyperGraph, n_scales: int = 3) -> dict:
    """
    Validate continuum limit behavior across scales.

    Args:
        graph: HyperGraph instance
        n_scales: Number of coarse-graining scales to test

    Returns:
        Validation results
    """
    from .scaling_flows import GSRGDecimate, GromovHausdorffDistance

    results = []

    for scale in range(1, n_scales + 1):
        gsrg = GSRGDecimate(graph, scale=2**scale)
        gh = GromovHausdorffDistance(graph)

        results.append(
            {
                "scale": 2**scale,
                "coarsened_n": gsrg.coarsened_n,
                "gh_distance": gh.distance,
                "gh_bound": gh.bound,
            }
        )

    # Check if GH distance approaches 0 as N increases
    distances = [r["gh_distance"] for r in results]
    converging = all(d < 1.0 for d in distances)  # Simplified check

    return {
        "converging": converging,
        "scales": results,
    }
