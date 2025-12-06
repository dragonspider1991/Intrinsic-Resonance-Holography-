"""
ncgg.py - Non-Commutative Graph Geometry (NCGG) Operators

This module implements NCGG operators for discrete quantum spacetime,
including gauge-covariant differentiation, momentum operators, and
canonical commutation relation (CCR) verification.

Key Operators:
- Position X_k: Projection onto k-th eigenspace
- Momentum P_k: -i ℏ_G / L_G * (D_k - D_k†) (gauge-covariant derivative)
- CCR: [X_k, P_j] = i ℏ_G δ_kj

Frustration Measure:
- F_uv = Im(W_uv): Phase frustration on edges

References:
- Non-commutative geometry (Connes)
- Graph quantum mechanics
- Discrete gauge theory
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

if TYPE_CHECKING:
    from .graph_state import HyperGraph


# Physical constants (stub values)
HBAR_G = 1.054571817e-34  # Reduced Planck constant (J·s)
L_G = 1.616255e-35  # Planck length (m)


@dataclass
class NCGGOperators:
    """Container for NCGG operator matrices."""

    X: list[NDArray[np.complex128]]  # Position operators X_k
    P: list[NDArray[np.complex128]]  # Momentum operators P_k
    D: list[NDArray[np.complex128]]  # Covariant derivatives D_k
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.complex128]


@dataclass
class CCRResult:
    """Result of CCR verification."""

    commutators: NDArray[np.complex128]  # [X_k, P_j] matrix
    expected: NDArray[np.complex128]  # i * hbar_G * delta_kj
    max_error: float
    passed: bool


@dataclass
class FrustrationResult:
    """Result of frustration analysis."""

    frustration_matrix: NDArray[np.float64]
    total_frustration: float
    max_frustration: float
    frustrated_edges: int


class NCGG:
    """
    Non-Commutative Graph Geometry operator algebra.

    Implements the discrete analog of quantum operators on the graph,
    including position, momentum, and their commutation relations.

    Attributes:
        graph: Associated HyperGraph
        hbar_G: Reduced Planck constant in geometric units
        L_G: Graph length scale
    """

    def __init__(
        self,
        graph: HyperGraph,
        hbar_G: float = HBAR_G,
        L_G: float = L_G,
    ) -> None:
        """
        Initialize NCGG operator algebra.

        Args:
            graph: HyperGraph instance
            hbar_G: Reduced Planck constant
            L_G: Length scale
        """
        self.graph = graph
        self.hbar_G = hbar_G
        self.L_G = L_G
        self.N = graph.N

        # Compute spectral decomposition
        L = graph.get_laplacian()
        self.eigenvalues, self.eigenvectors = eigh(L)

        # Build operators
        self._build_operators()

    def _build_operators(self) -> None:
        """Build position, momentum, and covariant derivative operators."""
        N = self.N
        n_modes = min(N, 10)  # Limit number of modes

        self._X = []  # Position operators
        self._P = []  # Momentum operators
        self._D = []  # Covariant derivatives

        for k in range(n_modes):
            # Position operator: projection onto k-th eigenspace
            # X_k = |φ_k⟩⟨φ_k| scaled by eigenvalue
            phi_k = self.eigenvectors[:, k : k + 1]
            X_k = phi_k @ phi_k.T.conj() * (self.eigenvalues[k] + 1e-10)
            self._X.append(X_k.astype(np.complex128))

            # Covariant derivative D_k
            D_k = self.gauge_covariant_diff_k(k)
            self._D.append(D_k)

            # Momentum operator: P_k = -i * hbar_G / L_G * (D_k - D_k†)
            D_k_dag = D_k.conj().T
            P_k = -1j * (self.hbar_G / self.L_G) * (D_k - D_k_dag)
            self._P.append(P_k)

    def gauge_covariant_diff_k(
        self, k: int, f: NDArray[np.complex128] | None = None
    ) -> NDArray[np.complex128]:
        """
        Compute gauge-covariant derivative in k-th mode direction.

        D_k f = Σ_{u ∈ N_k} (W_vu / |N_k|) * (f(u) - f(v))

        where N_k is the projected neighborhood onto k-th eigenvector.

        Args:
            k: Mode index
            f: Optional function values at nodes (default: identity)

        Returns:
            Covariant derivative operator matrix
        """
        N = self.N
        D_k = np.zeros((N, N), dtype=np.complex128)

        if k >= len(self.eigenvalues):
            return D_k

        # k-th eigenvector
        phi_k = self.eigenvectors[:, k]

        # Project neighborhood based on eigenvector components
        for v in range(N):
            # Get neighbors of v
            neighbors = self._get_neighbors(v)
            if len(neighbors) == 0:
                continue

            # Project neighbors using eigenvector weights
            weights = np.abs(phi_k[neighbors])
            if np.sum(weights) > 1e-10:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(neighbors)) / len(neighbors)

            # Covariant derivative contributions
            for i, u in enumerate(neighbors):
                edge = tuple(sorted([u, v]))
                W_uv = self.graph.W.get(edge, 1.0)
                D_k[v, u] += W_uv * weights[i]
                D_k[v, v] -= W_uv * weights[i]

        return D_k

    def _get_neighbors(self, v: int) -> list[int]:
        """Get neighbors of node v."""
        neighbors = []
        for edge in self.graph.E:
            if len(edge) == 2:
                if edge[0] == v:
                    neighbors.append(edge[1])
                elif edge[1] == v:
                    neighbors.append(edge[0])
        return neighbors

    def get_operators(self) -> NCGGOperators:
        """
        Get all NCGG operators.

        Returns:
            NCGGOperators container
        """
        return NCGGOperators(
            X=self._X,
            P=self._P,
            D=self._D,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
        )

    def commutator(
        self, A: NDArray[np.complex128], B: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """
        Compute commutator [A, B] = AB - BA.

        Args:
            A: First operator
            B: Second operator

        Returns:
            Commutator matrix
        """
        return A @ B - B @ A

    def verify_ccr(self, k: int, j: int) -> CCRResult:
        """
        Verify canonical commutation relation [X_k, P_j] = i ℏ_G δ_kj.

        Args:
            k: Position operator index
            j: Momentum operator index

        Returns:
            CCRResult with verification data
        """
        if k >= len(self._X) or j >= len(self._P):
            return CCRResult(
                commutators=np.array([]),
                expected=np.array([]),
                max_error=np.inf,
                passed=False,
            )

        # Compute [X_k, P_j]
        comm = self.commutator(self._X[k], self._P[j])

        # Expected: i * hbar_G * delta_kj * Identity
        delta_kj = 1.0 if k == j else 0.0
        expected = 1j * self.hbar_G * delta_kj * np.eye(self.N, dtype=np.complex128)

        # Check error
        error = np.max(np.abs(comm - expected))

        # Use relative tolerance based on hbar_G scale
        tol = max(1e-8, 1e-6 * self.hbar_G)
        passed = error < tol or (delta_kj == 0 and error < 1e-6)

        return CCRResult(
            commutators=comm,
            expected=expected,
            max_error=float(error),
            passed=passed,
        )

    def verify_all_ccr(self, max_modes: int = 5) -> dict:
        """
        Verify all CCR relations up to max_modes.

        Args:
            max_modes: Maximum number of modes to check

        Returns:
            Summary of CCR verification
        """
        n_modes = min(max_modes, len(self._X))
        results = {}
        all_passed = True

        for k in range(n_modes):
            for j in range(n_modes):
                result = self.verify_ccr(k, j)
                results[f"({k},{j})"] = {
                    "passed": result.passed,
                    "error": result.max_error,
                }
                if not result.passed:
                    all_passed = False

        return {
            "all_passed": all_passed,
            "n_modes": n_modes,
            "results": results,
        }


def frustration(graph: HyperGraph) -> FrustrationResult:
    """
    Compute edge frustration from phase factors.

    F_uv = Im(W_uv): The imaginary part captures phase frustration.

    Args:
        graph: HyperGraph instance

    Returns:
        FrustrationResult with frustration analysis
    """
    N = graph.N
    F = np.zeros((N, N), dtype=np.float64)

    frustrated_count = 0

    for edge, weight in graph.W.items():
        if len(edge) == 2:
            i, j = edge
            frustration_val = np.imag(weight)
            F[i, j] = frustration_val
            F[j, i] = -frustration_val  # Anti-symmetric

            if abs(frustration_val) > 1e-10:
                frustrated_count += 1

    total = np.sum(np.abs(F)) / 2  # Divide by 2 for symmetry
    max_f = np.max(np.abs(F))

    return FrustrationResult(
        frustration_matrix=F,
        total_frustration=float(total),
        max_frustration=float(max_f),
        frustrated_edges=frustrated_count,
    )


def gauge_covariant_derivative(
    graph: HyperGraph, f: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """
    Compute gauge-covariant derivative of function f on graph.

    (Df)_v = Σ_u W_vu * (f(u) - f(v))

    Args:
        graph: HyperGraph instance
        f: Function values at nodes

    Returns:
        Covariant derivative at each node
    """
    N = graph.N
    Df = np.zeros(N, dtype=np.complex128)

    for edge, weight in graph.W.items():
        if len(edge) == 2:
            i, j = edge
            Df[i] += weight * (f[j] - f[i])
            Df[j] += np.conj(weight) * (f[i] - f[j])

    return Df


def momentum_operator(
    graph: HyperGraph, hbar_G: float = HBAR_G, L_G: float = L_G
) -> NDArray[np.complex128]:
    """
    Construct momentum operator matrix.

    P = -i * hbar_G / L_G * D (gauge-covariant derivative)

    Args:
        graph: HyperGraph instance
        hbar_G: Reduced Planck constant
        L_G: Length scale

    Returns:
        Momentum operator matrix
    """
    N = graph.N
    D = np.zeros((N, N), dtype=np.complex128)

    for edge, weight in graph.W.items():
        if len(edge) == 2:
            i, j = edge
            D[i, j] = weight
            D[j, i] = np.conj(weight)
            D[i, i] -= abs(weight)
            D[j, j] -= abs(weight)

    P = -1j * (hbar_G / L_G) * D
    return P


def position_operators(graph: HyperGraph) -> list[NDArray[np.complex128]]:
    """
    Construct position operators from spectral decomposition.

    X_k = λ_k * |φ_k⟩⟨φ_k|

    Args:
        graph: HyperGraph instance

    Returns:
        List of position operator matrices
    """
    L = graph.get_laplacian()
    eigenvalues, eigenvectors = eigh(L)

    X_ops = []
    n_modes = min(graph.N, 10)

    for k in range(n_modes):
        phi_k = eigenvectors[:, k : k + 1]
        X_k = eigenvalues[k] * (phi_k @ phi_k.T.conj())
        X_ops.append(X_k.astype(np.complex128))

    return X_ops


def verify_2d_lattice_ccr() -> dict:
    """
    Verify CCR on 2D lattice (standard test case).

    For a 2D lattice, the commutation relations should match
    the discrete analog of [X, P] = i ℏ_G.

    Returns:
        Test results dictionary
    """
    from .graph_state import HyperGraph

    # Create 2D lattice
    N = 16  # 4x4 lattice
    graph = HyperGraph(N=N, topology="Lattice", seed=42)

    ncgg = NCGG(graph)
    result = ncgg.verify_all_ccr(max_modes=4)

    return result
