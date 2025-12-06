"""
quantum_mechanics.py - Quantum Mechanics Recovery Tests

Tests for recovering quantum mechanical behavior from graph states:
- Entanglement entropy
- Bell states
- GHZ states
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh, logm

if TYPE_CHECKING:
    from ..graph_state import HyperGraph


@dataclass
class EntanglementResult:
    """Result of entanglement test."""

    entropy: float
    concurrence: float
    is_entangled: bool
    bell_fidelity: float
    passed: bool


@dataclass
class GHZResult:
    """Result of GHZ state test."""

    fidelity: float
    n_qubits: int
    state_vector: NDArray[np.complex128]
    passed: bool


def entanglement_test(
    graph: "HyperGraph", W12: complex = complex(-1, 0)
) -> EntanglementResult:
    """
    Test entanglement for given edge weight configuration.

    For W12 = -1 (anti-ferromagnetic coupling), the ground state
    should exhibit maximal entanglement (Bell state).

    Args:
        graph: HyperGraph instance
        W12: Edge weight for the test edge

    Returns:
        EntanglementResult with entropy and fidelity
    """
    N = graph.N

    # Set specific edge weight if edges exist
    if len(graph.E) > 0:
        edge = graph.E[0]
        graph.W[edge] = W12
        graph._build_matrices()

    # Get ground state
    L = graph.get_weighted_laplacian()
    eigenvalues, eigenvectors = eigh(np.real(L))

    psi = eigenvectors[:, 0]
    psi = psi / np.linalg.norm(psi)

    # Compute entanglement entropy via reduced density matrix
    # For bipartition: subsystem A = first half, B = second half
    n_A = N // 2
    if n_A < 1:
        n_A = 1

    # Reshape into bipartite form (simplified)
    # Construct density matrix ρ = |ψ⟩⟨ψ|
    rho = np.outer(psi, np.conj(psi))

    # Partial trace over subsystem B
    # Simplified: use diagonal blocks
    rho_A = rho[:n_A, :n_A]
    rho_A = rho_A / (np.trace(rho_A) + 1e-15)

    # Von Neumann entropy: S = -Tr(ρ_A log ρ_A)
    eigs_A = np.linalg.eigvalsh(rho_A)
    eigs_A = eigs_A[eigs_A > 1e-15]
    entropy = -np.sum(eigs_A * np.log2(eigs_A + 1e-15))

    # Concurrence for 2-qubit systems
    if N == 4:
        concurrence = compute_concurrence(rho)
    else:
        concurrence = entropy / np.log2(min(n_A, N - n_A) + 1)  # Normalized proxy

    # Bell state fidelity
    bell_00_11 = np.array([1, 0, 0, 1]) / np.sqrt(2)  # |00⟩ + |11⟩
    if N >= 4:
        fidelity = np.abs(np.vdot(bell_00_11, psi[:4])) ** 2
    else:
        fidelity = 0.0

    is_entangled = entropy > 0.1 or concurrence > 0.1
    passed = is_entangled and (fidelity > 0.5 or entropy > 0.5)

    return EntanglementResult(
        entropy=float(entropy),
        concurrence=float(concurrence),
        is_entangled=is_entangled,
        bell_fidelity=float(fidelity),
        passed=passed,
    )


def compute_concurrence(rho: NDArray[np.complex128]) -> float:
    """
    Compute concurrence for a 4x4 density matrix.

    C = max(0, λ₁ - λ₂ - λ₃ - λ₄)

    where λᵢ are eigenvalues of sqrt(sqrt(ρ) ρ̃ sqrt(ρ)).

    Args:
        rho: 4x4 density matrix

    Returns:
        Concurrence value in [0, 1]
    """
    if rho.shape != (4, 4):
        return 0.0

    # Pauli Y tensor product
    sigma_y = np.array([[0, -1j], [1j, 0]])
    Y = np.kron(sigma_y, sigma_y)

    # Spin-flipped density matrix
    rho_tilde = Y @ np.conj(rho) @ Y

    # R = sqrt(sqrt(ρ) ρ̃ sqrt(ρ))
    sqrt_rho = matrix_sqrt(rho)
    R = matrix_sqrt(sqrt_rho @ rho_tilde @ sqrt_rho)

    eigenvalues = np.sort(np.real(np.linalg.eigvals(R)))[::-1]
    concurrence = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])

    return float(concurrence)


def matrix_sqrt(A: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Compute matrix square root via eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eig(A)
    sqrt_eigs = np.sqrt(np.maximum(eigenvalues, 0))
    return eigenvectors @ np.diag(sqrt_eigs) @ np.linalg.inv(eigenvectors)


def ghz_state_test(graph: "HyperGraph", n_qubits: int = 3) -> GHZResult:
    """
    Test GHZ state preparation from graph structure.

    GHZ state: |GHZ⟩ = (|00...0⟩ + |11...1⟩) / √2

    Args:
        graph: HyperGraph instance
        n_qubits: Number of qubits in GHZ state

    Returns:
        GHZResult with fidelity measure
    """
    # Target GHZ state
    dim = 2**n_qubits
    ghz = np.zeros(dim, dtype=np.complex128)
    ghz[0] = 1 / np.sqrt(2)  # |00...0⟩
    ghz[-1] = 1 / np.sqrt(2)  # |11...1⟩

    # Get graph ground state
    L = graph.get_laplacian()
    eigenvalues, eigenvectors = eigh(L)
    psi = eigenvectors[:, 0]
    psi = psi / np.linalg.norm(psi)

    # Map graph state to qubit space
    N = graph.N
    if N >= dim:
        psi_mapped = psi[:dim]
    else:
        psi_mapped = np.zeros(dim, dtype=np.complex128)
        psi_mapped[:N] = psi

    psi_mapped = psi_mapped / (np.linalg.norm(psi_mapped) + 1e-15)

    # Compute fidelity
    fidelity = np.abs(np.vdot(ghz, psi_mapped)) ** 2

    passed = fidelity > 0.5

    return GHZResult(
        fidelity=float(fidelity),
        n_qubits=n_qubits,
        state_vector=psi_mapped,
        passed=passed,
    )


def quantum_recovery_suite(graph: "HyperGraph") -> dict:
    """
    Run complete quantum mechanics recovery suite.

    Args:
        graph: HyperGraph instance

    Returns:
        Suite results
    """
    entanglement = entanglement_test(graph)
    ghz = ghz_state_test(graph)

    return {
        "entanglement": {
            "passed": entanglement.passed,
            "entropy": entanglement.entropy,
            "bell_fidelity": entanglement.bell_fidelity,
        },
        "ghz": {
            "passed": ghz.passed,
            "fidelity": ghz.fidelity,
        },
        "all_passed": entanglement.passed and ghz.passed,
    }
