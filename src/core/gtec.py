"""
gtec.py - Graph Topological Emergent Complexity (GTEC) Functional

RIRH v9.5 Quantum Emergence Framework

This module implements the GTEC functional for computing entanglement entropy
and verifying dark energy cancellation through vacuum entanglement.

Key Components:
- GTEC_Functional: Class for entanglement entropy and cancellation verification
- gtec_entanglement_energy: Function for negative vacuum energy calculation

The GTEC mechanism provides the negative energy contribution that cancels
the large positive QFT vacuum energy, resulting in the observed small
cosmological constant.

References:
- Holographic entanglement entropy
- Graph-theoretic quantum mechanics
- Dark energy cancellation via SOTE principle
"""

import numpy as np
from scipy.linalg import eigh
from scipy import sparse


class GTEC_Functional:
    """
    Graph Topological Emergent Complexity Functional.

    Computes entanglement entropy for bipartite graph partitions and
    verifies the dark energy cancellation mechanism.

    The entanglement entropy is computed by:
    1. Constructing the graph Hamiltonian (Laplacian)
    2. Finding the ground state |Ω⟩
    3. Computing the reduced density matrix ρ_A by tracing out region B
    4. Calculating S = -Tr(ρ_A log₂ ρ_A)

    Attributes:
        N: Number of nodes in the graph
        adj_matrix: Adjacency matrix
        laplacian: Graph Laplacian matrix
        ground_state: Ground state vector |Ω⟩
        mu: Coupling constant for GTEC energy (≈ 1/(N ln N))
    """

    def __init__(self, adj_matrix=None):
        """
        Initialize GTEC Functional.

        Args:
            adj_matrix: Optional adjacency matrix. If provided, computes
                       the ground state upon initialization.
        """
        self.adj_matrix = None
        self.laplacian = None
        self.ground_state = None
        self.N = 0
        self.mu = None

        if adj_matrix is not None:
            self._initialize_from_adjacency(adj_matrix)

    def _initialize_from_adjacency(self, adj_matrix):
        """Initialize from adjacency matrix."""
        if sparse.issparse(adj_matrix):
            self.adj_matrix = adj_matrix.toarray()
        else:
            self.adj_matrix = np.asarray(adj_matrix, dtype=float)

        self.N = self.adj_matrix.shape[0]

        # Compute Laplacian
        degrees = np.sum(self.adj_matrix, axis=1)
        D = np.diag(degrees)
        self.laplacian = D - self.adj_matrix

        # Compute ground state (smallest eigenvalue eigenvector)
        eigenvalues, eigenvectors = eigh(self.laplacian)
        self.ground_state = eigenvectors[:, 0]  # Ground state

        # Coupling constant derived from graph structure
        # mu ≈ 1 / (N ln N) as per SOTE principle
        if self.N > 1:
            self.mu = 1.0 / (self.N * np.log(self.N))
        else:
            self.mu = 1.0

    def compute_entanglement_entropy(self, adj_matrix, partition):
        """
        Compute entanglement entropy for a bipartite graph partition.

        Divides the graph into region A and region B, constructs the
        reduced density matrix ρ_A by tracing out B, and computes
        the von Neumann entropy S = -Tr(ρ_A log₂ ρ_A).

        Args:
            adj_matrix: Adjacency matrix (N x N)
            partition: Dictionary with keys 'A' and 'B' containing
                      lists of node indices for each region.
                      Example: {'A': [0, 1, 2], 'B': [3, 4, 5]}

        Returns:
            dict: Contains:
                - 'S_ent': Entanglement entropy in bits
                - 'rho_A': Reduced density matrix for region A
                - 'eigenvalues_A': Eigenvalues of ρ_A
                - 'ground_state': Ground state vector |Ω⟩
        """
        # Initialize if new adjacency matrix provided
        if adj_matrix is not None:
            self._initialize_from_adjacency(adj_matrix)

        region_A = partition.get('A', [])
        region_B = partition.get('B', [])

        if len(region_A) == 0:
            return {
                'S_ent': 0.0,
                'rho_A': np.array([[1.0]]),
                'eigenvalues_A': np.array([1.0]),
                'ground_state': self.ground_state
            }

        N = self.N
        n_A = len(region_A)
        n_B = len(region_B)

        # Ground state as density matrix: |Ω⟩⟨Ω|
        # For pure state, ρ = |Ω⟩⟨Ω|
        rho_full = np.outer(self.ground_state, self.ground_state)

        # Compute reduced density matrix ρ_A by tracing out region B
        # For a pure state |ψ⟩ = Σ_{ij} c_{ij} |i_A⟩|j_B⟩
        # ρ_A = Tr_B(|ψ⟩⟨ψ|) = Σ_j ⟨j_B|ψ⟩⟨ψ|j_B⟩

        # Reshape ground state for partial trace
        # We treat it as a bipartite system A ⊗ B
        # This is an approximation for graph states

        # Extract submatrices
        A_indices = np.array(region_A)
        B_indices = np.array(region_B)

        # Simple partial trace: sum over B indices
        rho_A = np.zeros((n_A, n_A), dtype=complex)

        for i, a_i in enumerate(region_A):
            for j, a_j in enumerate(region_A):
                # Sum over all B indices
                rho_A[i, j] = rho_full[a_i, a_j]
                # Add correlations through B
                for b in region_B:
                    rho_A[i, j] += rho_full[a_i, b] * rho_full[b, a_j]

        # Normalize ρ_A to have trace 1
        trace_rho_A = np.trace(rho_A)
        if np.abs(trace_rho_A) > 1e-12:
            rho_A = rho_A / trace_rho_A

        # Ensure Hermiticity
        rho_A = (rho_A + rho_A.conj().T) / 2

        # Compute eigenvalues
        eigenvalues_A = np.linalg.eigvalsh(rho_A)

        # Filter to valid probability eigenvalues
        eigenvalues_A = np.real(eigenvalues_A)
        eigenvalues_A = np.clip(eigenvalues_A, 0, 1)

        # Normalize eigenvalues
        sum_eig = np.sum(eigenvalues_A)
        if sum_eig > 1e-12:
            eigenvalues_A = eigenvalues_A / sum_eig

        # Von Neumann entropy: S = -Tr(ρ log₂ ρ) = -Σ λ_i log₂(λ_i)
        S_ent = 0.0
        for lam in eigenvalues_A:
            if lam > 1e-12:
                S_ent -= lam * np.log2(lam)

        return {
            'S_ent': float(S_ent),
            'rho_A': rho_A,
            'eigenvalues_A': eigenvalues_A,
            'ground_state': self.ground_state
        }

    def verify_cancellation(self, Lambda_QFT, S_ent):
        """
        Verify dark energy cancellation via GTEC mechanism.

        The GTEC mechanism provides negative energy E_GTEC = -μ S_ent
        that should cancel the large positive QFT vacuum energy.

        The observed cosmological constant is:
        Λ_obs = Λ_QFT + E_GTEC

        For successful cancellation, |Λ_obs| << |Λ_QFT|.

        Args:
            Lambda_QFT: Theoretical positive vacuum energy (scaled to graph size).
                       This represents the QFT prediction for vacuum energy.
            S_ent: Entanglement entropy computed from the graph partition.

        Returns:
            dict: Contains:
                - 'Lambda_obs': Residual observed cosmological constant
                - 'E_GTEC': Negative energy from entanglement
                - 'Lambda_QFT': Input QFT vacuum energy
                - 'cancellation_ratio': |Λ_obs| / |Λ_QFT| (smaller is better)
                - 'successful': True if cancellation is significant
        """
        # GTEC negative energy contribution
        # E_GTEC = -μ * S_ent
        if self.mu is None:
            # Use default coupling if not initialized
            self.mu = 0.1

        E_GTEC = -self.mu * S_ent

        # Observed cosmological constant (residual)
        Lambda_obs = Lambda_QFT + E_GTEC

        # Cancellation ratio
        if np.abs(Lambda_QFT) > 1e-12:
            cancellation_ratio = np.abs(Lambda_obs) / np.abs(Lambda_QFT)
        else:
            cancellation_ratio = 0.0

        # Consider cancellation successful if ratio < 0.1 (90% cancellation)
        successful = cancellation_ratio < 0.1

        return {
            'Lambda_obs': float(Lambda_obs),
            'E_GTEC': float(E_GTEC),
            'Lambda_QFT': float(Lambda_QFT),
            'cancellation_ratio': float(cancellation_ratio),
            'successful': successful,
            'mu': float(self.mu)
        }


def gtec_entanglement_energy(eigenvalues, coupling_mu, L_G, hbar_G):
    """
    Explicitly calculates the negative energy contribution from vacuum entanglement.

    Formalism v9.5: E_GTEC = - mu * S_ent

    Args:
        eigenvalues (np.array): Normalized spectrum of the entanglement Hamiltonian.
        coupling_mu (float): Derived coupling constant (approx 1/(N ln N)).
        L_G (float): Emergent graph scale (dimensionless).
        hbar_G (float): Emergent action quantum (dimensionless).

    Returns:
        float: Negative energy value (E_gtec).
    """
    # Filter zeros to ensure log stability
    spectrum = eigenvalues[eigenvalues > 0]

    # Von Neumann Entropy (bits)
    S_ent = -np.sum(spectrum * np.log2(spectrum))

    # Thermodynamic relation: Energy = - mu * Entropy
    # The negative sign is crucial for Dark Energy cancellation.
    E_gtec = -(L_G / hbar_G) * coupling_mu * S_ent

    return E_gtec


if __name__ == "__main__":
    # Verification test for GTEC functional
    print("=" * 60)
    print("GTEC Functional Verification")
    print("=" * 60)
    
    # Test 1: Uniform eigenvalue distribution
    print("\nTest 1: Uniform distribution (max entropy)")
    eigenvalues = np.array([0.25, 0.25, 0.25, 0.25])
    E_gtec = gtec_entanglement_energy(eigenvalues, coupling_mu=0.1, L_G=1.0, hbar_G=1.0)
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  E_GTEC = {E_gtec:.6f}")
    print(f"  Expected: -0.2 (entropy = 2 bits)")
    
    # Test 2: Non-uniform distribution
    print("\nTest 2: Non-uniform distribution")
    eigenvalues = np.array([0.5, 0.3, 0.15, 0.05])
    E_gtec = gtec_entanglement_energy(eigenvalues, coupling_mu=0.1, L_G=1.0, hbar_G=1.0)
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  E_GTEC = {E_gtec:.6f}")
    
    # Test 3: GTEC Functional class
    print("\nTest 3: GTEC Functional class on 4x4 lattice")
    N = 16
    adj = np.zeros((N, N))
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            if j < 3:
                adj[idx, idx + 1] = 1
                adj[idx + 1, idx] = 1
            if i < 3:
                adj[idx, idx + 4] = 1
                adj[idx + 4, idx] = 1
    
    gtec = GTEC_Functional(adj)
    print(f"  Graph size N = {gtec.N}")
    print(f"  Coupling mu = {gtec.mu:.6f} (expected: 1/(N*ln(N)) = {1/(16*np.log(16)):.6f})")
    
    # Compute entanglement entropy
    partition = {'A': list(range(8)), 'B': list(range(8, 16))}
    result = gtec.compute_entanglement_entropy(adj, partition)
    print(f"  Entanglement entropy S_ent = {result['S_ent']:.6f} bits")
    
    # Verify cancellation
    Lambda_QFT = 10.0
    cancel_result = gtec.verify_cancellation(Lambda_QFT, result['S_ent'])
    print(f"  Lambda_QFT = {Lambda_QFT}")
    print(f"  E_GTEC = {cancel_result['E_GTEC']:.6f}")
    print(f"  Lambda_obs = {cancel_result['Lambda_obs']:.6f}")
    print(f"  Cancellation ratio = {cancel_result['cancellation_ratio']:.4f}")
    
    print("\n" + "=" * 60)
    print("GTEC Verification Complete")
    print("=" * 60)
