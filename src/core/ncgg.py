"""
ncgg.py - Non-Commutative Graph Geometry (NCGG) Operators

RIRH v9.5 Quantum Emergence Framework

This module implements NCGG operators for discrete quantum spacetime,
including position (X) and momentum (P) operators constructed via
Laplacian spectral decomposition and gauge-covariant differences.

Key Components:
- NCGG_Operator_Algebra: Class for constructing X, P operators and commutators
- ncgg_covariant_derivative: Discrete gauge-covariant derivative function

References:
- Non-commutative geometry (Connes)
- Graph quantum mechanics
- Discrete gauge theory
"""

import numpy as np
from scipy import sparse
from scipy.linalg import eigh

# Alignment threshold for directional filtering (cos(60°) ≈ 0.5)
ALIGNMENT_THRESHOLD = 0.5


class NCGG_Operator_Algebra:
    """
    Non-Commutative Graph Geometry Operator Algebra.

    Constructs position (X) and momentum (P) operators on a graph using
    the spectral eigenvector method (Laplacian eigen-decomposition) and
    gauge-covariant difference operators.

    The construction follows the GNS (Gelfand-Naimark-Segal) paradigm:
    - Position basis ψ_k defined by Laplacian eigenvectors
    - X operator: multiplication by eigenvalue-weighted position
    - P operator: gauge-covariant finite difference operator

    Attributes:
        N: Number of nodes in the graph
        adj_matrix: Adjacency matrix (sparse or dense)
        eigenvalues: Laplacian eigenvalues
        eigenvectors: Laplacian eigenvectors (position basis ψ_k)
        X: Position operator (sparse matrix)
        P: Momentum operator (sparse matrix)
        hbar_G: Emergent Planck constant (computed from commutator)
    """

    def __init__(self, adj_matrix):
        """
        Initialize NCGG Operator Algebra from adjacency matrix.

        Args:
            adj_matrix: Adjacency matrix (N x N), can be dense or sparse.
                        Must be symmetric for undirected graphs.
        """
        # Convert to dense numpy array if sparse
        if sparse.issparse(adj_matrix):
            self.adj_matrix = adj_matrix.toarray()
        else:
            self.adj_matrix = np.asarray(adj_matrix, dtype=float)

        self.N = self.adj_matrix.shape[0]
        self.eigenvalues = None
        self.eigenvectors = None
        self.X = None
        self.P = None
        self.hbar_G = None

        # Construct operators upon initialization
        self.construct_operators(self.adj_matrix)

    def construct_operators(self, adj_matrix):
        """
        Construct sparse X (position) and P (momentum) operators.

        Uses the spectral eigenvector method:
        1. Compute graph Laplacian L = D - A
        2. Perform eigen-decomposition to get eigenvalues λ_k and eigenvectors ψ_k
        3. Position operator X: diagonal in position basis, scaled by coordinates
        4. Momentum operator P: gauge-covariant difference operator

        Args:
            adj_matrix: Adjacency matrix (N x N)

        Returns:
            tuple: (X, P) sparse position and momentum operators
        """
        N = self.N
        A = np.asarray(adj_matrix, dtype=float)

        # Compute graph Laplacian: L = D - A
        degrees = np.sum(A, axis=1)
        D = np.diag(degrees)
        L = D - A

        # Eigen-decomposition of Laplacian
        # Eigenvalues are non-negative, smallest is 0 for connected graphs
        self.eigenvalues, self.eigenvectors = eigh(L)

        # Position operator X: Defined in spectral basis
        # X_ij = sum_k λ_k ψ_k(i) ψ_k(j)
        # This gives position weighted by spectral coordinates
        # We use a simplified construction: X = Ψ Λ Ψ^T (spectral reconstruction)
        Lambda = np.diag(self.eigenvalues)
        Psi = self.eigenvectors

        # Position operator: spectral embedding coordinates
        # For a simple position operator, use first d_s non-trivial eigenvectors
        d_s = min(4, N - 1)  # Use up to 4 spectral dimensions
        X_dense = np.zeros((N, N), dtype=float)
        for k in range(1, d_s + 1):  # Skip zero eigenvalue (k=0)
            if k < N:
                psi_k = Psi[:, k]
                # Position operator contribution from k-th mode
                X_dense += self.eigenvalues[k] * np.outer(psi_k, psi_k)

        self.X = sparse.csr_matrix(X_dense)

        # Momentum operator P: Gauge-covariant difference
        # P = -i * (D_+ - D_-) where D_± are forward/backward differences
        # In graph context: P_ij = -i * A_ij * sign(j-i) for connected nodes
        P_dense = np.zeros((N, N), dtype=complex)

        for i in range(N):
            for j in range(N):
                if A[i, j] > 0 and i != j:
                    # Gauge-covariant difference operator
                    # Direction determined by spectral embedding
                    embedding_i = Psi[i, 1:d_s + 1] if d_s + 1 <= N else Psi[i, 1:]
                    embedding_j = Psi[j, 1:d_s + 1] if d_s + 1 <= N else Psi[j, 1:]

                    # Direction vector in spectral space
                    direction = embedding_j - embedding_i
                    norm = np.linalg.norm(direction)

                    if norm > 1e-12:
                        # Gauge connection: phase from spectral direction
                        # P_ij = -i * A_ij * (direction component)
                        P_dense[i, j] = -1j * A[i, j] * np.sign(np.sum(direction))

        # Antisymmetrize to ensure Hermiticity of iP
        P_dense = (P_dense - P_dense.T.conj()) / 2

        self.P = sparse.csr_matrix(P_dense)

        return self.X, self.P

    def compute_commutator(self, X=None, P=None):
        """
        Compute the commutator [X, P] = XP - PX.

        The commutator measures the non-commutativity of position and momentum.
        For canonical quantization, [X, P] = i ℏ_G where ℏ_G is the emergent
        Planck constant derived from the graph structure.

        Args:
            X: Position operator (default: self.X)
            P: Momentum operator (default: self.P)

        Returns:
            dict: Contains:
                - 'commutator': The full commutator matrix C = [X, P]
                - 'hbar_G': Estimated ℏ_G from Tr(C)/N or diagonal average
                - 'trace': Trace of the commutator
                - 'diagonal_avg': Average of diagonal elements
        """
        if X is None:
            X = self.X
        if P is None:
            P = self.P

        # Convert to dense if sparse for computation
        if sparse.issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = np.asarray(X)

        if sparse.issparse(P):
            P_dense = P.toarray()
        else:
            P_dense = np.asarray(P)

        # Commutator: C = XP - PX
        C = X_dense @ P_dense - P_dense @ X_dense

        # Extract hbar_G estimate
        # For canonical quantization: [X, P] = i * hbar_G * I
        # So hbar_G ≈ -i * Tr(C) / N or average of -i * C_ii
        trace_C = np.trace(C)
        diagonal_avg = np.mean(np.diag(C))

        # hbar_G is the imaginary part (since [X,P] should be purely imaginary)
        # Estimate from diagonal: hbar_G = Im(diagonal_avg) or -i * diagonal_avg
        hbar_G_estimate = np.abs(np.imag(diagonal_avg))

        # Alternative: from trace
        hbar_G_from_trace = np.abs(np.imag(trace_C)) / self.N

        # Use the more stable estimate
        self.hbar_G = max(hbar_G_estimate, hbar_G_from_trace)

        return {
            'commutator': C,
            'hbar_G': self.hbar_G,
            'trace': trace_C,
            'diagonal_avg': diagonal_avg,
            'hbar_G_from_trace': hbar_G_from_trace,
            'hbar_G_from_diagonal': hbar_G_estimate
        }

    def get_spectral_embedding(self, n_dims=4):
        """
        Get spectral embedding coordinates for all nodes.

        Args:
            n_dims: Number of spectral dimensions to use

        Returns:
            ndarray: (N x n_dims) embedding coordinates
        """
        if self.eigenvectors is None:
            raise ValueError("Operators not constructed. Call construct_operators first.")

        n_dims = min(n_dims, self.N - 1)
        # Skip first eigenvector (constant for connected graphs)
        return self.eigenvectors[:, 1:n_dims + 1]


def ncgg_covariant_derivative(f, W, adj_list, embedding, k, v):
    """
    Constructs the discrete gauge-covariant derivative D_k f(v).
    Uses spectral embedding to identify directional neighbors via projection.

    Formalism v9.5 Section IV.A

    Args:
        f (array): Scalar field on nodes.
        W (matrix): Complex edge weights.
        adj_list (list of lists): Adjacency list.
        embedding (matrix): Spectral embedding coordinates [N, d_s].
        k (int): Direction index.
        v (int): Node index.

    Returns:
        complex: The gauge-covariant derivative value D_k f(v).
    """
    neighbors = adj_list[v]
    if not neighbors:
        return 0.0 + 0.0j

    # Basis vector for direction k
    vec_k = np.zeros(embedding.shape[1])
    vec_k[k] = 1.0

    sum_val = 0.0 + 0.0j
    count = 0
    total_weight = 0.0

    for u in neighbors:
        # Check alignment via projection
        edge_vec = embedding[u] - embedding[v]
        # Directional filter (alignment threshold corresponds to ~60° cone)
        if np.dot(edge_vec, vec_k) > ALIGNMENT_THRESHOLD:
            # Parallel Transport: phase of W_vu
            w_vu = W[v, u]
            # Handle potential zero weight
            if np.abs(w_vu) > 1e-12:
                phase = w_vu / np.abs(w_vu)
                sum_val += f[u] * phase - f[v]
                total_weight += np.abs(w_vu)
                count += 1

    if count == 0:
        return 0.0 + 0.0j

    # Scale by average connection strength
    avg_weight = total_weight / count
    return (avg_weight / count) * sum_val


if __name__ == "__main__":
    # Verification test for NCGG operators
    print("=" * 60)
    print("NCGG Operator Algebra Verification")
    print("=" * 60)
    
    # Test 1: Cycle graph C_10
    print("\nTest 1: Cycle graph C_10")
    N = 10
    adj = np.zeros((N, N))
    for i in range(N):
        adj[i, (i + 1) % N] = 1
        adj[(i + 1) % N, i] = 1
    
    algebra = NCGG_Operator_Algebra(adj)
    print(f"  Graph size N = {algebra.N}")
    print(f"  X operator shape: {algebra.X.shape}")
    print(f"  P operator shape: {algebra.P.shape}")
    
    # Expected eigenvalues for cycle graph: λ_k = 2(1 - cos(2πk/N))
    expected = sorted([2 * (1 - np.cos(2 * np.pi * k / N)) for k in range(N)])
    computed = sorted(algebra.eigenvalues)
    match = np.allclose(computed, expected, atol=1e-10)
    print(f"  Eigenvalue spectrum correct: {match}")
    
    # Test 2: Commutator
    print("\nTest 2: Commutator [X, P]")
    result = algebra.compute_commutator()
    print(f"  Trace of [X,P] = {result['trace']:.6f}")
    print(f"  hbar_G (from diagonal) = {result['hbar_G_from_diagonal']:.6f}")
    print(f"  hbar_G (from trace) = {result['hbar_G_from_trace']:.6f}")
    
    # Test 3: Complete graph K_8
    print("\nTest 3: Complete graph K_8")
    N = 8
    adj = np.ones((N, N)) - np.eye(N)
    algebra = NCGG_Operator_Algebra(adj)
    print(f"  Graph size N = {algebra.N}")
    print(f"  Smallest eigenvalue (should be ~0): {algebra.eigenvalues[0]:.10f}")
    print(f"  Other eigenvalues (should be N=8): {algebra.eigenvalues[1]:.6f}")
    
    # Test 4: Spectral embedding
    print("\nTest 4: Spectral embedding (4x4 lattice)")
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
    
    algebra = NCGG_Operator_Algebra(adj)
    embedding = algebra.get_spectral_embedding(n_dims=4)
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  First node embedding: {embedding[0, :]}")
    
    # Test 5: Covariant derivative
    print("\nTest 5: Gauge-covariant derivative")
    f = np.ones(N, dtype=complex)
    W = adj.astype(complex)
    adj_list = [np.where(adj[v] > 0)[0].tolist() for v in range(N)]
    D_0_f = ncgg_covariant_derivative(f, W, adj_list, embedding, k=0, v=0)
    print(f"  D_0 f(0) for constant field = {D_0_f}")
    
    print("\n" + "=" * 60)
    print("NCGG Verification Complete")
    print("=" * 60)
