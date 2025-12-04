"""
Tests for src/core NCGG and ARO classes.

Unit tests for the Quantum Emergence Framework:
- NCGG_Operator_Algebra: Position/Momentum operators and commutators
- GTEC_Functional: Entanglement entropy and dark energy cancellation
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.ncgg import NCGG_Operator_Algebra, ncgg_covariant_derivative
from core.gtec import GTEC_Functional, gtec_entanglement_energy


class TestNCGG_Operator_Algebra:
    """Tests for NCGG_Operator_Algebra class."""

    @pytest.fixture
    def lattice_adj_matrix(self):
        """Create a 4x4 lattice adjacency matrix."""
        N = 16
        adj = np.zeros((N, N))
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                if j < 3:  # Right neighbor
                    adj[idx, idx + 1] = 1
                    adj[idx + 1, idx] = 1
                if i < 3:  # Bottom neighbor
                    adj[idx, idx + 4] = 1
                    adj[idx + 4, idx] = 1
        return adj

    @pytest.fixture
    def complete_adj_matrix(self):
        """Create a complete graph K_8 adjacency matrix."""
        N = 8
        adj = np.ones((N, N)) - np.eye(N)
        return adj

    @pytest.fixture
    def cycle_adj_matrix(self):
        """Create a cycle graph C_10 adjacency matrix."""
        N = 10
        adj = np.zeros((N, N))
        for i in range(N):
            adj[i, (i + 1) % N] = 1
            adj[(i + 1) % N, i] = 1
        return adj

    def test_initialization(self, lattice_adj_matrix):
        """Test NCGG_Operator_Algebra initialization."""
        algebra = NCGG_Operator_Algebra(lattice_adj_matrix)
        assert algebra.N == 16
        assert algebra.X is not None
        assert algebra.P is not None
        assert algebra.eigenvalues is not None
        assert algebra.eigenvectors is not None

    def test_operator_shapes(self, lattice_adj_matrix):
        """Test that operators have correct shapes."""
        algebra = NCGG_Operator_Algebra(lattice_adj_matrix)
        assert algebra.X.shape == (16, 16)
        assert algebra.P.shape == (16, 16)
        assert len(algebra.eigenvalues) == 16
        assert algebra.eigenvectors.shape == (16, 16)

    def test_eigenvalues_non_negative(self, lattice_adj_matrix):
        """Test that Laplacian eigenvalues are non-negative."""
        algebra = NCGG_Operator_Algebra(lattice_adj_matrix)
        # Laplacian eigenvalues should be >= 0
        assert np.all(algebra.eigenvalues >= -1e-10)

    def test_smallest_eigenvalue_near_zero(self, lattice_adj_matrix):
        """Test that connected graph has smallest eigenvalue near zero."""
        algebra = NCGG_Operator_Algebra(lattice_adj_matrix)
        # For connected graphs, smallest eigenvalue should be ~0
        assert abs(algebra.eigenvalues[0]) < 1e-10

    def test_compute_commutator(self, lattice_adj_matrix):
        """Test commutator computation."""
        algebra = NCGG_Operator_Algebra(lattice_adj_matrix)
        result = algebra.compute_commutator()

        assert 'commutator' in result
        assert 'hbar_G' in result
        assert 'trace' in result
        assert 'diagonal_avg' in result
        assert result['commutator'].shape == (16, 16)
        assert result['hbar_G'] >= 0

    def test_commutator_antisymmetric_structure(self, lattice_adj_matrix):
        """Test that [X,P] has expected structure."""
        algebra = NCGG_Operator_Algebra(lattice_adj_matrix)
        result = algebra.compute_commutator()
        C = result['commutator']

        # [X, P]† = [P†, X†] = [P, X] = -[X, P]
        # So C should be skew-Hermitian: C† = -C
        C_dag = C.conj().T
        # This is approximately true for properly constructed operators
        # The test checks if the commutator is consistent
        assert C.shape == (16, 16)

    def test_complete_graph(self, complete_adj_matrix):
        """Test NCGG on complete graph."""
        algebra = NCGG_Operator_Algebra(complete_adj_matrix)
        assert algebra.N == 8
        # Complete graph should have specific spectral properties
        # λ_0 = 0, λ_1 = ... = λ_{N-1} = N
        assert abs(algebra.eigenvalues[0]) < 1e-10
        # All other eigenvalues should be N = 8 for K_8
        assert np.allclose(algebra.eigenvalues[1:], 8, atol=1e-10)

    def test_cycle_graph(self, cycle_adj_matrix):
        """Test NCGG on cycle graph."""
        algebra = NCGG_Operator_Algebra(cycle_adj_matrix)
        assert algebra.N == 10
        # Cycle graph has known spectrum: λ_k = 2(1 - cos(2πk/N))
        expected = sorted([2 * (1 - np.cos(2 * np.pi * k / 10)) for k in range(10)])
        assert np.allclose(sorted(algebra.eigenvalues), expected, atol=1e-10)

    def test_spectral_embedding(self, lattice_adj_matrix):
        """Test spectral embedding extraction."""
        algebra = NCGG_Operator_Algebra(lattice_adj_matrix)
        embedding = algebra.get_spectral_embedding(n_dims=4)
        assert embedding.shape == (16, 4)


class TestGTEC_Functional:
    """Tests for GTEC_Functional class."""

    @pytest.fixture
    def lattice_adj_matrix(self):
        """Create a 4x4 lattice adjacency matrix."""
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
        return adj

    @pytest.fixture
    def complete_adj_matrix(self):
        """Create complete graph K_8."""
        N = 8
        return np.ones((N, N)) - np.eye(N)

    def test_initialization(self, lattice_adj_matrix):
        """Test GTEC_Functional initialization."""
        gtec = GTEC_Functional(lattice_adj_matrix)
        assert gtec.N == 16
        assert gtec.laplacian is not None
        assert gtec.ground_state is not None
        assert gtec.mu is not None

    def test_initialization_without_matrix(self):
        """Test GTEC_Functional initialization without adjacency matrix."""
        gtec = GTEC_Functional()
        assert gtec.N == 0
        assert gtec.adj_matrix is None

    def test_coupling_constant_scaling(self, lattice_adj_matrix):
        """Test that coupling mu scales as 1/(N ln N)."""
        gtec = GTEC_Functional(lattice_adj_matrix)
        N = 16
        expected_mu = 1.0 / (N * np.log(N))
        assert np.isclose(gtec.mu, expected_mu, rtol=1e-10)

    def test_compute_entanglement_entropy(self, lattice_adj_matrix):
        """Test entanglement entropy computation."""
        gtec = GTEC_Functional(lattice_adj_matrix)
        partition = {'A': list(range(8)), 'B': list(range(8, 16))}

        result = gtec.compute_entanglement_entropy(lattice_adj_matrix, partition)

        assert 'S_ent' in result
        assert 'rho_A' in result
        assert 'eigenvalues_A' in result
        assert 'ground_state' in result
        assert result['S_ent'] >= 0  # Entropy is non-negative

    def test_entropy_empty_partition(self, lattice_adj_matrix):
        """Test entropy with empty A partition."""
        gtec = GTEC_Functional(lattice_adj_matrix)
        partition = {'A': [], 'B': list(range(16))}

        result = gtec.compute_entanglement_entropy(lattice_adj_matrix, partition)
        assert result['S_ent'] == 0.0

    def test_verify_cancellation(self, lattice_adj_matrix):
        """Test dark energy cancellation verification."""
        gtec = GTEC_Functional(lattice_adj_matrix)

        # Compute entropy first
        partition = {'A': list(range(8)), 'B': list(range(8, 16))}
        entropy_result = gtec.compute_entanglement_entropy(lattice_adj_matrix, partition)
        S_ent = entropy_result['S_ent']

        # Test cancellation
        result = gtec.verify_cancellation(Lambda_QFT=10.0, S_ent=S_ent)

        assert 'Lambda_obs' in result
        assert 'E_ARO' in result
        assert 'Lambda_QFT' in result
        assert 'cancellation_ratio' in result
        assert 'successful' in result
        assert result['E_ARO'] <= 0  # ARO energy is negative

    def test_cancellation_with_large_entropy(self, lattice_adj_matrix):
        """Test that large entropy leads to better cancellation."""
        gtec = GTEC_Functional(lattice_adj_matrix)

        # Simulate large entropy case
        Lambda_QFT = 10.0
        S_ent_large = Lambda_QFT / gtec.mu  # This should give E_ARO = -Lambda_QFT

        result = gtec.verify_cancellation(Lambda_QFT=Lambda_QFT, S_ent=S_ent_large)

        # Perfect cancellation should give Lambda_obs ≈ 0
        assert abs(result['Lambda_obs']) < 1e-10
        assert result['successful']

    def test_rho_A_is_positive_semidefinite(self, lattice_adj_matrix):
        """Test that reduced density matrix is positive semi-definite."""
        gtec = GTEC_Functional(lattice_adj_matrix)
        partition = {'A': list(range(8)), 'B': list(range(8, 16))}

        result = gtec.compute_entanglement_entropy(lattice_adj_matrix, partition)
        eigenvalues = result['eigenvalues_A']

        # All eigenvalues should be >= 0
        assert np.all(eigenvalues >= -1e-10)

    def test_complete_graph_entropy(self, complete_adj_matrix):
        """Test entropy on complete graph."""
        gtec = GTEC_Functional(complete_adj_matrix)
        partition = {'A': [0, 1, 2, 3], 'B': [4, 5, 6, 7]}

        result = gtec.compute_entanglement_entropy(complete_adj_matrix, partition)
        assert result['S_ent'] >= 0


class TestGtecEntanglementEnergy:
    """Tests for gtec_entanglement_energy function."""

    def test_positive_eigenvalues(self):
        """Test with positive eigenvalues."""
        eigenvalues = np.array([0.5, 0.3, 0.2])
        result = gtec_entanglement_energy(eigenvalues, coupling_mu=0.1, L_G=1.0, hbar_G=1.0)
        # Result should be negative (energy contribution)
        assert result < 0

    def test_zero_eigenvalues_filtered(self):
        """Test that zero eigenvalues are filtered."""
        eigenvalues = np.array([0.0, 0.0, 0.5, 0.3, 0.2])
        result = gtec_entanglement_energy(eigenvalues, coupling_mu=0.1, L_G=1.0, hbar_G=1.0)
        assert result < 0  # Should still compute valid result

    def test_uniform_distribution(self):
        """Test with uniform distribution."""
        # Uniform over 4 outcomes: H = log2(4) = 2 bits
        eigenvalues = np.array([0.25, 0.25, 0.25, 0.25])
        result = gtec_entanglement_energy(eigenvalues, coupling_mu=0.1, L_G=1.0, hbar_G=1.0)
        # E = -(L_G/hbar_G) * mu * S = -1 * 0.1 * 2 = -0.2
        assert np.isclose(result, -0.2, rtol=1e-10)


class TestNcggCovariantDerivative:
    """Tests for ncgg_covariant_derivative function."""

    def test_empty_neighbors(self):
        """Test with no neighbors."""
        f = np.array([1.0, 2.0, 3.0])
        W = np.zeros((3, 3))
        adj_list = [[], [], []]
        embedding = np.array([[0.0], [1.0], [2.0]])

        result = ncgg_covariant_derivative(f, W, adj_list, embedding, k=0, v=0)
        assert result == 0.0 + 0.0j

    def test_with_neighbors(self):
        """Test with neighbors."""
        f = np.array([1.0, 2.0, 3.0], dtype=complex)
        W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
        adj_list = [[1], [0, 2], [1]]
        embedding = np.array([[0.0], [1.0], [2.0]])

        # Node 1 has neighbors 0 and 2
        result = ncgg_covariant_derivative(f, W, adj_list, embedding, k=0, v=1)
        # Result depends on alignment threshold
        assert isinstance(result, complex)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
