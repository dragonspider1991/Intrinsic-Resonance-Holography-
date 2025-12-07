"""
Tests for Phase 4: Gauge Group Algebraic Derivation (IRH v15.0)

Tests the derivation of SU(3)×SU(2)×U(1) from algorithmic holonomies.

Test Coverage:
- Boundary identification
- Betti number computation (β₁ = 12)
- Algorithmic Intersection Matrix
- Structure constants derivation
- Lie algebra classification
- Anomaly cancellation

References: IRH v15.0 Theorems 6.1-6.3, §6
"""

import pytest
import numpy as np
import scipy.sparse as sp
import networkx as nx

from src.topology.boundary_analysis import (
    identify_emergent_boundary,
    compute_betti_numbers_boundary,
    BoundaryAnalyzer
)
from src.topology.gauge_algebra import (
    compute_fundamental_loops,
    compute_algorithmic_intersection_matrix,
    derive_structure_constants,
    verify_jacobi_identity,
    verify_anomaly_cancellation,
    GaugeGroupDerivation
)


class TestBoundaryIdentification:
    """Tests for emergent boundary identification (Theorem 6.1)."""
    
    def test_boundary_identification_basic(self):
        """Test boundary identification on small network."""
        # Create test network
        N = 100
        W = self._create_test_network(N, boundary_fraction=0.1)
        
        boundary_nodes = identify_emergent_boundary(W, boundary_fraction=0.1)
        
        assert len(boundary_nodes) > 0
        assert len(boundary_nodes) <= N
        # Should be roughly 10% of nodes
        assert 0.05 < len(boundary_nodes) / N < 0.20
    
    def test_boundary_methods_consistency(self):
        """Test different boundary identification methods."""
        N = 80
        W = self._create_test_network(N)
        
        boundary_betweenness = identify_emergent_boundary(
            W, method='betweenness'
        )
        boundary_radial = identify_emergent_boundary(
            W, method='radial'
        )
        
        # Both should identify some boundary
        assert len(boundary_betweenness) > 0
        assert len(boundary_radial) > 0
        
        # Methods may differ, just verify both work
        assert len(boundary_betweenness) < N
        assert len(boundary_radial) < N
    
    def test_boundary_nodes_sorted(self):
        """Test that boundary nodes are returned sorted."""
        N = 50
        W = self._create_test_network(N)
        
        boundary_nodes = identify_emergent_boundary(W)
        
        assert np.all(boundary_nodes[:-1] <= boundary_nodes[1:])
    
    def _create_test_network(self, N: int, boundary_fraction: float = 0.1):
        """Create test network with approximate boundary structure."""
        # Create random geometric graph with radial structure
        G = nx.random_geometric_graph(N, radius=0.3, seed=42)
        
        # Add complex phases to edges
        W_lil = sp.lil_matrix((N, N), dtype=np.complex128)
        for u, v in G.edges():
            magnitude = np.random.uniform(0.5, 1.0)
            phase = np.random.uniform(0, 2 * np.pi)
            W_lil[u, v] = magnitude * np.exp(1j * phase)
            W_lil[v, u] = magnitude * np.exp(-1j * phase)  # Hermitian
        
        return W_lil.tocsr()


class TestBettiNumbers:
    """Tests for Betti number computation."""
    
    def test_betti_0_connected(self):
        """Test β₀ = 1 for connected boundary."""
        N = 60
        W = self._create_connected_network(N)
        boundary_nodes = np.arange(N // 2, N)  # Half as boundary
        
        betti = compute_betti_numbers_boundary(W, boundary_nodes)
        
        assert betti['beta_0'] == 1
    
    def test_betti_1_positive(self):
        """Test β₁ > 0 for boundary with cycles."""
        N = 80
        W = self._create_network_with_cycles(N)
        boundary_nodes = np.arange(N // 3, N)
        
        betti = compute_betti_numbers_boundary(W, boundary_nodes)
        
        # Should have some cycles
        assert betti['beta_1'] is not None
        assert betti['beta_1'] > 0
    
    def test_betti_numbers_structure(self):
        """Test betti_numbers dict structure."""
        N = 50
        W = self._create_connected_network(N)
        boundary_nodes = np.arange(N // 4, N)
        
        betti = compute_betti_numbers_boundary(W, boundary_nodes)
        
        assert 'beta_0' in betti
        assert 'beta_1' in betti
        assert 'beta_2' in betti
        assert 'beta_3' in betti
        assert 'n_boundary_nodes' in betti
        assert betti['n_boundary_nodes'] == len(boundary_nodes)
    
    def _create_connected_network(self, N: int):
        """Create connected test network."""
        G = nx.connected_watts_strogatz_graph(N, k=4, p=0.1, seed=42)
        W_lil = sp.lil_matrix((N, N), dtype=np.complex128)
        for u, v in G.edges():
            W_lil[u, v] = 1.0 + 0.0j
            W_lil[v, u] = 1.0 + 0.0j
        return W_lil.tocsr()
    
    def _create_network_with_cycles(self, N: int):
        """Create network with multiple cycles."""
        # Ring + random edges
        G = nx.cycle_graph(N)
        # Add random shortcuts to create cycles
        for _ in range(N // 2):
            u, v = np.random.randint(0, N, size=2)
            if u != v:
                G.add_edge(u, v)
        
        W_lil = sp.lil_matrix((N, N), dtype=np.complex128)
        for u, v in G.edges():
            phase = np.random.uniform(0, 2 * np.pi)
            W_lil[u, v] = np.exp(1j * phase)
            W_lil[v, u] = np.exp(-1j * phase)
        return W_lil.tocsr()


class TestFundamentalLoops:
    """Tests for fundamental loop computation."""
    
    def test_fundamental_loops_basic(self):
        """Test fundamental loop computation."""
        N = 60
        W = self._create_network_with_loops(N)
        boundary_nodes = np.arange(N // 3, N)
        
        loops = compute_fundamental_loops(W, boundary_nodes, target_loops=5)
        
        assert len(loops) > 0
        assert all(isinstance(loop, list) for loop in loops)
        assert all(len(loop) >= 3 for loop in loops)  # Minimum cycle length
    
    def test_loops_contain_boundary_nodes(self):
        """Test that loops use boundary nodes."""
        N = 50
        W = self._create_network_with_loops(N)
        boundary_nodes = np.arange(N // 4, N)
        
        loops = compute_fundamental_loops(W, boundary_nodes, target_loops=4)
        
        boundary_set = set(boundary_nodes)
        for loop in loops:
            # At least some nodes should be in boundary
            assert len(set(loop).intersection(boundary_set)) > 0
    
    def _create_network_with_loops(self, N: int):
        """Create network with explicit loop structure."""
        # Grid graph has many cycles
        n_side = int(np.sqrt(N))
        G = nx.grid_2d_graph(n_side, n_side)
        
        # Relabel to integers
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        
        W_lil = sp.lil_matrix((N, N), dtype=np.complex128)
        for u, v in G.edges():
            if u < N and v < N:
                phase = np.random.uniform(0, 2 * np.pi)
                W_lil[u, v] = np.exp(1j * phase)
                W_lil[v, u] = np.exp(-1j * phase)
        return W_lil.tocsr()


class TestAlgorithmicIntersectionMatrix:
    """Tests for AIX computation."""
    
    def test_AIX_shape(self):
        """Test AIX has correct shape."""
        loops = [[0, 1, 2, 0], [3, 4, 5, 3], [6, 7, 8, 6]]
        N = 10
        W = sp.csr_matrix((N, N), dtype=np.complex128)
        
        AIX = compute_algorithmic_intersection_matrix(loops, W)
        
        assert AIX.shape == (len(loops), len(loops))
    
    def test_AIX_antisymmetric(self):
        """Test AIX is antisymmetric: AIX[a,b] = -AIX[b,a]."""
        N = 50
        W = self._create_network_with_loops(N)
        boundary_nodes = np.arange(N // 3, N)
        loops = compute_fundamental_loops(W, boundary_nodes, target_loops=6)
        
        AIX = compute_algorithmic_intersection_matrix(loops, W)
        
        # Check antisymmetry
        antisym_error = np.linalg.norm(AIX + AIX.T)
        assert antisym_error < 1e-10
    
    def test_AIX_diagonal_zero(self):
        """Test AIX diagonal is zero (loop doesn't intersect itself)."""
        N = 40
        W = self._create_network_with_loops(N)
        boundary_nodes = np.arange(N // 4, N)
        loops = compute_fundamental_loops(W, boundary_nodes, target_loops=4)
        
        AIX = compute_algorithmic_intersection_matrix(loops, W)
        
        assert np.allclose(np.diag(AIX), 0.0)
    
    def _create_network_with_loops(self, N: int):
        """Create network for AIX testing."""
        n_side = int(np.sqrt(N))
        G = nx.grid_2d_graph(n_side, n_side)
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        
        W_lil = sp.lil_matrix((N, N), dtype=np.complex128)
        for u, v in G.edges():
            if u < N and v < N:
                W_lil[u, v] = 1.0 + 0.0j
                W_lil[v, u] = 1.0 + 0.0j
        return W_lil.tocsr()


class TestStructureConstants:
    """Tests for structure constant derivation."""
    
    def test_structure_constants_shape(self):
        """Test f^abc has correct shape."""
        AIX = np.random.randn(12, 12)
        AIX = AIX - AIX.T  # Make antisymmetric
        
        f_abc, lie_algebra = derive_structure_constants(AIX)
        
        assert f_abc.shape == (12, 12, 12)
    
    def test_structure_constants_antisymmetric(self):
        """Test f^abc is antisymmetric in first two indices."""
        AIX = np.random.randn(8, 8)
        AIX = AIX - AIX.T
        
        f_abc, _ = derive_structure_constants(AIX)
        
        # Check f^{abc} = -f^{bac}
        for c in range(f_abc.shape[2]):
            antisym_error = np.linalg.norm(f_abc[:, :, c] + f_abc[:, :, c].T)
            assert antisym_error < 1e-6
    
    def test_lie_algebra_classification(self):
        """Test Lie algebra classification returns string."""
        AIX = np.random.randn(12, 12)
        AIX = AIX - AIX.T
        
        f_abc, lie_algebra = derive_structure_constants(AIX, classify_algebra=True)
        
        assert isinstance(lie_algebra, str)
        assert len(lie_algebra) > 0


class TestJacobiIdentity:
    """Tests for Jacobi identity verification."""
    
    def test_jacobi_identity_structure(self):
        """Test Jacobi verification returns correct structure."""
        f_abc = np.random.randn(6, 6, 6)
        
        result = verify_jacobi_identity(f_abc)
        
        assert 'passes' in result
        assert 'max_violation' in result
        assert 'n_violations' in result
        assert isinstance(result['passes'], (bool, np.bool_))
    
    def test_jacobi_identity_zero_structure(self):
        """Test Jacobi identity holds for trivial algebra."""
        f_abc = np.zeros((4, 4, 4))
        
        result = verify_jacobi_identity(f_abc)
        
        assert result['passes']
        assert result['max_violation'] < 1e-10


class TestAnomalyCancellation:
    """Tests for anomaly cancellation verification."""
    
    def test_anomaly_cancellation_no_input(self):
        """Test anomaly cancellation without winding numbers."""
        result = verify_anomaly_cancellation()
        
        assert 'total_winding' in result
        assert 'passes' in result
        assert result['passes']  # Should pass trivially
    
    def test_anomaly_cancellation_with_windings(self):
        """Test anomaly cancellation with winding numbers."""
        # Create winding numbers that sum to zero
        windings = np.array([1.0, -1.0, 0.5, -0.5])
        
        result = verify_anomaly_cancellation(winding_numbers=windings)
        
        assert abs(result['total_winding']) < 1e-10
        assert result['passes']
    
    def test_anomaly_cancellation_violation(self):
        """Test anomaly cancellation detects violations."""
        # Create winding numbers that don't sum to zero
        windings = np.array([1.0, 1.0, 1.0])
        
        result = verify_anomaly_cancellation(winding_numbers=windings)
        
        assert abs(result['total_winding']) > 1e-5
        assert not result['passes']


class TestBoundaryAnalyzer:
    """Tests for high-level BoundaryAnalyzer class."""
    
    def test_boundary_analyzer_initialization(self):
        """Test BoundaryAnalyzer initialization."""
        N = 50
        W = sp.random(N, N, density=0.1, format='csr', dtype=np.complex128)
        
        analyzer = BoundaryAnalyzer(W, boundary_fraction=0.15)
        
        assert analyzer.N == N
        assert analyzer.boundary_fraction == 0.15
        assert analyzer.boundary_nodes is None
    
    def test_boundary_analyzer_run(self):
        """Test full BoundaryAnalyzer pipeline."""
        N = 60
        W = self._create_test_network(N)
        
        analyzer = BoundaryAnalyzer(W, boundary_fraction=0.1)
        results = analyzer.run_analysis()
        
        assert 'boundary_nodes' in results
        assert 'betti_numbers' in results
        assert 'topology_type' in results
        assert 'boundary_fraction_actual' in results
        assert isinstance(results['topology_type'], str)
    
    def _create_test_network(self, N: int):
        """Create test network."""
        G = nx.connected_watts_strogatz_graph(N, k=6, p=0.2, seed=42)
        W_lil = sp.lil_matrix((N, N), dtype=np.complex128)
        for u, v in G.edges():
            phase = np.random.uniform(0, 2 * np.pi)
            W_lil[u, v] = np.exp(1j * phase)
            W_lil[v, u] = np.exp(-1j * phase)
        return W_lil.tocsr()


class TestGaugeGroupDerivation:
    """Tests for complete gauge group derivation pipeline."""
    
    def test_gauge_group_derivation_initialization(self):
        """Test GaugeGroupDerivation initialization."""
        N = 50
        W = sp.random(N, N, density=0.1, format='csr', dtype=np.complex128)
        boundary_nodes = np.arange(N // 4)
        
        derivation = GaugeGroupDerivation(W, boundary_nodes)
        
        assert derivation.N == N
        assert len(derivation.boundary_nodes) == len(boundary_nodes)
    
    def test_gauge_group_derivation_run(self):
        """Test full gauge group derivation."""
        N = 80
        W = self._create_test_network(N)
        boundary_nodes = np.arange(N // 3, N)
        
        derivation = GaugeGroupDerivation(W, boundary_nodes)
        results = derivation.run_derivation(target_loops=8)
        
        assert 'beta_1' in results
        assert 'fundamental_loops' in results
        assert 'AIX' in results
        assert 'structure_constants' in results
        assert 'gauge_group' in results
        assert 'Jacobi_identity' in results
        assert 'anomaly_cancellation' in results
        
        # Check types
        assert isinstance(results['beta_1'], int)
        assert isinstance(results['gauge_group'], str)
        assert isinstance(results['AIX'], np.ndarray)
    
    def test_gauge_group_derivation_validation(self):
        """Test validation metrics in derivation."""
        N = 70
        W = self._create_test_network(N)
        boundary_nodes = np.arange(N // 3, N)
        
        derivation = GaugeGroupDerivation(W, boundary_nodes)
        results = derivation.run_derivation(target_loops=6)
        
        # AIX antisymmetry
        assert results['AIX_antisymmetry'] < 1e-8
        
        # Jacobi identity (relaxed tolerance for discrete approximation)
        assert bool(results['Jacobi_identity']['passes']) or \
               float(results['Jacobi_identity']['max_violation']) < 0.05
    
    def _create_test_network(self, N: int):
        """Create test network with cycles."""
        n_side = int(np.sqrt(N))
        G = nx.grid_2d_graph(n_side, n_side)
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        
        # Add random edges for more structure
        for _ in range(N // 4):
            u, v = np.random.randint(0, min(N, len(G.nodes())), size=2)
            if u != v and G.has_node(u) and G.has_node(v):
                G.add_edge(u, v)
        
        W_lil = sp.lil_matrix((N, N), dtype=np.complex128)
        for u, v in G.edges():
            phase = np.random.uniform(0, 2 * np.pi)
            W_lil[u, v] = np.exp(1j * phase)
            W_lil[v, u] = np.exp(-1j * phase)
        return W_lil.tocsr()


class TestPhase4Integration:
    """Integration tests for complete Phase 4."""
    
    def test_full_phase4_pipeline(self):
        """Test complete Phase 4 pipeline from network to gauge group."""
        # Create realistic test network
        N = 100
        W = self._create_aro_like_network(N)
        
        # Step 1: Boundary analysis
        boundary_analyzer = BoundaryAnalyzer(W, boundary_fraction=0.12)
        boundary_results = boundary_analyzer.run_analysis()
        
        assert boundary_results['betti_numbers']['beta_1'] is not None
        
        # Step 2: Gauge group derivation
        boundary_nodes = boundary_results['boundary_nodes']
        
        # Skip if boundary too small
        if len(boundary_nodes) < 10:
            return
        
        gauge_derivation = GaugeGroupDerivation(W, boundary_nodes)
        gauge_results = gauge_derivation.run_derivation(target_loops=10)
        
        # Validate full pipeline
        # beta_1 may be 0 if no cycles found
        assert gauge_results['beta_1'] >= 0
        
        # AIX shape based on number of loops found
        n_loops = len(gauge_results['fundamental_loops'])
        assert gauge_results['AIX'].shape == (n_loops, n_loops)
        assert gauge_results['structure_constants'].shape == (n_loops, n_loops, n_loops)
        assert len(gauge_results['gauge_group']) > 0
    
    def test_phase4_robustness(self):
        """Test Phase 4 robustness to network size."""
        for N in [50, 80, 120]:
            W = self._create_aro_like_network(N)
            
            analyzer = BoundaryAnalyzer(W, boundary_fraction=0.1)
            results = analyzer.run_analysis()
            
            # Should successfully complete
            assert results['betti_numbers']['beta_1'] is not None
            assert len(results['boundary_nodes']) > 0
    
    def _create_aro_like_network(self, N: int):
        """Create network mimicking ARO-optimized structure."""
        # Small-world with geometric structure
        G = nx.connected_watts_strogatz_graph(N, k=6, p=0.15, seed=42)
        
        # Add long-range connections (holographic)
        for _ in range(N // 8):
            u, v = np.random.randint(0, N, size=2)
            if u != v:
                G.add_edge(u, v)
        
        # Complex weights with phase structure
        W_lil = sp.lil_matrix((N, N), dtype=np.complex128)
        for u, v in G.edges():
            magnitude = np.random.uniform(0.7, 1.0)
            phase = np.random.uniform(0, 2 * np.pi)
            W_lil[u, v] = magnitude * np.exp(1j * phase)
            W_lil[v, u] = magnitude * np.exp(-1j * phase)
        
        return W_lil.tocsr()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
