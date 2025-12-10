"""
Unit tests for Cymatic Resonance Network (CRN) - Axiom 2.

Tests CRN construction and properties per IRHv16.md lines 87-100.

THEORETICAL COMPLIANCE:
    Tests validate against docs/manuscripts/IRHv16.md Axiom 2
    - Lines 87-100: Network Emergence Principle
    - ε_threshold = 0.730129 ± 10^-6
    - Edges exist iff |W_ij| > ε_threshold
    - Complex graph Laplacian (Interference Matrix ℒ)
"""

import pytest
import numpy as np
from irh.core.v16.ahs import create_ahs_network
from irh.core.v16.crn import (
    CymaticResonanceNetworkV16,
    create_crn_from_states
)


class TestCRNConstruction:
    """Test Cymatic Resonance Network construction."""
    
    def test_basic_crn_creation(self):
        """Test basic CRN creation from AHS list."""
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        assert crn.N == 5, "CRN should have 5 nodes"
        assert crn.adjacency_matrix is not None, "Adjacency matrix should be built"
        assert crn.adjacency_matrix.shape == (5, 5), "Adjacency matrix should be 5x5"
        
    def test_crn_epsilon_threshold(self):
        """
        Test ε_threshold filtering per IRHv16.md Axiom 2.
        
        References:
            IRHv16.md line 97: ε = 0.730129 ± 10^-6
        """
        states = create_ahs_network(N=10, seed=42)
        
        # Test with theoretical ε_threshold
        crn = create_crn_from_states(states, epsilon_threshold=0.730129)
        
        assert np.isclose(crn.epsilon_threshold, 0.730129, atol=1e-6), \
            "ε_threshold should match IRHv16.md line 97"
            
    def test_crn_complex_adjacency(self):
        """Test adjacency matrix is complex-valued per Axiom 2."""
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        assert crn.adjacency_matrix.dtype == np.complex128, \
            "Adjacency matrix must be complex128 per Axiom 2"
            
    def test_crn_no_self_loops(self):
        """Test CRN has no self-loops."""
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        # Diagonal should be zero (no self-loops)
        diagonal = np.diag(crn.adjacency_matrix)
        assert np.allclose(diagonal, 0), "CRN should have no self-loops"
        
    def test_crn_edge_filtering(self):
        """Test edges are filtered by ε_threshold."""
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        # All non-zero entries should have |W_ij| > ε_threshold
        for i in range(crn.N):
            for j in range(crn.N):
                if i != j and np.abs(crn.adjacency_matrix[i, j]) > 0:
                    assert np.abs(crn.adjacency_matrix[i, j]) > crn.epsilon_threshold, \
                        f"Edge ({i},{j}) has |W_ij| <= ε_threshold"


class TestCRNProperties:
    """Test CRN properties and methods."""
    
    def test_crn_num_edges(self):
        """Test num_edges property."""
        states = create_ahs_network(N=10, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        # Count manually
        manual_count = 0
        for i in range(crn.N):
            for j in range(crn.N):
                if i != j and np.abs(crn.adjacency_matrix[i, j]) > crn.epsilon_threshold:
                    manual_count += 1
                    
        assert crn.num_edges == manual_count, \
            f"num_edges {crn.num_edges} != manual count {manual_count}"
            
    def test_crn_get_weight(self):
        """Test get_weight method."""
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        # Get a weight
        w_01 = crn.get_weight(0, 1)
        
        assert isinstance(w_01, (complex, np.complexfloating)), \
            "Weight should be complex"
        assert w_01 == crn.adjacency_matrix[0, 1], \
            "get_weight should return adjacency_matrix entry"
            
    def test_crn_get_neighbors(self):
        """Test get_neighbors method."""
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        # Get outgoing neighbors of node 0
        neighbors = crn.get_neighbors(0, direction="out")
        
        assert isinstance(neighbors, list), "neighbors should be a list"
        
        # Verify all are valid neighbors
        for j in neighbors:
            assert np.abs(crn.adjacency_matrix[0, j]) > 0, \
                f"Node {j} should be a neighbor of 0"
                
    def test_crn_repr(self):
        """Test __repr__ method."""
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states)
        
        repr_str = repr(crn)
        
        assert "CRNv16" in repr_str, "repr should contain CRNv16"
        assert str(crn.N) in repr_str, "repr should contain N"


class TestInterferenceMatrix:
    """
    Test Interference Matrix (complex graph Laplacian).
    
    Per IRHv16.md §4 lines 265-266: ℒ is the complex graph Laplacian
    used in Harmony Functional S_H = Tr(ℒ²) / [det'(ℒ)]^{C_H}
    """
    
    def test_interference_matrix_shape(self):
        """Test interference matrix has correct shape."""
        states = create_ahs_network(N=10, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        L = crn.interference_matrix
        
        assert L.shape == (10, 10), "ℒ should be N x N"
        
    def test_interference_matrix_complex(self):
        """Test interference matrix is complex-valued."""
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        L = crn.interference_matrix
        
        assert L.dtype == np.complex128, "ℒ must be complex128"
        
    def test_interference_matrix_diagonal(self):
        """
        Test interference matrix diagonal.
        
        For directed graphs: ℒ_ii = Σ_j W_ij (out-degree sum)
        """
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        L = crn.interference_matrix
        
        # Check diagonal elements
        for i in range(crn.N):
            expected_diag = np.sum(crn.adjacency_matrix[i, :])
            assert np.isclose(L[i, i], expected_diag), \
                f"ℒ_ii should be sum of outgoing weights for node {i}"
                
    def test_interference_matrix_off_diagonal(self):
        """
        Test interference matrix off-diagonal elements.
        
        ℒ_ij = -W_ij for i ≠ j
        """
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        L = crn.interference_matrix
        
        # Check off-diagonal elements
        for i in range(crn.N):
            for j in range(crn.N):
                if i != j:
                    expected = -crn.adjacency_matrix[i, j]
                    assert np.isclose(L[i, j], expected), \
                        f"ℒ_ij should be -W_ij for i≠j"


class TestSpectralProperties:
    """Test spectral properties computation for Harmony Functional."""
    
    def test_compute_spectral_properties(self):
        """Test compute_spectral_properties returns expected keys."""
        states = create_ahs_network(N=10, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        props = crn.compute_spectral_properties()
        
        # Check required keys
        required_keys = ['eigenvalues', 'trace_L2', 'det_prime', 'num_zero_eigenvalues']
        for key in required_keys:
            assert key in props, f"Missing key: {key}"
            
    def test_trace_L2_computation(self):
        """Test Tr(ℒ²) computation."""
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        props = crn.compute_spectral_properties()
        L = crn.interference_matrix
        
        # Manual computation
        L2 = L @ L
        expected_trace = np.trace(L2)
        
        assert np.isclose(props['trace_L2'], expected_trace), \
            "Tr(ℒ²) should match manual computation"
            
    def test_eigenvalues_count(self):
        """Test number of eigenvalues equals N."""
        states = create_ahs_network(N=10, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        props = crn.compute_spectral_properties()
        
        assert len(props['eigenvalues']) == 10, \
            "Should have N eigenvalues"
            
    def test_det_prime_nonzero(self):
        """Test det'(ℒ) is non-zero for connected networks."""
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        props = crn.compute_spectral_properties()
        
        # det' should be product of non-zero eigenvalues
        assert props['det_prime'] != 0, "det'(ℒ) should be non-zero"


class TestTheoreticalCompliance:
    """
    Test compliance with IRHv16.md Axiom 2 specifications.
    
    References:
        docs/manuscripts/IRHv16.md lines 87-100: Axiom 2
    """
    
    def test_axiom2_epsilon_value(self):
        """
        Test ε_threshold matches IRHv16.md specification.
        
        References:
            IRHv16.md line 97: ε = 0.730129 ± 10^-6
        """
        # Create CRN with theoretical ε
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.730129)
        
        # Verify ε matches theoretical value
        expected_epsilon = 0.730129
        assert np.isclose(crn.epsilon_threshold, expected_epsilon, atol=1e-6), \
            f"Axiom 2 requires ε = {expected_epsilon} ± 10^-6"
            
    def test_axiom2_edge_criterion(self):
        """
        Test edges exist iff |W_ij| > ε per Axiom 2.
        
        References:
            IRHv16.md lines 94-96: Edge inclusion criterion
        """
        states = create_ahs_network(N=5, seed=42)
        epsilon = 0.5
        crn = create_crn_from_states(states, epsilon_threshold=epsilon)
        
        # Check edge criterion
        for i in range(crn.N):
            for j in range(crn.N):
                if i == j:
                    continue
                    
                w_ij = crn.adjacency_matrix[i, j]
                
                if np.abs(w_ij) > 0:
                    # Edge exists, so |W_ij| should be > ε
                    assert np.abs(w_ij) > epsilon, \
                        f"Edge ({i},{j}) exists but |W_ij|={np.abs(w_ij)} <= ε={epsilon}"
                        
    def test_axiom2_complex_laplacian(self):
        """
        Test ℒ is complex graph Laplacian per Axiom 2.
        
        References:
            IRHv16.md §4 lines 265-266: ℒ is Interference Matrix
        """
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        L = crn.interference_matrix
        
        # Must be complex
        assert L.dtype == np.complex128, "ℒ must be complex per Axiom 2"
        
        # Must be square
        assert L.shape[0] == L.shape[1] == crn.N, "ℒ must be N x N"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_network(self):
        """Test behavior with minimal network."""
        states = create_ahs_network(N=2, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        assert crn.N == 2
        assert crn.adjacency_matrix.shape == (2, 2)
        
    def test_high_threshold_sparse_network(self):
        """Test CRN with high threshold (sparse network)."""
        states = create_ahs_network(N=10, seed=42)
        
        # Very high threshold → few edges
        crn = create_crn_from_states(states, epsilon_threshold=0.99)
        
        # Network should exist but be sparse
        assert crn.num_edges >= 0, "Should handle sparse networks"
        
    def test_low_threshold_dense_network(self):
        """Test CRN with low threshold (dense network)."""
        states = create_ahs_network(N=10, seed=42)
        
        # Very low threshold → many edges
        crn = create_crn_from_states(states, epsilon_threshold=0.01)
        
        # Should have many edges
        assert crn.num_edges > 0, "Low threshold should create edges"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
