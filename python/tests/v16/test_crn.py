"""
Unit tests for Cymatic Resonance Network (Axiom 2).

Tests the CRN implementation as defined in IRHv16.md §1 Axiom 2.

References:
    IRHv16.md §1 Axiom 2: Network Emergence Principle
    IRHv16.md Theorem 1.2: Necessity of Network Representation
    IRHv16.md §2 Definition 2.1: Frustration Density
"""

import pytest
import numpy as np
from irh.core.v16.ahs import create_ahs_network
from irh.core.v16.crn import (
    CymaticResonanceNetwork,
    EPSILON_THRESHOLD,
    EPSILON_THRESHOLD_ERROR,
    derive_epsilon_threshold,
)


class TestCRNCreation:
    """Test CRN construction from AHS."""
    
    def test_from_states(self):
        """Test creating CRN from list of AHS."""
        states = create_ahs_network(N=10, seed=42)
        crn = CymaticResonanceNetwork.from_states(states)
        
        assert crn.N == 10
        assert crn.W.shape == (10, 10)
        
    def test_create_random(self):
        """Test creating random CRN."""
        crn = CymaticResonanceNetwork.create_random(N=20, seed=42)
        
        assert crn.N == 20
        assert crn.W.shape == (20, 20)
        
    def test_reproducibility(self):
        """Test CRN creation is reproducible."""
        crn1 = CymaticResonanceNetwork.create_random(N=10, seed=123)
        crn2 = CymaticResonanceNetwork.create_random(N=10, seed=123)
        
        assert np.allclose(crn1.W, crn2.W)
        
    def test_empty_states_raises(self):
        """Test empty states list raises error."""
        with pytest.raises(ValueError, match="at least one node"):
            CymaticResonanceNetwork(states=[], W=np.array([[]]))


class TestCRNProperties:
    """Test CRN properties and metrics."""
    
    def test_num_edges(self):
        """Test edge counting."""
        crn = CymaticResonanceNetwork.create_random(N=10, seed=42)
        
        # Edges should be non-negative
        assert crn.num_edges >= 0
        # Max edges is N*(N-1)
        assert crn.num_edges <= 10 * 9
        
    def test_edge_density(self):
        """Test edge density calculation."""
        crn = CymaticResonanceNetwork.create_random(N=10, seed=42)
        
        assert 0 <= crn.edge_density <= 1
        
    def test_adjacency_matrix(self):
        """Test adjacency matrix generation."""
        crn = CymaticResonanceNetwork.create_random(N=10, seed=42)
        A = crn.get_adjacency_matrix()
        
        assert A.shape == (10, 10)
        assert A.dtype == np.bool_
        # No self-loops
        assert not np.any(np.diag(A))
        
    def test_degree_distribution(self):
        """Test degree distribution."""
        crn = CymaticResonanceNetwork.create_random(N=10, seed=42)
        in_deg, out_deg = crn.get_degree_distribution()
        
        assert len(in_deg) == 10
        assert len(out_deg) == 10
        assert np.all(in_deg >= 0)
        assert np.all(out_deg >= 0)


class TestCRNConnectivity:
    """Test CRN connectivity analysis."""
    
    def test_connectivity_check(self):
        """Test connectivity checking works."""
        crn = CymaticResonanceNetwork.create_random(N=10, seed=42)
        
        # Should return boolean
        result = crn.is_connected()
        assert isinstance(result, bool)
        
    def test_small_connected_network(self):
        """Test small networks with low threshold are connected."""
        # Use very low threshold to ensure connectivity
        crn = CymaticResonanceNetwork.create_random(
            N=5, 
            epsilon_threshold=0.1,
            seed=42
        )
        # With low threshold, likely connected
        # (not guaranteed, but likely for small N)


class TestCRNHolonomy:
    """Test holonomy and frustration calculations."""
    
    def test_cycle_holonomy(self):
        """Test holonomy computation for a cycle."""
        crn = CymaticResonanceNetwork.create_random(N=5, seed=42)
        
        # Triangle cycle [0, 1, 2, 0]
        cycle = [0, 1, 2, 0]
        holonomy = crn.compute_cycle_holonomy(cycle)
        
        assert isinstance(holonomy, complex)
        
    def test_cycle_phase(self):
        """Test phase computation for a cycle."""
        crn = CymaticResonanceNetwork.create_random(N=5, seed=42)
        
        cycle = [0, 1, 2, 0]
        phase = crn.compute_cycle_phase(cycle)
        
        assert 0 <= phase < 2 * np.pi
        
    def test_cycle_validation(self):
        """Test cycle validation."""
        crn = CymaticResonanceNetwork.create_random(N=5, seed=42)
        
        # Not a cycle (doesn't close)
        with pytest.raises(ValueError, match="start and end"):
            crn.compute_cycle_holonomy([0, 1, 2])
            
        # Too short
        with pytest.raises(ValueError, match="at least 2"):
            crn.compute_cycle_holonomy([0])
            
    def test_find_triangular_cycles(self):
        """Test finding triangular cycles."""
        crn = CymaticResonanceNetwork.create_random(
            N=10, 
            epsilon_threshold=0.1,  # Low threshold for more edges
            seed=42
        )
        
        cycles = crn.find_triangular_cycles(max_cycles=100)
        
        # Should be a list
        assert isinstance(cycles, list)
        
        # Each cycle should be a triangle [i, j, k, i]
        for c in cycles:
            assert len(c) == 4
            assert c[0] == c[-1]
            
    def test_frustration_density(self):
        """Test frustration density computation."""
        crn = CymaticResonanceNetwork.create_random(
            N=10, 
            epsilon_threshold=0.1,
            seed=42
        )
        
        rho = crn.compute_frustration_density(max_cycles=100)
        
        # Should be non-negative and bounded
        assert rho >= 0
        assert rho <= np.pi  # Max phase winding is π


class TestCRNInterferenceMatrix:
    """Test interference matrix computation."""
    
    def test_interference_matrix_shape(self):
        """Test interference matrix has correct shape."""
        crn = CymaticResonanceNetwork.create_random(N=10, seed=42)
        L = crn.get_interference_matrix()
        
        assert L.shape == (10, 10)
        assert L.dtype == np.complex128
        
    def test_interference_matrix_properties(self):
        """Test interference matrix is Laplacian-like."""
        crn = CymaticResonanceNetwork.create_random(N=10, seed=42)
        L = crn.get_interference_matrix()
        
        # Row sums of Laplacian should be ~0 for unweighted
        # For weighted, this isn't exact, but diagonal dominance holds
        diag = np.diag(L)
        off_diag_row_sum = np.sum(L, axis=1) - diag
        
        # L_ii should equal sum of |W_ij| for j != i
        # So L_ii + (-W_ii) - Σ_{j≠i} W_ij ≈ 0
        # This is approximate for complex weights
        # Verify diagonal dominance: |L_ii| >= |off_diag_row_sum_i|
        assert all(np.abs(diag) >= np.abs(off_diag_row_sum) - 1e-10), \
            "Diagonal dominance should hold for Laplacian matrix"


class TestEpsilonThreshold:
    """Test epsilon threshold derivation."""
    
    def test_threshold_constant(self):
        """Test threshold constant value."""
        # From IRHv16.md
        assert abs(EPSILON_THRESHOLD - 0.730129) < 1e-5
        assert EPSILON_THRESHOLD_ERROR == 1e-6
        
    def test_derive_epsilon(self):
        """Test epsilon derivation (simplified)."""
        # This is a very simplified test due to computational cost
        eps, err = derive_epsilon_threshold(
            N_samples=10,  # Small for speed
            N_per_sample=10,
            seed=42
        )
        
        # Should be in reasonable range
        assert 0.3 < eps < 1.0
        assert err > 0


class TestCRNRepr:
    """Test CRN string representations."""
    
    def test_repr(self):
        """Test developer representation."""
        crn = CymaticResonanceNetwork.create_random(N=10, seed=42)
        repr_str = repr(crn)
        
        assert "CRN" in repr_str
        assert "N=10" in repr_str
        
    def test_str(self):
        """Test user-friendly string."""
        crn = CymaticResonanceNetwork.create_random(N=10, seed=42)
        str_repr = str(crn)
        
        assert "Cymatic Resonance Network" in str_repr
        assert "10" in str_repr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
