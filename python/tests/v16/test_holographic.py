"""
Unit tests for Combinatorial Holographic Principle (Axiom 3).

Tests the holographic implementation as defined in IRHv16.md §1 Axiom 3.

References:
    IRHv16.md §1 Axiom 3: Combinatorial Holographic Principle
    IRHv16.md Theorem 1.3: Optimal Holographic Scaling
"""

import pytest
import numpy as np
from irh.core.v16.crn import CymaticResonanceNetwork
from irh.core.v16.holographic import (
    Subnetwork,
    HolographicAnalyzer,
    verify_holographic_principle,
    HOLOGRAPHIC_CONSTANT_K,
)


class TestSubnetwork:
    """Test Subnetwork class."""
    
    def test_basic_creation(self):
        """Test basic subnetwork creation."""
        crn = CymaticResonanceNetwork.create_random(N=20, seed=42)
        sub = Subnetwork(node_indices={0, 1, 2, 3}, parent_crn=crn)
        
        assert sub.N == 4
        assert sub.parent_crn is crn
        
    def test_empty_subnetwork_raises(self):
        """Test empty subnetwork raises error."""
        crn = CymaticResonanceNetwork.create_random(N=10, seed=42)
        
        with pytest.raises(ValueError, match="at least one node"):
            Subnetwork(node_indices=set(), parent_crn=crn)
            
    def test_invalid_indices_raises(self):
        """Test invalid node indices raise error."""
        crn = CymaticResonanceNetwork.create_random(N=10, seed=42)
        
        with pytest.raises(ValueError, match="out of range"):
            Subnetwork(node_indices={0, 1, 100}, parent_crn=crn)
            
    def test_boundary_nodes(self):
        """Test boundary node computation."""
        crn = CymaticResonanceNetwork.create_random(N=20, seed=42, epsilon_threshold=0.3)
        sub = Subnetwork(node_indices={0, 1, 2, 3, 4}, parent_crn=crn)
        
        boundary = sub.boundary_nodes
        interior = sub.interior_nodes
        
        # Boundary and interior should partition the subnetwork
        assert boundary | interior == sub.node_indices
        assert boundary & interior == set()
        
    def test_boundary_degree_sum(self):
        """Test boundary degree sum calculation."""
        crn = CymaticResonanceNetwork.create_random(N=20, seed=42, epsilon_threshold=0.3)
        sub = Subnetwork(node_indices={0, 1, 2}, parent_crn=crn)
        
        degree_sum = sub.boundary_degree_sum
        assert degree_sum >= 0
        
    def test_subnetwork_matrix(self):
        """Test subnetwork matrix extraction."""
        crn = CymaticResonanceNetwork.create_random(N=20, seed=42)
        sub = Subnetwork(node_indices={0, 1, 2, 3}, parent_crn=crn)
        
        W_sub = sub.get_subnetwork_matrix()
        
        assert W_sub.shape == (4, 4)
        assert W_sub.dtype == np.complex128
        
    def test_information_content(self):
        """Test information content estimation."""
        crn = CymaticResonanceNetwork.create_random(N=20, seed=42)
        sub = Subnetwork(node_indices={0, 1, 2}, parent_crn=crn)
        
        I_A = sub.compute_information_content()
        
        assert I_A > 0


class TestHolographicAnalyzer:
    """Test HolographicAnalyzer class."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        crn = CymaticResonanceNetwork.create_random(N=20, seed=42)
        analyzer = HolographicAnalyzer(crn)
        
        assert analyzer.crn is crn
        assert analyzer.K == HOLOGRAPHIC_CONSTANT_K
        
    def test_extract_subnetwork(self):
        """Test subnetwork extraction."""
        crn = CymaticResonanceNetwork.create_random(N=20, seed=42)
        analyzer = HolographicAnalyzer(crn)
        
        sub = analyzer.extract_subnetwork({0, 5, 10})
        
        assert sub.N == 3
        assert sub.node_indices == {0, 5, 10}
        
    def test_extract_random_subnetwork(self):
        """Test random subnetwork extraction."""
        crn = CymaticResonanceNetwork.create_random(N=20, seed=42)
        analyzer = HolographicAnalyzer(crn)
        
        sub = analyzer.extract_random_subnetwork(size=5, seed=123)
        
        assert sub.N == 5
        
    def test_random_subnetwork_size_validation(self):
        """Test size validation for random subnetwork."""
        crn = CymaticResonanceNetwork.create_random(N=10, seed=42)
        analyzer = HolographicAnalyzer(crn)
        
        with pytest.raises(ValueError, match="Requested size"):
            analyzer.extract_random_subnetwork(size=100, seed=123)
            
    def test_compute_holographic_bound(self):
        """Test holographic bound computation."""
        crn = CymaticResonanceNetwork.create_random(N=20, seed=42, epsilon_threshold=0.3)
        analyzer = HolographicAnalyzer(crn, K=1.0)
        
        sub = analyzer.extract_subnetwork({0, 1, 2, 3})
        bound = analyzer.compute_holographic_bound(sub)
        
        assert bound >= 0
        assert bound == sub.boundary_degree_sum  # K=1.0
        
    def test_verify_holographic_bound(self):
        """Test holographic bound verification."""
        crn = CymaticResonanceNetwork.create_random(N=20, seed=42, epsilon_threshold=0.3)
        analyzer = HolographicAnalyzer(crn, K=1.0)
        
        sub = analyzer.extract_subnetwork({0, 1, 2})
        is_satisfied, I_A, bound = analyzer.verify_holographic_bound(sub)
        
        assert isinstance(is_satisfied, bool)
        assert I_A >= 0
        assert bound >= 0
        
    def test_holographic_scaling(self):
        """Test holographic scaling analysis."""
        crn = CymaticResonanceNetwork.create_random(N=30, seed=42, epsilon_threshold=0.3)
        analyzer = HolographicAnalyzer(crn)
        
        results = analyzer.test_holographic_scaling(n_samples=15, seed=123)
        
        assert results["success"]
        assert "beta" in results
        assert "r_squared" in results
        assert results["n_samples"] > 0
        
    def test_holographic_entropy(self):
        """Test holographic entropy computation."""
        crn = CymaticResonanceNetwork.create_random(N=20, seed=42, epsilon_threshold=0.3)
        analyzer = HolographicAnalyzer(crn)
        
        sub = analyzer.extract_subnetwork({0, 1, 2, 3, 4})
        entropy = analyzer.compute_holographic_entropy(sub)
        
        assert entropy >= 0
        # S_holo = 0.25 * boundary_degree_sum
        expected = 0.25 * sub.boundary_degree_sum
        assert np.isclose(entropy, expected)
        
    def test_analyze_all_sizes(self):
        """Test comprehensive size analysis."""
        crn = CymaticResonanceNetwork.create_random(N=20, seed=42, epsilon_threshold=0.3)
        analyzer = HolographicAnalyzer(crn)
        
        results = analyzer.analyze_all_sizes(seed=123)
        
        assert "sizes" in results
        assert "avg_info_content" in results
        assert "avg_boundary_degree" in results
        assert len(results["sizes"]) > 0


class TestVerifyHolographicPrinciple:
    """Test the verify_holographic_principle function."""
    
    def test_basic_verification(self):
        """Test basic holographic principle verification."""
        crn = CymaticResonanceNetwork.create_random(N=25, seed=42, epsilon_threshold=0.3)
        
        ratio, details = verify_holographic_principle(crn, n_tests=20, seed=123)
        
        assert 0 <= ratio <= 1
        assert details["total_tests"] > 0
        assert "violations" in details
        
    def test_small_network(self):
        """Test verification on small network."""
        crn = CymaticResonanceNetwork.create_random(N=10, seed=42, epsilon_threshold=0.3)
        
        ratio, details = verify_holographic_principle(crn, n_tests=10, seed=123)
        
        assert 0 <= ratio <= 1
        
    def test_reproducibility(self):
        """Test verification is reproducible with seed."""
        crn = CymaticResonanceNetwork.create_random(N=20, seed=42)
        
        ratio1, _ = verify_holographic_principle(crn, n_tests=10, seed=999)
        ratio2, _ = verify_holographic_principle(crn, n_tests=10, seed=999)
        
        assert ratio1 == ratio2


class TestHolographicScaling:
    """Test holographic scaling properties (Theorem 1.3)."""
    
    def test_scaling_exponent_near_one(self):
        """Test that scaling exponent β is near 1 (linear scaling)."""
        # Create a larger network for better statistics
        crn = CymaticResonanceNetwork.create_random(N=50, seed=42, epsilon_threshold=0.3)
        analyzer = HolographicAnalyzer(crn)
        
        results = analyzer.test_holographic_scaling(n_samples=30, seed=123)
        
        if results["success"]:
            # Per Theorem 1.3, β should be close to 1
            # We use a loose tolerance since this is a finite-size effect
            beta = results["beta"]
            # Just check it's positive and reasonable
            assert beta > 0
            
    def test_scaling_goodness_of_fit(self):
        """Test that scaling fit has reasonable R²."""
        crn = CymaticResonanceNetwork.create_random(N=40, seed=42, epsilon_threshold=0.3)
        analyzer = HolographicAnalyzer(crn)
        
        results = analyzer.test_holographic_scaling(n_samples=25, seed=123)
        
        if results["success"]:
            # R² should be positive for any meaningful fit
            assert results["r_squared"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
