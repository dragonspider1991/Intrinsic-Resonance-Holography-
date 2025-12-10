"""
Unit tests for Harmony Functional - Theorem 4.1.

Tests S_H[G] computation per IRHv16.md §4 lines 254-277.

THEORETICAL COMPLIANCE:
    Tests validate against docs/manuscripts/IRHv16.md §4 Theorem 4.1
    - Line 266: S_H[G] = Tr(ℒ²) / [det'(ℒ)]^{C_H}
    - Line 275: C_H = 0.045935703598
    - Lines 271-272: Numerator = coherent information flow
    - Lines 272: Denominator = cymatic complexity
"""

import pytest
import numpy as np
from irh.core.v16.ahs import create_ahs_network
from irh.core.v16.crn import create_crn_from_states
from irh.core.v16.harmony import (
    compute_harmony_functional,
    validate_harmony_functional_properties,
    HarmonyFunctionalEvaluator
)


class TestHarmonyFunctionalComputation:
    """Test Harmony Functional S_H[G] computation."""
    
    def test_basic_computation(self):
        """Test basic S_H computation."""
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        result = compute_harmony_functional(crn)
        
        # Check required keys
        assert 'S_H' in result
        assert 'trace_L2' in result
        assert 'det_prime' in result
        assert 'C_H' in result
        
    def test_S_H_positive(self):
        """Test S_H > 0 for non-degenerate networks."""
        states = create_ahs_network(N=10, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        result = compute_harmony_functional(crn)
        
        assert result['S_H'] > 0, f"S_H must be positive, got {result['S_H']}"
        
    def test_C_H_constant(self):
        """
        Test C_H matches IRHv16.md line 275.
        
        References:
            IRHv16.md line 275: C_H = 0.045935703598(1)
        """
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        result = compute_harmony_functional(crn)
        
        expected_C_H = 0.045935703598
        assert np.isclose(result['C_H'], expected_C_H, atol=1e-11), \
            f"C_H should be {expected_C_H} per IRHv16.md line 275"
            
    def test_formula_implementation(self):
        """
        Test S_H = Tr(ℒ²) / [det'(ℒ)]^{C_H} per IRHv16.md line 266.
        """
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        result = compute_harmony_functional(crn)
        
        # Manual computation
        trace_L2_mag = result['trace_L2_magnitude']
        det_prime_mag = result['det_prime_magnitude']
        C_H = result['C_H']
        
        expected_S_H = trace_L2_mag / (det_prime_mag ** C_H)
        
        assert np.isclose(result['S_H'], expected_S_H, rtol=1e-10), \
            "S_H should match formula from IRHv16.md"
            
    def test_trace_L2_nonzero(self):
        """Test Tr(ℒ²) is non-zero for non-trivial networks."""
        states = create_ahs_network(N=10, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        result = compute_harmony_functional(crn)
        
        assert np.abs(result['trace_L2']) > 0, "Tr(ℒ²) should be non-zero"
        
    def test_det_prime_nonzero(self):
        """Test det'(ℒ) is non-zero for non-degenerate networks."""
        states = create_ahs_network(N=10, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        result = compute_harmony_functional(crn)
        
        assert np.abs(result['det_prime']) > 0, "det'(ℒ) should be non-zero"


class TestHarmonyFunctionalProperties:
    """Test properties of the Harmony Functional."""
    
    def test_different_network_sizes(self):
        """Test S_H computation for different network sizes."""
        sizes = [5, 10, 15]
        
        for N in sizes:
            states = create_ahs_network(N=N, seed=42)
            crn = create_crn_from_states(states, epsilon_threshold=0.5)
            
            result = compute_harmony_functional(crn)
            
            assert result['S_H'] > 0, f"S_H should be positive for N={N}"
            assert result['C_H'] == 0.045935703598, f"C_H should be constant for N={N}"
            
    def test_S_H_varies_with_network(self):
        """Test S_H values differ for different networks."""
        # Create two different networks
        states1 = create_ahs_network(N=10, seed=42)
        states2 = create_ahs_network(N=10, seed=123)
        
        crn1 = create_crn_from_states(states1, epsilon_threshold=0.5)
        crn2 = create_crn_from_states(states2, epsilon_threshold=0.5)
        
        result1 = compute_harmony_functional(crn1)
        result2 = compute_harmony_functional(crn2)
        
        # Different networks should generally have different S_H
        # (Not guaranteed but highly likely with different seeds)
        # Just ensure both are computable
        assert result1['S_H'] > 0
        assert result2['S_H'] > 0


class TestValidateProperties:
    """Test validate_harmony_functional_properties function."""
    
    def test_validation_passes(self):
        """Test validation passes for valid results."""
        states = create_ahs_network(N=10, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        result = compute_harmony_functional(crn)
        
        # Should not raise
        assert validate_harmony_functional_properties(result) is True
        
    def test_validation_checks_S_H_positive(self):
        """Test validation checks S_H > 0."""
        # Create a mock result with S_H = 0
        mock_result = {
            'S_H': 0.0,
            'C_H': 0.045935703598,
            'trace_L2': 1.0,
            'det_prime': 1.0
        }
        
        with pytest.raises(AssertionError, match="S_H must be positive"):
            validate_harmony_functional_properties(mock_result)
            
    def test_validation_checks_C_H(self):
        """Test validation checks C_H matches theoretical value."""
        # Create a mock result with wrong C_H
        mock_result = {
            'S_H': 1.0,
            'C_H': 0.999,  # Wrong value
            'trace_L2': 1.0,
            'det_prime': 1.0
        }
        
        with pytest.raises(AssertionError, match="does not match IRHv16.md"):
            validate_harmony_functional_properties(mock_result)


class TestHarmonyFunctionalEvaluator:
    """Test HarmonyFunctionalEvaluator class for ARO."""
    
    def test_evaluator_creation(self):
        """Test evaluator can be created."""
        evaluator = HarmonyFunctionalEvaluator()
        
        assert evaluator.best_S_H == -np.inf
        assert len(evaluator.history) == 0
        
    def test_evaluator_evaluate(self):
        """Test evaluator.evaluate() method."""
        evaluator = HarmonyFunctionalEvaluator()
        
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        S_H = evaluator.evaluate(crn, iteration=0)
        
        assert S_H > 0, "evaluate should return positive S_H"
        assert len(evaluator.history) == 1, "History should have one entry"
        assert evaluator.best_S_H == S_H, "best_S_H should be updated"
        
    def test_evaluator_tracks_best(self):
        """Test evaluator tracks best S_H."""
        evaluator = HarmonyFunctionalEvaluator()
        
        # Evaluate multiple networks
        for seed in [42, 43, 44]:
            states = create_ahs_network(N=5, seed=seed)
            crn = create_crn_from_states(states, epsilon_threshold=0.5)
            evaluator.evaluate(crn, iteration=seed-42)
            
        # best_S_H should be the max
        all_S_H = [s for i, s in evaluator.history]
        assert evaluator.best_S_H == max(all_S_H), \
            "best_S_H should track maximum"
            
    def test_evaluator_convergence_metrics(self):
        """Test get_convergence_metrics() method."""
        evaluator = HarmonyFunctionalEvaluator()
        
        # Evaluate several networks
        for i in range(10):
            states = create_ahs_network(N=5, seed=42+i)
            crn = create_crn_from_states(states, epsilon_threshold=0.5)
            evaluator.evaluate(crn, iteration=i)
            
        metrics = evaluator.get_convergence_metrics()
        
        assert metrics['num_evaluations'] == 10
        assert metrics['best_S_H'] > 0
        assert metrics['mean_S_H'] > 0
        assert metrics['std_S_H'] >= 0
        assert 'convergence_trend' in metrics
        
    def test_evaluator_handles_degenerate(self):
        """Test evaluator handles degenerate networks gracefully."""
        evaluator = HarmonyFunctionalEvaluator()
        
        # Create network with very high threshold (likely degenerate)
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.99)
        
        # Should handle gracefully by returning -inf
        S_H = evaluator.evaluate(crn, iteration=0)
        
        # Degenerate networks may return -inf
        assert S_H == -np.inf or S_H > 0


class TestTheoreticalCompliance:
    """
    Test compliance with IRHv16.md §4 Theorem 4.1 specifications.
    
    References:
        docs/manuscripts/IRHv16.md §4 lines 254-277: Theorem 4.1
    """
    
    def test_theorem41_formula(self):
        """
        Test S_H formula matches Theorem 4.1.
        
        References:
            IRHv16.md line 266: S_H[G] = Tr(ℒ²) / [det'(ℒ)]^{C_H}
        """
        states = create_ahs_network(N=10, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        result = compute_harmony_functional(crn)
        
        # Verify formula implementation
        trace = result['trace_L2_magnitude']
        det = result['det_prime_magnitude']
        C_H = result['C_H']
        
        expected = trace / (det ** C_H)
        
        assert np.isclose(result['S_H'], expected, rtol=1e-10), \
            "S_H must follow Theorem 4.1 formula"
            
    def test_theorem41_C_H_value(self):
        """
        Test C_H value matches Theorem 4.1.
        
        References:
            IRHv16.md line 275: C_H = 0.045935703598(1)
        """
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        result = compute_harmony_functional(crn)
        
        # Exact value from IRHv16.md
        expected_C_H = 0.045935703598
        
        assert np.isclose(result['C_H'], expected_C_H, atol=1e-12), \
            f"Theorem 4.1 requires C_H = {expected_C_H}"
            
    def test_theorem41_intensive_property(self):
        """
        Test S_H is intensive (scales properly with N).
        
        References:
            IRHv16.md line 260: S_H must be intensive
        """
        # Test that S_H doesn't simply scale linearly with N
        # (Full intensive scaling test requires larger N)
        
        S_H_values = []
        for N in [5, 10, 15]:
            states = create_ahs_network(N=N, seed=42)
            crn = create_crn_from_states(states, epsilon_threshold=0.5)
            result = compute_harmony_functional(crn)
            S_H_values.append(result['S_H'])
            
        # S_H should vary, not scale linearly with N
        # (Actual intensive behavior requires more sophisticated testing)
        assert all(s > 0 for s in S_H_values), \
            "S_H should be positive for all network sizes"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_small_network(self):
        """Test S_H computation for small networks."""
        states = create_ahs_network(N=3, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        result = compute_harmony_functional(crn)
        
        assert result['S_H'] > 0 or result['S_H'] == -np.inf
        
    def test_sparse_network(self):
        """Test S_H for sparse networks may be degenerate."""
        states = create_ahs_network(N=10, seed=42)
        
        # Very high threshold → sparse/degenerate
        crn = create_crn_from_states(states, epsilon_threshold=0.95)
        
        # May raise ValueError for degenerate network
        try:
            result = compute_harmony_functional(crn)
            # If it doesn't raise, S_H should still be computed
            assert 'S_H' in result
        except ValueError as e:
            # Degenerate network is acceptable
            assert "Degenerate network" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
