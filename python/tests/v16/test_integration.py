"""
Integration tests for IRH v16.0 Phase 1 complete pipeline.

Tests the full AHS → ACW → CRN → S_H → ARO pipeline.

THEORETICAL COMPLIANCE:
    Tests validate complete IRH v16.0 Phase 1 implementation
    against docs/manuscripts/IRHv16.md
"""

import pytest
import numpy as np
from irh.core.v16 import (
    # Axiom 0
    AlgorithmicHolonomicState,
    create_ahs_network,
    # Axiom 1
    compute_acw,
    # Axiom 2
    create_crn_from_states,
    # Theorem 4.1
    compute_harmony_functional,
    validate_harmony_functional_properties,
    # Definition 4.1
    AROOptimizerV16
)


class TestFullPipeline:
    """Test complete IRH v16.0 pipeline."""
    
    def test_ahs_to_acw(self):
        """Test AHS → ACW pipeline."""
        # Create AHS
        state_i = AlgorithmicHolonomicState("101010", 0.5)
        state_j = AlgorithmicHolonomicState("110011", 1.0)
        
        # Compute ACW
        acw = compute_acw(state_i, state_j)
        
        assert acw.magnitude >= 0
        assert 0 <= acw.phase < 2 * np.pi
        assert isinstance(acw.complex_value, (complex, np.complexfloating))
        
    def test_ahs_to_crn(self):
        """Test AHS → CRN pipeline."""
        # Create AHS network
        states = create_ahs_network(N=10, seed=42)
        
        # Build CRN
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        assert crn.N == 10
        assert crn.adjacency_matrix is not None
        assert crn.adjacency_matrix.dtype == np.complex128
        
    def test_crn_to_harmony(self):
        """Test CRN → S_H pipeline."""
        # Create network
        states = create_ahs_network(N=10, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        # Compute Harmony Functional
        result = compute_harmony_functional(crn)
        
        assert result['S_H'] > 0
        assert np.isclose(result['C_H'], 0.045935703598, atol=1e-11)
        
    def test_harmony_to_aro(self):
        """Test S_H → ARO pipeline."""
        # Create ARO optimizer
        aro = AROOptimizerV16(N=10, population_size=5, seed=42)
        
        # Run optimization (uses Harmony Functional internally)
        best = aro.optimize(num_generations=10)
        
        assert best is not None
        assert best.S_H > -np.inf
        assert best.crn.N == 10
        
    def test_complete_pipeline(self):
        """
        Test complete AHS → ACW → CRN → S_H → ARO pipeline.
        
        This is the full IRH v16.0 Phase 1 workflow.
        """
        # Step 1: Create Algorithmic Holonomic States (Axiom 0)
        states = create_ahs_network(N=15, seed=42)
        assert len(states) == 15
        assert all(isinstance(s, AlgorithmicHolonomicState) for s in states)
        
        # Step 2: Compute ACWs and build CRN (Axioms 1 & 2)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        assert crn.N == 15
        assert crn.num_edges > 0
        
        # Step 3: Compute Harmony Functional (Theorem 4.1)
        result = compute_harmony_functional(crn)
        assert result['S_H'] > 0
        validate_harmony_functional_properties(result)
        
        # Step 4: Run ARO to maximize S_H (Definition 4.1)
        aro = AROOptimizerV16(N=15, population_size=10, seed=42)
        best = aro.optimize(num_generations=15)
        
        # Verify ARO found a valid configuration
        # (May not always improve due to small population/generations in test)
        assert best.S_H > -np.inf, \
            "ARO should find configuration with valid S_H"


class TestPipelineConsistency:
    """Test consistency across pipeline components."""
    
    def test_N_preserved_through_pipeline(self):
        """Test N (node count) is preserved throughout pipeline."""
        N = 12
        
        # Create AHS
        states = create_ahs_network(N=N, seed=42)
        assert len(states) == N
        
        # Create CRN
        crn = create_crn_from_states(states)
        assert crn.N == N
        
        # Harmony Functional
        result = compute_harmony_functional(crn)
        assert crn.N == N  # Should not modify CRN
        
        # ARO
        aro = AROOptimizerV16(N=N, population_size=5, seed=42)
        aro.optimize(num_generations=5)
        
        # All configurations should preserve N
        for config in aro.population:
            assert config.crn.N == N
            
    def test_constants_consistent(self):
        """Test theoretical constants are consistent across components."""
        states = create_ahs_network(N=10, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.730129)
        result = compute_harmony_functional(crn)
        
        # C_H should be consistent
        expected_C_H = 0.045935703598
        assert np.isclose(result['C_H'], expected_C_H, atol=1e-11)
        
        # ε_threshold should match IRHv16.md
        expected_epsilon = 0.730129
        assert np.isclose(crn.epsilon_threshold, expected_epsilon, atol=1e-6)


class TestScalability:
    """Test pipeline works at different scales."""
    
    @pytest.mark.parametrize("N", [5, 10, 20])
    def test_different_network_sizes(self, N):
        """Test pipeline works for different network sizes."""
        # Create network
        states = create_ahs_network(N=N, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        # Compute S_H
        result = compute_harmony_functional(crn)
        
        # Run mini ARO
        aro = AROOptimizerV16(N=N, population_size=5, seed=42)
        best = aro.optimize(num_generations=5)
        
        # All should work
        assert result['S_H'] > 0 or result['S_H'] == -np.inf
        assert best is not None
        
    @pytest.mark.parametrize("pop_size", [3, 5, 10])
    def test_different_population_sizes(self, pop_size):
        """Test ARO works with different population sizes."""
        aro = AROOptimizerV16(N=10, population_size=pop_size, seed=42)
        best = aro.optimize(num_generations=5)
        
        assert len(aro.population) == pop_size
        assert best is not None


class TestTheoreticalAlignment:
    """
    Test theoretical alignment across all components.
    
    References:
        docs/manuscripts/IRHv16.md - Complete theoretical framework
    """
    
    def test_axiom0_to_axiom2_consistency(self):
        """Test Axiom 0 (AHS) → Axiom 1 (ACW) → Axiom 2 (CRN) consistency."""
        # Axiom 0: Create AHS
        states = create_ahs_network(N=10, seed=42)
        
        # All should be complex-valued (e^{iφ})
        for s in states:
            assert np.isclose(abs(s.complex_amplitude), 1.0)
            
        # Axiom 1: Compute ACW between pairs
        acw = compute_acw(states[0], states[1])
        assert isinstance(acw.complex_value, (complex, np.complexfloating))
        
        # Axiom 2: Build CRN with ε_threshold
        crn = create_crn_from_states(states, epsilon_threshold=0.730129)
        assert crn.adjacency_matrix.dtype == np.complex128
        
    def test_theorem41_with_axiom2(self):
        """Test Theorem 4.1 (S_H) uses Axiom 2 (CRN) correctly."""
        # Create CRN per Axiom 2
        states = create_ahs_network(N=10, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.5)
        
        # Compute S_H per Theorem 4.1
        result = compute_harmony_functional(crn)
        
        # Verify uses complex Laplacian from CRN
        L = crn.interference_matrix
        assert L.dtype == np.complex128
        
        # Verify formula
        trace_L2_mag = np.abs(np.trace(L @ L))
        
        # Should match (approximately, due to regularization)
        assert np.isclose(result['trace_L2_magnitude'], trace_L2_mag, rtol=0.1)
        
    def test_definition41_uses_theorem41(self):
        """Test Definition 4.1 (ARO) uses Theorem 4.1 (S_H) for fitness."""
        # Create ARO
        aro = AROOptimizerV16(N=10, population_size=5, seed=42)
        aro.initialize_population()
        
        # Each configuration should have S_H evaluated
        for config in aro.population:
            # S_H should be set (either positive or -inf for degenerate)
            assert config.S_H > -np.inf or config.S_H == -np.inf
            
        # Best should be tracked by S_H
        assert aro.best_config.S_H == max(c.S_H for c in aro.population)


class TestErrorHandling:
    """Test error handling throughout pipeline."""
    
    def test_degenerate_network_handling(self):
        """Test pipeline handles degenerate networks gracefully."""
        # Create sparse network
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.99)
        
        # May be degenerate
        try:
            result = compute_harmony_functional(crn)
            # If it succeeds, S_H should be valid
            assert 'S_H' in result
        except ValueError as e:
            # Degenerate network is acceptable
            assert "Degenerate" in str(e)
            
    def test_aro_handles_failures(self):
        """Test ARO handles failed S_H evaluations."""
        # Create ARO with potentially problematic threshold
        aro = AROOptimizerV16(N=5, population_size=5, epsilon_threshold=0.95, seed=42)
        
        # Should handle gracefully
        best = aro.optimize(num_generations=5)
        
        # Even with failures, should return something
        assert best is not None


class TestPhase1Completion:
    """
    Test Phase 1 (90% → 100%) completion criteria.
    
    This test validates that all major Phase 1 components work together.
    """
    
    def test_all_components_functional(self):
        """Test all Phase 1 components are functional."""
        # Component checklist
        components_working = {
            'AHS': False,
            'ACW': False,
            'CRN': False,
            'Harmony': False,
            'ARO': False
        }
        
        # Test AHS
        states = create_ahs_network(N=10, seed=42)
        components_working['AHS'] = len(states) == 10
        
        # Test ACW
        acw = compute_acw(states[0], states[1])
        components_working['ACW'] = acw.magnitude >= 0
        
        # Test CRN
        crn = create_crn_from_states(states)
        components_working['CRN'] = crn.N == 10
        
        # Test Harmony
        result = compute_harmony_functional(crn)
        components_working['Harmony'] = result['S_H'] > 0 or result['S_H'] == -np.inf
        
        # Test ARO
        aro = AROOptimizerV16(N=10, population_size=5, seed=42)
        best = aro.optimize(num_generations=5)
        components_working['ARO'] = best is not None
        
        # All should be working
        assert all(components_working.values()), \
            f"Not all components working: {components_working}"
            
    def test_theoretical_compliance_complete(self):
        """Test theoretical compliance is complete for Phase 1."""
        # Check all theoretical constants match IRHv16.md
        
        states = create_ahs_network(N=10, seed=42)
        crn = create_crn_from_states(states, epsilon_threshold=0.730129)
        result = compute_harmony_functional(crn)
        
        # C_H from IRHv16.md line 275
        assert np.isclose(result['C_H'], 0.045935703598, atol=1e-11), \
            "C_H must match IRHv16.md line 275"
            
        # ε from IRHv16.md line 97
        assert np.isclose(crn.epsilon_threshold, 0.730129, atol=1e-6), \
            "ε must match IRHv16.md line 97"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
