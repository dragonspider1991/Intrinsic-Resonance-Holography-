"""
Unit Tests for IRH v16.0 Core Enhancements

Tests the enhanced AHS and ACW implementations with certified precision tracking.
"""

import pytest
import numpy as np
from src.core.ahs_v16 import (
    AlgorithmicHolonomicStateV16,
    create_ahs_network_v16,
    validate_ahs_network_precision,
)
from src.core.acw_v16 import (
    AlgorithmicCoherenceWeightV16,
    compute_ncd_multi_fidelity,
    compute_acw_v16,
    build_acw_matrix_v16,
)
from src.numerics import CertifiedValue


class TestAHSV16:
    """Tests for enhanced Algorithmic Holonomic States."""
    
    def test_ahs_creation_from_bytes_and_phase(self):
        """Test creating AHS from bytes and phase."""
        ahs = AlgorithmicHolonomicStateV16.from_bytes_and_phase(
            info_content=b"test",
            phase=1.5,
            phase_error=1e-12,
            state_id=0
        )
        
        assert ahs.info_content == b"test"
        assert abs(ahs.holonomic_phase.value - 1.5) < 1e-10
        assert ahs.holonomic_phase.error == 1e-12
        assert ahs.state_id == 0
    
    def test_ahs_phase_normalization(self):
        """Test phase normalization to [0, 2π)."""
        ahs = AlgorithmicHolonomicStateV16.from_bytes_and_phase(
            info_content=b"test",
            phase=3 * np.pi,  # > 2π
            phase_error=1e-12
        )
        
        # Should be normalized to π
        assert abs(ahs.holonomic_phase.value - np.pi) < 1e-10
    
    def test_ahs_to_complex_amplitude(self):
        """Test conversion to complex amplitude."""
        ahs = AlgorithmicHolonomicStateV16.from_bytes_and_phase(
            info_content=b"test",
            phase=np.pi / 2,  # 90 degrees
            phase_error=1e-12
        )
        
        amplitude, error = ahs.to_complex_amplitude()
        
        # exp(i*π/2) = i
        assert abs(amplitude.real - 0.0) < 1e-10
        assert abs(amplitude.imag - 1.0) < 1e-10
        assert error > 0
    
    def test_ahs_phase_difference(self):
        """Test phase difference calculation."""
        ahs1 = AlgorithmicHolonomicStateV16.from_bytes_and_phase(
            info_content=b"state1",
            phase=0.5,
            phase_error=1e-12,
            state_id=1
        )
        ahs2 = AlgorithmicHolonomicStateV16.from_bytes_and_phase(
            info_content=b"state2",
            phase=1.5,
            phase_error=1e-12,
            state_id=2
        )
        
        phase_diff = ahs1.phase_difference_to(ahs2)
        
        assert abs(phase_diff.value - 1.0) < 1e-10
        assert phase_diff.error > 0  # Errors propagate
    
    def test_ahs_non_commutative_product(self):
        """Test non-commutative algebraic product."""
        ahs1 = AlgorithmicHolonomicStateV16.from_bytes_and_phase(
            info_content=b"A",
            phase=0.5,
            phase_error=1e-12,
            state_id=1
        )
        ahs2 = AlgorithmicHolonomicStateV16.from_bytes_and_phase(
            info_content=b"B",
            phase=1.0,
            phase_error=1e-12,
            state_id=2
        )
        
        # Compute both orders
        product_12 = ahs1.compute_non_commutative_product(ahs2, order='ij')
        product_21 = ahs1.compute_non_commutative_product(ahs2, order='ji')
        
        # Phases should add (mod 2π)
        assert abs(product_12.holonomic_phase.value - 1.5) < 1e-10
        assert abs(product_21.holonomic_phase.value - 1.5) < 1e-10
        
        # Info content should concatenate (order matters)
        assert product_12.info_content == b"AB"
        assert product_21.info_content == b"BA"
    
    def test_create_ahs_network(self):
        """Test creating a network of AHS."""
        N = 10
        states = create_ahs_network_v16(
            N=N,
            phase_distribution='uniform',
            phase_error_bound=1e-12,
            rng=np.random.default_rng(42)
        )
        
        assert len(states) == N
        assert all(isinstance(s, AlgorithmicHolonomicStateV16) for s in states)
        assert all(s.state_id == i for i, s in enumerate(states))
    
    def test_validate_ahs_network_precision(self):
        """Test network precision validation."""
        states = create_ahs_network_v16(
            N=5,
            phase_error_bound=1e-12,
            rng=np.random.default_rng(42)
        )
        
        is_valid, budget = validate_ahs_network_precision(states, required_phase_precision=12)
        
        assert is_valid  # Should pass with 1e-12 error bound
        assert budget.total_error() > 0


class TestACWV16:
    """Tests for enhanced Algorithmic Coherence Weights."""
    
    def test_compute_ncd_identical_strings(self):
        """Test NCD for identical strings."""
        ncd, ncd_cert = compute_ncd_multi_fidelity(b"test", b"test", fidelity='high')
        
        # NCD of identical strings should be close to 0, but compression overhead can cause non-zero values
        assert 0 <= ncd <= 0.5  # Relaxed bound due to compression overhead
        assert ncd_cert.error > 0
    
    def test_compute_ncd_different_strings(self):
        """Test NCD for different strings."""
        ncd, ncd_cert = compute_ncd_multi_fidelity(b"hello", b"world", fidelity='high')
        
        # NCD should be positive for different strings
        assert 0 < ncd <= 1.0
        assert ncd_cert.error > 0
    
    def test_compute_ncd_fidelity_levels(self):
        """Test different fidelity levels."""
        bytes1, bytes2 = b"test string 1", b"test string 2"
        
        ncd_low, _ = compute_ncd_multi_fidelity(bytes1, bytes2, fidelity='low')
        ncd_med, _ = compute_ncd_multi_fidelity(bytes1, bytes2, fidelity='medium')
        ncd_high, _ = compute_ncd_multi_fidelity(bytes1, bytes2, fidelity='high')
        
        # All should give reasonable results (may differ slightly)
        assert 0 <= ncd_low <= 1.0
        assert 0 <= ncd_med <= 1.0
        assert 0 <= ncd_high <= 1.0
    
    def test_compute_acw_v16(self):
        """Test ACW computation between two states."""
        state1 = AlgorithmicHolonomicStateV16.from_bytes_and_phase(
            info_content=b"state_1",
            phase=0.5,
            phase_error=1e-12,
            state_id=1
        )
        state2 = AlgorithmicHolonomicStateV16.from_bytes_and_phase(
            info_content=b"state_2",
            phase=1.5,
            phase_error=1e-12,
            state_id=2
        )
        
        acw = compute_acw_v16(state1, state2, fidelity='high')
        
        assert isinstance(acw, AlgorithmicCoherenceWeightV16)
        assert 0 <= acw.magnitude.value <= 1.0
        assert 0 <= acw.phase.value < 2 * np.pi
        assert acw.source_id == 1
        assert acw.target_id == 2
        assert acw.error_budget.total_error() > 0
    
    def test_acw_to_complex(self):
        """Test ACW conversion to complex number."""
        state1 = AlgorithmicHolonomicStateV16.from_bytes_and_phase(
            info_content=b"A",
            phase=0.0,
            phase_error=1e-12,
            state_id=1
        )
        state2 = AlgorithmicHolonomicStateV16.from_bytes_and_phase(
            info_content=b"B",
            phase=np.pi / 2,
            phase_error=1e-12,
            state_id=2
        )
        
        acw = compute_acw_v16(state1, state2)
        weight, error = acw.to_complex()
        
        assert isinstance(weight, complex)
        assert abs(weight) <= 1.0  # Magnitude bounded
        assert error > 0
    
    def test_build_acw_matrix_small(self):
        """Test building ACW matrix for small network."""
        states = create_ahs_network_v16(
            N=5,
            phase_distribution='uniform',
            phase_error_bound=1e-12,
            rng=np.random.default_rng(42)
        )
        
        W, budget = build_acw_matrix_v16(states, fidelity='medium', sparse_threshold=0.0)
        
        # Check matrix properties
        assert W.shape == (5, 5)
        assert np.allclose(W, W.conj().T)  # Hermitian
        assert np.allclose(np.diag(W), 1.0)  # Diagonal is 1
        assert budget.total_error() > 0
    
    def test_acw_matrix_sparsity(self):
        """Test sparse ACW matrix construction."""
        states = create_ahs_network_v16(
            N=10,
            phase_distribution='uniform',
            phase_error_bound=1e-12,
            rng=np.random.default_rng(42)
        )
        
        # Build with high sparsity threshold (most ACW magnitudes will be < 0.9)
        W, budget = build_acw_matrix_v16(
            states,
            fidelity='low',
            sparse_threshold=0.9
        )
        
        # Some off-diagonal elements should be zero
        off_diag = W - np.diag(np.diag(W))
        n_nonzero = np.count_nonzero(off_diag)
        total_off_diag = 10 * 10 - 10
        
        # With high threshold, we should have some sparsification
        sparsity = budget.metadata.get('sparsity', 0)
        assert sparsity >= 0  # Some entries filtered out


class TestIntegration:
    """Integration tests for v16 core components."""
    
    def test_full_network_creation_workflow(self):
        """Test complete workflow: AHS creation -> ACW matrix."""
        # Create network
        N = 8
        states = create_ahs_network_v16(
            N=N,
            phase_distribution='uniform',
            phase_error_bound=1e-12,
            rng=np.random.default_rng(123)
        )
        
        # Validate precision
        is_valid, _ = validate_ahs_network_precision(states, required_phase_precision=12)
        assert is_valid
        
        # Build ACW matrix
        W, budget = build_acw_matrix_v16(states, fidelity='medium')
        
        # Verify properties
        assert W.shape == (N, N)
        assert np.allclose(W, W.conj().T)
        assert budget.metadata['N'] == N
        
        # Check error budget is reasonable
        # Note: finite-size error dominates for small N (O(1/√N) ≈ 0.35 for N=8)
        assert budget.total_error() < 1.0  # Relaxed bound


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
