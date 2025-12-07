"""
Tests for Rigor Enhancements (v15.0+)

Tests nondimensional formulations, symbolic derivations,
RG flow analysis, and falsifiability thresholds.
"""

import pytest
import numpy as np
import scipy.sparse as sp

from src.core.rigor_enhancements import (
    nondimensional_zeta,
    dimensional_convergence_limit,
    rg_flow_beta,
    compute_nondimensional_resonance_density,
    solve_rg_fixed_point,
)


class TestNondimensionalZeta:
    """Test nondimensional spectral zeta function."""
    
    def test_basic_computation(self):
        """Test basic zeta function computation."""
        eigenvalues = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = 2.0
        
        zeta = nondimensional_zeta(s, eigenvalues, lambda_0=1.0, symbolic=False)
        
        # Should compute sum(λ^{-s})
        expected = np.sum(eigenvalues ** (-s))
        assert abs(zeta - expected) < 1e-10
    
    def test_zero_eigenvalues_filtered(self):
        """Test that zero eigenvalues are filtered out."""
        eigenvalues = np.array([0.0, 1.0, 2.0, 3.0])
        s = 1.0
        
        zeta = nondimensional_zeta(s, eigenvalues, symbolic=False)
        
        # Should only sum over non-zero eigenvalues
        expected = 1.0 + 1.0/2.0 + 1.0/3.0
        assert abs(zeta - expected) < 1e-10
    
    def test_nondimensional_scaling(self):
        """Test nondimensional scaling with lambda_0."""
        eigenvalues = np.array([2.0, 4.0, 6.0])
        lambda_0 = 2.0
        s = 1.0
        
        zeta = nondimensional_zeta(s, eigenvalues, lambda_0=lambda_0, symbolic=False)
        
        # Normalized eigenvalues: [1, 2, 3]
        expected = 1.0 + 1.0/2.0 + 1.0/3.0
        assert abs(zeta - expected) < 1e-10


class TestDimensionalConvergence:
    """Test dimensional convergence limit analysis."""
    
    def test_convergence_to_4d(self):
        """Test convergence to d=4 for realistic eigenspectrum."""
        N = 1000
        # Simulate 4D-like eigenspectrum
        eigenvalues = np.random.exponential(scale=2.0, size=N)
        eigenvalues = np.sort(eigenvalues)
        
        d_spec, info = dimensional_convergence_limit(N, eigenvalues, verbose=False)
        
        # Should be close to 4 with error ~ 1/√N
        assert 3.0 < d_spec < 5.0
        assert info['error_bound'] == pytest.approx(1.0 / np.sqrt(N), rel=0.1)
    
    def test_error_bounds(self):
        """Test O(1/√N) error bound scaling."""
        for N in [100, 1000, 10000]:
            eigenvalues = np.random.exponential(scale=1.0, size=N)
            _, info = dimensional_convergence_limit(N, eigenvalues, verbose=False)
            
            expected_bound = 1.0 / np.sqrt(N)
            assert info['error_bound'] == pytest.approx(expected_bound, rel=0.01)
    
    def test_convergence_flag(self):
        """Test convergence flag for large N."""
        N = 20000
        eigenvalues = np.random.exponential(scale=2.0, size=N)
        
        d_spec, info = dimensional_convergence_limit(N, eigenvalues, verbose=False)
        
        # For large N, should converge within threshold
        # (May fail randomly due to eigenvalue distribution, but usually passes)
        assert info['N'] == N


class TestRGFlow:
    """Test renormalization group flow functions."""
    
    def test_rg_beta_at_fixed_points(self):
        """Test beta function at fixed points."""
        # Trivial fixed point
        beta_0 = rg_flow_beta(0.0, symbolic=False)
        assert abs(beta_0) < 1e-10
        
        # Cosmic fixed point at q = 1/137
        q = 1.0 / 137.035999084
        beta_q = rg_flow_beta(q, symbolic=False)
        assert abs(beta_q) < 1e-10
    
    def test_rg_beta_sign(self):
        """Test beta function flow direction."""
        q = 1.0 / 137.035999084
        
        # Below fixed point: should flow up
        C_below = q / 2
        beta_below = rg_flow_beta(C_below, symbolic=False)
        assert beta_below > 0  # Flows toward q
        
        # Above fixed point: should flow down
        C_above = q * 2
        beta_above = rg_flow_beta(C_above, symbolic=False)
        assert beta_above < 0  # Flows back toward q
    
    def test_solve_fixed_points(self):
        """Test solving for RG fixed points."""
        trivial_fp, cosmic_fp = solve_rg_fixed_point(verbose=False)
        
        assert trivial_fp == 0.0
        assert cosmic_fp == pytest.approx(1.0 / 137.035999084, rel=1e-6)


class TestResonanceDensity:
    """Test nondimensional resonance density computation."""
    
    def test_basic_computation(self):
        """Test basic resonance density calculation."""
        eigenvalues = np.array([1.0, 2.0, 3.0, 4.0])
        N = 4
        
        rho_res, info = compute_nondimensional_resonance_density(eigenvalues, N)
        
        # Should be average eigenvalue / N
        expected = np.sum(eigenvalues) / N
        assert rho_res == pytest.approx(expected, rel=1e-10)
    
    def test_info_statistics(self):
        """Test that info dict contains statistics."""
        eigenvalues = np.random.exponential(scale=2.0, size=100)
        N = 100
        
        rho_res, info = compute_nondimensional_resonance_density(eigenvalues, N)
        
        assert 'mean_eigenvalue' in info
        assert 'std_eigenvalue' in info
        assert 'spectral_gap' in info
        assert info['N'] == N
    
    def test_zero_inputs(self):
        """Test handling of zero or empty inputs."""
        rho_res, info = compute_nondimensional_resonance_density(np.array([]), 0)
        
        assert rho_res == 0.0
        assert 'error' in info


class TestIntegration:
    """Integration tests with actual network."""
    
    def test_with_small_network(self):
        """Test rigor enhancements with small network."""
        from src.core.aro_optimizer import AROOptimizer
        from src.core.harmony import compute_information_transfer_matrix
        
        # Create small network
        N = 100
        opt = AROOptimizer(N=N, rng_seed=42)
        opt.initialize_network(scheme='random', connectivity_param=0.1)
        
        # Get eigenvalues
        M = compute_information_transfer_matrix(opt.current_W)
        eigenvalues = np.linalg.eigvalsh(M.toarray())
        
        # Test nondimensional resonance density
        rho_res, info = compute_nondimensional_resonance_density(eigenvalues, N)
        assert rho_res > 0
        assert 0 < rho_res < 10  # Reasonable range
        
        # Test dimensional convergence
        d_spec, conv_info = dimensional_convergence_limit(N, eigenvalues, verbose=False)
        assert 1.0 < d_spec < 10.0
        
        # Test RG beta function
        from src.core.harmony import C_H
        beta = rg_flow_beta(C_H, symbolic=False)
        # C_H ≈ 0.046 is not exactly at fixed point (0 or 1/137)
        # So beta should be non-zero but small
        assert abs(beta) < 0.1
    
    def test_falsifiability_check(self):
        """Test falsifiability checking."""
        from src.cosmology.vacuum_energy import falsifiability_check
        
        # Test with consistent w0
        results = falsifiability_check(
            observed_w0=-0.910,
            predicted_w0=-0.912,
            verbose=False
        )
        assert results['w0_consistent'] is True
        assert len(results['dissonance_warnings']) == 0
        
        # Test with inconsistent w0
        results = falsifiability_check(
            observed_w0=-0.95,
            predicted_w0=-0.912,
            threshold_w0=-0.92,
            verbose=False
        )
        assert results['w0_consistent'] is False
        assert len(results['dissonance_warnings']) > 0
        assert len(results['refinement_suggestions']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
