"""
Tests for IRH v17.0 Beta Functions Module

Tests the implementation of:
- One-loop β-functions (Eq.1.13)
- Fixed-point computation (Eq.1.14)
- Stability matrix analysis

References:
    IRH v17.0 Manuscript: docs/manuscripts/IRHv17.md, Section 1.2
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose


class TestBetaFunctions:
    """Tests for beta function implementations."""
    
    def test_beta_lambda_at_fixed_point(self):
        """β_λ should vanish at the fixed point."""
        from irh.core.v17.beta_functions import (
            beta_lambda,
            FIXED_POINT_LAMBDA,
        )
        
        result = beta_lambda(FIXED_POINT_LAMBDA)
        # β_λ = -2λ̃ + (9/8π²)λ̃² should be 0 at λ̃*
        assert_allclose(result, 0.0, atol=1e-10)
    
    def test_beta_lambda_formula(self):
        """Test β_λ formula at arbitrary points."""
        from irh.core.v17.beta_functions import beta_lambda
        
        # At λ̃ = 0, β_λ = 0
        assert beta_lambda(0.0) == 0.0
        
        # At small λ̃, β_λ ≈ -6λ̃ (negative)
        lambda_small = 1.0
        result = beta_lambda(lambda_small)
        assert result < 0  # -6(1) + (9/8π²)(1)² < 0
    
    def test_beta_gamma_formula(self):
        """Test β_γ formula."""
        from irh.core.v17.beta_functions import (
            beta_gamma,
            FIXED_POINT_LAMBDA,
            FIXED_POINT_GAMMA,
        )
        
        # β_γ = -2γ̃ + (3/4π²)λ̃γ̃
        result = beta_gamma(FIXED_POINT_LAMBDA, FIXED_POINT_GAMMA)
        expected = -2.0 * FIXED_POINT_GAMMA + (3.0 / (4.0 * np.pi**2)) * FIXED_POINT_LAMBDA * FIXED_POINT_GAMMA
        assert_allclose(result, expected, rtol=1e-12)
    
    def test_beta_mu_formula(self):
        """Test β_μ formula."""
        from irh.core.v17.beta_functions import (
            beta_mu,
            FIXED_POINT_LAMBDA,
            FIXED_POINT_MU,
        )
        
        # β_μ = -4μ̃ + (1/2π²)λ̃μ̃
        result = beta_mu(FIXED_POINT_LAMBDA, None, FIXED_POINT_MU)
        expected = -4.0 * FIXED_POINT_MU + (1.0 / (2.0 * np.pi**2)) * FIXED_POINT_LAMBDA * FIXED_POINT_MU
        assert_allclose(result, expected, rtol=1e-12)
    
    def test_fixed_point_values(self):
        """Test that fixed-point values match Eq.1.14."""
        from irh.core.v17.beta_functions import (
            FIXED_POINT_LAMBDA,
            FIXED_POINT_GAMMA,
            FIXED_POINT_MU,
        )
        
        pi_sq = np.pi ** 2
        
        # λ̃* = 48π²/9
        expected_lambda = 48.0 * pi_sq / 9.0
        assert_allclose(FIXED_POINT_LAMBDA, expected_lambda, rtol=1e-12)
        
        # γ̃* = 32π²/3
        expected_gamma = 32.0 * pi_sq / 3.0
        assert_allclose(FIXED_POINT_GAMMA, expected_gamma, rtol=1e-12)
        
        # μ̃* = 16π²
        expected_mu = 16.0 * pi_sq
        assert_allclose(FIXED_POINT_MU, expected_mu, rtol=1e-12)
    
    def test_compute_fixed_point(self):
        """Test numerical fixed-point computation."""
        from irh.core.v17.beta_functions import (
            compute_fixed_point,
            FIXED_POINT_LAMBDA,
            FIXED_POINT_GAMMA,
            FIXED_POINT_MU,
        )
        
        lambda_star, gamma_star, mu_star = compute_fixed_point()
        
        # Should match the analytic values
        assert_allclose(lambda_star, FIXED_POINT_LAMBDA, rtol=1e-10)
        assert_allclose(gamma_star, FIXED_POINT_GAMMA, rtol=1e-10)
        assert_allclose(mu_star, FIXED_POINT_MU, rtol=1e-10)
    
    def test_stability_matrix(self):
        """Test stability matrix computation at fixed point."""
        from irh.core.v17.beta_functions import (
            compute_stability_matrix,
            FIXED_POINT_LAMBDA,
            FIXED_POINT_GAMMA,
            FIXED_POINT_MU,
        )
        
        M = compute_stability_matrix(
            FIXED_POINT_LAMBDA,
            FIXED_POINT_GAMMA,
            FIXED_POINT_MU,
        )
        
        # Matrix should be 3x3
        assert M.shape == (3, 3)
        
        # Check specific elements
        # ∂β_λ/∂λ = -6 + (9/4π²)λ*
        expected_00 = -6.0 + (9.0 / (4.0 * np.pi**2)) * FIXED_POINT_LAMBDA
        assert_allclose(M[0, 0], expected_00, rtol=1e-10)
    
    def test_symbolic_fixed_point(self):
        """Test symbolic computation of fixed point."""
        from irh.core.v17.beta_functions import compute_fixed_point_symbolic
        import sympy as sp
        
        lambda_star, gamma_star, mu_star = compute_fixed_point_symbolic()
        
        # Evaluate symbolically
        pi = sp.pi
        expected_lambda = 48 * pi**2 / 9
        expected_gamma = 32 * pi**2 / 3
        expected_mu = 16 * pi**2
        
        assert sp.simplify(lambda_star - expected_lambda) == 0
        assert sp.simplify(gamma_star - expected_gamma) == 0
        assert sp.simplify(mu_star - expected_mu) == 0


class TestBetaFunctionNumericalProperties:
    """Tests for numerical properties of beta functions."""
    
    def test_beta_lambda_positive_for_large_coupling(self):
        """β_λ should be positive for large λ̃ (UV-unstable direction)."""
        from irh.core.v17.beta_functions import beta_lambda, FIXED_POINT_LAMBDA
        
        # For λ̃ > λ̃*, β_λ > 0
        large_lambda = 2 * FIXED_POINT_LAMBDA
        result = beta_lambda(large_lambda)
        assert result > 0
    
    def test_beta_lambda_negative_for_small_coupling(self):
        """β_λ should be negative for small λ̃."""
        from irh.core.v17.beta_functions import beta_lambda
        
        # For small λ̃, β_λ ≈ -2λ̃ < 0
        small_lambda = 1.0
        result = beta_lambda(small_lambda)
        assert result < 0
    
    def test_fixed_point_numerical_values(self):
        """Test numerical values of fixed point constants."""
        from irh.core.v17.beta_functions import (
            FIXED_POINT_LAMBDA,
            FIXED_POINT_GAMMA,
            FIXED_POINT_MU,
        )
        
        # Approximate numerical values
        assert 52 < FIXED_POINT_LAMBDA < 53
        assert 105 < FIXED_POINT_GAMMA < 106
        assert 157 < FIXED_POINT_MU < 158
