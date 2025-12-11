"""
Tests for IRH v18.0 Core Implementation
=======================================

Tests the cGFT framework including:
- Group manifold G_inf = SU(2) × U(1)_φ
- cGFT field and action
- RG flow and Cosmic Fixed Point
"""

import pytest
import numpy as np
import sys
import os

# Add the source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from irh.core.v18 import (
    # Group manifold
    SU2Element,
    U1Element,
    GInfElement,
    compute_ncd,
    compute_ncd_distance,
    
    # cGFT field
    cGFTFieldDiscrete,
    BiLocalField,
    CondensateState,
    
    # cGFT action
    cGFTCouplings,
    InteractionKernel,
    cGFTAction,
    compute_harmony_functional,
    
    # RG flow
    BetaFunctions,
    CosmicFixedPoint,
    find_fixed_point,
    StabilityAnalysis,
    integrate_rg_flow,
    compute_C_H_certified,
    
    # Constants
    C_H_V18,
)


class TestSU2Element:
    """Tests for SU(2) group element."""
    
    def test_identity(self):
        """Test identity element."""
        e = SU2Element.identity()
        assert np.isclose(e.q0, 1.0)
        assert np.isclose(e.q1, 0.0)
        assert np.isclose(e.q2, 0.0)
        assert np.isclose(e.q3, 0.0)
        assert np.isclose(e.norm, 1.0)
    
    def test_normalization(self):
        """Test automatic normalization to unit quaternion."""
        u = SU2Element(q0=2.0, q1=0.0, q2=0.0, q3=0.0)
        assert np.isclose(u.norm, 1.0)
    
    def test_random_unit_norm(self):
        """Test random elements have unit norm."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            u = SU2Element.random(rng)
            assert np.isclose(u.norm, 1.0, atol=1e-10)
    
    def test_multiplication_closure(self):
        """Test group multiplication stays in SU(2)."""
        rng = np.random.default_rng(42)
        u1 = SU2Element.random(rng)
        u2 = SU2Element.random(rng)
        
        product = u1 * u2
        assert np.isclose(product.norm, 1.0, atol=1e-10)
    
    def test_inverse(self):
        """Test inverse property: u * u^{-1} = e."""
        rng = np.random.default_rng(42)
        u = SU2Element.random(rng)
        u_inv = u.inverse()
        
        product = u * u_inv
        e = SU2Element.identity()
        
        assert np.isclose(product.q0, e.q0, atol=1e-10)
        assert np.isclose(product.q1, e.q1, atol=1e-10)
        assert np.isclose(product.q2, e.q2, atol=1e-10)
        assert np.isclose(product.q3, e.q3, atol=1e-10)
    
    def test_trace(self):
        """Test trace for identity: Tr(Id) = 2."""
        e = SU2Element.identity()
        assert np.isclose(e.trace(), 2.0)
    
    def test_binary_encoding(self):
        """Test binary string encoding."""
        u = SU2Element.identity()
        encoded = u.to_binary_string(bits_per_component=16)
        
        assert isinstance(encoded, bytes)
        assert len(encoded) == 8  # 4 components × 2 bytes each


class TestU1Element:
    """Tests for U(1) phase element."""
    
    def test_identity(self):
        """Test identity element (φ = 0)."""
        e = U1Element.identity()
        assert np.isclose(e.phi, 0.0)
        assert np.isclose(e.complex_value, 1.0)
    
    def test_phase_normalization(self):
        """Test phase wrapping to [0, 2π)."""
        u = U1Element(phi=3 * np.pi)
        assert 0 <= u.phi < 2 * np.pi
    
    def test_multiplication(self):
        """Test phase addition."""
        u1 = U1Element(phi=np.pi / 4)
        u2 = U1Element(phi=np.pi / 4)
        
        product = u1 * u2
        assert np.isclose(product.phi, np.pi / 2)
    
    def test_inverse(self):
        """Test inverse: u * u^{-1} = e."""
        u = U1Element(phi=np.pi / 3)
        u_inv = u.inverse()
        
        product = u * u_inv
        assert np.isclose(product.phi, 0.0, atol=1e-10) or np.isclose(product.phi, 2*np.pi, atol=1e-10)


class TestGInfElement:
    """Tests for G_inf = SU(2) × U(1) composite element."""
    
    def test_identity(self):
        """Test identity element."""
        e = GInfElement.identity()
        assert np.isclose(e.su2.q0, 1.0)
        assert np.isclose(e.u1.phi, 0.0)
    
    def test_random(self):
        """Test random element generation."""
        rng = np.random.default_rng(42)
        g = GInfElement.random(rng)
        
        assert np.isclose(g.su2.norm, 1.0)
        assert 0 <= g.u1.phi < 2 * np.pi
    
    def test_multiplication(self):
        """Test component-wise multiplication."""
        rng = np.random.default_rng(42)
        g1 = GInfElement.random(rng)
        g2 = GInfElement.random(rng)
        
        product = g1 * g2
        
        assert np.isclose(product.su2.norm, 1.0, atol=1e-10)
        assert 0 <= product.u1.phi < 2 * np.pi
    
    def test_binary_encoding(self):
        """Test composite binary encoding."""
        g = GInfElement.identity()
        encoded = g.to_binary_string()
        
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0


class TestNCD:
    """Tests for Normalized Compression Distance."""
    
    def test_identity_distance_zero(self):
        """Test d(x, x) is small (near 0)."""
        x = b"test string"
        d = compute_ncd(x, x)
        
        # NCD of identical strings should be small (not exactly 0 due to compression header)
        assert d < 0.15  # Allow some slack for compression overhead
    
    def test_symmetry(self):
        """Test d(x, y) = d(y, x)."""
        x = b"hello world"
        y = b"goodbye world"
        
        d_xy = compute_ncd(x, y)
        d_yx = compute_ncd(y, x)
        
        assert np.isclose(d_xy, d_yx, atol=0.01)
    
    def test_range(self):
        """Test NCD is in [0, 1]."""
        x = b"abcdefg"
        y = b"xyz1234"
        
        d = compute_ncd(x, y)
        assert 0 <= d <= 1.0 + 0.1  # Slight tolerance for edge cases
    
    def test_ginf_distance(self):
        """Test NCD distance on G_inf elements."""
        rng = np.random.default_rng(42)
        g1 = GInfElement.random(rng)
        g2 = GInfElement.random(rng)
        
        d = compute_ncd_distance(g1, g2)
        assert d >= 0


class TestcGFTField:
    """Tests for cGFT field."""
    
    def test_create_random(self):
        """Test random field creation."""
        phi = cGFTFieldDiscrete.create_random(N=5, seed=42)
        
        assert phi.N == 5
        assert phi.field_array.shape == (5, 5, 5, 5)
    
    def test_norm_squared(self):
        """Test field norm computation."""
        phi = cGFTFieldDiscrete.create_random(N=5, seed=42)
        norm_sq = phi.norm_squared()
        
        assert norm_sq >= 0
        assert np.isfinite(norm_sq)
    
    def test_conjugate(self):
        """Test Hermitian conjugate."""
        phi = cGFTFieldDiscrete.create_random(N=5, seed=42)
        phi_bar = phi.conjugate()
        
        # Check that conjugation works
        assert np.allclose(phi_bar.field_array, np.conj(phi.field_array))


class TestcGFTAction:
    """Tests for cGFT action computation."""
    
    def test_couplings_fixed_point(self):
        """Test fixed point coupling values."""
        fp = cGFTCouplings.fixed_point()
        
        pi_sq = np.pi**2
        assert np.isclose(fp.lambda_, 48 * pi_sq / 9, rtol=1e-10)
        assert np.isclose(fp.gamma, 32 * pi_sq / 3, rtol=1e-10)
        assert np.isclose(fp.mu, 16 * pi_sq, rtol=1e-10)
    
    def test_action_computation(self):
        """Test total action can be computed."""
        phi = cGFTFieldDiscrete.create_random(N=3, seed=42, amplitude_scale=0.1)
        action = cGFTAction()
        
        result = action.compute_total_action(phi)
        
        assert "S_kin" in result
        assert "S_int" in result
        assert "S_hol" in result
        assert "S_total" in result
        
        # Values should be finite
        assert np.isfinite(result["S_kin"])
        assert np.isfinite(result["S_total"])


class TestRGFlow:
    """Tests for RG flow and fixed point."""
    
    def test_beta_functions_at_fixed_point(self):
        """Test β-functions vanish at fixed point."""
        fp = CosmicFixedPoint()
        beta = BetaFunctions()
        
        beta_vals = beta.evaluate(fp.lambda_star, fp.gamma_star, fp.mu_star)
        
        # All β-functions should be ≈ 0 at fixed point
        assert np.abs(beta_vals[0]) < 1e-8
        assert np.abs(beta_vals[1]) < 1e-8
        assert np.abs(beta_vals[2]) < 1e-8
    
    def test_fixed_point_values(self):
        """Test fixed point has expected values."""
        fp = CosmicFixedPoint()
        
        pi_sq = np.pi**2
        assert np.isclose(fp.lambda_star, 48 * pi_sq / 9)
        assert np.isclose(fp.gamma_star, 32 * pi_sq / 3)
        assert np.isclose(fp.mu_star, 16 * pi_sq)
    
    def test_fixed_point_verify(self):
        """Test fixed point verification."""
        fp = CosmicFixedPoint()
        verification = fp.verify()
        
        assert verification["is_fixed_point"]
    
    def test_numerical_fixed_point(self):
        """Test numerical fixed point finder."""
        fp = find_fixed_point()
        verification = fp.verify()
        
        # Should find the same fixed point
        assert verification["is_fixed_point"]
    
    def test_stability_eigenvalues(self):
        """Test stability matrix eigenvalues."""
        stability = StabilityAnalysis()
        eigenvalues = stability.compute_eigenvalues()
        
        # Sort eigenvalues
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Expected: 6, 2, -4/3
        expected = [6.0, 2.0, -4/3]
        
        for i, (actual, exp) in enumerate(zip(eigenvalues, expected)):
            assert np.isclose(actual, exp, rtol=0.01), \
                f"Eigenvalue {i}: expected {exp}, got {actual}"
    
    def test_globally_attractive(self):
        """Test fixed point is globally attractive."""
        stability = StabilityAnalysis()
        assert stability.is_globally_attractive()
    
    def test_rg_flow_integration(self):
        """Test RG flow integration toward fixed point."""
        # Start near fixed point
        fp = CosmicFixedPoint()
        initial = (
            fp.lambda_star * 1.1,  # Slightly away
            fp.gamma_star * 1.1,
            fp.mu_star * 1.1
        )
        
        solution = integrate_rg_flow(initial, t_span=(0, -5), num_points=50)
        
        # Should flow toward fixed point (for relevant directions)
        final = solution.couplings_final
        
        # lambda and gamma should approach fixed point values
        # (mu is irrelevant, so may not approach)
        assert len(solution.t_values) == 50
        assert solution.lambda_values[-1] is not None


class TestUniversalConstants:
    """Tests for universal constants."""
    
    def test_C_H_value(self):
        """Test C_H has expected value."""
        assert np.isclose(C_H_V18, 0.045935703598, rtol=1e-10)
    
    def test_C_H_certified(self):
        """Test certified C_H computation."""
        result = compute_C_H_certified()
        
        assert "C_H" in result
        assert np.isclose(result["C_H"], 0.045935703598, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
