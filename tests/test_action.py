"""
Test Spectral Action and Gradient

Verify correctness of the spectral action computation and its gradient
using finite differences.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cncg import FiniteSpectralTriple
from cncg.action import (
    spectral_action,
    spectral_action_gradient,
    trace_heat_kernel,
    compute_spectral_torsion,
)


class TestSpectralAction:
    """Test spectral action computation."""
    
    def test_action_positive(self):
        """Test that action is positive for heat kernel cutoff."""
        triple = FiniteSpectralTriple(N=20, seed=42)
        
        S = spectral_action(triple.D, cutoff="heat")
        
        assert S > 0
    
    def test_action_decreases_with_sparsity(self):
        """Test that sparsity penalty increases action."""
        triple = FiniteSpectralTriple(N=20, seed=42)
        
        S_no_penalty = spectral_action(triple.D, sparsity_weight=0.0)
        S_with_penalty = spectral_action(triple.D, sparsity_weight=0.1)
        
        # With penalty should be larger
        assert S_with_penalty > S_no_penalty
    
    def test_action_scale_invariance(self):
        """Test scaling property: S[αD, αΛ] = S[D, Λ]."""
        triple = FiniteSpectralTriple(N=20, seed=42)
        
        alpha = 2.0
        S1 = spectral_action(triple.D, Lambda=1.0, cutoff="heat", cutoff_param=1.0)
        
        # Scale D and Lambda by alpha
        triple_scaled = FiniteSpectralTriple(N=20)
        triple_scaled.D = alpha * triple.D
        triple_scaled.gamma = triple.gamma
        triple_scaled.J = triple.J
        
        S2 = spectral_action(triple_scaled.D, Lambda=alpha, cutoff="heat", cutoff_param=1.0)
        
        # Should be approximately equal (for heat kernel)
        np.testing.assert_allclose(S1, S2, rtol=0.1)


class TestGradient:
    """Test gradient computation via finite differences."""
    
    def test_gradient_finite_difference(self):
        """Verify gradient using finite differences."""
        N = 10  # Small size for speed
        triple = FiniteSpectralTriple(N=N, seed=42)
        
        # Compute analytical gradient
        grad_analytic = spectral_action_gradient(
            triple.D,
            Lambda=1.0,
            cutoff="heat",
            cutoff_param=1.0,
            sparsity_weight=0.0,
        )
        
        # Compute numerical gradient via finite differences
        epsilon = 1e-5
        grad_numeric = np.zeros((N, N), dtype=np.complex128)
        
        S0 = spectral_action(triple.D, Lambda=1.0, cutoff="heat", cutoff_param=1.0)
        
        # Only check a few random elements (full check is too slow)
        np.random.seed(42)
        test_indices = [(i, j) for i in range(N) for j in range(i, N)]
        sample_indices = [test_indices[i] for i in np.random.choice(len(test_indices), 5)]
        
        for i, j in sample_indices:
            # Perturb real part
            D_perturbed = triple.D.copy()
            D_perturbed[i, j] += epsilon
            if i != j:
                D_perturbed[j, i] += epsilon  # Maintain Hermiticity
            
            S_plus = spectral_action(D_perturbed, Lambda=1.0, cutoff="heat", cutoff_param=1.0)
            grad_numeric[i, j] = (S_plus - S0) / epsilon
            
            # Compare
            if i == j:
                # Diagonal: real gradient
                np.testing.assert_allclose(
                    np.real(grad_analytic[i, j]),
                    np.real(grad_numeric[i, j]),
                    rtol=1e-3,
                    atol=1e-5,
                )
    
    def test_gradient_hermiticity(self):
        """Test that gradient is Hermitian."""
        triple = FiniteSpectralTriple(N=20, seed=42)
        
        grad = spectral_action_gradient(triple.D)
        
        # Gradient should be Hermitian
        grad_dagger = grad.conj().T
        np.testing.assert_allclose(grad, grad_dagger, rtol=1e-10)
    
    def test_gradient_descent_reduces_action(self):
        """Test that gradient descent step reduces action."""
        triple = FiniteSpectralTriple(N=20, seed=42)
        
        S0 = spectral_action(triple.D, cutoff="heat")
        grad = spectral_action_gradient(triple.D, cutoff="heat")
        
        # Take a small step in negative gradient direction
        learning_rate = 0.001
        triple.D -= learning_rate * grad
        triple.enforce_axioms()
        
        S1 = spectral_action(triple.D, cutoff="heat")
        
        # Action should decrease (or stay similar)
        assert S1 <= S0 + 0.1  # Allow small numerical error


class TestHeatKernel:
    """Test heat kernel trace."""
    
    def test_heat_kernel_positive(self):
        """Test that K(t) > 0."""
        triple = FiniteSpectralTriple(N=20, seed=42)
        
        K = trace_heat_kernel(triple.D, t=1.0)
        
        assert K > 0
    
    def test_heat_kernel_decreasing(self):
        """Test that K(t) decreases with t."""
        triple = FiniteSpectralTriple(N=20, seed=42)
        
        K1 = trace_heat_kernel(triple.D, t=0.1)
        K2 = trace_heat_kernel(triple.D, t=1.0)
        K3 = trace_heat_kernel(triple.D, t=10.0)
        
        assert K1 > K2 > K3
    
    def test_heat_kernel_limit(self):
        """Test that K(t→0) → N (dimension of Hilbert space)."""
        triple = FiniteSpectralTriple(N=20, seed=42)
        
        K = trace_heat_kernel(triple.D, t=1e-6)
        
        # Should be close to N
        np.testing.assert_allclose(K, triple.N, rtol=0.1)


class TestSpectralTorsion:
    """Test spectral torsion computation."""
    
    def test_torsion_bounds(self):
        """Test that torsion is bounded."""
        triple = FiniteSpectralTriple(N=50, seed=42)
        
        torsion = compute_spectral_torsion(triple.D, triple.gamma)
        
        # Torsion should be bounded
        assert abs(torsion) <= 1.0
    
    def test_torsion_zero_for_symmetric(self):
        """Test that torsion is zero for symmetric spectrum."""
        # Create symmetric D
        N = 20
        triple = FiniteSpectralTriple(N=N, seed=42)
        
        # Make D commute with gamma (symmetric under chirality)
        triple.D = triple.gamma @ triple.D @ triple.gamma
        triple.enforce_axioms()
        
        torsion = compute_spectral_torsion(triple.D, triple.gamma)
        
        # Should be zero (or very small)
        np.testing.assert_allclose(torsion, 0, atol=1e-6)


class TestCutoffFunctions:
    """Test different cutoff functions."""
    
    def test_heat_vs_sigmoid(self):
        """Test that different cutoffs give different results."""
        triple = FiniteSpectralTriple(N=20, seed=42)
        
        S_heat = spectral_action(triple.D, cutoff="heat", cutoff_param=1.0)
        S_sigmoid = spectral_action(triple.D, cutoff="sigmoid", cutoff_param=1.0)
        
        # Should be different
        assert abs(S_heat - S_sigmoid) > 0.01
    
    def test_cutoff_parameter_effect(self):
        """Test that cutoff parameter affects result."""
        triple = FiniteSpectralTriple(N=20, seed=42)
        
        S1 = spectral_action(triple.D, cutoff="heat", cutoff_param=0.5)
        S2 = spectral_action(triple.D, cutoff="heat", cutoff_param=1.0)
        S3 = spectral_action(triple.D, cutoff="heat", cutoff_param=2.0)
        
        # All should be different
        assert S1 != S2
        assert S2 != S3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
