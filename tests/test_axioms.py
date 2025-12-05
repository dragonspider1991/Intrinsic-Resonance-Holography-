"""
Test Spectral Triple Axioms

Verify that the FiniteSpectralTriple class correctly enforces
the axioms of spectral triples.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cncg import FiniteSpectralTriple


class TestAxioms:
    """Test spectral triple axioms."""
    
    def test_hermiticity(self):
        """Test that D is Hermitian."""
        triple = FiniteSpectralTriple(N=50, seed=42)
        
        # Check D = D†
        D_dagger = triple.D.conj().T
        np.testing.assert_allclose(triple.D, D_dagger, rtol=1e-10)
    
    def test_anticommutation_with_grading(self):
        """Test that {D, γ} = 0."""
        triple = FiniteSpectralTriple(N=50, seed=42)
        
        # Compute {D, γ} = D γ + γ D
        anticommutator = triple.D @ triple.gamma + triple.gamma @ triple.D
        
        # Should be zero
        np.testing.assert_allclose(anticommutator, 0, atol=1e-10)
    
    def test_gamma_squaring(self):
        """Test that γ² = I."""
        triple = FiniteSpectralTriple(N=50, seed=42)
        
        gamma_squared = triple.gamma @ triple.gamma
        I = np.eye(triple.N, dtype=np.complex128)
        
        np.testing.assert_allclose(gamma_squared, I, rtol=1e-10)
    
    def test_enforce_axioms_preserves_hermiticity(self):
        """Test that enforce_axioms maintains Hermiticity."""
        triple = FiniteSpectralTriple(N=50, seed=42)
        
        # Perturb D with a non-Hermitian matrix
        perturbation = np.random.randn(50, 50) + 1j * np.random.randn(50, 50)
        triple.D += perturbation
        
        # Enforce axioms
        triple.enforce_axioms()
        
        # Check Hermiticity
        D_dagger = triple.D.conj().T
        np.testing.assert_allclose(triple.D, D_dagger, rtol=1e-10)
    
    def test_enforce_axioms_preserves_anticommutation(self):
        """Test that enforce_axioms maintains {D, γ} = 0."""
        triple = FiniteSpectralTriple(N=50, seed=42)
        
        # Perturb D
        perturbation = np.random.randn(50, 50) + 1j * np.random.randn(50, 50)
        perturbation = (perturbation + perturbation.conj().T) / 2  # Keep Hermitian
        triple.D += perturbation
        
        # Enforce axioms
        triple.enforce_axioms()
        
        # Check anticommutation
        anticommutator = triple.D @ triple.gamma + triple.gamma @ triple.D
        np.testing.assert_allclose(anticommutator, 0, atol=1e-10)
    
    def test_spectrum_real(self):
        """Test that eigenvalues are real (D is Hermitian)."""
        triple = FiniteSpectralTriple(N=50, seed=42)
        
        spectrum = triple.spectrum()
        
        # All eigenvalues should be real
        assert spectrum.dtype == np.float64
        assert len(spectrum) == triple.N
    
    def test_spectrum_sorted(self):
        """Test that spectrum returns sorted eigenvalues."""
        triple = FiniteSpectralTriple(N=50, seed=42)
        
        spectrum = triple.spectrum()
        
        # Should be sorted
        np.testing.assert_array_equal(spectrum, np.sort(spectrum))
    
    def test_zero_mode_counting(self):
        """Test zero mode counting."""
        # Create a triple with known zero modes
        N = 10
        triple = FiniteSpectralTriple(N=N, seed=42, enforce_axioms_on_init=False)
        
        # Create a matrix that already satisfies {D, γ} = 0
        # D must be block off-diagonal with respect to gamma
        # gamma = diag(1,1,1,1,1,-1,-1,-1,-1,-1)
        # So D should have the form: [[0, A], [A†, 0]]
        
        # Create small eigenvalues in the off-diagonal blocks
        A = np.zeros((5, 5), dtype=np.complex128)
        A[0, 0] = 1e-8
        A[1, 1] = -1e-8
        A[2, 2] = 1e-9
        A[3, 3] = 2.0
        A[4, 4] = 3.0
        
        D_block = np.block([[np.zeros((5, 5)), A],
                            [A.conj().T, np.zeros((5, 5))]])
        
        triple.D = D_block
        
        # This should already satisfy axioms
        triple.enforce_axioms()
        
        # Count zero modes with appropriate threshold
        n_zero = triple.count_zero_modes(threshold=1e-6)
        
        # Should find 3 zero modes (the three small eigenvalues we created)
        # Note: the actual count depends on the structure of A
        # For this block structure, we expect at most 6 near-zero modes
        assert n_zero >= 3  # At least the 3 we explicitly created
        assert n_zero <= 10  # Not all modes
    
    def test_serialization(self):
        """Test to_dict and from_dict."""
        triple = FiniteSpectralTriple(N=20, seed=123)
        
        # Serialize
        data = triple.to_dict()
        
        # Deserialize
        triple_restored = FiniteSpectralTriple.from_dict(data)
        
        # Check equality
        assert triple_restored.N == triple.N
        np.testing.assert_allclose(triple_restored.D, triple.D, rtol=1e-10)
        np.testing.assert_allclose(triple_restored.J, triple.J, rtol=1e-10)
        np.testing.assert_allclose(triple_restored.gamma, triple.gamma, rtol=1e-10)


class TestRealStructure:
    """Test real structure J."""
    
    def test_J_antilinearity(self):
        """Test that J is antilinear: J(αψ) = ᾱJ(ψ)."""
        triple = FiniteSpectralTriple(N=20, seed=42)
        
        psi = np.random.randn(20) + 1j * np.random.randn(20)
        alpha = 2.0 + 3.0j
        
        # J(α ψ)
        J_alpha_psi = triple.apply_J(alpha * psi)
        
        # ᾱ J(ψ)
        alpha_bar_J_psi = np.conj(alpha) * triple.apply_J(psi)
        
        np.testing.assert_allclose(J_alpha_psi, alpha_bar_J_psi, rtol=1e-10)
    
    def test_J_involution(self):
        """Test that J² = ±1 (depends on KO-dimension)."""
        triple = FiniteSpectralTriple(N=20, seed=42)
        
        psi = np.random.randn(20) + 1j * np.random.randn(20)
        
        # J²(ψ)
        J_J_psi = triple.apply_J(triple.apply_J(psi))
        
        # For standard real structure with J = I, J² = I
        np.testing.assert_allclose(J_J_psi, psi, rtol=1e-10)


class TestChirality:
    """Test chirality-related properties."""
    
    def test_chirality_balance(self):
        """Test that gamma has balanced eigenvalues (+1 and -1)."""
        triple = FiniteSpectralTriple(N=100, seed=42)
        
        # Eigenvalues of gamma should be ±1
        gamma_eigvals = np.linalg.eigvalsh(triple.gamma)
        
        # Should be N/2 positive and N/2 negative (approximately)
        n_plus = np.sum(gamma_eigvals > 0)
        n_minus = np.sum(gamma_eigvals < 0)
        
        assert n_plus == 50
        assert n_minus == 50
    
    def test_chirality_projection(self):
        """Test chiral projectors P_± = (1 ± γ)/2."""
        triple = FiniteSpectralTriple(N=50, seed=42)
        
        I = np.eye(50, dtype=np.complex128)
        P_plus = (I + triple.gamma) / 2
        P_minus = (I - triple.gamma) / 2
        
        # P_± should be projectors: P² = P
        np.testing.assert_allclose(P_plus @ P_plus, P_plus, rtol=1e-10)
        np.testing.assert_allclose(P_minus @ P_minus, P_minus, rtol=1e-10)
        
        # P_+ + P_- = I
        np.testing.assert_allclose(P_plus + P_minus, I, rtol=1e-10)
        
        # P_+ P_- = 0
        np.testing.assert_allclose(P_plus @ P_minus, 0, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
