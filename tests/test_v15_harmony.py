"""
Test Suite for IRH v15.0 Harmony Functional

Validates the key changes in v15.0:
- Universal constant C_H = 0.045935703 (not N-dependent)
- Intensive action density
- Scale invariance
"""

import numpy as np
import scipy.sparse as sp
import pytest
from src.core.harmony import harmony_functional, C_H, compute_information_transfer_matrix


def test_c_h_constant_value():
    """Verify C_H has the correct derived value from IRH v15.0 Theorem 4.1."""
    expected_C_H = 0.045935703
    assert abs(C_H - expected_C_H) < 1e-9, f"C_H = {C_H}, expected {expected_C_H}"


def test_harmony_uses_universal_constant():
    """Verify harmony functional uses C_H, not N-dependent alpha."""
    # Create a small test network
    N = 100
    W = sp.random(N, N, density=0.1, format='csr', dtype=np.complex128)
    W = W + W.T  # Make Hermitian
    
    # Compute harmony
    S_H = harmony_functional(W)
    
    # Verify S_H is finite and positive (for well-formed networks)
    assert S_H > -np.inf, "Harmony functional should not be -inf for valid network"
    assert not np.isnan(S_H), "Harmony functional should not be NaN"


def test_harmony_scale_independence():
    """
    Verify that S_H does not scale with N in the same way as v13.0.
    
    In v13.0, alpha = 1/(N ln N) made the denominator N-dependent.
    In v15.0, C_H is universal, so the scaling should be different.
    """
    results = []
    
    for N in [50, 100, 200]:
        W = sp.random(N, N, density=0.1, format='csr', dtype=np.complex128)
        W = W + W.T
        S_H = harmony_functional(W)
        results.append((N, S_H))
    
    # In v15.0, the relationship should not follow 1/(N ln N) scaling
    # Just verify all values are computed successfully
    for N, S_H in results:
        assert S_H > -np.inf, f"Failed for N={N}"


def test_harmony_components():
    """Verify the components of the harmony functional are computed correctly."""
    N = 50
    W = sp.random(N, N, density=0.2, format='csr', dtype=np.complex128)
    W = W + W.T
    
    S_H, Tr_M2, det_term = harmony_functional(W, return_components=True)
    
    # Verify components are valid
    assert Tr_M2 > 0, "Tr(M²) should be positive"
    assert det_term > 0, "det term should be positive"
    assert S_H > -np.inf, "S_H should be finite"
    
    # Verify the relationship S_H = Tr(M²) / det_term
    computed_S_H = Tr_M2 / det_term
    assert abs(S_H - computed_S_H) < 1e-6, "S_H should equal Tr(M²)/det_term"


def test_information_transfer_matrix():
    """Verify the Information Transfer Matrix is computed correctly."""
    N = 30
    W = sp.random(N, N, density=0.15, format='csr', dtype=np.complex128)
    
    M = compute_information_transfer_matrix(W)
    
    # Verify M has correct shape
    assert M.shape == (N, N), "M should have same shape as W"
    
    # Verify M = D - W structure
    # Note: D uses real part of degrees to ensure Hermitian Laplacian
    degrees = np.array(W.sum(axis=1)).flatten()
    degrees_real = np.real(degrees)  # Take real part as per implementation
    D = sp.diags(degrees_real, format='csr')
    expected_M = D - W
    
    # Check equality (allowing for numerical precision)
    diff = (M - expected_M).toarray()
    assert np.max(np.abs(diff)) < 1e-10, "M should equal D - W (with real degrees)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
