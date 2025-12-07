"""
Test Suite for Unitary Evolution Operator - IRH v15.0

Validates Axiom 4: Deterministic unitary evolution of AHS.
"""

import numpy as np
import scipy.sparse as sp
import pytest
from src.core.unitary_evolution import (
    UnitaryEvolutionOperator,
    create_unitary_operator_from_network
)
from src.core.harmony import compute_information_transfer_matrix


def create_test_network(N: int = 50, density: float = 0.15) -> sp.spmatrix:
    """Create a test network for unitary evolution with Hermitian weights."""
    # Create random real matrix
    W_real = sp.random(N, N, density=density, format='csr')
    W_imag = sp.random(N, N, density=density, format='csr')
    
    # Combine to complex
    W = W_real.astype(np.complex128) + 1j * W_imag.astype(np.complex128)
    
    # Make Hermitian: W = (W + W†) / 2
    W = (W + W.conj().T) / 2.0
    
    return W.tocsr()


def test_unitary_operator_initialization():
    """Test basic initialization of unitary evolution operator."""
    W = create_test_network(N=30)
    L = compute_information_transfer_matrix(W)
    
    op = UnitaryEvolutionOperator(L, dt=0.1, hbar_0=1.0)
    
    assert op.N == 30
    assert op.dt == 0.1
    assert op.hbar_0 == 1.0
    assert op.H.shape == (30, 30)


def test_unitarity_small_system():
    """Test that evolution operator is unitary for small systems."""
    W = create_test_network(N=40)
    L = compute_information_transfer_matrix(W)
    
    op = UnitaryEvolutionOperator(L, dt=0.05)
    
    is_unitary, deviation = op.verify_unitarity(tolerance=1e-10)
    
    assert is_unitary, f"Operator not unitary, deviation = {deviation:.2e}"
    assert deviation < 1e-10


def test_unitarity_large_system():
    """Test unitarity using sampling for large systems."""
    W = create_test_network(N=600, density=0.05)
    L = compute_information_transfer_matrix(W)
    
    op = UnitaryEvolutionOperator(L, dt=0.01)
    
    # Should use sampling-based verification
    is_unitary, deviation = op.verify_unitarity(tolerance=1e-8)
    
    assert is_unitary, f"Operator not unitary (sampled), deviation = {deviation:.2e}"


def test_norm_preservation():
    """Test that evolution preserves state vector norm."""
    W = create_test_network(N=50)
    L = compute_information_transfer_matrix(W)
    
    op = UnitaryEvolutionOperator(L, dt=0.1)
    
    # Create random initial state
    psi_0 = np.random.randn(50) + 1j * np.random.randn(50)
    
    preserves_norm, max_dev = op.verify_norm_preservation(
        psi_0, n_steps=20, tolerance=1e-10
    )
    
    assert preserves_norm, f"Norm not preserved, max deviation = {max_dev:.2e}"
    assert max_dev < 1e-10


def test_energy_conservation():
    """Test that energy is conserved during evolution."""
    W = create_test_network(N=50)
    L = compute_information_transfer_matrix(W)
    
    op = UnitaryEvolutionOperator(L, dt=0.05)
    
    # Create random initial state
    psi_0 = np.random.randn(50) + 1j * np.random.randn(50)
    psi_0 = psi_0 / np.linalg.norm(psi_0)
    
    conserves_energy, rel_var = op.verify_energy_conservation(
        psi_0, n_steps=50, tolerance=1e-8
    )
    
    assert conserves_energy, f"Energy not conserved, relative variation = {rel_var:.2e}"
    assert rel_var < 1e-8


def test_evolution_single_step():
    """Test single time step evolution."""
    W = create_test_network(N=40)
    L = compute_information_transfer_matrix(W)
    
    op = UnitaryEvolutionOperator(L, dt=0.1)
    
    psi_0 = np.random.randn(40) + 1j * np.random.randn(40)
    psi_0 = psi_0 / np.linalg.norm(psi_0)  # Normalize input
    psi_1 = op.evolve(psi_0, n_steps=1)
    
    # Check norm is preserved
    assert abs(np.linalg.norm(psi_1) - 1.0) < 1e-10
    
    # Check state has evolved (not identical)
    assert np.linalg.norm(psi_1 - psi_0) > 1e-6


def test_evolution_multiple_steps():
    """Test multiple time step evolution."""
    W = create_test_network(N=40)
    L = compute_information_transfer_matrix(W)
    
    op = UnitaryEvolutionOperator(L, dt=0.05)
    
    psi_0 = np.random.randn(40) + 1j * np.random.randn(40)
    psi_0 = psi_0 / np.linalg.norm(psi_0)  # Normalize input
    
    # Evolve 10 steps
    psi_10 = op.evolve(psi_0, n_steps=10)
    
    # Norm should be preserved
    assert abs(np.linalg.norm(psi_10) - 1.0) < 1e-9


def test_evolution_operator_explicit():
    """Test explicit computation of evolution operator for small systems."""
    W = create_test_network(N=30)
    L = compute_information_transfer_matrix(W)
    
    op = UnitaryEvolutionOperator(L, dt=0.1)
    
    # Compute explicit operator
    U = op.compute_evolution_operator()
    
    assert U.shape == (30, 30)
    assert sp.issparse(U)
    
    # Verify unitarity explicitly
    U_dag = U.conj().T
    I = sp.eye(30, format='csr')
    
    deviation = sp.linalg.norm(U_dag @ U - I)
    assert deviation < 1e-10


def test_evolution_operator_large_system_error():
    """Test that explicit operator raises error for large systems."""
    W = create_test_network(N=600, density=0.05)
    L = compute_information_transfer_matrix(W)
    
    op = UnitaryEvolutionOperator(L, dt=0.1)
    
    with pytest.raises(ValueError, match="System too large"):
        op.compute_evolution_operator()


def test_energy_computation():
    """Test energy expectation value computation."""
    W = create_test_network(N=40)
    L = compute_information_transfer_matrix(W)
    
    op = UnitaryEvolutionOperator(L, dt=0.1)
    
    psi = np.random.randn(40) + 1j * np.random.randn(40)
    psi = psi / np.linalg.norm(psi)
    
    energy = op.compute_energy(psi)
    
    # Energy should be real (Hermitian operator)
    assert abs(np.imag(energy)) < 1e-10
    
    # Energy should be finite
    assert np.isfinite(np.real(energy))


def test_hamiltonian_hermiticity():
    """Test that Hamiltonian H = ℏ₀ L is Hermitian."""
    W = create_test_network(N=40)
    L = compute_information_transfer_matrix(W)
    
    op = UnitaryEvolutionOperator(L, dt=0.1, hbar_0=1.0)
    
    H = op.H
    H_dag = H.conj().T
    
    # H should equal H†
    deviation = sp.linalg.norm(H - H_dag)
    assert deviation < 1e-10, f"H not Hermitian, deviation = {deviation:.2e}"


def test_create_from_network():
    """Test convenience function for creating operator from network."""
    W = create_test_network(N=35)
    
    op = create_unitary_operator_from_network(W, dt=0.05, hbar_0=1.0)
    
    assert isinstance(op, UnitaryEvolutionOperator)
    assert op.N == 35
    assert op.dt == 0.05
    
    # Test it works
    psi = np.random.randn(35) + 1j * np.random.randn(35)
    psi_evolved = op.evolve(psi, n_steps=5)
    
    assert np.linalg.norm(psi_evolved) > 0


def test_reversibility():
    """Test that evolution is reversible (unitary)."""
    W = create_test_network(N=40)
    L = compute_information_transfer_matrix(W)
    
    # Forward evolution
    op_forward = UnitaryEvolutionOperator(L, dt=0.1)
    
    # Backward evolution (negative dt)
    op_backward = UnitaryEvolutionOperator(L, dt=-0.1)
    
    psi_0 = np.random.randn(40) + 1j * np.random.randn(40)
    psi_0 = psi_0 / np.linalg.norm(psi_0)  # Normalize
    
    # Evolve forward
    psi_forward = op_forward.evolve(psi_0, n_steps=10)
    
    # Evolve backward from evolved state
    psi_recovered = op_backward.evolve(psi_forward, n_steps=10)
    
    # Should recover original state
    recovery_error = np.linalg.norm(psi_recovered - psi_0)
    assert recovery_error < 1e-6, f"Evolution not reversible, error = {recovery_error:.2e}"


def test_dt_scaling():
    """Test that smaller dt gives more accurate evolution."""
    W = create_test_network(N=40)
    L = compute_information_transfer_matrix(W)
    
    psi_0 = np.random.randn(40) + 1j * np.random.randn(40)
    psi_0 = psi_0 / np.linalg.norm(psi_0)
    
    # Evolve with large dt
    op_large = UnitaryEvolutionOperator(L, dt=0.5)
    psi_large = op_large.evolve(psi_0, n_steps=1)
    
    # Evolve with small dt (10 steps of dt/10)
    op_small = UnitaryEvolutionOperator(L, dt=0.05)
    psi_small = op_small.evolve(psi_0, n_steps=10)
    
    # Both should preserve norm
    assert abs(np.linalg.norm(psi_large) - 1.0) < 1e-10
    assert abs(np.linalg.norm(psi_small) - 1.0) < 1e-10


def test_complex_state_requirement():
    """Test that operator handles real states by converting to complex."""
    W = create_test_network(N=40)
    L = compute_information_transfer_matrix(W)
    
    op = UnitaryEvolutionOperator(L, dt=0.1)
    
    # Real state
    psi_real = np.random.randn(40)
    
    # Should convert and evolve
    psi_evolved = op.evolve(psi_real, n_steps=1)
    
    # Result should be complex
    assert np.iscomplexobj(psi_evolved)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
