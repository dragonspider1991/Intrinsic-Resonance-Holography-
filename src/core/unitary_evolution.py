"""
Unitary Evolution Operator for IRH v15.0

Implements Axiom 4: Deterministic, unitary evolution of Algorithmic Holonomic States.

The evolution operator U acts on complex state vectors:
    Ψ(τ+1) = U(τ) Ψ(τ)

where U is derived from the Interference Matrix (complex graph Laplacian):
    U(dt) = exp(-i dt H / ℏ₀)
    H = ℏ₀ L

This replaces the classical greedy update of v14.0 with fundamentally unitary,
deterministic evolution on complex states, directly mandated by Axiom 0-1.

References: IRH v15.0 Axiom 4, §1
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, expm
from typing import Tuple, Optional
from numpy.typing import NDArray


class UnitaryEvolutionOperator:
    """
    Implements deterministic unitary evolution of Algorithmic Holonomic States.
    
    Attributes
    ----------
    L : sp.spmatrix
        Interference Matrix (complex graph Laplacian)
    dt : float
        Discrete time step
    hbar_0 : float
        Fundamental action scale (default: 1.0)
    H : sp.spmatrix
        Hamiltonian H = ℏ₀ L
    
    References
    ----------
    IRH v15.0 Axiom 4: Algorithmic Coherent Evolution
    """
    
    def __init__(
        self,
        interference_matrix: sp.spmatrix,
        dt: float = 1.0,
        hbar_0: float = 1.0
    ):
        """
        Initialize unitary evolution operator.
        
        Parameters
        ----------
        interference_matrix : sp.spmatrix
            Complex graph Laplacian L = D - W
        dt : float
            Discrete time step for evolution
        hbar_0 : float
            Fundamental action scale (emerges from ARO)
        """
        self.L = interference_matrix.tocsr()
        self.dt = dt
        self.hbar_0 = hbar_0
        self.N = self.L.shape[0]
        
        # Hamiltonian H = ℏ₀ L
        self.H = self.hbar_0 * self.L
        
        # Cache for evolution operator (if small enough)
        self._U_cached = None
        self._use_cache = (self.N < 500)  # Only cache for small systems
    
    def evolve(
        self,
        state_vector: np.ndarray,
        n_steps: int = 1
    ) -> np.ndarray:
        """
        Apply n time steps of unitary evolution.
        
        Ψ(τ + n*dt) = U^n Ψ(τ)
        
        Parameters
        ----------
        state_vector : np.ndarray
            Complex state vector Ψ(τ)
        n_steps : int
            Number of time steps
        
        Returns
        -------
        evolved_state : np.ndarray
            State Ψ(τ + n*dt) after evolution
            
        Notes
        -----
        For large systems, uses Krylov methods (expm_multiply) which are
        more efficient than computing the full matrix exponential.
        """
        # Verify state is complex
        if not np.iscomplexobj(state_vector):
            state_vector = state_vector.astype(np.complex128)
        
        # Normalize input
        norm = np.linalg.norm(state_vector)
        if norm > 0:
            state_vector = state_vector / norm
        
        # Evolution operator: U = exp(-i dt H / ℏ₀) = exp(-i dt L)
        # Note: H = ℏ₀ L, so H/ℏ₀ = L
        
        evolved = state_vector.copy()
        
        for step in range(n_steps):
            # Use Krylov method for large sparse matrices
            # expm_multiply computes exp(A) @ v efficiently
            evolved = expm_multiply(
                -1j * self.dt * self.L,
                evolved
            )
        
        return evolved
    
    def compute_evolution_operator(self) -> sp.spmatrix:
        """
        Compute the discrete unitary operator U = exp(-i dt L).
        
        Returns
        -------
        U : sp.spmatrix
            Unitary evolution operator
            
        Notes
        -----
        Only use for small systems (N < 500). For larger systems,
        use evolve() which applies the operator via Krylov methods.
        """
        if self._U_cached is not None:
            return self._U_cached
        
        if self.N > 500:
            raise ValueError(
                f"System too large (N={self.N}) for explicit operator. "
                "Use evolve() instead which uses Krylov methods."
            )
        
        # Compute matrix exponential for small systems
        L_dense = self.L.toarray()
        U_dense = expm(-1j * self.dt * L_dense)
        U = sp.csr_matrix(U_dense)
        
        if self._use_cache:
            self._U_cached = U
        
        return U
    
    def verify_unitarity(
        self,
        tolerance: float = 1e-10
    ) -> Tuple[bool, float]:
        """
        Verify that evolution operator is unitary: U†U = I.
        
        Parameters
        ----------
        tolerance : float
            Maximum allowed deviation from identity
        
        Returns
        -------
        is_unitary : bool
            True if ||U†U - I|| < tolerance
        deviation : float
            Actual deviation ||U†U - I||
        """
        if self.N > 500:
            # Sample verification for large systems
            # Test on random vectors
            test_vectors = 10
            max_deviation = 0.0
            
            for _ in range(test_vectors):
                v = np.random.randn(self.N) + 1j * np.random.randn(self.N)
                v = v / np.linalg.norm(v)
                
                # Apply U and U†
                Uv = self.evolve(v, n_steps=1)
                
                # U†Uv should equal v
                # U† = exp(+i dt L)
                U_dag_Uv = expm_multiply(
                    +1j * self.dt * self.L,
                    Uv
                )
                
                deviation = np.linalg.norm(U_dag_Uv - v)
                max_deviation = max(max_deviation, deviation)
            
            return max_deviation < tolerance, max_deviation
        else:
            # Exact verification for small systems
            U = self.compute_evolution_operator()
            U_dag = U.conj().T
            
            # Compute U†U
            U_dag_U = U_dag @ U
            
            # Should be identity
            I = sp.eye(self.N, format='csr')
            
            deviation = sp.linalg.norm(U_dag_U - I)
            
            return deviation < tolerance, deviation
    
    def verify_norm_preservation(
        self,
        state_vector: np.ndarray,
        n_steps: int = 10,
        tolerance: float = 1e-10
    ) -> Tuple[bool, float]:
        """
        Verify that evolution preserves norm: ||Ψ(τ+dt)|| = ||Ψ(τ)||.
        
        Parameters
        ----------
        state_vector : np.ndarray
            Initial state
        n_steps : int
            Number of evolution steps to test
        tolerance : float
            Maximum allowed deviation
        
        Returns
        -------
        preserves_norm : bool
            True if norm is preserved
        max_deviation : float
            Maximum observed deviation
        """
        initial_norm = np.linalg.norm(state_vector)
        max_deviation = 0.0
        
        current_state = state_vector / initial_norm  # Normalize
        
        for step in range(n_steps):
            current_state = self.evolve(current_state, n_steps=1)
            current_norm = np.linalg.norm(current_state)
            
            deviation = abs(current_norm - 1.0)
            max_deviation = max(max_deviation, deviation)
        
        return max_deviation < tolerance, max_deviation
    
    def compute_energy(
        self,
        state_vector: np.ndarray
    ) -> complex:
        """
        Compute energy expectation value ⟨Ψ|H|Ψ⟩.
        
        Parameters
        ----------
        state_vector : np.ndarray
            State vector
        
        Returns
        -------
        energy : complex
            Energy expectation value (should be real)
        """
        # Normalize state
        psi = state_vector / np.linalg.norm(state_vector)
        
        # Compute H|Ψ⟩
        H_psi = self.H @ psi
        
        # Compute ⟨Ψ|H|Ψ⟩
        energy = np.vdot(psi, H_psi)
        
        return energy
    
    def verify_energy_conservation(
        self,
        state_vector: np.ndarray,
        n_steps: int = 100,
        tolerance: float = 1e-8
    ) -> Tuple[bool, float]:
        """
        Verify that energy is conserved during evolution.
        
        Parameters
        ----------
        state_vector : np.ndarray
            Initial state
        n_steps : int
            Number of steps to test
        tolerance : float
            Maximum allowed relative variation
        
        Returns
        -------
        conserves_energy : bool
            True if energy is conserved
        relative_variation : float
            Relative variation in energy
        """
        energies = []
        current_state = state_vector.copy()
        
        for step in range(n_steps):
            energy = self.compute_energy(current_state)
            energies.append(np.real(energy))
            current_state = self.evolve(current_state, n_steps=1)
        
        energies = np.array(energies)
        mean_energy = np.mean(energies)
        
        if abs(mean_energy) < 1e-14:
            # Energy near zero, check absolute variation
            variation = np.std(energies)
            relative_variation = variation
        else:
            relative_variation = np.std(energies) / abs(mean_energy)
        
        return relative_variation < tolerance, relative_variation


def create_unitary_operator_from_network(
    W: sp.spmatrix,
    dt: float = 1.0,
    hbar_0: float = 1.0
) -> UnitaryEvolutionOperator:
    """
    Create unitary evolution operator from network adjacency matrix.
    
    Parameters
    ----------
    W : sp.spmatrix
        Complex adjacency matrix
    dt : float
        Time step
    hbar_0 : float
        Fundamental action scale
    
    Returns
    -------
    operator : UnitaryEvolutionOperator
        Configured evolution operator
    """
    from .harmony import compute_information_transfer_matrix
    
    L = compute_information_transfer_matrix(W)
    
    return UnitaryEvolutionOperator(L, dt=dt, hbar_0=hbar_0)


# Export
__all__ = [
    'UnitaryEvolutionOperator',
    'create_unitary_operator_from_network'
]
