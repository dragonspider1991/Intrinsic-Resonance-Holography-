"""
Axiom 4: Algorithmic Coherent Evolution - Network Dynamics

This module implements the coherent evolution dynamics for the Cymatic
Resonance Network (CRN) as defined in IRHv16.md §1 Axiom 4.

Key Concepts (from IRHv16.md §1 Axiom 4):
    - Deterministic, unitary evolution of AHS and ACW
    - Governed by maximal coherent information transfer
    - Local information preservation, global Harmony optimization
    - Evolution operator U derived from Interference Matrix L

Implementation Status: Phase 3 Implementation
    - Evolution operator: IMPLEMENTED (basic)
    - Time stepping: IMPLEMENTED
    - Unitarity verification: IMPLEMENTED

References:
    IRHv16.md §1 Axiom 4: Algorithmic Coherent Evolution
    IRHv16.md §4: Harmony Functional
    [IRH-PHYS-2025-03]: Quantum Mechanics from Algorithmic Path Integrals
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable
import numpy as np
from numpy.typing import NDArray
from numpy.linalg import eigh, norm
import scipy.sparse as sp
from scipy.linalg import expm

from .ahs import AlgorithmicHolonomicState
from .crn import CymaticResonanceNetwork


# Numerical thresholds for stability
EIGENVALUE_ZERO_THRESHOLD = 1e-10  # Below this, eigenvalue is considered zero
HARMONY_SINGULARITY_THRESHOLD = 1e-100  # Below this, determinant is singular


@dataclass
class EvolutionState:
    """
    State of the CRN at a given time step.
    
    Captures the full configuration needed for evolution.
    
    Attributes:
        W: Complex ACW matrix at time τ
        tau: Current discrete time step
        harmony: Harmony functional value S_H
        energy: Total "energy" (related to L eigenvalues)
    """
    W: NDArray[np.complex128]
    tau: int = 0
    harmony: Optional[float] = None
    energy: Optional[float] = None
    
    @property
    def N(self) -> int:
        """Network size."""
        return self.W.shape[0]


class CoherentEvolution:
    """
    Implements coherent evolution dynamics for the CRN.
    
    Per IRHv16.md §1 Axiom 4:
        "The CRN undergoes deterministic, unitary evolution of its AHS s_i
        and their ACW W_ij in discrete time steps τ. This evolution is
        governed by the principle of maximal coherent information transfer,
        locally preserving information while globally optimizing the
        Harmony Functional."
    
    The evolution operator is derived from the Interference Matrix L:
        U = exp(-i * dt * L)
    
    Attributes:
        crn: The Cymatic Resonance Network being evolved
        dt: Time step size
        history: List of EvolutionState at each step
        
    References:
        IRHv16.md §1 Axiom 4: Evolution definition
        IRHv16.md §4: Harmony Functional
    """
    
    def __init__(
        self,
        crn: CymaticResonanceNetwork,
        dt: float = 0.1,
    ):
        """
        Initialize coherent evolution.
        
        Args:
            crn: The CRN to evolve
            dt: Time step size (default 0.1)
        """
        self.crn = crn
        self.dt = dt
        self.history: List[EvolutionState] = []
        
        # Store initial state
        self._current_W = crn.W.copy() if not sp.issparse(crn.W) else crn.W.toarray().copy()
        self._tau = 0
        
        # Record initial state
        self._record_state()
    
    def _get_interference_matrix(self, W: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Compute interference matrix L = D - W.
        
        Args:
            W: Current ACW matrix
            
        Returns:
            Interference matrix L
        """
        degrees = np.sum(np.abs(W), axis=1)
        D = np.diag(degrees)
        L = D - W
        return L
    
    def _compute_evolution_operator(
        self,
        L: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        """
        Compute unitary evolution operator U = exp(-i * dt * L).
        
        Per IRHv16.md §1 Axiom 4:
            "The global evolution operator U is an N×N matrix whose elements
            are directly derived from the Interference Matrix L of the CRN."
        
        Args:
            L: Interference matrix
            
        Returns:
            Unitary evolution operator U
            
        References:
            IRHv16.md §1 Axiom 4: Evolution operator
        """
        # U = exp(-i * dt * L)
        # Using scipy.linalg.expm for matrix exponential
        U = expm(-1j * self.dt * L)
        return U
    
    def _apply_evolution(
        self,
        W: NDArray[np.complex128],
        U: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        """
        Apply evolution operator to ACW matrix.
        
        Per IRHv16.md §1 Axiom 4:
            s_i(τ+1) = U_i({s_j(τ)}_{j ∈ N(i)}, {W_ij(τ)}_{j ∈ N(i)})
        
        The ACW matrix evolves as:
            W(τ+1) = U @ W @ U†
            
        This is a similarity transformation preserving eigenvalues.
        
        Args:
            W: Current ACW matrix
            U: Evolution operator
            
        Returns:
            Evolved ACW matrix W(τ+1)
        """
        # Conjugate transpose of U
        U_dagger = U.conj().T
        
        # W' = U @ W @ U†
        W_new = U @ W @ U_dagger
        
        return W_new
    
    def _compute_harmony(self, W: NDArray[np.complex128]) -> float:
        """
        Compute Harmony Functional S_H.
        
        Per IRHv16.md §4 (Theorem 4.1):
            S_H[G] = Tr(L²) / (det'(L))^α
            
        Simplified version uses eigenvalue-based computation:
            S_H ≈ Σ λ_i² / Π_{λ_i≠0} |λ_i|^(α/N)
            
        where α = 1/(N ln N) for spectral zeta regularization.
        
        Args:
            W: ACW matrix
            
        Returns:
            Harmony functional value
            
        References:
            IRHv16.md §4 Theorem 4.1: Harmony Functional
        """
        L = self._get_interference_matrix(W)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(L)
        
        # Tr(L²) = Σ λ_i²
        tr_L2 = np.sum(eigenvalues ** 2)
        
        # det'(L) = product of non-zero eigenvalues
        # Use threshold to identify non-zero eigenvalues
        nonzero_eig = eigenvalues[np.abs(eigenvalues) > EIGENVALUE_ZERO_THRESHOLD]
        
        if len(nonzero_eig) == 0:
            return 0.0
        
        # Spectral zeta regularization exponent
        N = len(eigenvalues)
        if N > 1:
            alpha = 1.0 / (N * np.log(N))
        else:
            alpha = 1.0
        
        # det'(L)^α using log for numerical stability
        log_det_prime = np.sum(np.log(np.abs(nonzero_eig)))
        det_prime_alpha = np.exp(alpha * log_det_prime)
        
        # If determinant is essentially zero, return infinity
        # This indicates a singular configuration
        if det_prime_alpha < HARMONY_SINGULARITY_THRESHOLD:
            return float('inf')
        
        harmony = tr_L2 / det_prime_alpha
        return float(harmony)
    
    def _compute_energy(self, W: NDArray[np.complex128]) -> float:
        """
        Compute total network "energy" from eigenvalues.
        
        E = Tr(L) = Σ λ_i
        
        Args:
            W: ACW matrix
            
        Returns:
            Network energy
        """
        L = self._get_interference_matrix(W)
        return float(np.trace(L).real)
    
    def _record_state(self):
        """Record current state to history."""
        state = EvolutionState(
            W=self._current_W.copy(),
            tau=self._tau,
            harmony=self._compute_harmony(self._current_W),
            energy=self._compute_energy(self._current_W),
        )
        self.history.append(state)
    
    def step(self) -> EvolutionState:
        """
        Perform one evolution step τ → τ+1.
        
        Per IRHv16.md §1 Axiom 4:
            "The evolution from τ → τ+1 is an iterative application of a
            unitary operator U that maximizes the change in local algorithmic
            mutual information."
        
        Returns:
            New EvolutionState after evolution
        """
        # Get interference matrix
        L = self._get_interference_matrix(self._current_W)
        
        # Compute evolution operator
        U = self._compute_evolution_operator(L)
        
        # Apply evolution
        self._current_W = self._apply_evolution(self._current_W, U)
        self._tau += 1
        
        # Record state
        self._record_state()
        
        return self.history[-1]
    
    def evolve(self, n_steps: int) -> List[EvolutionState]:
        """
        Evolve the network for n_steps.
        
        Args:
            n_steps: Number of evolution steps
            
        Returns:
            List of EvolutionState objects for all steps
        """
        for _ in range(n_steps):
            self.step()
        return self.history
    
    def check_unitarity(self) -> Tuple[bool, float]:
        """
        Verify that evolution preserves unitarity.
        
        Per IRHv16.md §1 Axiom 4:
            "This is a fundamentally unitary, deterministic evolution."
        
        Checks that U @ U† = I
        
        Returns:
            (is_unitary, deviation) where deviation is ||U@U† - I||
        """
        L = self._get_interference_matrix(self._current_W)
        U = self._compute_evolution_operator(L)
        
        # Check U @ U† = I
        product = U @ U.conj().T
        identity = np.eye(U.shape[0], dtype=np.complex128)
        deviation = norm(product - identity)
        
        is_unitary = deviation < 1e-10
        return is_unitary, float(deviation)
    
    def check_information_conservation(self) -> Tuple[bool, float]:
        """
        Verify information conservation across evolution.
        
        Total information should be conserved: Tr(W†W) = const
        
        Returns:
            (is_conserved, relative_change)
        """
        if len(self.history) < 2:
            return True, 0.0
        
        initial_info = np.trace(self.history[0].W.conj().T @ self.history[0].W).real
        current_info = np.trace(self._current_W.conj().T @ self._current_W).real
        
        if initial_info < 1e-10:
            return True, 0.0
        
        relative_change = abs(current_info - initial_info) / initial_info
        is_conserved = relative_change < 1e-6
        
        return is_conserved, float(relative_change)
    
    def get_harmony_history(self) -> NDArray[np.float64]:
        """
        Get history of Harmony functional values.
        
        Returns:
            Array of S_H values at each time step
        """
        return np.array([s.harmony for s in self.history if s.harmony is not None])
    
    def get_energy_history(self) -> NDArray[np.float64]:
        """
        Get history of energy values.
        
        Returns:
            Array of energy values at each time step
        """
        return np.array([s.energy for s in self.history if s.energy is not None])


class AdaptiveResonanceOptimization:
    """
    Adaptive Resonance Optimization (ARO) - the global optimization process.
    
    Per IRHv16.md, ARO is the process by which the CRN evolves toward
    the Cosmic Fixed Point, maximizing the Harmony Functional.
    
    This is a simplified implementation. Full v16.0 requires:
    - Exascale distributed computation
    - Certified numerical precision
    - Multi-fidelity NCD evaluation
    
    Attributes:
        evolution: CoherentEvolution instance
        best_harmony: Best harmony value found
        best_state: State with best harmony
        
    References:
        IRHv16.md §4: Adaptive Resonance Optimization
        IRHv16.md Theorem 10.1: Cosmic Fixed Point
    """
    
    def __init__(
        self,
        crn: CymaticResonanceNetwork,
        dt: float = 0.1,
    ):
        """
        Initialize ARO.
        
        Args:
            crn: CRN to optimize
            dt: Evolution time step
        """
        self.evolution = CoherentEvolution(crn, dt=dt)
        self.best_harmony: Optional[float] = None
        self.best_state: Optional[EvolutionState] = None
        
        self._update_best()
    
    def _update_best(self):
        """Update best state if current is better."""
        current = self.evolution.history[-1]
        if current.harmony is not None:
            if self.best_harmony is None or current.harmony > self.best_harmony:
                self.best_harmony = current.harmony
                self.best_state = EvolutionState(
                    W=current.W.copy(),
                    tau=current.tau,
                    harmony=current.harmony,
                    energy=current.energy,
                )
    
    def optimize(
        self,
        max_steps: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = False,
    ) -> EvolutionState:
        """
        Run ARO optimization.
        
        Per IRHv16.md §4:
            "ARO drives the CRN toward the unique Cosmic Fixed Point
            configuration that maximizes the Harmony Functional."
        
        Args:
            max_steps: Maximum evolution steps
            tolerance: Convergence tolerance on harmony change
            verbose: Print progress
            
        Returns:
            Best EvolutionState found
        """
        prev_harmony = self.evolution.history[-1].harmony
        
        for step in range(max_steps):
            self.evolution.step()
            current = self.evolution.history[-1]
            self._update_best()
            
            if verbose and step % 10 == 0:
                print(f"Step {step}: S_H = {current.harmony:.6f}, E = {current.energy:.6f}")
            
            # Check convergence
            if prev_harmony is not None and current.harmony is not None:
                delta = abs(current.harmony - prev_harmony)
                if delta < tolerance:
                    if verbose:
                        print(f"Converged at step {step} with delta={delta:.2e}")
                    break
            
            prev_harmony = current.harmony
        
        return self.best_state if self.best_state else current
    
    def get_convergence_metrics(self) -> dict:
        """
        Get metrics about the optimization convergence.
        
        Returns:
            Dict with convergence statistics
        """
        harmony = self.evolution.get_harmony_history()
        energy = self.evolution.get_energy_history()
        
        return {
            "n_steps": len(self.evolution.history),
            "final_harmony": float(harmony[-1]) if len(harmony) > 0 else None,
            "best_harmony": self.best_harmony,
            "harmony_std": float(np.std(harmony)) if len(harmony) > 1 else 0.0,
            "energy_change": float(energy[-1] - energy[0]) if len(energy) > 1 else 0.0,
            "is_converged": len(harmony) > 1 and abs(harmony[-1] - harmony[-2]) < 1e-6,
        }


__version__ = "16.0.0-dev"
__status__ = "Phase 3 Implementation - Evolution dynamics and ARO"

__all__ = [
    "EvolutionState",
    "CoherentEvolution", 
    "AdaptiveResonanceOptimization",
]
