"""
cGFT Action for IRH v18.0
=========================

Implements the complete cGFT action S[φ,φ̄] = S_kin + S_int + S_hol
as defined in IRHv18.md Section 1.1.1.

THEORETICAL COMPLIANCE:
    This implementation strictly follows docs/manuscripts/IRHv18.md
    - Eq. 1.1: Kinetic term with group Laplacian
    - Eq. 1.2-1.3: Interaction term with NCD-weighted kernel
    - Eq. 1.4: Holographic measure term

Key Components:
    - S_kin: Complex group Laplacian (discrete analogue of Tr(L²))
    - S_int: Phase-coherent, NCD-weighted 4-vertex
    - S_hol: Combinatorial boundary regulator (Axiom 3)

References:
    docs/manuscripts/IRHv18.md:
        - §1.1.1: The cGFT Action (Equations 1.1-1.4)
        - Appendix G: Weyl ordering for Laplacian
        - Appendix A: NCD metric in kernel
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import numpy as np
from numpy.typing import NDArray

from .group_manifold import (
    GInfElement, SU2Element, U1Element,
    compute_ncd_distance, haar_integrate_ginf
)
from .cgft_field import cGFTFieldDiscrete, BiLocalField


# =============================================================================
# cGFT Coupling Constants
# =============================================================================

@dataclass
class cGFTCouplings:
    """
    Running coupling constants for the cGFT.
    
    These couplings flow under RG to the Cosmic Fixed Point values:
    λ̃* = 48π²/9, γ̃* = 32π²/3, μ̃* = 16π²
    
    Attributes:
        lambda_: Interaction coupling (4-vertex strength)
        gamma: NCD kernel coupling (algorithmic locality)
        mu: Holographic measure coupling (boundary regulator)
        
    References:
        IRHv18.md Eq. 1.14: Fixed point values
        IRHv18.md §1.2: RG flow
    """
    
    lambda_: float = 1.0  # Interaction coupling
    gamma: float = 1.0    # NCD kernel coupling
    mu: float = 1.0       # Holographic measure coupling
    
    @classmethod
    def fixed_point(cls) -> 'cGFTCouplings':
        """
        Return the Cosmic Fixed Point coupling values.
        
        These are the unique infrared-attractive fixed point values
        derived analytically in IRHv18.md Eq. 1.14.
        """
        pi_sq = np.pi**2
        return cls(
            lambda_=48 * pi_sq / 9,   # λ̃* = 48π²/9 ≈ 52.64
            gamma=32 * pi_sq / 3,      # γ̃* = 32π²/3 ≈ 105.28
            mu=16 * pi_sq              # μ̃* = 16π² ≈ 157.91
        )
    
    def dimensionless(self, k: float = 1.0) -> 'cGFTCouplings':
        """
        Return dimensionless couplings at scale k.
        
        The canonical dimensions are:
        d_λ = -2, d_γ = 0, d_μ = 2
        """
        return cGFTCouplings(
            lambda_=self.lambda_ * k**2,   # d_λ = -2
            gamma=self.gamma,               # d_γ = 0
            mu=self.mu * k**(-2)           # d_μ = 2
        )


# =============================================================================
# Interaction Kernel
# =============================================================================

@dataclass
class InteractionKernel:
    """
    Complex interaction kernel K(g₁, g₂, g₃, g₄) for cGFT.
    
    K(g₁,g₂,g₃,g₄) = e^{i(φ₁+φ₂+φ₃-φ₄)} × exp[-γ Σ d_NCD(gᵢgⱼ⁻¹)]
    
    Components:
    - Phase factor: Ensures phase coherence (U(1) conservation)
    - NCD exponential: Enforces algorithmic locality
    
    Attributes:
        gamma: NCD kernel coupling
        
    References:
        IRHv18.md Eq. 1.3: Kernel definition
        IRHv18.md Appendix A: NCD distance
    """
    
    gamma: float = 1.0
    
    def __call__(
        self,
        g1: GInfElement,
        g2: GInfElement,
        g3: GInfElement,
        g4: GInfElement
    ) -> complex:
        """
        Evaluate kernel at (g₁, g₂, g₃, g₄).
        
        Returns:
            Complex kernel value
        """
        # Phase factor: e^{i(φ₁+φ₂+φ₃-φ₄)}
        phase_sum = g1.phase + g2.phase + g3.phase - g4.phase
        phase_factor = np.exp(1j * phase_sum)
        
        # NCD exponential: exp[-γ Σ_{i<j} d_NCD(gᵢgⱼ⁻¹)]
        ncd_sum = 0.0
        pairs = [(g1, g2), (g1, g3), (g1, g4), (g2, g3), (g2, g4), (g3, g4)]
        
        for gi, gj in pairs:
            ncd_sum += compute_ncd_distance(gi, gj)
        
        ncd_factor = np.exp(-self.gamma * ncd_sum)
        
        return phase_factor * ncd_factor
    
    def evaluate_discrete(
        self,
        samples: List[GInfElement],
        cache: bool = True
    ) -> NDArray[np.complex128]:
        """
        Evaluate kernel on discrete grid of samples.
        
        Returns K[i,j,k,l] = K(g_i, g_j, g_k, g_l)
        
        Args:
            samples: List of group elements
            cache: Whether to cache NCD computations
            
        Returns:
            4D complex array of kernel values
        """
        N = len(samples)
        K = np.zeros((N, N, N, N), dtype=np.complex128)
        
        # Precompute NCD distances if caching
        if cache:
            ncd_cache = {}
            for i in range(N):
                for j in range(N):
                    if i != j:
                        key = (i, j) if i < j else (j, i)
                        if key not in ncd_cache:
                            ncd_cache[key] = compute_ncd_distance(
                                samples[i], samples[j]
                            )
        
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        # Phase factor
                        phase = (samples[i].phase + samples[j].phase + 
                                samples[k].phase - samples[l].phase)
                        phase_factor = np.exp(1j * phase)
                        
                        # NCD factor
                        if cache:
                            pairs = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
                            ncd_sum = sum(
                                ncd_cache.get(
                                    (min(a,b), max(a,b)), 
                                    0.0 if a == b else compute_ncd_distance(samples[a], samples[b])
                                )
                                for a, b in pairs
                            )
                        else:
                            ncd_sum = sum(
                                compute_ncd_distance(samples[a], samples[b])
                                for a, b in [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
                                if a != b
                            )
                        
                        ncd_factor = np.exp(-self.gamma * ncd_sum)
                        K[i, j, k, l] = phase_factor * ncd_factor
        
        return K


# =============================================================================
# Holographic Measure
# =============================================================================

def smooth_step(x: float, width: float = 0.1) -> float:
    """
    Smooth step function Θ for holographic constraint.
    
    Θ(x) ≈ 0 for x << 0, Θ(x) ≈ 1 for x >> 0
    with smooth transition of specified width.
    
    Args:
        x: Input value
        width: Transition width
        
    Returns:
        Smooth step value in [0, 1]
    """
    return 0.5 * (1.0 + np.tanh(x / width))


def holographic_constraint(
    g1: GInfElement,
    g2: GInfElement,
    g3: GInfElement,
    g4: GInfElement
) -> float:
    """
    Evaluate holographic constraint Π_i Θ(Tr(gᵢg_{i+1}⁻¹)).
    
    This enforces the Combinatorial Holographic Principle (Axiom 3)
    at the level of individual 4-simplices.
    
    Args:
        g1, g2, g3, g4: Vertex group elements
        
    Returns:
        Constraint value in [0, 1]
    """
    # Get SU(2) components
    elements = [g1, g2, g3, g4]
    
    constraint = 1.0
    for i in range(4):
        gi = elements[i]
        gi_next = elements[(i + 1) % 4]
        
        # Compute Tr(gᵢ g_{i+1}⁻¹) for SU(2) part
        product = gi.su2 * gi_next.su2.inverse()
        trace_val = product.trace()  # Tr(U) = 2*Re(α) for SU(2)
        
        # Apply smooth step (threshold at trace = 0)
        constraint *= smooth_step(trace_val)
    
    return constraint


# =============================================================================
# cGFT Action Components
# =============================================================================

class cGFTAction:
    """
    Complete cGFT action S[φ,φ̄] = S_kin + S_int + S_hol.
    
    This is the fundamental action defining the quantum field theory
    on G_inf^4 from which all physics emerges.
    
    At the Cosmic Fixed Point, the effective action reduces to the
    Harmony Functional for the bilocal field Σ (Theorem 1.1).
    
    References:
        IRHv18.md §1.1.1: Complete action definition
        IRHv18.md Theorem 1.1: Emergence of Harmony Functional
    """
    
    def __init__(self, couplings: Optional[cGFTCouplings] = None):
        """
        Initialize cGFT action.
        
        Args:
            couplings: Coupling constants (default: unit couplings)
        """
        self.couplings = couplings or cGFTCouplings()
        self.kernel = InteractionKernel(gamma=self.couplings.gamma)
    
    def compute_kinetic_term(
        self,
        phi: cGFTFieldDiscrete,
        method: str = "discrete_laplacian"
    ) -> complex:
        """
        Compute kinetic term S_kin.
        
        S_kin = ∫ φ̄(g₁,...,g₄) [Σ_a Σ_i Δ_a^(i)] φ(g₁,...,g₄) dg
        
        For discrete fields, uses finite difference approximation
        of the Laplace-Beltrami operator.
        
        Args:
            phi: cGFT field
            method: Computation method
            
        Returns:
            Complex kinetic term value
        """
        N = phi.N
        phi_array = phi.field_array
        phi_bar = np.conj(phi_array)
        
        # Discrete Laplacian approximation
        # For group manifold, approximate as second derivative
        # Δφ ≈ Σ_neighbors [φ(neighbor) - φ(current)] / h²
        
        laplacian_phi = np.zeros_like(phi_array)
        
        for idx in np.ndindex(phi_array.shape):
            i, j, k, l = idx
            
            # Sum over all 4 arguments (each has 3 SU(2) generators + 1 U(1))
            lap_val = 0.0
            
            # Finite difference in each direction
            for dim in range(4):
                indices_plus = list(idx)
                indices_minus = list(idx)
                
                indices_plus[dim] = (idx[dim] + 1) % N
                indices_minus[dim] = (idx[dim] - 1) % N
                
                # Second derivative: (f(x+h) + f(x-h) - 2f(x)) / h²
                lap_val += (
                    phi_array[tuple(indices_plus)] +
                    phi_array[tuple(indices_minus)] -
                    2 * phi_array[idx]
                )
            
            laplacian_phi[idx] = lap_val
        
        # S_kin = ∫ φ̄ Δ φ ≈ Σ φ̄ * Δφ / N^4
        S_kin = np.sum(phi_bar * laplacian_phi) / N**4
        
        return S_kin
    
    def compute_interaction_term(
        self,
        phi: cGFTFieldDiscrete,
        num_samples: int = 100
    ) -> complex:
        """
        Compute interaction term S_int.
        
        S_int = λ ∫ K(g₁h₁⁻¹,...) φ̄(g₁,...) φ(h₁,...) dg dh
        
        Uses Monte Carlo integration over the double group integral.
        
        Args:
            phi: cGFT field
            num_samples: MC samples for integration
            
        Returns:
            Complex interaction term value
        """
        N = phi.N
        samples = phi.group_samples
        phi_array = phi.field_array
        
        rng = np.random.default_rng()
        
        total = 0.0 + 0.0j
        
        for _ in range(num_samples):
            # Random indices for g and h
            gi = [rng.integers(0, N) for _ in range(4)]
            hi = [rng.integers(0, N) for _ in range(4)]
            
            # Get group elements
            g = [samples[gi[j]] for j in range(4)]
            h = [samples[hi[j]] for j in range(4)]
            
            # Compute kernel arguments: gⱼ hⱼ⁻¹
            kernel_args = [g[j] * h[j].inverse() for j in range(4)]
            
            # Evaluate kernel
            K_val = self.kernel(kernel_args[0], kernel_args[1], 
                               kernel_args[2], kernel_args[3])
            
            # Field values
            phi_bar_g = np.conj(phi_array[gi[0], gi[1], gi[2], gi[3]])
            phi_h = phi_array[hi[0], hi[1], hi[2], hi[3]]
            
            total += K_val * phi_bar_g * phi_h
        
        # Normalize by number of samples
        S_int = self.couplings.lambda_ * total / num_samples
        
        return S_int
    
    def compute_holographic_term(
        self,
        phi: cGFTFieldDiscrete
    ) -> complex:
        """
        Compute holographic measure term S_hol.
        
        S_hol = μ ∫ |φ|² Π_i Θ(Tr(gᵢg_{i+1}⁻¹)) dg
        
        Args:
            phi: cGFT field
            
        Returns:
            Complex holographic term value
        """
        N = phi.N
        samples = phi.group_samples
        phi_array = phi.field_array
        
        total = 0.0
        
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        # |φ|²
                        phi_sq = np.abs(phi_array[i, j, k, l])**2
                        
                        # Holographic constraint
                        constraint = holographic_constraint(
                            samples[i], samples[j], 
                            samples[k], samples[l]
                        )
                        
                        total += phi_sq * constraint
        
        # Normalize
        S_hol = self.couplings.mu * total / N**4
        
        return S_hol
    
    def compute_total_action(
        self,
        phi: cGFTFieldDiscrete,
        verbose: bool = False
    ) -> Dict[str, complex]:
        """
        Compute complete cGFT action S = S_kin + S_int + S_hol.
        
        Args:
            phi: cGFT field
            verbose: Whether to print intermediate results
            
        Returns:
            Dictionary with total and individual terms
        """
        S_kin = self.compute_kinetic_term(phi)
        S_int = self.compute_interaction_term(phi)
        S_hol = self.compute_holographic_term(phi)
        
        S_total = S_kin + S_int + S_hol
        
        if verbose:
            print(f"S_kin = {S_kin:.6f}")
            print(f"S_int = {S_int:.6f}")
            print(f"S_hol = {S_hol:.6f}")
            print(f"S_total = {S_total:.6f}")
        
        return {
            "S_kin": S_kin,
            "S_int": S_int,
            "S_hol": S_hol,
            "S_total": S_total
        }


# =============================================================================
# Effective Action (Harmony Functional)
# =============================================================================

def compute_effective_laplacian(
    sigma: BiLocalField,
    samples: List[GInfElement]
) -> NDArray[np.complex128]:
    """
    Compute emergent graph Laplacian L[Σ] from bilocal field.
    
    This is the Laplacian appearing in the Harmony Functional
    (Theorem 1.1): Γ[Σ] = Tr(L²) - C_H log det'(L)
    
    Args:
        sigma: Bilocal field
        samples: Group element discretization
        
    Returns:
        Complex Laplacian matrix
    """
    N = len(samples)
    
    # Adjacency from bilocal field
    W = sigma.to_matrix(samples)
    
    # Laplacian: L_ii = Σ_j W_ij, L_ij = -W_ij
    L = np.zeros((N, N), dtype=np.complex128)
    
    for i in range(N):
        for j in range(N):
            if i == j:
                L[i, i] = np.sum(W[i, :])
            else:
                L[i, j] = -W[i, j]
    
    return L


def compute_harmony_functional(
    sigma: BiLocalField,
    samples: List[GInfElement],
    C_H: float = 0.045935703598
) -> Dict[str, complex]:
    """
    Compute Harmony Functional Γ[Σ] = Tr(L²) - C_H log det'(L).
    
    This is the effective action for the bilocal field at the
    Cosmic Fixed Point (Theorem 1.1, Eq. 1.5).
    
    Args:
        sigma: Bilocal field
        samples: Group element discretization
        C_H: Universal exponent (from Eq. 1.16)
        
    Returns:
        Dictionary with Harmony Functional and components
    """
    L = compute_effective_laplacian(sigma, samples)
    
    # Tr(L²)
    L2 = L @ L
    trace_L2 = np.trace(L2)
    
    # det'(L): product of non-zero eigenvalues
    eigenvalues = np.linalg.eigvals(L)
    nonzero_eigs = eigenvalues[np.abs(eigenvalues) > 1e-10]
    
    if len(nonzero_eigs) > 0:
        det_prime = np.prod(nonzero_eigs)
        log_det_prime = np.log(np.abs(det_prime))
    else:
        det_prime = 1.0
        log_det_prime = 0.0
    
    # Γ[Σ] = Tr(L²) - C_H log det'(L)
    # Note: Using magnitude for log determinant
    Gamma = trace_L2 - C_H * log_det_prime
    
    return {
        "Gamma": Gamma,
        "trace_L2": trace_L2,
        "det_prime": det_prime,
        "log_det_prime": log_det_prime,
        "C_H": C_H,
        "num_zero_eigenvalues": len(eigenvalues) - len(nonzero_eigs)
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'cGFTCouplings',
    'InteractionKernel',
    'smooth_step',
    'holographic_constraint',
    'cGFTAction',
    'compute_effective_laplacian',
    'compute_harmony_functional',
]
