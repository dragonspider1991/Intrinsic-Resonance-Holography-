"""
cGFT Field for IRH v18.0
========================

Implements the fundamental complex scalar field φ(g₁,g₂,g₃,g₄) ∈ ℂ
and the bilocal field Σ(g,g') as defined in IRHv18.md Section 1.1.

THEORETICAL COMPLIANCE:
    This implementation strictly follows docs/manuscripts/IRHv18.md
    - Section 1.1: Fundamental field on G_inf^4
    - Section 1.1.2: Bilocal field definition
    - Theorem 1.1: Harmony Functional from cGFT

Key Concepts:
    - 4-valent vertex: φ(g₁,g₂,g₃,g₄) represents connections between 4 group elements
    - Hermitian conjugate: φ̄ for complex conjugate field
    - Bilocal field: Σ(g,g') = emergent edge representation

References:
    docs/manuscripts/IRHv18.md:
        - §1.1: Field definition
        - §1.1.2: Σ(g,g') definition (Eq. for bilocal field)
        - Theorem 1.1: Effective action derivation
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Callable
import numpy as np
from numpy.typing import NDArray

from .group_manifold import GInfElement, SU2Element, U1Element, haar_integrate_ginf


# =============================================================================
# Fundamental cGFT Field
# =============================================================================

@dataclass
class cGFTField:
    """
    Fundamental complex scalar field φ(g₁,g₂,g₃,g₄) ∈ ℂ.
    
    This is the elementary building block of the cGFT, representing
    a 4-valent vertex connecting four group elements. In the emergent
    CRN picture, these vertices combine to form the network structure.
    
    The field satisfies gauge invariance under simultaneous left-multiplication:
    φ(kg₁, kg₂, kg₃, kg₄) = φ(g₁, g₂, g₃, g₄) for all k ∈ G_inf
    
    Attributes:
        values: Dictionary mapping 4-tuples of GInfElements to complex values
        gauge_invariant: Whether to enforce gauge invariance
        
    References:
        IRHv18.md §1.1: φ(g₁,g₂,g₃,g₄) ∈ ℂ definition
        IRHv18.md §1.1.1: Gauge invariance
    """
    
    _values: Dict[Tuple, complex] = field(default_factory=dict)
    gauge_invariant: bool = True
    
    def __call__(
        self, 
        g1: GInfElement, 
        g2: GInfElement, 
        g3: GInfElement, 
        g4: GInfElement
    ) -> complex:
        """
        Evaluate field at (g₁, g₂, g₃, g₄).
        
        Returns the complex amplitude at the specified vertex.
        """
        key = self._make_key(g1, g2, g3, g4)
        return self._values.get(key, 0.0 + 0.0j)
    
    def set_value(
        self,
        g1: GInfElement,
        g2: GInfElement, 
        g3: GInfElement,
        g4: GInfElement,
        value: complex
    ) -> None:
        """Set field value at specified vertex."""
        key = self._make_key(g1, g2, g3, g4)
        self._values[key] = value
    
    def _make_key(
        self,
        g1: GInfElement,
        g2: GInfElement,
        g3: GInfElement,
        g4: GInfElement
    ) -> Tuple:
        """Create hashable key from group elements."""
        # Use binary representation as key
        return (
            g1.to_binary_string(),
            g2.to_binary_string(),
            g3.to_binary_string(),
            g4.to_binary_string()
        )
    
    def conjugate(self) -> 'cGFTField':
        """
        Return Hermitian conjugate field φ̄.
        
        φ̄(g₁,g₂,g₃,g₄) = [φ(g₁,g₂,g₃,g₄)]*
        """
        conj_field = cGFTField(gauge_invariant=self.gauge_invariant)
        for key, value in self._values.items():
            conj_field._values[key] = np.conj(value)
        return conj_field
    
    @property
    def num_vertices(self) -> int:
        """Number of non-zero vertices in the field."""
        return len(self._values)
    
    def norm_squared(self) -> float:
        """
        Compute ||φ||² = Σ |φ(g₁,g₂,g₃,g₄)|²
        
        This is the field norm used in action terms.
        """
        return sum(np.abs(v)**2 for v in self._values.values())


@dataclass
class cGFTFieldDiscrete:
    """
    Discretized cGFT field for numerical computation.
    
    Instead of continuous functions on G_inf^4, this stores field
    values on a discrete lattice of sampled group elements.
    
    Attributes:
        group_samples: List of sampled GInfElements
        field_array: Complex array of shape (N, N, N, N)
        N: Number of samples per group argument
        
    Notes:
        For N samples, memory scales as O(N⁴) which limits practical
        calculations to N ≲ 50 without optimization.
    """
    
    group_samples: List[GInfElement] = field(default_factory=list)
    field_array: Optional[NDArray[np.complex128]] = None
    
    @classmethod
    def create_random(
        cls,
        N: int = 10,
        seed: Optional[int] = None,
        amplitude_scale: float = 1.0
    ) -> 'cGFTFieldDiscrete':
        """
        Create discretized field with random sampling.
        
        Args:
            N: Number of group samples
            seed: Random seed
            amplitude_scale: Scale for random field values
            
        Returns:
            Initialized discrete cGFT field
        """
        rng = np.random.default_rng(seed)
        
        # Sample group elements
        samples = [GInfElement.random(rng) for _ in range(N)]
        
        # Initialize random field values
        # Real and imaginary parts from standard normal
        real_part = rng.standard_normal((N, N, N, N))
        imag_part = rng.standard_normal((N, N, N, N))
        field_array = amplitude_scale * (real_part + 1j * imag_part)
        
        return cls(group_samples=samples, field_array=field_array)
    
    @property
    def N(self) -> int:
        """Number of samples."""
        return len(self.group_samples)
    
    def __call__(self, i: int, j: int, k: int, l: int) -> complex:
        """Get field value at discrete indices."""
        return self.field_array[i, j, k, l]
    
    def get_group_element(self, i: int) -> GInfElement:
        """Get group element at index."""
        return self.group_samples[i]
    
    def norm_squared(self) -> float:
        """Compute discretized ||φ||²."""
        return np.sum(np.abs(self.field_array)**2) / self.N**4
    
    def conjugate(self) -> 'cGFTFieldDiscrete':
        """Return conjugate field."""
        return cGFTFieldDiscrete(
            group_samples=self.group_samples,
            field_array=np.conj(self.field_array)
        )


# =============================================================================
# Bilocal Field Σ(g, g')
# =============================================================================

@dataclass
class BiLocalField:
    """
    Bilocal field Σ(g, g') representing emergent edges.
    
    Defined from the fundamental field as:
    Σ(g, g') = ∫ φ(g, ·, ·, ·) φ̄(·, ·, ·, g') Π_{k=2}^3 dg_k
    
    This is the effective two-point correlation from which the
    emergent CRN structure and Harmony Functional are derived.
    
    In the infrared limit, Σ determines:
    - The emergent graph Laplacian L[Σ]
    - The effective metric tensor g_μν
    - The Harmony Functional Γ[Σ]
    
    Attributes:
        values: Dictionary mapping (g, g') pairs to complex values
        
    References:
        IRHv18.md §1.1.2: Definition via trace over internal indices
        IRHv18.md Theorem 1.1: 1PI effective action for Σ
    """
    
    _values: Dict[Tuple[bytes, bytes], complex] = field(default_factory=dict)
    
    def __call__(self, g: GInfElement, g_prime: GInfElement) -> complex:
        """Evaluate bilocal field at (g, g')."""
        key = (g.to_binary_string(), g_prime.to_binary_string())
        return self._values.get(key, 0.0 + 0.0j)
    
    def set_value(self, g: GInfElement, g_prime: GInfElement, value: complex) -> None:
        """Set bilocal field value."""
        key = (g.to_binary_string(), g_prime.to_binary_string())
        self._values[key] = value
    
    @classmethod
    def from_cgft_field(
        cls,
        phi: cGFTFieldDiscrete,
        num_integration_samples: int = 100
    ) -> 'BiLocalField':
        """
        Construct bilocal field from fundamental cGFT field.
        
        Computes Σ(g, g') = ∫ φ(g, g₂, g₃, g₄) φ̄(g₂, g₃, g₄, g') dg₂ dg₃
        
        using Monte Carlo integration over internal indices.
        
        Args:
            phi: Discretized cGFT field
            num_integration_samples: Samples for Monte Carlo integration
            
        Returns:
            Bilocal field Σ
        """
        rng = np.random.default_rng()
        sigma = cls()
        
        N = phi.N
        
        # For each pair of external group elements
        for i in range(N):
            for j in range(N):
                g = phi.get_group_element(i)
                g_prime = phi.get_group_element(j)
                
                # Monte Carlo integration over internal indices
                total = 0.0 + 0.0j
                for _ in range(num_integration_samples):
                    # Random internal indices
                    i2 = rng.integers(0, N)
                    i3 = rng.integers(0, N)
                    i4 = rng.integers(0, N)
                    
                    # φ(g, g₂, g₃, g₄)
                    phi_val = phi(i, i2, i3, i4)
                    
                    # φ̄(g₂, g₃, g₄, g')
                    phi_bar_val = np.conj(phi(i2, i3, i4, j))
                    
                    total += phi_val * phi_bar_val
                
                # Normalize
                sigma.set_value(g, g_prime, total / num_integration_samples)
        
        return sigma
    
    def to_matrix(self, group_samples: List[GInfElement]) -> NDArray[np.complex128]:
        """
        Convert bilocal field to matrix representation.
        
        Args:
            group_samples: List of group elements defining the discretization
            
        Returns:
            Complex matrix Σ_ij = Σ(g_i, g_j)
        """
        N = len(group_samples)
        matrix = np.zeros((N, N), dtype=np.complex128)
        
        for i, g in enumerate(group_samples):
            for j, g_prime in enumerate(group_samples):
                matrix[i, j] = self(g, g_prime)
        
        return matrix


# =============================================================================
# Condensate State
# =============================================================================

@dataclass
class CondensateState:
    """
    Condensate state ⟨φ⟩ for the cGFT.
    
    At the Cosmic Fixed Point, the cGFT develops a non-trivial
    condensate ⟨φ⟩ ≠ 0, breaking the symmetries of G_inf.
    This condensate defines the emergent spacetime geometry.
    
    Attributes:
        expectation_value: Function giving ⟨φ⟩(g₁, g₂, g₃, g₄)
        order_parameter: Magnitude of condensate
        
    References:
        IRHv18.md §2.2.1: Emergent metric from condensate
        IRHv18.md §3.2: Higgs VEV as condensate order parameter
    """
    
    _condensate_field: cGFTFieldDiscrete = None
    order_parameter: float = 0.0
    
    @classmethod
    def create_homogeneous(
        cls,
        N: int = 10,
        condensate_value: complex = 1.0,
        seed: Optional[int] = None
    ) -> 'CondensateState':
        """
        Create homogeneous condensate state.
        
        This represents the simplest condensate where ⟨φ⟩ is
        approximately constant across G_inf^4.
        
        Args:
            N: Discretization size
            condensate_value: Constant condensate amplitude
            seed: Random seed for group sampling
            
        Returns:
            Homogeneous condensate state
        """
        rng = np.random.default_rng(seed)
        samples = [GInfElement.random(rng) for _ in range(N)]
        
        # Homogeneous: constant value everywhere
        field_array = np.full((N, N, N, N), condensate_value, dtype=np.complex128)
        
        condensate_field = cGFTFieldDiscrete(
            group_samples=samples,
            field_array=field_array
        )
        
        return cls(
            _condensate_field=condensate_field,
            order_parameter=np.abs(condensate_value)
        )
    
    def get_bilocal_sigma(self) -> BiLocalField:
        """
        Get bilocal field Σ from the condensate.
        
        This is the effective propagator structure in the condensate phase.
        """
        return BiLocalField.from_cgft_field(self._condensate_field)
    
    @property
    def field(self) -> cGFTFieldDiscrete:
        """Access underlying field."""
        return self._condensate_field


# =============================================================================
# Field Fluctuations
# =============================================================================

def compute_fluctuation_field(
    phi: cGFTFieldDiscrete,
    condensate: CondensateState
) -> cGFTFieldDiscrete:
    """
    Compute fluctuation field δφ = φ - ⟨φ⟩.
    
    Physical excitations (gravitons, gauge bosons, fermions)
    correspond to fluctuations around the condensate.
    
    Args:
        phi: Full cGFT field
        condensate: Condensate state
        
    Returns:
        Fluctuation field δφ
    """
    delta_array = phi.field_array - condensate.field.field_array
    
    return cGFTFieldDiscrete(
        group_samples=phi.group_samples,
        field_array=delta_array
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'cGFTField',
    'cGFTFieldDiscrete',
    'BiLocalField',
    'CondensateState',
    'compute_fluctuation_field',
]
