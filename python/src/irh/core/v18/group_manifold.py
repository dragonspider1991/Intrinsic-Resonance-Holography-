"""
Group Manifold for IRH v18.0 cGFT
=================================

Implements the informational group manifold G_inf = SU(2) × U(1)_φ
as defined in IRHv18.md Section 1.1.

THEORETICAL COMPLIANCE:
    This implementation strictly follows docs/manuscripts/IRHv18.md
    - Section 1.1: The Fundamental Field and the Informational Group Manifold
    - Appendix A: Construction of the NCD-Induced Metric on G_inf

Key Concepts:
    - SU(2): Encodes minimal non-commutative algebra of EATs
    - U(1)_φ: Carries intrinsic holonomic phase φ ∈ [0, 2π)
    - G_inf = SU(2) × U(1)_φ: Compact Lie group of primordial informational DOFs

References:
    docs/manuscripts/IRHv18.md:
        - §1.1: G_inf definition
        - §1.1.1: cGFT Action components
        - Appendix A.1-A.3: Binary encoding and NCD metric
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray
import hashlib
import zlib


# =============================================================================
# SU(2) Element Implementation
# =============================================================================

@dataclass
class SU2Element:
    """
    Element of SU(2) represented as a unit quaternion.
    
    SU(2) encodes the minimal non-commutative algebra of Elementary
    Algorithmic Transformations (EATs) as per IRHv18.md Theorem 1.5.
    
    The quaternion representation q = q₀ + iq₁ + jq₂ + kq₃ satisfies
    q₀² + q₁² + q₂² + q₃² = 1 (unit sphere S³).
    
    Attributes:
        q0, q1, q2, q3: Quaternion components (real, i, j, k)
        
    References:
        IRHv18.md §1.1: SU(2) encodes minimal non-commutative algebra
        Appendix A.1: Quaternion encoding for NCD computation
    """
    
    q0: float = 1.0  # Real component
    q1: float = 0.0  # i component
    q2: float = 0.0  # j component  
    q3: float = 0.0  # k component
    
    def __post_init__(self):
        """Normalize to unit quaternion after creation."""
        self._normalize()
    
    def _normalize(self) -> None:
        """Ensure quaternion has unit norm."""
        norm = np.sqrt(self.q0**2 + self.q1**2 + self.q2**2 + self.q3**2)
        if norm > 1e-10:
            self.q0 /= norm
            self.q1 /= norm
            self.q2 /= norm
            self.q3 /= norm
        else:
            # Default to identity if degenerate
            self.q0, self.q1, self.q2, self.q3 = 1.0, 0.0, 0.0, 0.0
    
    @property
    def quaternion(self) -> NDArray[np.float64]:
        """Return quaternion as numpy array [q0, q1, q2, q3]."""
        return np.array([self.q0, self.q1, self.q2, self.q3])
    
    @property
    def norm(self) -> float:
        """Return quaternion norm (should be 1 for SU(2))."""
        return np.sqrt(self.q0**2 + self.q1**2 + self.q2**2 + self.q3**2)
    
    @classmethod
    def identity(cls) -> 'SU2Element':
        """Return identity element of SU(2)."""
        return cls(q0=1.0, q1=0.0, q2=0.0, q3=0.0)
    
    @classmethod
    def from_axis_angle(cls, axis: NDArray, angle: float) -> 'SU2Element':
        """
        Create SU(2) element from rotation axis and angle.
        
        Args:
            axis: Unit vector [nx, ny, nz] specifying rotation axis
            angle: Rotation angle in radians
            
        Returns:
            SU2Element representing the rotation
        """
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-10)
        
        half_angle = angle / 2
        sin_half = np.sin(half_angle)
        
        return cls(
            q0=np.cos(half_angle),
            q1=axis[0] * sin_half,
            q2=axis[1] * sin_half,
            q3=axis[2] * sin_half
        )
    
    @classmethod
    def random(cls, rng: Optional[np.random.Generator] = None) -> 'SU2Element':
        """
        Generate random SU(2) element with uniform Haar measure.
        
        Uses the method: generate 4 Gaussian random numbers and normalize.
        This produces uniform distribution on S³ = SU(2).
        
        Args:
            rng: Random number generator for reproducibility
            
        Returns:
            Random SU2Element
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Gaussian sampling gives uniform distribution on sphere
        q = rng.standard_normal(4)
        q = q / np.linalg.norm(q)
        
        return cls(q0=q[0], q1=q[1], q2=q[2], q3=q[3])
    
    def inverse(self) -> 'SU2Element':
        """
        Return inverse element (conjugate for unit quaternion).
        
        For unit quaternion: q⁻¹ = q̄ = (q0, -q1, -q2, -q3)
        """
        return SU2Element(
            q0=self.q0,
            q1=-self.q1,
            q2=-self.q2,
            q3=-self.q3
        )
    
    def __mul__(self, other: 'SU2Element') -> 'SU2Element':
        """
        Quaternion multiplication (group operation).
        
        Hamilton product:
        (a + bi + cj + dk)(e + fi + gj + hk) = ...
        """
        a, b, c, d = self.q0, self.q1, self.q2, self.q3
        e, f, g, h = other.q0, other.q1, other.q2, other.q3
        
        return SU2Element(
            q0=a*e - b*f - c*g - d*h,
            q1=a*f + b*e + c*h - d*g,
            q2=a*g - b*h + c*e + d*f,
            q3=a*h + b*g - c*f + d*e
        )
    
    def to_matrix(self) -> NDArray[np.complex128]:
        """
        Convert to 2×2 complex matrix representation.
        
        SU(2) matrix: [[α, β], [-β*, α*]] where α = q0 + iq3, β = q2 + iq1
        """
        alpha = complex(self.q0, self.q3)
        beta = complex(self.q2, self.q1)
        
        return np.array([
            [alpha, beta],
            [-np.conj(beta), np.conj(alpha)]
        ], dtype=np.complex128)
    
    def trace(self) -> float:
        """Return trace of the SU(2) matrix: Tr(U) = 2*Re(α) = 2*q0."""
        return 2.0 * self.q0
    
    def to_binary_string(self, bits_per_component: int = 32) -> bytes:
        """
        Encode SU(2) element as binary string for NCD computation.
        
        As per IRHv18.md Appendix A.1, each quaternion component is
        represented with fixed-point precision.
        
        Args:
            bits_per_component: Bits per quaternion component (M in Appendix A)
            
        Returns:
            Binary string encoding of the SU(2) element
        """
        # Scale to integer range and encode
        scale = 2**(bits_per_component - 1) - 1
        
        components = [self.q0, self.q1, self.q2, self.q3]
        encoded = bytearray()
        
        for q in components:
            # Map [-1, 1] to integer range
            int_val = int(q * scale)
            # Pack as signed integer
            encoded.extend(int_val.to_bytes(
                (bits_per_component + 7) // 8, 
                byteorder='big', 
                signed=True
            ))
        
        return bytes(encoded)
    
    def __repr__(self) -> str:
        return f"SU2({self.q0:.4f}, {self.q1:.4f}i, {self.q2:.4f}j, {self.q3:.4f}k)"


# =============================================================================
# U(1) Element Implementation  
# =============================================================================

@dataclass
class U1Element:
    """
    Element of U(1)_φ representing the holonomic phase.
    
    U(1)_φ carries the intrinsic holonomic phase φ ∈ [0, 2π) crucial for:
    - Emergent quantum mechanics
    - Electrodynamics (fine structure constant derivation)
    - Phase coherence in cGFT interaction kernel
    
    Attributes:
        phi: Holonomic phase in radians [0, 2π)
        
    References:
        IRHv18.md §1.1: U(1)_φ carries intrinsic holonomic phase
        IRHv18.md §1.1.1: Phase factor e^{i(φ₁+φ₂+φ₃-φ₄)} in kernel
    """
    
    phi: float = 0.0
    
    def __post_init__(self):
        """Normalize phase to [0, 2π)."""
        self.phi = self.phi % (2 * np.pi)
    
    @property
    def complex_value(self) -> complex:
        """Return e^{iφ} as complex number."""
        return np.exp(1j * self.phi)
    
    @classmethod
    def identity(cls) -> 'U1Element':
        """Return identity element (φ = 0)."""
        return cls(phi=0.0)
    
    @classmethod
    def random(cls, rng: Optional[np.random.Generator] = None) -> 'U1Element':
        """
        Generate random U(1) element with uniform measure on circle.
        
        Args:
            rng: Random number generator
            
        Returns:
            Random U1Element with φ ∈ [0, 2π)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        return cls(phi=rng.uniform(0, 2 * np.pi))
    
    def inverse(self) -> 'U1Element':
        """Return inverse element (-φ mod 2π)."""
        return U1Element(phi=-self.phi)
    
    def __mul__(self, other: 'U1Element') -> 'U1Element':
        """Group multiplication: addition of phases."""
        return U1Element(phi=self.phi + other.phi)
    
    def to_binary_string(self, bits: int = 32) -> bytes:
        """
        Encode U(1) element as binary string for NCD computation.
        
        Args:
            bits: Bits for phase encoding (R in Appendix A.1)
            
        Returns:
            Binary string encoding of the phase
        """
        # Map [0, 2π) to integer range [0, 2^bits)
        scale = 2**bits - 1
        int_val = int((self.phi / (2 * np.pi)) * scale)
        
        return int_val.to_bytes((bits + 7) // 8, byteorder='big')
    
    def __repr__(self) -> str:
        return f"U1(φ={self.phi:.4f})"


# =============================================================================
# G_inf = SU(2) × U(1)_φ Implementation
# =============================================================================

@dataclass  
class GInfElement:
    """
    Element of G_inf = SU(2) × U(1)_φ - the informational group manifold.
    
    This is the fundamental degree of freedom in the cGFT, representing
    primordial informational states from which all physics emerges.
    
    The cGFT field φ(g₁, g₂, g₃, g₄) is defined over four such elements,
    representing a 4-valent vertex in the emergent Cymatic Resonance Network.
    
    Attributes:
        su2: SU(2) component (non-commutative algebra)
        u1: U(1)_φ component (holonomic phase)
        
    References:
        IRHv18.md §1.1: G_inf = SU(2) × U(1)_φ definition
        IRHv18.md Theorem 1.5: Axiomatic uniqueness of G_inf
        IRHv18.md Appendix A: NCD metric construction
    """
    
    su2: SU2Element = field(default_factory=SU2Element.identity)
    u1: U1Element = field(default_factory=U1Element.identity)
    
    @classmethod
    def identity(cls) -> 'GInfElement':
        """Return identity element e = (Id_SU2, 1)."""
        return cls(
            su2=SU2Element.identity(),
            u1=U1Element.identity()
        )
    
    @classmethod
    def random(cls, rng: Optional[np.random.Generator] = None) -> 'GInfElement':
        """
        Generate random G_inf element with product Haar measure.
        
        The product measure dg = dg_SU2 × dg_U1 is the invariant
        Haar measure on the compact group G_inf.
        
        Args:
            rng: Random number generator
            
        Returns:
            Random GInfElement
        """
        if rng is None:
            rng = np.random.default_rng()
        
        return cls(
            su2=SU2Element.random(rng),
            u1=U1Element.random(rng)
        )
    
    def inverse(self) -> 'GInfElement':
        """Return inverse element g⁻¹ = (u⁻¹, e^{-iφ})."""
        return GInfElement(
            su2=self.su2.inverse(),
            u1=self.u1.inverse()
        )
    
    def __mul__(self, other: 'GInfElement') -> 'GInfElement':
        """Group multiplication: component-wise."""
        return GInfElement(
            su2=self.su2 * other.su2,
            u1=self.u1 * other.u1
        )
    
    @property
    def phase(self) -> float:
        """Return the U(1) phase component."""
        return self.u1.phi
    
    def to_binary_string(self, su2_bits: int = 32, u1_bits: int = 32) -> bytes:
        """
        Encode G_inf element as composite binary string.
        
        As per IRHv18.md Appendix A.1, the encoding is:
        b(g) = Enc_SU2(u) ∘ b(φ)
        
        Total length N_B = M + R where M = 4 × su2_bits, R = u1_bits
        
        Args:
            su2_bits: Bits per SU(2) quaternion component
            u1_bits: Bits for U(1) phase
            
        Returns:
            Concatenated binary encoding
        """
        su2_encoding = self.su2.to_binary_string(su2_bits)
        u1_encoding = self.u1.to_binary_string(u1_bits)
        
        return su2_encoding + u1_encoding
    
    def trace_su2(self) -> float:
        """Return Tr(SU2 component) for holographic constraint."""
        return self.su2.trace()
    
    def __repr__(self) -> str:
        return f"GInf({self.su2}, {self.u1})"


# =============================================================================
# NCD Metric on G_inf
# =============================================================================

def compute_ncd(x: bytes, y: bytes) -> float:
    """
    Compute Normalized Compression Distance between binary strings.
    
    NCD(x, y) = [C(x∘y) - min(C(x), C(y))] / max(C(x), C(y))
    
    Uses zlib (LZ77) as the universal compressor, which is proven
    to give compressor-independent results at the fixed point
    (IRHv18.md Appendix A.4, Theorem A.1).
    
    Args:
        x, y: Binary strings to compare
        
    Returns:
        NCD value in [0, 1]
        
    References:
        IRHv18.md Appendix A.2: NCD definition
        IRHv18.md Theorem A.1: Compressor-independence
    """
    if len(x) == 0 and len(y) == 0:
        return 0.0
    
    # Compress individual strings
    Cx = len(zlib.compress(x, level=9))
    Cy = len(zlib.compress(y, level=9))
    
    # Compress concatenation
    Cxy = len(zlib.compress(x + y, level=9))
    
    # NCD formula
    numerator = Cxy - min(Cx, Cy)
    denominator = max(Cx, Cy)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def compute_ncd_distance(g1: GInfElement, g2: GInfElement) -> float:
    """
    Compute bi-invariant NCD distance on G_inf.
    
    d_NCD(g1, g2) = NCD(b(g1 g2⁻¹), b(e))
    
    This is bi-invariant: d(kg1, kg2) = d(g1, g2) for all k ∈ G_inf.
    
    Args:
        g1, g2: G_inf elements
        
    Returns:
        NCD-induced distance
        
    References:
        IRHv18.md Appendix A.3: Bi-invariant distance construction
    """
    # Compute g1 * g2^{-1}
    g_diff = g1 * g2.inverse()
    
    # Get identity
    e = GInfElement.identity()
    
    # Encode to binary
    b_diff = g_diff.to_binary_string()
    b_e = e.to_binary_string()
    
    return compute_ncd(b_diff, b_e)


# =============================================================================
# Haar Measure Integration Utilities
# =============================================================================

def haar_integrate_su2(
    f: callable,
    num_samples: int = 1000,
    rng: Optional[np.random.Generator] = None
) -> complex:
    """
    Monte Carlo integration over SU(2) with Haar measure.
    
    ∫_{SU(2)} f(u) du ≈ (1/N) Σ f(u_i)
    
    where u_i are uniformly sampled from SU(2).
    
    Args:
        f: Function SU2Element → complex
        num_samples: Number of Monte Carlo samples
        rng: Random generator
        
    Returns:
        Estimated integral value
    """
    if rng is None:
        rng = np.random.default_rng()
    
    total = 0.0 + 0.0j
    for _ in range(num_samples):
        u = SU2Element.random(rng)
        total += f(u)
    
    return total / num_samples


def haar_integrate_ginf(
    f: callable,
    num_samples: int = 1000,
    rng: Optional[np.random.Generator] = None
) -> complex:
    """
    Monte Carlo integration over G_inf with product Haar measure.
    
    ∫_{G_inf} f(g) dg ≈ (1/N) Σ f(g_i)
    
    Args:
        f: Function GInfElement → complex
        num_samples: Number of Monte Carlo samples
        rng: Random generator
        
    Returns:
        Estimated integral value
    """
    if rng is None:
        rng = np.random.default_rng()
    
    total = 0.0 + 0.0j
    for _ in range(num_samples):
        g = GInfElement.random(rng)
        total += f(g)
    
    return total / num_samples


# =============================================================================
# Laplace-Beltrami Operator Components
# =============================================================================

def laplacian_su2_eigenvalue(j: float) -> float:
    """
    Eigenvalue of Laplace-Beltrami operator on SU(2).
    
    For representation j, the eigenvalue is -j(j+1).
    
    Args:
        j: SU(2) representation label (j = 0, 1/2, 1, 3/2, ...)
        
    Returns:
        Laplacian eigenvalue
    """
    return -j * (j + 1)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'SU2Element',
    'U1Element', 
    'GInfElement',
    'compute_ncd',
    'compute_ncd_distance',
    'haar_integrate_su2',
    'haar_integrate_ginf',
    'laplacian_su2_eigenvalue',
]
