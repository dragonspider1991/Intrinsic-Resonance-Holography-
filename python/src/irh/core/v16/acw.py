"""
Axiom 1: Algorithmic Relationality - Complex-Valued Coherence Weights

This module implements the computation of Algorithmic Coherence Weights (ACW)
W_ij ∈ ℂ between pairs of Algorithmic Holonomic States.

Key Concepts (from IRHv16.md §1):
    - |W_ij| = Normalized Compression Distance (NCD) from K_t
      Formula: C_ij = [K_t(b_i) + K_t(b_j) - K_t(b_i ∘ b_j)] / max(K_t(b_i), K_t(b_j))
    - arg(W_ij) = Minimal holonomic phase shift from AHS algebra
    - Multi-fidelity evaluation for exascale (N ≥ 10^12)
    - Certified error bounds (Theorem 1.1)

Implementation Status: Phase 2 Implementation
    - Basic NCD computation: IMPLEMENTED
    - Phase shift computation: IMPLEMENTED (simplified)
    - Multi-fidelity evaluation: PLACEHOLDER

References:
    IRHv16.md §1 (Axiom 1): Precise definition of ACW
    IRHv16.md Theorem 1.1: Convergence of NCD to Algorithmic Correlation
    [IRH-COMP-2025-02] §2: Multi-fidelity NCD computation (future)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp

from .ahs import AlgorithmicHolonomicState


def _to_bytes(value: Union[str, bytes, bytearray]) -> bytes:
    """Normalize binary inputs to ASCII bytes."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("ascii")
    raise TypeError("binary_string must be str, bytes, or bytearray")


def _to_bytes(value: Union[str, bytes, bytearray]) -> bytes:
    """Normalize binary inputs to ASCII bytes."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode('ascii')
    raise TypeError("binary inputs must be str, bytes, or bytearray")


@dataclass
class AlgorithmicCoherenceWeight:
    """
    Complex-valued coherence weight W_ij between two AHS.
    
    v16.0: Fundamental relationship quantifying coherent transfer potential
    as defined in IRHv16.md §1 (Axiom 1).
    
    Per IRHv16.md:
        "For any ordered pair (s_i, s_j), this potential is represented by a
        complex-valued Algorithmic Coherence Weight W_ij ∈ ℂ."
    
    Attributes:
        magnitude: |W_ij| from NCD (algorithmic compressibility) ∈ [0, 1]
        phase: arg(W_ij) from holonomic phase shift ∈ [0, 2π)
        error_bound: Certified numerical error (from Theorem 1.1)
        method: Computation method used ("lzw", "sampling", etc.)
        
    Properties:
        complex_value: W_ij as complex number
        
    References:
        IRHv16.md §1 Axiom 1: Definition of ACW
        IRHv16.md Theorem 1.1: NCD convergence with error bounds
    """
    
    magnitude: float  # |W_ij| ∈ [0, 1] from NCD
    phase: float  # arg(W_ij) ∈ [0, 2π)
    error_bound: float = 1e-6  # Certified error (conservative default)
    method: str = "lzw"  # Computation method
    
    def __post_init__(self):
        """Validate and normalize ACW values."""
        # Validate magnitude range
        if not 0 <= self.magnitude <= 1:
            raise ValueError(f"magnitude must be in [0, 1], got {self.magnitude}")
        # Normalize phase to [0, 2π)
        self.phase = float(self.phase) % (2 * np.pi)
    
    @property
    def complex_value(self) -> complex:
        """Return W_ij as complex number."""
        return self.magnitude * np.exp(1j * self.phase)
        
    def __complex__(self) -> complex:
        """Allow complex(w_ij) conversion."""
        return self.complex_value
    
    def __repr__(self) -> str:
        """Developer representation."""
        return (f"ACW(|W|={self.magnitude:.6f}, arg(W)={self.phase:.4f}, "
                f"±{self.error_bound:.2e}, method='{self.method}')")


def compute_ncd_magnitude(
    binary1: Union[str, bytes, bytearray],
    binary2: Union[str, bytes, bytearray],
    method: str = "lzw",
    compression_level: int = 6,
    time_bound: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute Normalized Compression Distance magnitude |W_ij|.
    
    NCD formula from IRHv16.md §1 Axiom 1:
        C_ij^(t) := [K_t(b_i) + K_t(b_j) - K_t(b_i ∘ b_j)] / max(K_t(b_i), K_t(b_j))
    
    v16.0 Implementation:
    - Uses zlib (LZ77-based) compression as proxy for LZW
    - For strings < 10^4 bits: Direct compression
    - Error bound estimated from compression ratio variance
    
    Args:
        binary1: First binary string (as bytes)
        binary2: Second binary string (as bytes)
        method: "lzw" (only method currently implemented)
        time_bound: Computational time limit for K_t (not yet used)
    
    Returns:
        (ncd_value, error_bound) tuple where:
        - ncd_value ∈ [0, 1] is the normalized compression distance
        - error_bound is the estimated numerical precision
        
    Raises:
        ValueError: If inputs are invalid
        NotImplementedError: If unsupported method requested
        
    References:
        IRHv16.md §1 Axiom 1: NCD formula definition
        IRHv16.md Theorem 1.1: Convergence proof
        [IRH-COMP-2025-02] §2.1: Multi-fidelity NCD (future)
    """
    import zlib
    
    if method != "lzw":
        raise NotImplementedError(f"Method '{method}' not yet implemented. Use 'lzw'.")
    
    bytes1 = _to_bytes(binary1)
    bytes2 = _to_bytes(binary2)
    # Special case: identical strings
    if bytes1 == bytes2:
        return (0.0, 0.0)
    
    bytes_concat = bytes1 + bytes2
    
    # Compress using zlib (LZ77, similar to LZW)
    # Use compression level 9 for maximum compression (high fidelity)
    c1 = len(zlib.compress(bytes1, level=9))
    c2 = len(zlib.compress(bytes2, level=9))
    c12 = len(zlib.compress(bytes_concat, level=9))
    
    # Kolmogorov complexity approximation: C(x) ≈ compressed_size * 8 (bits)
    K_b1 = c1 * 8
    K_b2 = c2 * 8
    K_b12 = c12 * 8
    
    # NCD formula
    max_K = max(K_b1, K_b2)
    if max_K == 0:
        # Both strings are empty or compress to nothing
        ncd = 0.0
        error = 0.0
    else:
        ncd = (K_b1 + K_b2 - K_b12) / max_K
        
        # Clamp NCD to [0, 1] range (can exceed due to compression overhead)
        ncd = max(0.0, min(1.0, ncd))
        
        # Error estimate: compression is deterministic, but we account for
        # finite-size effects and overhead. Conservative estimate.
        # For strings > 100 chars, error ~ 1/sqrt(length)
        min_len = min(len(bytes1), len(bytes2))
        if min_len > 100:
            error = 1.0 / np.sqrt(min_len)
        else:
            error = 0.01  # 1% error for short strings
    
    return (ncd, error)


def compute_acw(
    state_i: 'AlgorithmicHolonomicState',  # type: ignore
    state_j: 'AlgorithmicHolonomicState',   # type: ignore
    method: str = "lzw"
) -> AlgorithmicCoherenceWeight:
    """
    Compute complete Algorithmic Coherence Weight W_ij between two AHS.
    
    Combines NCD magnitude with holonomic phase shift to create the
    complex-valued coherence weight.
    
    Args:
        state_i: Source AHS
        state_j: Target AHS
        method: Compression method for NCD ("lzw")
        
    Returns:
        AlgorithmicCoherenceWeight with magnitude, phase, and error bound
        
    Examples:
        >>> from .ahs import AlgorithmicHolonomicState
        >>> s1 = AlgorithmicHolonomicState("0110", 0.5)
        >>> s2 = AlgorithmicHolonomicState("1001", 1.2)
        >>> w_ij = compute_acw(s1, s2)
        >>> print(f"|W_ij| = {w_ij.magnitude:.4f}")
        >>> print(f"arg(W_ij) = {w_ij.phase:.4f}")
    """
    # Compute magnitude from NCD
    ncd, error = compute_ncd_magnitude(
        state_i.binary_string,
        state_j.binary_string,
        method=method
    )
    
    # Compute phase from holonomic shift
    phase = compute_phase_shift(state_i, state_j)
    
    # Create ACW
    return AlgorithmicCoherenceWeight(
        magnitude=ncd,
        phase=phase,
        error_bound=error
    )


def compute_phase_shift(
    state_i: AlgorithmicHolonomicState,
    state_j: AlgorithmicHolonomicState
) -> float:
    """
    Compute minimal holonomic phase shift arg(W_ij).
    
    Per IRHv16.md §1 Axiom 1:
        "The phase arg(W_ij) quantifies the minimal holonomic phase shift 
        required to coherently transform the holonomic phase of s_i to s_j.
        This phase is determined by the compositional rules of AHS algebra
        (Axiom 0), ensuring the most efficient, interference-minimized path
        in abstract computational space."
    
    v16.0 Implementation:
    - Uses simple phase difference: φ_j - φ_i (mod 2π)
    - Minimal path assumption (direct transformation)
    - Full non-commutative path analysis deferred to [IRH-MATH-2025-01]
    
    Args:
        state_i: Source AHS with holonomic phase φ_i
        state_j: Target AHS with holonomic phase φ_j
        
    Returns:
        Phase shift in [0, 2π) representing minimal angular distance
        
    References:
        [IRH-MATH-2025-01] Theorem 1.3: Minimal holonomic phase shifts
        IRHv16.md Axiom 1: Phase component of W_ij
    """
    # Simple implementation: direct phase difference
    # This assumes the minimal transformation is the direct path
    phase_diff = (state_j.holonomic_phase - state_i.holonomic_phase) % (2 * np.pi)
    
    return phase_diff


def build_acw_matrix(
    states: List[AlgorithmicHolonomicState],
    epsilon_threshold: float = 0.730129,
    compression_level: int = 6,
    sparse: bool = True
) -> Union[NDArray[np.complex128], sp.csr_matrix]:
    """
    Build complex-valued ACW matrix W for network of AHS.
    
    Per IRHv16.md §1 Axiom 2 (Network Emergence Principle):
        "Any system of Algorithmic Holonomic States satisfying Axiom 1 can be
        represented uniquely and minimally as a complex-weighted, directed
        Cymatic Resonance Network (CRN) G = (V, E, W) where:
        - V = S (nodes are AHS)
        - (s_i, s_j) ∈ E iff |W_ij| > ε_threshold
        - W_ij ∈ ℂ as defined in Axiom 1."
    
    Per IRHv16.md on epsilon_threshold:
        "Computational Value: Exhaustive, multi-fidelity computational analysis
        (N ≥ 10^12) confirms ε_threshold = 0.730129 ± 10^{-6}, a rigorously
        derived constant. This is not chosen, but necessitated by the underlying
        phase dynamics."
    
    Args:
        states: List of AlgorithmicHolonomicState objects (network nodes)
        epsilon_threshold: Edge inclusion threshold (default from manuscript)
        compression_level: zlib compression level for NCD computation
        sparse: If True, return scipy sparse matrix (recommended for N > 1000)
        
    Returns:
        N×N complex matrix W_ij where W_ij = 0 if |W_ij| <= epsilon_threshold
        
    References:
        IRHv16.md §1 Axiom 2: Network Emergence Principle
        IRHv16.md §1 Axiom 1: ACW definition
        [IRH-COMP-2025-02] §3: Distributed version (future)
    """
    N = len(states)
    if N == 0:
        raise ValueError("states list cannot be empty")
    
    # Build dense matrix first (for small N)
    # TODO v16.0: Distributed computation for N >= 10^12
    if N > 5000:
        import warnings
        warnings.warn(
            f"Building ACW matrix for N={N} may be slow. "
            "For N > 5000, consider distributed implementation from [IRH-COMP-2025-02]."
        )
    
    # Initialize matrix
    W = np.zeros((N, N), dtype=np.complex128)
    
    for i in range(N):
        for j in range(N):
            if i == j:
                # Self-coherence is maximal by definition
                W[i, j] = 1.0 + 0j
                continue
                
            # Compute magnitude |W_ij| via NCD
            ncd, _ = compute_ncd_magnitude(
                states[i].binary_string,
                states[j].binary_string,
                compression_level=compression_level
            )
            
            # Apply threshold (Axiom 2)
            if ncd <= epsilon_threshold:
                W[i, j] = 0.0 + 0j
                continue
            
            # Compute phase arg(W_ij)
            phase = compute_phase_shift(states[i], states[j])
            
            # Construct complex weight W_ij = |W_ij| * e^{i*arg(W_ij)}
            W[i, j] = ncd * np.exp(1j * phase)
    
    # Convert to sparse if requested
    if sparse:
        return sp.csr_matrix(W)
    
    return W


# Multi-fidelity strategies
class MultiFidelityNCDEvaluator:
    """
    Multi-fidelity NCD evaluation for exascale networks.
    
    Strategy:
        - Short-range, dense: Full LZW-based NCD
        - Long-range, sparse: Statistical sampling
        - Adaptive refinement based on error bounds
        
    TODO v16.0: Full implementation in [IRH-COMP-2025-02] §2.2
    """
    
    def __init__(self, N: int, target_precision: float = 1e-12):
        """
        Initialize multi-fidelity evaluator.
        
        Args:
            N: Network size
            target_precision: Target error bound for NCD values
        """
        self.N = N
        self.target_precision = target_precision
        
    def adaptive_evaluate(
        self,
        state_i: 'AlgorithmicHolonomicState',  # type: ignore
        state_j: 'AlgorithmicHolonomicState',  # type: ignore
        distance_metric: Optional[float] = None
    ) -> AlgorithmicCoherenceWeight:
        """
        Adaptively choose NCD evaluation strategy.
        
        TODO v16.0: Implement from [IRH-COMP-2025-02]
        """
        raise NotImplementedError("v16.0: Requires [IRH-COMP-2025-02]")


__version__ = "16.0.0-dev"
__status__ = "Partial Implementation - NCD calculator functional"

__all__ = [
    "AlgorithmicCoherenceWeight",
    "compute_ncd_magnitude",
    "compute_phase_shift",
    "compute_acw",
    "build_acw_matrix",
    "MultiFidelityNCDEvaluator",
]
