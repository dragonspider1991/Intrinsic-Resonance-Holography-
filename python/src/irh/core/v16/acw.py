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
from typing import Optional, Tuple, List, Union
import zlib
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
        # Cast to built-in complex to avoid returning numpy subclasses
        return complex(self.magnitude * np.exp(1j * self.phase))
        
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
    
    where K_t is the resource-bounded Kolmogorov complexity computed using
    a universal Turing machine within time bound t = O(N log N).
    
    Per IRHv16.md Theorem 1.1 (Convergence of NCD to Algorithmic Correlation):
        "Proven using Certified Numerical Analysis on randomized data streams
        of increasing length (up to L=10^8 bits), demonstrating convergence of
        statistical moments and error bounds for C_ij^(t) to within 10^{-12}
        for typical CRN states."
    
    Current implementation uses zlib (LZ77-based) as proxy for LZW.
    
    Args:
        binary1: First binary string b_i
        binary2: Second binary string b_j
        method: Compression method - "lzw" (default) or "sampling"
        compression_level: zlib compression level (1=low/fast, 9=high/slow)
                          Per memory: level 1 ~5% error, 6 ~2% error, 9 ~1% error
        time_bound: Computational time limit for K_t (unused in current impl)
        
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
    if method == "sampling":
        raise NotImplementedError(
            "v16.0: Statistical sampling method requires [IRH-COMP-2025-02]"
        )
    if method != "lzw":
        raise ValueError(f"Unknown method '{method}', supported: 'lzw'")
    
    # Validate inputs
    b1 = _to_bytes(binary1)
    b2 = _to_bytes(binary2)
    if not b1 or not b2:
        raise ValueError("Binary strings cannot be empty")
    try:
        decoded1 = b1.decode("ascii")
        decoded2 = b2.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("binary strings must be ASCII-encodable") from exc
    if not all(c in '01' for c in decoded1):
        raise ValueError("binary1 must contain only '0' and '1'")
    if not all(c in '01' for c in decoded2):
        raise ValueError("binary2 must contain only '0' and '1'")
    
    # Convert to bytes for compression
    b_concat = b1 + b2
    
    # Compute compressed sizes K_t approximations
    # Using zlib as proxy for LZW (both are LZ-based)
    c1 = len(zlib.compress(b1, level=compression_level))
    c2 = len(zlib.compress(b2, level=compression_level))
    c_concat = len(zlib.compress(b_concat, level=compression_level))
    
    # NCD formula: [K(x) + K(y) - K(xy)] / max(K(x), K(y))
    # This measures how much joint compression saves vs independent
    numerator = c1 + c2 - c_concat
    denominator = max(c1, c2)
    
    if denominator == 0:
        # Edge case: both strings compress to nothing
        ncd = 0.0
    else:
        ncd = numerator / denominator
        
    # Clamp to [0, 1] - NCD should be in this range but numerical
    # issues can cause slight excursions
    ncd = max(0.0, min(1.0, ncd))
    
    # Error estimate based on compression level
    # These are empirical estimates, not certified bounds
    # Per repository memory: "level 1 ~5% error, 6 ~2% error, 9 ~1% error"
    error_estimates = {
        1: 0.05,  # Low fidelity, ~5% error
        2: 0.04,
        3: 0.035,
        4: 0.03,
        5: 0.025,
        6: 0.02,  # Medium fidelity, ~2% error
        7: 0.015,
        8: 0.012,
        9: 0.01,  # High fidelity, ~1% error
    }
    error_bound = error_estimates.get(compression_level, 0.02)
    
    return ncd, error_bound


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
    
    Current implementation: Simplified version using angular difference.
    Full v16.0 requires non-commutative path integral from [IRH-MATH-2025-01].
    
    Args:
        state_i: Source AHS with holonomic phase φ_i
        state_j: Target AHS with holonomic phase φ_j
        
    Returns:
        Phase shift in [0, 2π) representing minimal angular distance
        
    References:
        IRHv16.md §1 Axiom 1: Phase definition
        [IRH-MATH-2025-01] Theorem 1.3: Full non-commutative derivation (future)
    """
    # Simple implementation: minimal angular distance (modular arithmetic)
    # This is the "most efficient path" in the simplified case
    delta = state_j.holonomic_phase - state_i.holonomic_phase
    
    # Normalize to [0, 2π) per manuscript definition
    delta = delta % (2 * np.pi)
    
    return delta


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
__status__ = "Phase 2 Implementation - Core functions implemented, multi-fidelity pending"

__all__ = [
    "AlgorithmicCoherenceWeight",
    "compute_ncd_magnitude",
    "compute_phase_shift",
    "build_acw_matrix",
    "MultiFidelityNCDEvaluator",
]
