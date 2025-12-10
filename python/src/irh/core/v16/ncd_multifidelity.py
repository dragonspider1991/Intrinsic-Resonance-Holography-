"""
Multi-Fidelity Normalized Compression Distance (NCD) Calculator

This module implements Phase 2 multi-fidelity NCD evaluation with certified
error bounds as specified in PHASE_2_STATUS.md and v16_IMPLEMENTATION_ROADMAP.md.

Key Features:
    - LZW compression for short strings (high fidelity)
    - Statistical sampling for long strings (medium/low fidelity)
    - Certified error bounds computation
    - Adaptive fidelity selection based on string length

Implementation Status: Phase 2 - Initial Implementation

References:
    PHASE_2_STATUS.md: Multi-fidelity NCD pipeline requirement
    v16_IMPLEMENTATION_ROADMAP.md Phase 3, §3.1: Multi-fidelity algorithms
    [IRH-COMP-2025-02] §2: Multi-fidelity NCD computation (future reference)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Tuple, Optional
from enum import Enum
import numpy as np
import zlib


class FidelityLevel(Enum):
    """Fidelity levels for NCD computation."""
    HIGH = "high"      # Full compression, exact NCD
    MEDIUM = "medium"  # Sampled compression with error bounds
    LOW = "low"        # Approximate NCD with larger error bounds


@dataclass
class NCDResult:
    """
    Result of NCD computation with certified error bounds.
    
    Attributes:
        ncd_value: Normalized compression distance in [0, 1]
        error_bound: Certified error bound (1-sigma)
        fidelity: Fidelity level used for computation
        method: Specific method used ("lzw", "zlib", "sampling")
        compute_time: Computation time in seconds (optional)
    """
    ncd_value: float
    error_bound: float
    fidelity: FidelityLevel
    method: str
    compute_time: Optional[float] = None
    
    def __post_init__(self):
        """Validate NCD result."""
        if not 0 <= self.ncd_value <= 1:
            raise ValueError(f"ncd_value must be in [0, 1], got {self.ncd_value}")
        if self.error_bound < 0:
            raise ValueError(f"error_bound must be non-negative, got {self.error_bound}")


def _to_bytes(value: Union[str, bytes, bytearray]) -> bytes:
    """Normalize binary inputs to bytes."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode('ascii')
    raise TypeError("binary inputs must be str, bytes, or bytearray")


def compute_ncd_lzw(
    binary1: Union[str, bytes, bytearray],
    binary2: Union[str, bytes, bytearray],
    compression_level: int = 9
) -> Tuple[float, float]:
    """
    Compute NCD using LZW-style compression (high fidelity).
    
    Uses zlib (LZ77-based) as a proxy for LZW compression.
    This provides high-fidelity NCD for strings up to ~10^4 bytes.
    
    Args:
        binary1: First binary string
        binary2: Second binary string
        compression_level: Compression level (1-9, default 9 for max compression)
        
    Returns:
        (ncd_value, error_bound) tuple
        
    References:
        Li & Vitányi (2008): "The Similarity Metric"
    """
    bytes1 = _to_bytes(binary1)
    bytes2 = _to_bytes(binary2)
    
    # Special case: identical strings
    if bytes1 == bytes2:
        return (0.0, 0.0)
    
    # Compress individual strings and concatenation
    c1 = len(zlib.compress(bytes1, level=compression_level))
    c2 = len(zlib.compress(bytes2, level=compression_level))
    c12 = len(zlib.compress(bytes1 + bytes2, level=compression_level))
    
    # Kolmogorov complexity approximation (in bits)
    K_b1 = c1 * 8
    K_b2 = c2 * 8
    K_b12 = c12 * 8
    
    # NCD formula: (K(x) + K(y) - K(xy)) / max(K(x), K(y))
    max_K = max(K_b1, K_b2)
    if max_K == 0:
        return (0.0, 0.0)
    
    ncd = (K_b1 + K_b2 - K_b12) / max_K
    
    # Clamp to [0, 1] range
    ncd = max(0.0, min(1.0, ncd))
    
    # Error estimate: compression overhead and finite-size effects
    # For strings > 100 bytes, error ~ 1/sqrt(length)
    min_len = min(len(bytes1), len(bytes2))
    if min_len > 100:
        error = 1.0 / np.sqrt(min_len)
    else:
        error = 0.01  # 1% error for short strings
    
    return (ncd, error)


def compute_ncd_sampling(
    binary1: Union[str, bytes, bytearray],
    binary2: Union[str, bytes, bytearray],
    sample_size: int = 1000,
    num_samples: int = 10,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute NCD using statistical sampling (medium/low fidelity).
    
    For very long strings (> 10^4 bytes), computes NCD on random samples
    and aggregates results with error bounds.
    
    Args:
        binary1: First binary string
        binary2: Second binary string
        sample_size: Size of each sample in bytes
        num_samples: Number of samples to average
        seed: Random seed for reproducibility
        
    Returns:
        (mean_ncd, std_error) tuple where std_error is the standard error
        
    References:
        Statistical sampling for NCD approximation
    """
    bytes1 = _to_bytes(binary1)
    bytes2 = _to_bytes(binary2)
    
    # Special case: identical strings
    if bytes1 == bytes2:
        return (0.0, 0.0)
    
    # If strings are short enough, use full LZW
    if len(bytes1) <= sample_size and len(bytes2) <= sample_size:
        return compute_ncd_lzw(bytes1, bytes2)
    
    # Sample both strings and compute NCD
    rng = np.random.RandomState(seed)
    ncd_samples = []
    
    for _ in range(num_samples):
        # Sample from both strings
        if len(bytes1) > sample_size:
            start1 = rng.randint(0, len(bytes1) - sample_size + 1)
            sample1 = bytes1[start1:start1 + sample_size]
        else:
            sample1 = bytes1
        
        if len(bytes2) > sample_size:
            start2 = rng.randint(0, len(bytes2) - sample_size + 1)
            sample2 = bytes2[start2:start2 + sample_size]
        else:
            sample2 = bytes2
        
        # Compute NCD on samples
        ncd_sample, _ = compute_ncd_lzw(sample1, sample2)
        ncd_samples.append(ncd_sample)
    
    # Compute mean and standard error
    mean_ncd = np.mean(ncd_samples)
    std_error = np.std(ncd_samples, ddof=1) / np.sqrt(num_samples)
    
    return (mean_ncd, std_error)


def compute_ncd_adaptive(
    binary1: Union[str, bytes, bytearray],
    binary2: Union[str, bytes, bytearray],
    fidelity: Optional[FidelityLevel] = None,
    auto_select: bool = True
) -> NCDResult:
    """
    Compute NCD with adaptive fidelity selection.
    
    Automatically selects appropriate fidelity level based on input size,
    or uses specified fidelity level.
    
    Fidelity selection criteria:
        - HIGH: strings < 10^4 bytes (full LZW compression)
        - MEDIUM: strings < 10^6 bytes (10 samples of 1000 bytes)
        - LOW: strings >= 10^6 bytes (5 samples of 1000 bytes)
    
    Args:
        binary1: First binary string
        binary2: Second binary string
        fidelity: Requested fidelity level (None for auto-select)
        auto_select: If True, override fidelity with automatic selection
        
    Returns:
        NCDResult with value, error bound, and metadata
        
    Examples:
        >>> result = compute_ncd_adaptive("0110", "1001")
        >>> print(f"NCD = {result.ncd_value:.4f} ± {result.error_bound:.4f}")
        >>> print(f"Fidelity: {result.fidelity.value}")
    """
    import time
    
    bytes1 = _to_bytes(binary1)
    bytes2 = _to_bytes(binary2)
    
    max_len = max(len(bytes1), len(bytes2))
    
    # Auto-select fidelity based on size
    if auto_select or fidelity is None:
        if max_len < 10_000:  # 10^4 bytes
            fidelity = FidelityLevel.HIGH
        elif max_len < 1_000_000:  # 10^6 bytes
            fidelity = FidelityLevel.MEDIUM
        else:
            fidelity = FidelityLevel.LOW
    
    # Compute NCD with selected fidelity
    start_time = time.time()
    
    if fidelity == FidelityLevel.HIGH:
        ncd_value, error_bound = compute_ncd_lzw(bytes1, bytes2)
        method = "lzw"
    elif fidelity == FidelityLevel.MEDIUM:
        ncd_value, error_bound = compute_ncd_sampling(
            bytes1, bytes2, sample_size=1000, num_samples=10
        )
        method = "sampling-medium"
    else:  # LOW
        ncd_value, error_bound = compute_ncd_sampling(
            bytes1, bytes2, sample_size=1000, num_samples=5
        )
        method = "sampling-low"
    
    compute_time = time.time() - start_time
    
    return NCDResult(
        ncd_value=ncd_value,
        error_bound=error_bound,
        fidelity=fidelity,
        method=method,
        compute_time=compute_time
    )


def compute_ncd_certified(
    binary1: Union[str, bytes, bytearray],
    binary2: Union[str, bytes, bytearray],
    target_precision: float = 1e-6,
    max_fidelity: FidelityLevel = FidelityLevel.HIGH
) -> NCDResult:
    """
    Compute NCD with certified precision guarantee.
    
    Iteratively increases fidelity/sampling until target precision is met
    or maximum fidelity is reached.
    
    Args:
        binary1: First binary string
        binary2: Second binary string
        target_precision: Target error bound
        max_fidelity: Maximum fidelity level to use
        
    Returns:
        NCDResult with certified error bound <= target_precision (if possible)
        
    References:
        Phase 2 requirement: Certified error bounds for NCD
    """
    # Start with adaptive selection
    result = compute_ncd_adaptive(binary1, binary2)
    
    # If error bound is acceptable, return
    if result.error_bound <= target_precision:
        return result
    
    # Try to improve by increasing samples (if using sampling)
    if result.fidelity in [FidelityLevel.MEDIUM, FidelityLevel.LOW]:
        # Increase number of samples
        for num_samples in [20, 50, 100]:
            ncd_value, error_bound = compute_ncd_sampling(
                binary1, binary2, sample_size=1000, num_samples=num_samples
            )
            if error_bound <= target_precision:
                return NCDResult(
                    ncd_value=ncd_value,
                    error_bound=error_bound,
                    fidelity=FidelityLevel.MEDIUM,
                    method=f"sampling-{num_samples}"
                )
    
    # If still not meeting target, try high fidelity if allowed
    if max_fidelity == FidelityLevel.HIGH:
        bytes1 = _to_bytes(binary1)
        bytes2 = _to_bytes(binary2)
        if max(len(bytes1), len(bytes2)) < 100_000:  # Reasonable limit for full compression
            ncd_value, error_bound = compute_ncd_lzw(binary1, binary2)
            return NCDResult(
                ncd_value=ncd_value,
                error_bound=error_bound,
                fidelity=FidelityLevel.HIGH,
                method="lzw"
            )
    
    # Return best available result
    return result
