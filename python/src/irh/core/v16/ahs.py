"""
Axiom 0: Algorithmic Holonomic Substrate

This module implements the fundamental ontological primitive of IRH v16.0:
Algorithmic Holonomic States (AHS) with intrinsic complex-valued nature.

Key Concepts:
    - Each AHS is a pair (binary_string, holonomic_phase)
    - Complex nature emerges from non-commutative algebraic structure
    - Phase is derived from path-dependent composition rules
    - No pre-geometric, pre-temporal assumptions

Implementation Status: PLACEHOLDER - Requires [IRH-MATH-2025-01]

References:
    [IRH-MATH-2025-01] §1: Rigorous derivation of complex numbers from
                           non-commutative algorithmic transformations
    Main Manuscript §1: Axiom 0 statement and justification
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

PHASE_TOLERANCE = 1e-10  # Absolute tolerance for phase equality


@dataclass
class AlgorithmicHolonomicState:
    """
    Algorithmic Holonomic State - fundamental ontological primitive.
    
    v16.0: Intrinsically complex-valued information process embodying both
    informational content and holonomic phase degree of freedom.
    
    Attributes:
        binary_string: Finite binary informational content (immutable)
        holonomic_phase: Intrinsic phase φ ∈ [0, 2π) from non-commutative algebra
        complexity_Kt: Resource-bounded Kolmogorov complexity K_t(binary_string)
        
    Properties:
        complex_amplitude: e^{iφ} representation
        information_content: |binary_string| in bits
        
    Notes:
        - Two states are distinguishable if binary_string OR phase differs
        - Phase quantization occurs through ARO optimization (Theorem 2.1)
        - Algebraic operations defined in AHSAlgebra class
        
    TODO v16.0:
        - Implement non-commutative composition operator
        - Add phase interference calculation
        - Implement resource-bounded K_t computation
        - Add serialization for distributed computing
    """
    
    binary_string: Union[str, bytes, bytearray]  # Binary informational content
    holonomic_phase: float  # φ ∈ [0, 2π)
    complexity_Kt: Optional[float] = None  # Computed on demand
    
    def __post_init__(self):
        """Validate and normalize AHS."""
        # Validate binary string and normalize to consistent type
        allowed_chars = {"0", "1"}
        original_binary = self.binary_string
        if isinstance(original_binary, bytes):
            try:
                normalized_bytes = original_binary
                normalized_str = normalized_bytes.decode("ascii")
            except UnicodeDecodeError as exc:
                raise ValueError("binary_string must be ASCII-encodable") from exc
        elif isinstance(original_binary, bytearray):
            try:
                normalized_bytes = bytes(original_binary)
                normalized_str = normalized_bytes.decode("ascii")
            except UnicodeDecodeError as exc:
                raise ValueError("binary_string must be ASCII-encodable") from exc
        elif isinstance(original_binary, str):
            try:
                normalized_str = original_binary
                normalized_bytes = original_binary.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ValueError("binary_string must be ASCII-encodable") from exc
        else:
            raise TypeError("binary_string must be str, bytes, or bytearray")

        if not normalized_str:  # Empty string
            raise ValueError("binary_string cannot be empty")
        if not all(c in allowed_chars for c in normalized_str):
            raise ValueError("binary_string must contain only '0' and '1'")

        # Preserve original type (str stays str, bytes/bytearray stay bytes)
        if isinstance(original_binary, str):
            self.binary_string = normalized_str
        else:
            self.binary_string = normalized_bytes
        
        # Validate and normalize phase
        if not isinstance(self.holonomic_phase, (int, float)):
            raise TypeError("holonomic_phase must be numeric")
        self.holonomic_phase = float(self.holonomic_phase) % (2 * np.pi)
        
        # Compute complexity if not provided
        if self.complexity_Kt is None:
            # For now, use simple estimate (length)
            # TODO v16.0: Replace with proper K_t computation
            self.complexity_Kt = float(len(self._as_bytes()))

    def _as_bytes(self) -> bytes:
        """Return binary_string normalized to ASCII bytes."""
        if isinstance(self.binary_string, bytes):
            return self.binary_string
        if isinstance(self.binary_string, bytearray):
            return bytes(self.binary_string)
        return self.binary_string.encode("ascii")
        
    @property
    def complex_amplitude(self) -> complex:
        """
        Complex amplitude representation: e^{iφ}.
        
        Returns:
            Complex number on unit circle
        """
        return np.exp(1j * self.holonomic_phase)
        
    @property
    def information_content(self) -> int:
        """
        Information content in bits.
        
        Returns:
            Length of binary_string
        """
        return len(self._as_bytes())
        
    def compute_complexity(self, time_bound: int = 1000) -> float:
        """
        Estimate K_t using simple LZW compression.
        
        NOTE: This is a PLACEHOLDER. v16.0 requires certified multi-fidelity
        evaluation from [IRH-COMP-2025-02].
        
        Args:
            time_bound: Computational time limit (in operations)
            
        Returns:
            K_t estimate via compression
            
        References:
            [IRH-COMP-2025-02] §2.1: Multi-fidelity K_t approximation
        """
        import zlib
        # Use zlib (LZ77-based) as proxy for LZW
        compressed = zlib.compress(self._as_bytes())
        self.complexity_Kt = float(len(compressed) * 8)  # bits
        return self.complexity_Kt

    @staticmethod
    def _wrapped_phase_difference(phase_a: float, phase_b: float) -> float:
        """
        Compute minimal wrapped phase difference in [-π, π].

        The sign follows (phase_a - phase_b); when the unwrapped difference is
        exactly π, the wrapped value is reported as -π to keep the interval
        closed at -π.
        """
        return (phase_a - phase_b + np.pi) % (2 * np.pi) - np.pi
    
    def __eq__(self, other: object) -> bool:
        """Two AHS are equal if info and phase match."""
        if not isinstance(other, AlgorithmicHolonomicState):
            return NotImplemented
        phase_diff = self._wrapped_phase_difference(
            self.holonomic_phase, other.holonomic_phase
        )
        self_bytes = self._as_bytes()
        other_bytes = other._as_bytes()
        return self_bytes == other_bytes and abs(phase_diff) <= PHASE_TOLERANCE

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        # Hash binary string and quantized phase
        phase_quant = int(self.holonomic_phase * 1e10)  # 10 decimal places
        return hash((self._as_bytes(), phase_quant))
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        if isinstance(self.binary_string, bytes):
            info_source = self.binary_string.decode("ascii")
        else:
            info_source = self.binary_string
        info = info_source[:8] + "..." if len(info_source) > 8 else info_source
        return f"AHS(info={info}, φ={self.holonomic_phase:.4f}, K_t={self.complexity_Kt:.1f})"

    def __str__(self) -> str:
        """User-friendly representation."""
        return f"AHS[{len(self.binary_string)}bits, φ={self.holonomic_phase:.3f}rad]"


class AHSAlgebra:
    """
    Non-commutative algebra of Algorithmic Holonomic State transformations.
    
    Implements the algebraic structure from which complex numbers emerge
    as a necessity (proven in [IRH-MATH-2025-01]).
    
    Operations:
        compose: Non-commutative composition T1 ∘ T2
        interference: Coherent interference of transformation paths
        unitary_check: Verify information conservation
        
    TODO v16.0: Full implementation requires [IRH-MATH-2025-01] §2-3
    """
    
    @staticmethod
    def compose(
        state1: AlgorithmicHolonomicState,
        state2: AlgorithmicHolonomicState,
        transformation: str = "sequential"
    ) -> AlgorithmicHolonomicState:
        """
        Compose two AHS via non-commutative transformation.
        
        The phase of the result depends on transformation path, inherently
        requiring complex algebra for coherent composition.
        
        TODO v16.0: Implement path-dependent phase composition from
        [IRH-MATH-2025-01] Theorem 1.1
        
        Args:
            state1: First AHS
            state2: Second AHS  
            transformation: Type of composition rule
            
        Returns:
            Composed AHS with path-dependent phase
            
        Raises:
            NotImplementedError: Awaiting [IRH-MATH-2025-01]
        """
        raise NotImplementedError(
            "v16.0: Requires non-commutative composition rules from [IRH-MATH-2025-01]"
        )
        
    @staticmethod
    def interference(
        paths: list[AlgorithmicHolonomicState]
    ) -> complex:
        """
        Compute coherent interference amplitude for multiple transformation paths.
        
        This is where complex numbers become NECESSARY (not assumed).
        
        TODO v16.0: Implement using [IRH-MATH-2025-01] Theorem 1.2
        
        Args:
            paths: List of alternative AHS paths leading to same final state
            
        Returns:
            Complex interference amplitude
            
        References:
            [IRH-MATH-2025-01] Theorem 1.2: Necessity of complex algebra
        """
        raise NotImplementedError(
            "v16.0: Requires interference formula from [IRH-MATH-2025-01]"
        )


def create_ahs_network(
    N: int,
    phase_distribution: str = "uniform",
    info_distribution: str = "random",
    seed: Optional[int] = None
) -> list[AlgorithmicHolonomicState]:
    """
    Create a network of N Algorithmic Holonomic States.
    
    v16.0: This initializes the ontological substrate before ARO optimization.
    
    Args:
        N: Number of AHS to create
        phase_distribution: How to initialize phases ("uniform", "quantized")
        info_distribution: How to generate binary strings ("random", "structured")
        seed: Random seed for reproducibility
        
    Returns:
        List of N AlgorithmicHolonomicState objects
        
    TODO v16.0:
        - Add structured information generation
        - Implement phase initialization strategies from [IRH-COMP-2025-02]
        - Add validation of network properties
    """
    rng = np.random.default_rng(seed)
    
    states = []
    for i in range(N):
        # Generate random binary string (10-20 bits for demonstration)
        bit_length = rng.integers(10, 21)
        binary_str = ''.join(str(rng.integers(0, 2)) for _ in range(bit_length))
        
        # Initialize phase
        if phase_distribution == "uniform":
            phase = rng.uniform(0, 2 * np.pi)
        elif phase_distribution == "quantized":
            # TODO: Implement phase quantization from Theorem 2.1
            phase = rng.uniform(0, 2 * np.pi)
        else:
            raise ValueError(f"Unknown phase_distribution: {phase_distribution}")
            
        states.append(AlgorithmicHolonomicState(binary_str, phase))
        
    return states


# Version and status
__version__ = "16.0.0-dev"
__status__ = "PLACEHOLDER - Requires [IRH-MATH-2025-01]"

__all__ = [
    "AlgorithmicHolonomicState",
    "AHSAlgebra",
    "create_ahs_network",
]
