"""
Enhanced Algorithmic Holonomic States (AHS) - IRH v16.0

Extends the v15.0 AHS implementation with:
- Explicit holonomic_phase storage and tracking
- Integration with certified numerics for precision tracking
- Non-commutative algebraic operations
- Phase coherence quantification with error bounds

References:
- IRH v16.0 Axiom 0: Algorithmic Holonomic Substrate
- [IRH-MATH-2025-01] Section 1: Non-Commutative AHS Algebra
"""

import numpy as np
from typing import Optional, List, Union, Tuple
from dataclasses import dataclass, field
from numpy.typing import NDArray

# Import v16.0 certified numerics
from ..numerics import CertifiedValue, ErrorBudget


@dataclass
class AlgorithmicHolonomicStateV16:
    """
    Enhanced AHS with v16.0 precision tracking.
    
    Attributes
    ----------
    info_content : bytes
        Finite binary string representing informational content (b_i).
    holonomic_phase : CertifiedValue
        Intrinsic phase degree of freedom φ_i ∈ [0, 2π) with certified error.
    state_id : int, optional
        Unique identifier for this state in the network.
    metadata : dict
        Additional state metadata (creation timestamp, lineage, etc).
        
    Notes
    -----
    v16.0 Enhancement: Phase is stored as CertifiedValue to track
    numerical precision and propagate errors through algebraic operations.
    
    The complex nature arises from non-commutative algebra of elementary
    algorithmic transformations (Axiom 0).
    
    References
    ----------
    IRH v16.0 Axiom 0: Algorithmic Holonomic Substrate
    """
    
    info_content: bytes
    holonomic_phase: CertifiedValue
    state_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize the state."""
        # Ensure phase is CertifiedValue
        if not isinstance(self.holonomic_phase, CertifiedValue):
            # Convert float to CertifiedValue with machine precision error
            eps = np.finfo(np.float64).eps
            self.holonomic_phase = CertifiedValue.from_value_and_error(
                self.holonomic_phase,
                eps,
                "initial_phase"
            )
        
        # Normalize phase to [0, 2π) while preserving error bound
        normalized_value = self.holonomic_phase.value % (2 * np.pi)
        self.holonomic_phase = CertifiedValue.from_value_and_error(
            normalized_value,
            self.holonomic_phase.error,
            self.holonomic_phase.source
        )
    
    @classmethod
    def from_bytes_and_phase(
        cls,
        info_content: bytes,
        phase: float,
        phase_error: Optional[float] = None,
        state_id: Optional[int] = None
    ) -> 'AlgorithmicHolonomicStateV16':
        """
        Create AHS from bytes and phase with optional error specification.
        
        Parameters
        ----------
        info_content : bytes
            Binary string content.
        phase : float
            Holonomic phase value in radians.
        phase_error : float, optional
            Absolute error in phase. If None, uses machine epsilon.
        state_id : int, optional
            State identifier.
            
        Returns
        -------
        ahs : AlgorithmicHolonomicStateV16
            New AHS instance.
        """
        if phase_error is None:
            phase_error = np.finfo(np.float64).eps
        
        certified_phase = CertifiedValue.from_value_and_error(
            phase,
            phase_error,
            "explicit_phase"
        )
        
        return cls(
            info_content=info_content,
            holonomic_phase=certified_phase,
            state_id=state_id
        )
    
    def to_complex_amplitude(self) -> Tuple[complex, float]:
        """
        Convert to complex amplitude representation with error.
        
        Returns
        -------
        amplitude : complex
            Unit complex number exp(i φ_i).
        amplitude_error : float
            Absolute error in complex amplitude magnitude.
            
        Notes
        -----
        Error in phase propagates to error in complex amplitude:
        |exp(iφ + iδφ) - exp(iφ)| ≈ |δφ| for small δφ
        """
        amplitude = np.exp(1j * self.holonomic_phase.value)
        
        # Error propagation: d|exp(iφ)|/dφ = 0 (magnitude is constant)
        # But real/imag parts have error: δRe ≈ sin(φ)δφ, δIm ≈ cos(φ)δφ
        amplitude_error = self.holonomic_phase.error
        
        return amplitude, amplitude_error
    
    def phase_difference_to(
        self,
        other: 'AlgorithmicHolonomicStateV16'
    ) -> CertifiedValue:
        """
        Compute minimal phase shift to coherently transform to another state.
        
        Parameters
        ----------
        other : AlgorithmicHolonomicStateV16
            Target state.
            
        Returns
        -------
        phase_diff : CertifiedValue
            Minimal computational phase shift in [0, 2π) with error bound.
            
        Notes
        -----
        This is the foundation for φ_ij in Algorithmic Coherence Weights.
        Errors from both phases combine in the difference.
        """
        diff_value = (other.holonomic_phase.value - self.holonomic_phase.value) % (2 * np.pi)
        
        # Errors add in subtraction
        diff_error = self.holonomic_phase.error + other.holonomic_phase.error
        
        return CertifiedValue.from_value_and_error(
            diff_value,
            diff_error,
            f"phase_diff({self.state_id},{other.state_id})"
        )
    
    def phase_coherence_with(
        self,
        other: 'AlgorithmicHolonomicStateV16'
    ) -> CertifiedValue:
        """
        Compute phase coherence |exp(i(φ_j - φ_i))| magnitude.
        
        Parameters
        ----------
        other : AlgorithmicHolonomicStateV16
            Other state.
            
        Returns
        -------
        coherence : CertifiedValue
            Phase coherence measure in [0, 1].
            
        Notes
        -----
        For pure phases, coherence magnitude is always 1.
        This becomes non-trivial when generalized to mixed states.
        """
        phase_diff = self.phase_difference_to(other)
        
        # For pure states, coherence is exp(i*phase_diff)
        # Magnitude is 1, but we track the phase uncertainty
        coherence_value = 1.0
        
        # Error in coherence magnitude is approximately the phase error
        coherence_error = phase_diff.error
        
        return CertifiedValue.from_value_and_error(
            coherence_value,
            coherence_error,
            f"coherence({self.state_id},{other.state_id})"
        )
    
    def compute_non_commutative_product(
        self,
        other: 'AlgorithmicHolonomicStateV16',
        order: str = 'ij'
    ) -> 'AlgorithmicHolonomicStateV16':
        """
        Compute non-commutative product of AHS (algebraic operation).
        
        Parameters
        ----------
        other : AlgorithmicHolonomicStateV16
            Second operand.
        order : str
            'ij' for self * other, 'ji' for other * self.
            
        Returns
        -------
        product : AlgorithmicHolonomicStateV16
            Resulting state with combined phase.
            
        Notes
        -----
        AHS algebra is non-commutative: T_i ∘ T_j ≠ T_j ∘ T_i in general.
        Phase accumulates: φ_product = φ_i + φ_j (mod 2π).
        Information content concatenates (with appropriate encoding).
        """
        if order == 'ij':
            # self * other
            first, second = self, other
        else:
            # other * self
            first, second = other, self
        
        # Phase addition (mod 2π) with error propagation
        combined_phase_value = (first.holonomic_phase.value + second.holonomic_phase.value) % (2 * np.pi)
        combined_phase_error = first.holonomic_phase.error + second.holonomic_phase.error
        
        combined_phase = CertifiedValue.from_value_and_error(
            combined_phase_value,
            combined_phase_error,
            f"product_{order}"
        )
        
        # Information content concatenation
        # Note: Real implementation would use proper compression/encoding
        combined_info = first.info_content + second.info_content
        
        return AlgorithmicHolonomicStateV16(
            info_content=combined_info,
            holonomic_phase=combined_phase,
            state_id=None,  # New composite state
            metadata={'operation': 'product', 'order': order}
        )
    
    def __repr__(self) -> str:
        """String representation."""
        info_len = len(self.info_content)
        phase_str = f"{self.holonomic_phase.value:.6f} ± {self.holonomic_phase.error:.2e}"
        return f"AHSv16(id={self.state_id}, info={info_len}B, φ={phase_str})"


def create_ahs_network_v16(
    N: int,
    phase_distribution: str = 'uniform',
    phase_error_bound: float = 1e-12,
    rng: Optional[np.random.Generator] = None
) -> List[AlgorithmicHolonomicStateV16]:
    """
    Create a network of N Algorithmic Holonomic States with certified precision.
    
    Parameters
    ----------
    N : int
        Number of states to create.
    phase_distribution : str
        Distribution for initial phases:
        - 'uniform': Uniform in [0, 2π)
        - 'gaussian': Gaussian centered at π
        - 'zero': All phases initialized to 0
    phase_error_bound : float
        Initial phase error bound (default: 1e-12 for high precision).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
        
    Returns
    -------
    states : List[AlgorithmicHolonomicStateV16]
        List of N AHS with unique IDs and certified phases.
        
    Notes
    -----
    This is the v16.0 enhanced version with certified precision tracking.
    Each state's phase error contributes to the total error budget.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    states = []
    
    for i in range(N):
        # Generate phase based on distribution
        if phase_distribution == 'uniform':
            phase = rng.uniform(0, 2 * np.pi)
        elif phase_distribution == 'gaussian':
            phase = rng.normal(np.pi, np.pi / 4) % (2 * np.pi)
        elif phase_distribution == 'zero':
            phase = 0.0
        else:
            raise ValueError(f"Unknown phase distribution: {phase_distribution}")
        
        # Generate minimal info content (could be more sophisticated)
        info_content = f"AHS_{i}".encode('utf-8')
        
        # Create state with certified phase
        state = AlgorithmicHolonomicStateV16.from_bytes_and_phase(
            info_content=info_content,
            phase=phase,
            phase_error=phase_error_bound,
            state_id=i
        )
        
        states.append(state)
    
    return states


def validate_ahs_network_precision(
    states: List[AlgorithmicHolonomicStateV16],
    required_phase_precision: int = 12
) -> Tuple[bool, ErrorBudget]:
    """
    Validate that AHS network meets precision requirements.
    
    Parameters
    ----------
    states : List[AlgorithmicHolonomicStateV16]
        Network of AHS to validate.
    required_phase_precision : int
        Required decimal places for phase precision.
        
    Returns
    -------
    is_valid : bool
        True if all states meet precision requirement.
    budget : ErrorBudget
        Combined error budget for the network.
    """
    from ..numerics import ErrorBudget, create_error_budget
    
    max_phase_error = 0.0
    total_phase_error_squared = 0.0
    
    for state in states:
        phase_error = state.holonomic_phase.error
        max_phase_error = max(max_phase_error, phase_error)
        total_phase_error_squared += phase_error**2
    
    # RMS phase error
    rms_phase_error = np.sqrt(total_phase_error_squared / len(states))
    
    # Check precision requirement
    required_error = 10 ** (-required_phase_precision)
    is_valid = max_phase_error <= required_error
    
    # Create error budget
    budget = ErrorBudget(
        numerical_error=rms_phase_error,
        statistical_error=0.0,  # No statistical sampling in initialization
        finite_size_error=1.0 / np.sqrt(len(states)),  # O(1/√N) scaling
        theoretical_error=0.0,
        metadata={
            'n_states': len(states),
            'max_phase_error': max_phase_error,
            'rms_phase_error': rms_phase_error,
            'required_precision': required_phase_precision
        }
    )
    
    return is_valid, budget


# Example usage
if __name__ == "__main__":
    print("IRH v16.0 Enhanced AHS Example")
    print("=" * 60)
    
    # Create a small network
    states = create_ahs_network_v16(
        N=10,
        phase_distribution='uniform',
        phase_error_bound=1e-12,
        rng=np.random.default_rng(42)
    )
    
    print(f"\nCreated {len(states)} AHS:")
    for state in states[:3]:
        print(f"  {state}")
    
    # Validate precision
    is_valid, budget = validate_ahs_network_precision(states, required_phase_precision=12)
    print(f"\nPrecision Validation: {'PASS' if is_valid else 'FAIL'}")
    print(f"Error Budget: {budget}")
    
    # Demonstrate non-commutative product
    print("\n" + "=" * 60)
    print("Non-Commutative Algebra:")
    s0, s1 = states[0], states[1]
    
    product_ij = s0.compute_non_commutative_product(s1, order='ij')
    product_ji = s1.compute_non_commutative_product(s0, order='ji')
    
    print(f"\ns0 * s1 = {product_ij}")
    print(f"s1 * s0 = {product_ji}")
    print(f"Commutator: Δφ = {abs(product_ij.holonomic_phase.value - product_ji.holonomic_phase.value):.6f}")
