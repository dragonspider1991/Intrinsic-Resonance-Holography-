"""
Algorithmic Holonomic States (AHS) - Core Data Structures for IRH v15.0

Implements Axiom 0: The fundamental ontological primitive of reality consists of 
Algorithmic Holonomic States (AHS), each embodying:
- Informational content (binary string)
- Intrinsic holonomic phase degree of freedom

This module provides the foundational data structures and operations on AHS
that enable the non-circular derivation of quantum mechanics and gauge theory.

References: IRH v15.0 §1, Axiom 0
"""

import numpy as np
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class AlgorithmicHolonomicState:
    """
    Fundamental information process with intrinsic complex nature.
    
    Attributes
    ----------
    info_content : bytes or str
        Finite binary string representing informational content (b_i).
    holonomic_phase : float
        Intrinsic phase degree of freedom φ_i ∈ [0, 2π).
        This arises from non-commutative algebra of elementary algorithmic
        transformations (Axiom 0).
    state_id : int, optional
        Unique identifier for this state in the network.
        
    Notes
    -----
    The complex nature is NOT imposed by quantum mechanics - it is derived
    from the non-commutative structure of algorithmic processing itself.
    Two states are distinguishable if info_content OR holonomic_phase differ.
    
    References
    ----------
    IRH v15.0 Axiom 0: Algorithmic Holonomic Substrate
    """
    
    info_content: Union[bytes, str]
    holonomic_phase: float
    state_id: Optional[int] = None
    
    def __post_init__(self):
        """Validate and normalize the state."""
        # Normalize phase to [0, 2π)
        self.holonomic_phase = self.holonomic_phase % (2 * np.pi)
        
        # Convert string to bytes if needed
        if isinstance(self.info_content, str):
            self.info_content = self.info_content.encode('utf-8')
    
    def to_complex_amplitude(self) -> complex:
        """
        Convert to complex amplitude representation.
        
        Returns
        -------
        amplitude : complex
            Unit complex number exp(i φ_i) representing the phase.
            
        Notes
        -----
        The magnitude is 1 because each AHS has equal "existence weight"
        in the fundamental substrate. Coherence weights arise from relations
        between states (Axiom 1).
        """
        return np.exp(1j * self.holonomic_phase)
    
    def phase_difference_to(self, other: 'AlgorithmicHolonomicState') -> float:
        """
        Compute minimal phase shift to coherently transform to another state.
        
        Parameters
        ----------
        other : AlgorithmicHolonomicState
            Target state.
            
        Returns
        -------
        phase_diff : float
            Minimal computational phase shift in [0, 2π).
            
        Notes
        -----
        This is the foundation for φ_ij in Algorithmic Coherence Weights.
        The phase arises from the non-commutative sequence of operations
        linking the two states in abstract computational space.
        """
        diff = (other.holonomic_phase - self.holonomic_phase) % (2 * np.pi)
        return diff
    
    def __eq__(self, other: 'AlgorithmicHolonomicState') -> bool:
        """Two AHS are equal if both info_content and phase match."""
        if not isinstance(other, AlgorithmicHolonomicState):
            return False
        return (self.info_content == other.info_content and 
                abs(self.holonomic_phase - other.holonomic_phase) < 1e-10)
    
    def __hash__(self) -> int:
        """Hash based on info_content and quantized phase."""
        # Quantize phase for hashing
        phase_quantized = int(self.holonomic_phase * 1e6) % int(2 * np.pi * 1e6)
        return hash((self.info_content, phase_quantized))
    
    def __repr__(self) -> str:
        """String representation."""
        info_preview = self.info_content[:20] if len(self.info_content) > 20 else self.info_content
        return (f"AHS(id={self.state_id}, info={info_preview}..., "
                f"φ={self.holonomic_phase:.6f})")


class AlgorithmicCoherenceWeight:
    """
    Complex-valued weight representing coherent transfer potential between AHS.
    
    Implements Axiom 1: W_ij = |W_ij| exp(i φ_ij)
    where:
    - |W_ij| quantifies algorithmic compressibility (NCD)
    - φ_ij quantifies minimal computational phase shift
    
    Attributes
    ----------
    magnitude : float
        Algorithmic correlation strength |W_ij| ∈ [0, 1].
    phase : float
        Coherent transfer phase φ_ij ∈ [0, 2π).
    source_id : int, optional
        ID of source AHS.
    target_id : int, optional
        ID of target AHS.
        
    References
    ----------
    IRH v15.0 Axiom 1: Algorithmic Relationality as Coherent Transfer Potential
    """
    
    def __init__(
        self,
        magnitude: float,
        phase: float,
        source_id: Optional[int] = None,
        target_id: Optional[int] = None
    ):
        """Initialize algorithmic coherence weight."""
        self.magnitude = np.clip(magnitude, 0.0, 1.0)
        self.phase = phase % (2 * np.pi)
        self.source_id = source_id
        self.target_id = target_id
    
    def to_complex(self) -> complex:
        """Convert to complex number representation."""
        return self.magnitude * np.exp(1j * self.phase)
    
    @classmethod
    def from_complex(
        cls,
        value: complex,
        source_id: Optional[int] = None,
        target_id: Optional[int] = None
    ) -> 'AlgorithmicCoherenceWeight':
        """
        Construct from complex value.
        
        Parameters
        ----------
        value : complex
            Complex weight.
        source_id, target_id : int, optional
            State identifiers.
            
        Returns
        -------
        weight : AlgorithmicCoherenceWeight
        """
        magnitude = np.abs(value)
        phase = np.angle(value) % (2 * np.pi)
        return cls(magnitude, phase, source_id, target_id)
    
    def conjugate(self) -> 'AlgorithmicCoherenceWeight':
        """
        Return complex conjugate (for reverse traversal).
        
        Notes
        -----
        W_ji = W_ij^* ensures phase consistency for closed loops.
        """
        return AlgorithmicCoherenceWeight(
            self.magnitude,
            -self.phase % (2 * np.pi),
            source_id=self.target_id,
            target_id=self.source_id
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"ACW({self.source_id}→{self.target_id}: "
                f"|W|={self.magnitude:.4f}, φ={self.phase:.4f})")


def create_ahs_network(
    N: int,
    info_generator: Optional[callable] = None,
    phase_distribution: str = 'uniform',
    rng: Optional[np.random.Generator] = None
) -> List[AlgorithmicHolonomicState]:
    """
    Create a collection of Algorithmic Holonomic States.
    
    Parameters
    ----------
    N : int
        Number of states.
    info_generator : callable, optional
        Function that generates informational content.
        If None, uses random binary strings.
    phase_distribution : str, default 'uniform'
        Distribution for initial phases: 'uniform', 'normal', 'fixed'.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
        
    Returns
    -------
    states : List[AlgorithmicHolonomicState]
        Collection of N algorithmic holonomic states.
        
    Notes
    -----
    The specific initial phases don't matter much - ARO will optimize them
    to the Cosmic Fixed Point (Theorem 10.1).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    states = []
    
    for i in range(N):
        # Generate informational content
        if info_generator is None:
            # Random binary string of length proportional to log(N)
            content_length = max(8, int(np.log2(N + 1)))
            info = rng.bytes(content_length)
        else:
            info = info_generator(i)
        
        # Generate initial phase
        if phase_distribution == 'uniform':
            phase = rng.uniform(0, 2 * np.pi)
        elif phase_distribution == 'normal':
            phase = rng.normal(np.pi, np.pi/2) % (2 * np.pi)
        elif phase_distribution == 'fixed':
            phase = 0.0
        else:
            phase = rng.uniform(0, 2 * np.pi)
        
        state = AlgorithmicHolonomicState(
            info_content=info,
            holonomic_phase=phase,
            state_id=i
        )
        states.append(state)
    
    return states


# Export key classes and functions
__all__ = [
    'AlgorithmicHolonomicState',
    'AlgorithmicCoherenceWeight',
    'create_ahs_network'
]
