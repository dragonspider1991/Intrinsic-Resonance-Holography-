"""
Harmony Functional for IRH v16.0

Implements the unique action functional S_H[G] that governs Adaptive Resonance
Optimization (ARO) and the emergence of physical law.

THEORETICAL COMPLIANCE:
    This implementation strictly follows docs/manuscripts/IRHv16.md §4
    - Lines 254-277: Theorem 4.1 - Unique Action Functional
    - Line 266: S_H[G] = Tr(ℒ²) / [det'(ℒ)]^{C_H}
    - Line 275: C_H = 0.045935703598(1) - Universal critical exponent
    - Lines 268: ℒ is the Interference Matrix (complex graph Laplacian)
    - Lines 272: det'(ℒ) excludes zero eigenvalues
    
Key Concepts:
    - S_H quantifies global efficiency of algorithmic information processing
    - Intensive scaling with network size (action density)
    - Renormalization-group invariant
    - Maximization yields Cosmic Fixed Point
    
Formula Breakdown:
    - Numerator Tr(ℒ²): Total coherent algorithmic information flow
    - Denominator det'(ℒ): Algorithmic configurational volume (Cymatic Complexity)
    - Exponent C_H: Universal critical exponent from phase transition to Harmonic Crystalization

References:
    docs/manuscripts/IRHv16.md:
        - §4 Theorem 4.1: Harmony Functional derivation
        - Lines 266-276: Formula and C_H derivation
        - Lines 282-299: ARO maximizes S_H
"""

from __future__ import annotations
from typing import Optional, Dict
import numpy as np
from numpy.typing import NDArray

# Import certified constant from numerics
# Try multiple import paths
C_H_CERTIFIED = None

try:
    # Try src/numerics first
    import sys
    from pathlib import Path
    repo_root = Path(__file__).parent.parent.parent.parent.parent
    sys.path.insert(0, str(repo_root / 'src'))
    from numerics import C_H_CERTIFIED
except ImportError:
    try:
        # Try relative import
        from ....numerics import C_H_CERTIFIED
    except ImportError:
        # Fallback: create simple certified value
        from dataclasses import dataclass
        
        @dataclass
        class CertifiedValueFallback:
            value: float
            error: float
            source: str
        
        C_H_CERTIFIED = CertifiedValueFallback(
            value=0.045935703598,
            error=1e-12,
            source="harmony_functional_rg_fixed_point"
        )

# Also try python/src location
try:
    from .crn import CymaticResonanceNetworkV16
except ImportError:
    # Handle import from different location
    pass


def compute_harmony_functional(
    crn: 'CymaticResonanceNetworkV16',
    use_certified_constant: bool = True,
    eigenvalue_threshold: float = 1e-10
) -> Dict[str, any]:
    """
    Compute the Harmony Functional S_H[G] for a Cymatic Resonance Network.
    
    Implements Theorem 4.1 from docs/manuscripts/IRHv16.md §4 (lines 254-277):
    
        S_H[G] = Tr(ℒ²) / [det'(ℒ)]^{C_H}
    
    Where:
        - ℒ: Complex graph Laplacian (Interference Matrix)
        - Tr(ℒ²): Total coherent algorithmic information flow
        - det'(ℒ): Regularized determinant (excluding zero eigenvalues)
        - C_H: Universal critical exponent = 0.045935703598
    
    Args:
        crn: CymaticResonanceNetworkV16 instance
        use_certified_constant: If True, use C_H from precision_constants.py
        eigenvalue_threshold: Threshold for considering eigenvalues as zero
        
    Returns:
        Dictionary containing:
            - S_H: Harmony Functional value
            - trace_L2: Tr(ℒ²)
            - det_prime: det'(ℒ) (excluding zero eigenvalues)
            - C_H: Value of C_H used
            - num_zero_eigenvalues: Number of zero eigenvalues excluded
            - eigenvalues: All eigenvalues of ℒ
            
    Raises:
        ValueError: If network not built or degenerate case
        
    Notes:
        This is a preliminary implementation for Phase 1. Full exascale
        implementation with distributed spectral solvers comes in Phase 2.
        
    References:
        docs/manuscripts/IRHv16.md §4 lines 254-277: Theorem 4.1
        PHASE_1_STATUS.md: Preliminary Harmony Functional task
    """
    # Get spectral properties from CRN
    props = crn.compute_spectral_properties()
    
    # Extract values
    trace_L2 = props['trace_L2']
    det_prime = props['det_prime']
    eigenvalues = props['eigenvalues']
    num_zero_eigs = props['num_zero_eigenvalues']
    
    # Get C_H value
    if use_certified_constant:
        C_H = C_H_CERTIFIED.value
    else:
        C_H = 0.045935703598  # Fallback to hardcoded value
    
    # Compute S_H
    # Handle complex trace and determinant
    trace_L2_mag = np.abs(trace_L2)
    det_prime_mag = np.abs(det_prime)
    
    if det_prime_mag < 1e-12:
        # Degenerate network - all eigenvalues zero or network disconnected
        raise ValueError(
            f"Degenerate network: det'(ℒ) = {det_prime_mag:.4e} is too small. "
            f"Network may be disconnected or have insufficient edges."
        )
    
    # S_H = Tr(ℒ²) / [det'(ℒ)]^{C_H}
    # Using magnitudes for numerical stability
    S_H = trace_L2_mag / (det_prime_mag ** C_H)
    
    return {
        'S_H': S_H,
        'trace_L2': trace_L2,
        'det_prime': det_prime,
        'C_H': C_H,
        'num_zero_eigenvalues': num_zero_eigs,
        'eigenvalues': eigenvalues,
        'trace_L2_magnitude': trace_L2_mag,
        'det_prime_magnitude': det_prime_mag,
    }


def validate_harmony_functional_properties(result: Dict[str, any]) -> bool:
    """
    Validate that Harmony Functional result has expected properties.
    
    Checks from IRHv16.md Theorem 4.1:
    - S_H is positive (measures information flow efficiency)
    - C_H matches theoretical value
    - Trace and determinant are computable
    
    Args:
        result: Output from compute_harmony_functional()
        
    Returns:
        True if all validations pass
        
    Raises:
        AssertionError: If validation fails
    """
    # S_H should be positive
    assert result['S_H'] > 0, f"S_H must be positive, got {result['S_H']}"
    
    # C_H should match theoretical value (from IRHv16.md line 275)
    expected_C_H = 0.045935703598
    assert abs(result['C_H'] - expected_C_H) < 1e-11, \
        f"C_H = {result['C_H']} does not match IRHv16.md value {expected_C_H}"
    
    # Trace should be non-zero for non-trivial networks
    assert abs(result['trace_L2']) > 0, "Tr(ℒ²) should be non-zero"
    
    # det' should be non-zero (already checked in compute function)
    assert abs(result['det_prime']) > 0, "det'(ℒ) should be non-zero"
    
    return True


class HarmonyFunctionalEvaluator:
    """
    Evaluator for computing and tracking Harmony Functional across iterations.
    
    Used by ARO to evaluate fitness of network configurations.
    
    Attributes:
        history: List of (iteration, S_H) tuples
        best_S_H: Best S_H value seen
        best_config: Configuration achieving best S_H
        
    References:
        docs/manuscripts/IRHv16.md §4 Definition 4.1: ARO maximizes S_H
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.history = []
        self.best_S_H = -np.inf
        self.best_config = None
        
    def evaluate(
        self,
        crn: 'CymaticResonanceNetworkV16',
        iteration: Optional[int] = None
    ) -> float:
        """
        Evaluate S_H for a network configuration.
        
        Args:
            crn: Network to evaluate
            iteration: Optional iteration number for tracking
            
        Returns:
            S_H value
        """
        try:
            result = compute_harmony_functional(crn)
            S_H = result['S_H']
            
            # Track history
            if iteration is not None:
                self.history.append((iteration, S_H))
            
            # Update best
            if S_H > self.best_S_H:
                self.best_S_H = S_H
                # Note: We don't store the full CRN to save memory
                # In production, would serialize to disk
                
            return S_H
            
        except ValueError as e:
            # Degenerate network - return very low fitness
            return -np.inf
    
    def get_convergence_metrics(self) -> Dict[str, any]:
        """
        Get convergence metrics for ARO.
        
        Returns:
            Dictionary with convergence statistics
        """
        if len(self.history) == 0:
            return {
                'num_evaluations': 0,
                'best_S_H': None,
                'mean_S_H': None,
                'std_S_H': None,
            }
        
        iterations, S_H_values = zip(*self.history)
        
        return {
            'num_evaluations': len(self.history),
            'best_S_H': self.best_S_H,
            'mean_S_H': np.mean(S_H_values),
            'std_S_H': np.std(S_H_values),
            'latest_S_H': S_H_values[-1] if S_H_values else None,
            'convergence_trend': self._compute_trend(),
        }
    
    def _compute_trend(self) -> str:
        """Compute whether S_H is increasing, decreasing, or stable."""
        if len(self.history) < 10:
            return "insufficient_data"
        
        # Look at last 10 iterations
        recent_values = [s for i, s in self.history[-10:]]
        early = np.mean(recent_values[:5])
        late = np.mean(recent_values[5:])
        
        if late > early * 1.01:
            return "increasing"
        elif late < early * 0.99:
            return "decreasing"
        else:
            return "stable"


__all__ = [
    "compute_harmony_functional",
    "validate_harmony_functional_properties",
    "HarmonyFunctionalEvaluator",
]

__version__ = "16.0.0-dev"
