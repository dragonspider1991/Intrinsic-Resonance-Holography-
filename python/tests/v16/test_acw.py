"""
Unit tests for Algorithmic Coherence Weights (ACW) - Axiom 1.

Tests NCD calculator and ACW computation per IRHv16.md lines 66-83.

THEORETICAL COMPLIANCE:
    Tests validate against docs/manuscripts/IRHv16.md Axiom 1
    - Lines 66-83: W_ij ∈ ℂ from NCD and phase shift
    - |W_ij| computed from NCD
    - arg(W_ij) = φ_j - φ_i (mod 2π)
"""

import pytest
import numpy as np
from irh.core.v16.ahs import AlgorithmicHolonomicState
from irh.core.v16.acw import (
    compute_ncd_magnitude,
    compute_phase_shift,
    compute_acw,
    AlgorithmicCoherenceWeight
)


class TestNCDMagnitude:
    """Test Normalized Compression Distance computation."""
    
    def test_identical_strings(self):
        """
        Test NCD(x, x) = 0 for identical strings.
        
        Per IRHv16.md: NCD measures algorithmic distance.
        Identical strings have zero distance.
        """
        binary1 = "10101010"
        binary2 = "10101010"
        
        ncd, error = compute_ncd_magnitude(binary1, binary2)
        
        assert ncd == 0.0, "NCD of identical strings must be 0"
        assert error >= 0, "Error bound must be non-negative"
    
    def test_completely_different_strings(self):
        """Test NCD for completely different strings."""
        binary1 = "00000000"
        binary2 = "11111111"
        
        ncd, error = compute_ncd_magnitude(binary1, binary2)
        
        # NCD is bounded in [0, 1]
        assert 0 <= ncd <= 1.0, f"NCD must be in [0,1], got {ncd}"
        assert error >= 0, "Error bound must be non-negative"
        
    def test_similar_strings(self):
        """Test NCD for similar strings (one bit different)."""
        binary1 = "10101010"
        binary2 = "10101011"  # Last bit different
        
        ncd, error = compute_ncd_magnitude(binary1, binary2)
        
        # Similar strings should have low NCD
        assert 0 < ncd < 1.0, "Similar strings should have 0 < NCD < 1"
        assert error >= 0
        
    def test_empty_strings(self):
        """Test NCD handles empty strings."""
        binary1 = ""
        binary2 = ""
        
        ncd, error = compute_ncd_magnitude(binary1, binary2)
        
        # Both empty should have NCD = 0
        assert ncd == 0.0
        
    def test_ncd_symmetry(self):
        """Test NCD(x, y) ≈ NCD(y, x) (approximate symmetry)."""
        binary1 = "101010"
        binary2 = "110011"
        
        ncd_xy, _ = compute_ncd_magnitude(binary1, binary2)
        ncd_yx, _ = compute_ncd_magnitude(binary2, binary1)
        
        # NCD should be approximately symmetric (may vary slightly with compression)
        assert np.isclose(ncd_xy, ncd_yx, rtol=0.15), \
            f"NCD should be approximately symmetric: {ncd_xy} vs {ncd_yx}"
        
    def test_ncd_bounds(self):
        """Test NCD is always in [0, 1] for various strings."""
        test_pairs = [
            ("1", "0"),
            ("101", "010"),
            ("111111", "000000"),
            ("1010", "1010"),
            ("random", "string"),
        ]
        
        for b1, b2 in test_pairs:
            ncd, _ = compute_ncd_magnitude(b1, b2)
            assert 0 <= ncd <= 1.0, f"NCD out of bounds for ({b1}, {b2}): {ncd}"


class TestPhaseShift:
    """Test holonomic phase shift computation."""
    
    def test_zero_phase_shift(self):
        """Test phase shift when both states have same phase."""
        state_i = AlgorithmicHolonomicState("101", 0.5)
        state_j = AlgorithmicHolonomicState("110", 0.5)
        
        phase_shift = compute_phase_shift(state_i, state_j)
        
        assert np.isclose(phase_shift, 0.0), "Same phases should give zero shift"
        
    def test_pi_phase_shift(self):
        """Test π phase shift."""
        state_i = AlgorithmicHolonomicState("101", 0.0)
        state_j = AlgorithmicHolonomicState("110", np.pi)
        
        phase_shift = compute_phase_shift(state_i, state_j)
        
        assert np.isclose(phase_shift, np.pi), f"Expected π, got {phase_shift}"
        
    def test_phase_shift_wraparound(self):
        """
        Test phase shift wraps around 2π.
        
        Per IRHv16.md: Phases are in [0, 2π), shift is mod 2π.
        """
        state_i = AlgorithmicHolonomicState("101", 0.1)
        state_j = AlgorithmicHolonomicState("110", 2 * np.pi - 0.1)
        
        phase_shift = compute_phase_shift(state_i, state_j)
        
        # Shift should be close to 2π - 0.2 ≈ 6.08
        assert 0 <= phase_shift < 2 * np.pi, "Phase shift must be in [0, 2π)"
        
    def test_phase_shift_in_range(self):
        """Test phase shift is always in [0, 2π)."""
        import random
        random.seed(42)
        
        for _ in range(10):
            phi_i = random.uniform(0, 2 * np.pi)
            phi_j = random.uniform(0, 2 * np.pi)
            
            state_i = AlgorithmicHolonomicState("1", phi_i)
            state_j = AlgorithmicHolonomicState("0", phi_j)
            
            phase_shift = compute_phase_shift(state_i, state_j)
            
            assert 0 <= phase_shift < 2 * np.pi, \
                f"Phase shift {phase_shift} out of range [0, 2π)"


class TestACWComputation:
    """Test complete Algorithmic Coherence Weight computation."""
    
    def test_acw_basic(self):
        """
        Test basic ACW computation.
        
        Per IRHv16.md Axiom 1: W_ij = |W_ij| e^{i·arg(W_ij)}
        where |W_ij| from NCD, arg(W_ij) = φ_j - φ_i
        """
        state_i = AlgorithmicHolonomicState("1010", 0.5)
        state_j = AlgorithmicHolonomicState("1100", 1.0)
        
        acw = compute_acw(state_i, state_j)
        
        assert isinstance(acw, AlgorithmicCoherenceWeight)
        assert acw.magnitude >= 0
        assert 0 <= acw.phase < 2 * np.pi
        assert isinstance(acw.complex_value, complex)
        
    def test_acw_magnitude_from_ncd(self):
        """Test ACW magnitude equals NCD."""
        state_i = AlgorithmicHolonomicState("101", 0.0)
        state_j = AlgorithmicHolonomicState("110", 0.0)
        
        acw = compute_acw(state_i, state_j)
        ncd, _ = compute_ncd_magnitude(state_i.binary_string, state_j.binary_string)
        
        assert np.isclose(acw.magnitude, ncd), \
            "ACW magnitude should equal NCD"
            
    def test_acw_phase_from_shift(self):
        """Test ACW phase equals holonomic phase shift."""
        state_i = AlgorithmicHolonomicState("101", 0.5)
        state_j = AlgorithmicHolonomicState("110", 1.5)
        
        acw = compute_acw(state_i, state_j)
        phase_shift = compute_phase_shift(state_i, state_j)
        
        assert np.isclose(acw.phase, phase_shift), \
            "ACW phase should equal phase shift"
            
    def test_acw_complex_value(self):
        """Test ACW complex value W_ij = |W_ij| e^{i·φ}."""
        state_i = AlgorithmicHolonomicState("1", np.pi/4)
        state_j = AlgorithmicHolonomicState("0", np.pi/2)
        
        acw = compute_acw(state_i, state_j)
        
        # Verify complex value matches magnitude and phase
        expected = acw.magnitude * np.exp(1j * acw.phase)
        
        assert np.isclose(acw.complex_value, expected), \
            "Complex value should be |W| e^{iφ}"
            
    def test_acw_identical_states(self):
        """Test ACW for identical states (NCD = 0)."""
        state_i = AlgorithmicHolonomicState("10101010", 0.5)
        state_j = AlgorithmicHolonomicState("10101010", 0.5)
        
        acw = compute_acw(state_i, state_j)
        
        # Identical binary strings → NCD = 0 → |W_ij| = 0
        assert acw.magnitude == 0.0, "Identical states should have zero ACW magnitude"
        
    def test_acw_error_bounds(self):
        """Test ACW includes error bounds."""
        state_i = AlgorithmicHolonomicState("101", 0.0)
        state_j = AlgorithmicHolonomicState("110", 0.0)
        
        acw = compute_acw(state_i, state_j)
        
        assert acw.error_bound >= 0, "Error bound must be non-negative"


class TestAlgorithmicCoherenceWeight:
    """Test ACW dataclass."""
    
    def test_acw_dataclass_creation(self):
        """Test ACW can be created directly."""
        acw = AlgorithmicCoherenceWeight(
            magnitude=0.5,
            phase=np.pi/4,
            error_bound=0.01
        )
        
        assert acw.magnitude == 0.5
        assert acw.phase == np.pi/4
        
    def test_acw_complex_value_property(self):
        """Test complex_value property computes correctly."""
        acw = AlgorithmicCoherenceWeight(
            magnitude=1.0,
            phase=np.pi/2,
            error_bound=0.0
        )
        
        # Should be approximately i
        assert np.isclose(acw.complex_value, 1j), \
            f"Expected i, got {acw.complex_value}"


class TestTheoreticalCompliance:
    """
    Test compliance with IRHv16.md Axiom 1 specifications.
    
    References:
        docs/manuscripts/IRHv16.md lines 66-83: Axiom 1
    """
    
    def test_axiom1_complex_valued(self):
        """Test W_ij ∈ ℂ per IRHv16.md line 68."""
        state_i = AlgorithmicHolonomicState("101", 0.5)
        state_j = AlgorithmicHolonomicState("110", 1.0)
        
        acw = compute_acw(state_i, state_j)
        
        # W_ij must be complex-valued
        assert isinstance(acw.complex_value, (complex, np.complexfloating)), \
            "W_ij must be complex-valued per Axiom 1"
            
    def test_axiom1_magnitude_from_ncd(self):
        """Test |W_ij| derived from NCD per IRHv16.md."""
        state_i = AlgorithmicHolonomicState("1010", 0.0)
        state_j = AlgorithmicHolonomicState("0101", 0.0)
        
        acw = compute_acw(state_i, state_j)
        ncd, _ = compute_ncd_magnitude(state_i.binary_string, state_j.binary_string)
        
        # Per Axiom 1: |W_ij| from NCD
        assert np.isclose(acw.magnitude, ncd), \
            "Axiom 1 requires |W_ij| from NCD"
            
    def test_axiom1_phase_from_holonomy(self):
        """Test arg(W_ij) from holonomic phases per IRHv16.md."""
        state_i = AlgorithmicHolonomicState("1", 0.0)
        state_j = AlgorithmicHolonomicState("0", np.pi)
        
        acw = compute_acw(state_i, state_j)
        
        # Per Axiom 1: arg(W_ij) = φ_j - φ_i
        expected_phase = np.pi
        assert np.isclose(acw.phase, expected_phase), \
            f"Axiom 1 requires arg(W_ij) = φ_j - φ_i, got {acw.phase}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
