"""
Unit tests for AlgorithmicHolonomicState (Axiom 0).

Tests the fundamental ontological primitive of IRH v16.0.
"""

import pytest
import numpy as np
from irh.core.v16.ahs import (
    AlgorithmicHolonomicState,
    create_ahs_network
)


class TestAlgorithmicHolonomicState:
    """Test AHS data structure and properties."""
    
    def test_basic_creation(self):
        """Test basic AHS creation."""
        ahs = AlgorithmicHolonomicState("101010", np.pi)
        assert ahs.binary_string == "101010"
        assert np.isclose(ahs.holonomic_phase, np.pi)
        assert ahs.information_content == 6
        
    def test_phase_normalization(self):
        """Test phase is normalized to [0, 2π)."""
        ahs = AlgorithmicHolonomicState("1", 3 * np.pi)
        assert 0 <= ahs.holonomic_phase < 2 * np.pi
        assert np.isclose(ahs.holonomic_phase, np.pi)
        
    def test_negative_phase_normalization(self):
        """Test negative phase normalization."""
        ahs = AlgorithmicHolonomicState("1", -np.pi/2)
        assert 0 <= ahs.holonomic_phase < 2 * np.pi
        assert np.isclose(ahs.holonomic_phase, 3*np.pi/2)
        
    def test_complex_amplitude(self):
        """Test complex amplitude e^{iφ}."""
        ahs = AlgorithmicHolonomicState("1", np.pi/4)
        amp = ahs.complex_amplitude
        assert np.isclose(abs(amp), 1.0)  # On unit circle
        assert np.isclose(np.angle(amp), np.pi/4)
        
    def test_complex_amplitude_various_phases(self):
        """Test complex amplitude for various phases."""
        phases = [0, np.pi/2, np.pi, 3*np.pi/2, 2 * np.pi - 1e-12]
        for phase in phases:
            ahs = AlgorithmicHolonomicState("1", phase)
            amp = ahs.complex_amplitude
            assert np.isclose(abs(amp), 1.0)
            wrapped_angle = np.mod(np.angle(amp), 2 * np.pi)
            assert np.isclose(wrapped_angle, np.mod(phase, 2 * np.pi))
            
    def test_invalid_binary_string_non_binary(self):
        """Test validation of binary string."""
        with pytest.raises(ValueError, match="only '0' and '1'"):
            AlgorithmicHolonomicState("012", 0.0)
            
    def test_invalid_binary_string_letters(self):
        """Test rejection of non-binary characters."""
        with pytest.raises(ValueError, match="only '0' and '1'"):
            AlgorithmicHolonomicState("abc", 0.0)
    
    def test_empty_binary_string(self):
        """Test rejection of empty binary string."""
        with pytest.raises(ValueError, match="cannot be empty"):
            AlgorithmicHolonomicState("", 0.0)
    
    def test_invalid_binary_string_type(self):
        """Test rejection of non-string binary_string."""
        with pytest.raises(TypeError, match="must be str"):
            AlgorithmicHolonomicState(101010, 0.0)
    
    def test_invalid_phase_type(self):
        """Test rejection of non-numeric phase."""
        with pytest.raises(TypeError, match="must be numeric"):
            AlgorithmicHolonomicState("101", "not a number")
            
    def test_equality(self):
        """Test AHS equality comparison."""
        ahs1 = AlgorithmicHolonomicState("101", 0.5)
        ahs2 = AlgorithmicHolonomicState("101", 0.5)
        ahs3 = AlgorithmicHolonomicState("101", 0.6)
        ahs4 = AlgorithmicHolonomicState("110", 0.5)
        
        assert ahs1 == ahs2
        assert ahs1 != ahs3  # Different phase
        assert ahs1 != ahs4  # Different info
        
    def test_equality_phase_tolerance(self):
        """Test equality with small phase differences."""
        ahs1 = AlgorithmicHolonomicState("101", 0.5)
        ahs2 = AlgorithmicHolonomicState("101", 0.5 + 1e-11)  # Within tolerance
        ahs3 = AlgorithmicHolonomicState("101", 0.5 + 1e-9)   # Outside tolerance
        
        assert ahs1 == ahs2
        assert ahs1 != ahs3
    
    def test_wrapped_phase_difference_pi(self):
        """Test that a π difference wraps to -π."""
        diff = AlgorithmicHolonomicState._wrapped_phase_difference(0.0, np.pi)
        assert np.isclose(diff, -np.pi)
        
    def test_hashing(self):
        """Test AHS can be used in sets/dicts."""
        ahs1 = AlgorithmicHolonomicState("101", 0.5)
        ahs2 = AlgorithmicHolonomicState("101", 0.5)
        
        ahs_set = {ahs1, ahs2}
        assert len(ahs_set) == 1  # Should be same
        
    def test_hashing_uniqueness(self):
        """Test different AHS have different hashes (usually)."""
        ahs1 = AlgorithmicHolonomicState("101", 0.5)
        ahs2 = AlgorithmicHolonomicState("110", 0.5)
        ahs3 = AlgorithmicHolonomicState("101", 0.6)
        
        # Different AHS should typically have different hashes
        assert hash(ahs1) != hash(ahs2)
        assert hash(ahs1) != hash(ahs3)
        
    def test_compute_complexity(self):
        """Test K_t complexity estimation."""
        ahs = AlgorithmicHolonomicState("0" * 100, 0.0)
        kt = ahs.compute_complexity()
        assert kt > 0
        # Highly compressible string should have low K_t
        assert kt < len("0" * 100) * 8  # Compressed < original
        
    def test_compute_complexity_random(self):
        """Test K_t for random-looking string."""
        # Alternating pattern is somewhat compressible
        ahs = AlgorithmicHolonomicState("01" * 50, 0.0)
        kt = ahs.compute_complexity()
        assert kt > 0
        # Should compress somewhat
        assert kt < len("01" * 50) * 8
        
    def test_complexity_auto_computed(self):
        """Test complexity is auto-computed on init."""
        ahs = AlgorithmicHolonomicState("1010", 0.0)
        assert ahs.complexity_Kt is not None
        assert ahs.complexity_Kt > 0
        
    def test_repr(self):
        """Test developer representation."""
        ahs = AlgorithmicHolonomicState("10101010", np.pi)
        repr_str = repr(ahs)
        assert "AHS" in repr_str
        assert "10101010" in repr_str or "..." in repr_str
        assert "φ=" in repr_str
        
    def test_str(self):
        """Test user-friendly string."""
        ahs = AlgorithmicHolonomicState("10101010", np.pi/2)
        str_repr = str(ahs)
        assert "AHS" in str_repr
        assert "8bits" in str_repr
        assert "φ=" in str_repr
        
    def test_information_content(self):
        """Test information_content property."""
        for length in [1, 10, 100]:
            ahs = AlgorithmicHolonomicState("0" * length, 0.0)
            assert ahs.information_content == length


class TestAHSNetwork:
    """Test AHS network creation utilities."""
    
    def test_create_network(self):
        """Test creating network of AHS."""
        states = create_ahs_network(N=10, seed=42)
        assert len(states) == 10
        assert all(isinstance(s, AlgorithmicHolonomicState) for s in states)
        
    def test_network_reproducibility(self):
        """Test network creation is reproducible with seed."""
        states1 = create_ahs_network(N=5, seed=123)
        states2 = create_ahs_network(N=5, seed=123)
        
        for s1, s2 in zip(states1, states2):
            assert s1 == s2
            
    def test_network_different_seeds(self):
        """Test different seeds produce different networks."""
        states1 = create_ahs_network(N=5, seed=123)
        states2 = create_ahs_network(N=5, seed=456)
        
        # At least some should be different
        differences = sum(1 for s1, s2 in zip(states1, states2) if s1 != s2)
        assert differences > 0
        
    def test_phase_distribution_uniform(self):
        """Test phase distribution options."""
        states_uniform = create_ahs_network(
            N=100, 
            phase_distribution="uniform",
            seed=42
        )
        
        phases = [s.holonomic_phase for s in states_uniform]
        # Should be roughly uniform in [0, 2π)
        assert min(phases) >= 0
        assert max(phases) < 2 * np.pi
        assert np.std(phases) > 0.5  # Not all the same
        
    def test_phase_distribution_coverage(self):
        """Test phases cover the full range."""
        states = create_ahs_network(N=1000, phase_distribution="uniform", seed=42)
        phases = [s.holonomic_phase for s in states]
        
        # Check we have phases in different quadrants
        quadrant_counts = [
            sum(1 for p in phases if 0 <= p < np.pi/2),
            sum(1 for p in phases if np.pi/2 <= p < np.pi),
            sum(1 for p in phases if np.pi <= p < 3*np.pi/2),
            sum(1 for p in phases if 3*np.pi/2 <= p < 2*np.pi),
        ]
        # Each quadrant should have some representation
        assert all(c > 0 for c in quadrant_counts)
        
    def test_binary_string_variety(self):
        """Test binary strings have variety."""
        states = create_ahs_network(N=50, seed=42)
        binary_strings = [s.binary_string for s in states]
        
        # Should have different lengths
        lengths = [len(b) for b in binary_strings]
        assert len(set(lengths)) > 1
        
        # Should have different content
        assert len(set(binary_strings)) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
