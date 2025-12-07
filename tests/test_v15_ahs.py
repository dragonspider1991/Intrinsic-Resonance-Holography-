"""
Test Suite for Algorithmic Holonomic States (AHS) - IRH v15.0

Validates Axiom 0 and Axiom 1 implementations.
"""

import numpy as np
import pytest
from src.core.ahs_v15 import (
    AlgorithmicHolonomicState,
    AlgorithmicCoherenceWeight,
    create_ahs_network
)


def test_ahs_creation():
    """Test basic creation of Algorithmic Holonomic States."""
    state = AlgorithmicHolonomicState(
        info_content=b"test_data",
        holonomic_phase=np.pi/4,
        state_id=0
    )
    
    assert state.info_content == b"test_data"
    assert abs(state.holonomic_phase - np.pi/4) < 1e-10
    assert state.state_id == 0


def test_ahs_phase_normalization():
    """Test that phases are normalized to [0, 2π)."""
    # Phase > 2π should be normalized
    state1 = AlgorithmicHolonomicState(
        info_content=b"data",
        holonomic_phase=3 * np.pi
    )
    assert 0 <= state1.holonomic_phase < 2 * np.pi
    assert abs(state1.holonomic_phase - np.pi) < 1e-10
    
    # Negative phase should be normalized
    state2 = AlgorithmicHolonomicState(
        info_content=b"data",
        holonomic_phase=-np.pi/2
    )
    assert 0 <= state2.holonomic_phase < 2 * np.pi
    assert abs(state2.holonomic_phase - 3*np.pi/2) < 1e-10


def test_ahs_string_conversion():
    """Test automatic conversion of string to bytes."""
    state = AlgorithmicHolonomicState(
        info_content="test string",
        holonomic_phase=0.5
    )
    
    assert isinstance(state.info_content, bytes)
    assert state.info_content == b"test string"


def test_ahs_complex_amplitude():
    """Test conversion to complex amplitude."""
    phase = np.pi/3
    state = AlgorithmicHolonomicState(
        info_content=b"data",
        holonomic_phase=phase
    )
    
    amplitude = state.to_complex_amplitude()
    
    # Should be unit magnitude
    assert abs(abs(amplitude) - 1.0) < 1e-10
    
    # Should have correct phase
    expected = np.exp(1j * phase)
    assert abs(amplitude - expected) < 1e-10


def test_ahs_phase_difference():
    """Test phase difference calculation between states."""
    state1 = AlgorithmicHolonomicState(
        info_content=b"state1",
        holonomic_phase=np.pi/4
    )
    
    state2 = AlgorithmicHolonomicState(
        info_content=b"state2",
        holonomic_phase=3*np.pi/4
    )
    
    diff = state1.phase_difference_to(state2)
    
    # Should be π/2
    assert abs(diff - np.pi/2) < 1e-10
    
    # Check wraparound
    state3 = AlgorithmicHolonomicState(
        info_content=b"state3",
        holonomic_phase=np.pi/4
    )
    
    state4 = AlgorithmicHolonomicState(
        info_content=b"state4",
        holonomic_phase=7*np.pi/4
    )
    
    diff_wrap = state3.phase_difference_to(state4)
    assert 0 <= diff_wrap < 2 * np.pi


def test_ahs_equality():
    """Test equality comparison for AHS."""
    state1 = AlgorithmicHolonomicState(
        info_content=b"data",
        holonomic_phase=np.pi/4
    )
    
    state2 = AlgorithmicHolonomicState(
        info_content=b"data",
        holonomic_phase=np.pi/4
    )
    
    state3 = AlgorithmicHolonomicState(
        info_content=b"different",
        holonomic_phase=np.pi/4
    )
    
    state4 = AlgorithmicHolonomicState(
        info_content=b"data",
        holonomic_phase=np.pi/2
    )
    
    # Same content and phase
    assert state1 == state2
    
    # Different content
    assert state1 != state3
    
    # Different phase
    assert state1 != state4


def test_algorithmic_coherence_weight_creation():
    """Test creation of Algorithmic Coherence Weights."""
    weight = AlgorithmicCoherenceWeight(
        magnitude=0.8,
        phase=np.pi/3,
        source_id=0,
        target_id=1
    )
    
    assert weight.magnitude == 0.8
    assert abs(weight.phase - np.pi/3) < 1e-10
    assert weight.source_id == 0
    assert weight.target_id == 1


def test_acw_magnitude_clipping():
    """Test that magnitudes are clipped to [0, 1]."""
    # Too large
    weight1 = AlgorithmicCoherenceWeight(magnitude=1.5, phase=0.0)
    assert weight1.magnitude == 1.0
    
    # Negative
    weight2 = AlgorithmicCoherenceWeight(magnitude=-0.5, phase=0.0)
    assert weight2.magnitude == 0.0


def test_acw_to_complex():
    """Test conversion of ACW to complex number."""
    magnitude = 0.7
    phase = np.pi/4
    
    weight = AlgorithmicCoherenceWeight(magnitude=magnitude, phase=phase)
    complex_val = weight.to_complex()
    
    expected = magnitude * np.exp(1j * phase)
    assert abs(complex_val - expected) < 1e-10


def test_acw_from_complex():
    """Test creating ACW from complex number."""
    complex_val = 0.6 * np.exp(1j * np.pi/6)
    
    weight = AlgorithmicCoherenceWeight.from_complex(
        complex_val,
        source_id=1,
        target_id=2
    )
    
    assert abs(weight.magnitude - 0.6) < 1e-10
    assert abs(weight.phase - np.pi/6) < 1e-10
    assert weight.source_id == 1
    assert weight.target_id == 2


def test_acw_conjugate():
    """Test conjugation of ACW (for reverse traversal)."""
    weight = AlgorithmicCoherenceWeight(
        magnitude=0.5,
        phase=np.pi/4,
        source_id=0,
        target_id=1
    )
    
    conj = weight.conjugate()
    
    # Magnitude unchanged
    assert conj.magnitude == weight.magnitude
    
    # Phase negated
    assert abs(conj.phase - (2*np.pi - weight.phase)) < 1e-10
    
    # IDs swapped
    assert conj.source_id == 1
    assert conj.target_id == 0


def test_create_ahs_network():
    """Test creation of AHS network."""
    N = 100
    states = create_ahs_network(N, phase_distribution='uniform', rng=np.random.default_rng(42))
    
    assert len(states) == N
    
    # Check all states are unique objects
    assert all(isinstance(s, AlgorithmicHolonomicState) for s in states)
    
    # Check state IDs are assigned
    assert all(s.state_id == i for i, s in enumerate(states))
    
    # Check phases are in valid range
    assert all(0 <= s.holonomic_phase < 2*np.pi for s in states)
    
    # Check info content exists
    assert all(len(s.info_content) > 0 for s in states)


def test_create_ahs_network_distributions():
    """Test different phase distributions."""
    N = 50
    rng = np.random.default_rng(42)
    
    # Uniform distribution
    states_uniform = create_ahs_network(N, phase_distribution='uniform', rng=rng)
    phases_uniform = [s.holonomic_phase for s in states_uniform]
    assert 0 < np.std(phases_uniform) < 2  # Should have some variance
    
    # Fixed distribution
    states_fixed = create_ahs_network(N, phase_distribution='fixed', rng=rng)
    phases_fixed = [s.holonomic_phase for s in states_fixed]
    assert all(p == 0.0 for p in phases_fixed)
    
    # Normal distribution
    states_normal = create_ahs_network(N, phase_distribution='normal', rng=rng)
    phases_normal = [s.holonomic_phase for s in states_normal]
    assert all(0 <= p < 2*np.pi for p in phases_normal)


def test_ahs_hash():
    """Test that AHS can be used in sets/dicts."""
    state1 = AlgorithmicHolonomicState(b"data1", 1.0)
    state2 = AlgorithmicHolonomicState(b"data1", 1.0)
    state3 = AlgorithmicHolonomicState(b"data2", 1.0)
    
    # Should be hashable
    state_set = {state1, state2, state3}
    
    # state1 and state2 are equal, so set should have 2 elements
    assert len(state_set) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
