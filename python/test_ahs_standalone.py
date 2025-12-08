"""
Standalone test script for AlgorithmicHolonomicState.
Tests Phase 1A implementation without requiring full package dependencies.
"""

import sys
sys.path.insert(0, '/home/runner/work/Intrinsic-Resonance-Holography-/Intrinsic-Resonance-Holography-/python/src')

import numpy as np
from irh.core.v16.ahs import AlgorithmicHolonomicState, create_ahs_network

def test_basic_creation():
    """Test basic AHS creation."""
    print("Test 1: Basic creation...")
    ahs = AlgorithmicHolonomicState("101010", np.pi)
    assert ahs.binary_string == "101010"
    assert np.isclose(ahs.holonomic_phase, np.pi)
    assert ahs.information_content == 6
    print(f"  ✓ Created: {ahs}")
    print(f"  ✓ Repr: {repr(ahs)}")

def test_phase_normalization():
    """Test phase normalization."""
    print("\nTest 2: Phase normalization...")
    ahs = AlgorithmicHolonomicState("1", 3 * np.pi)
    assert 0 <= ahs.holonomic_phase < 2 * np.pi
    assert np.isclose(ahs.holonomic_phase, np.pi)
    print(f"  ✓ Phase 3π normalized to: {ahs.holonomic_phase:.4f}")
    
def test_complex_amplitude():
    """Test complex amplitude."""
    print("\nTest 3: Complex amplitude...")
    ahs = AlgorithmicHolonomicState("1", np.pi/4)
    amp = ahs.complex_amplitude
    assert np.isclose(abs(amp), 1.0)
    assert np.isclose(np.angle(amp), np.pi/4)
    print(f"  ✓ Complex amplitude: {amp}")
    print(f"  ✓ Magnitude: {abs(amp):.6f}")

def test_validation():
    """Test validation."""
    print("\nTest 4: Validation...")
    
    # Invalid binary string
    try:
        AlgorithmicHolonomicState("012", 0.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Caught invalid binary: {e}")
    
    # Empty string
    try:
        AlgorithmicHolonomicState("", 0.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Caught empty string: {e}")
    
    # Invalid type
    try:
        AlgorithmicHolonomicState(123, 0.0)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        print(f"  ✓ Caught wrong type: {e}")

def test_equality():
    """Test equality."""
    print("\nTest 5: Equality and hashing...")
    ahs1 = AlgorithmicHolonomicState("101", 0.5)
    ahs2 = AlgorithmicHolonomicState("101", 0.5)
    ahs3 = AlgorithmicHolonomicState("110", 0.5)
    
    assert ahs1 == ahs2
    assert ahs1 != ahs3
    print(f"  ✓ Equality works")
    
    # Test hashing
    ahs_set = {ahs1, ahs2}
    assert len(ahs_set) == 1
    print(f"  ✓ Hashing works (set size: {len(ahs_set)})")

def test_complexity():
    """Test complexity computation."""
    print("\nTest 6: Complexity computation...")
    ahs = AlgorithmicHolonomicState("0" * 100, 0.0)
    kt = ahs.compute_complexity()
    assert kt > 0
    assert kt < len("0" * 100) * 8
    print(f"  ✓ Compressible string K_t: {kt:.1f} bits (vs {100*8} original)")

def test_network_creation():
    """Test network creation."""
    print("\nTest 7: Network creation...")
    states = create_ahs_network(N=10, seed=42)
    assert len(states) == 10
    assert all(isinstance(s, AlgorithmicHolonomicState) for s in states)
    print(f"  ✓ Created {len(states)} AHS")
    print(f"  ✓ Sample states:")
    for i, s in enumerate(states[:3]):
        print(f"     [{i}] {s}")
    
    # Test reproducibility
    states2 = create_ahs_network(N=10, seed=42)
    assert all(s1 == s2 for s1, s2 in zip(states, states2))
    print(f"  ✓ Network creation is reproducible")

def main():
    """Run all tests."""
    print("=" * 60)
    print("IRH v16.0 Phase 1A: AlgorithmicHolonomicState Tests")
    print("=" * 60)
    
    try:
        test_basic_creation()
        test_phase_normalization()
        test_complex_amplitude()
        test_validation()
        test_equality()
        test_complexity()
        test_network_creation()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
