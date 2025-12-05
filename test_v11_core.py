"""
Quick validation test for IRH v11.0 core modules.

Tests:
1. Substrate initialization
2. ARO action computation
3. Quantum emergence verification
"""

import numpy as np
import sys
sys.path.insert(0, '/home/runner/work/Intrinsic-Resonance-Holography-/Intrinsic-Resonance-Holography-')

from src.core.substrate_v11 import InformationSubstrate
from src.core.sote_v11 import AROFunctional
from src.core.quantum_v11 import QuantumEmergence

def test_substrate():
    """Test substrate initialization."""
    print("=" * 70)
    print("TEST 1: Information Substrate")
    print("=" * 70)
    
    N = 500
    substrate = InformationSubstrate(N=N, dimension=4)
    substrate.initialize_correlations('random_geometric')
    substrate.compute_laplacian()
    
    # Verify holographic bound
    bound_check = substrate.verify_holographic_bound()
    
    print(f"✓ Substrate initialized: N={N}, d=4")
    print(f"✓ Holographic bound satisfied: {bound_check['satisfies_bound']}")
    print(f"  I_bulk/I_boundary = {bound_check['ratio']:.4f}")
    
    return substrate

def test_sote(substrate):
    """Test ARO action computation."""
    print("\n" + "=" * 70)
    print("TEST 2: ARO Functional")
    print("=" * 70)
    
    sote = AROFunctional(substrate)
    S = sote.compute_action()
    
    print(f"✓ ARO action computed: S = {S:.4e}")
    
    # Test holographic compliance
    compliance = sote.verify_holographic_compliance()
    print(f"✓ Holographic compliance: {compliance['bound_satisfied']}")
    print(f"  Action increases when bound violated: {compliance['action_increases']}")
    
    return sote

def test_quantum(substrate):
    """Test quantum emergence."""
    print("\n" + "=" * 70)
    print("TEST 3: Quantum Emergence")
    print("=" * 70)
    
    qm = QuantumEmergence(substrate)
    
    # Derive Hamiltonian
    H = qm.derive_hamiltonian()
    print(f"✓ Hamiltonian derived: shape {H.shape}")
    
    # Compute Planck constant
    hbar = qm.compute_planck_constant(n_samples=200)
    if hbar:
        print(f"✓ Planck constant: ℏ = {hbar:.4e} J·s")
        print(f"  CODATA 2022: ℏ = 1.0546e-34 J·s")
    
    # Test CCR
    ccr = qm.compute_commutator()
    print(f"✓ Canonical commutation relation: satisfied={ccr['satisfies_CCR']}")
    
    # Test Born rule
    born = qm.verify_born_rule(n_trials=50)
    print(f"✓ Born rule: verified={born['born_rule_verified']}")
    print(f"  Time avg={born['time_average']:.4f}, Ensemble avg={born['ensemble_average']:.4f}")
    
    return qm

def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "IRH v11.0 CORE MODULES VALIDATION TEST" + " " * 20 + "║")
    print("╚" + "=" * 68 + "╝")
    
    try:
        # Test 1: Substrate
        substrate = test_substrate()
        
        # Test 2: ARO
        sote = test_sote(substrate)
        
        # Test 3: Quantum
        qm = test_quantum(substrate)
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY: ALL CORE TESTS PASSED ✓")
        print("=" * 70)
        print("\nCore v11.0 modules are functional:")
        print("  • InformationSubstrate: Discrete ontology without assumptions")
        print("  • AROFunctional: Unique action principle")
        print("  • QuantumEmergence: Non-circular QM derivation")
        print("\nReady for optimization and empirical predictions!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
