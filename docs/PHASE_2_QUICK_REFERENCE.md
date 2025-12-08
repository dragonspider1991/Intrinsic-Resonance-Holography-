# Phase 2 Quick Reference Guide

**Status**: ✅ COMPLETE  
**Date**: December 8, 2024

## Quick Start

Run the complete demonstration:
```bash
cd /path/to/Intrinsic-Resonance-Holography-
PYTHONPATH=. python examples/phase2_quantum_emergence_demo.py
```

Run all tests:
```bash
python -m pytest tests/test_v15_unitary_evolution.py tests/test_v15_quantum_emergence.py -v
```

## Core Modules

### Unitary Evolution
```python
from src.core.unitary_evolution import UnitaryEvolutionOperator

# Create operator from network
W = create_hermitian_network(N=100)
L = compute_information_transfer_matrix(W)
op = UnitaryEvolutionOperator(L, dt=0.1, hbar_0=1.0)

# Evolve state
psi = initial_state(N=100)
psi_evolved = op.evolve(psi, n_steps=10)

# Verify properties
is_unitary, _ = op.verify_unitarity()
conserves, _ = op.verify_energy_conservation(psi)
```

### Hilbert Space Emergence
```python
from src.physics.quantum_emergence import HilbertSpaceEmergence

# Run simulation
simulator = HilbertSpaceEmergence(N=50, M_ensemble=100)
results = simulator.run_emergence_simulation()

# Access results
C = results['correlation_matrix']  # Hermitian
basis = results['basis']            # Orthonormal
amplitudes = results['amplitudes']  # Normalized
```

### Hamiltonian & Born Rule
```python
from src.physics.quantum_emergence import (
    derive_hamiltonian,
    verify_born_rule
)

# Derive Hamiltonian
H = derive_hamiltonian(L, hbar_0=1.0)

# Verify Born rule
psi = random_state(N=100)
results = verify_born_rule(psi, measurements=10000)
print(f"p-value: {results['p_value']}")  # Should be > 0.05
```

## Success Metrics (All Passing)

| Metric | Target | Achieved |
|--------|--------|----------|
| Unitarity | < 1e-12 | 3.28e-15 ✅ |
| Energy Conservation | < 1e-10 | 2.17e-16 ✅ |
| Schrödinger Convergence | < 1e-6 | 2.63e-15 ✅ |
| Born Rule (p-value) | > 0.05 | 0.9931 ✅ |

## Test Coverage

- **Total**: 32 tests (Target: 20+) ✅
- **Unitary Evolution**: 15 tests
- **Quantum Emergence**: 17 tests
- **Pass Rate**: 100%

## Key Files

### Implementation
- `src/core/unitary_evolution.py` - Axiom 4 implementation
- `src/physics/quantum_emergence.py` - Theorems 3.1-3.3

### Testing
- `tests/test_v15_unitary_evolution.py` - 15 tests
- `tests/test_v15_quantum_emergence.py` - 17 tests

### Documentation
- `docs/PHASE_2_COMPLETION_REPORT.md` - Full completion report
- `examples/phase2_quantum_emergence_demo.py` - Working demonstration
- `.github/agents/PHASE_2_QUANTUM_EMERGENCE.md` - Original specification

## Common Tasks

### Run Single Task Tests
```bash
# Task 2.1: Unitary Evolution
python -m pytest tests/test_v15_unitary_evolution.py -v

# Tasks 2.2-2.4: Quantum Emergence
python -m pytest tests/test_v15_quantum_emergence.py -v
```

### Validate Metrics
```python
# Create validation script or use existing
PYTHONPATH=. python /tmp/validate_phase2.py
```

### Run Demonstration
```bash
PYTHONPATH=. python examples/phase2_quantum_emergence_demo.py
```

## Troubleshooting

**Import Errors**: Set PYTHONPATH
```bash
export PYTHONPATH=/path/to/Intrinsic-Resonance-Holography-
```

**Missing Dependencies**: Install requirements
```bash
pip install numpy scipy pytest networkx
```

**Test Failures**: Check versions
```bash
python --version  # Should be >= 3.8
pip list | grep -E "numpy|scipy"
```

## What's Next?

Phase 2 provides the quantum mechanics framework for:

1. **Phase 3**: General Relativity derivation
2. **Phase 4**: Gauge group algebraic derivation
3. **Phase 5**: Fermion generation topology
4. **Phase 6**: Cosmological constant problem
5. **Phase 7**: Exascale implementation
6. **Phase 8**: Final validation

All Phase 2 deliverables are complete and ready for use in subsequent phases.

---

For detailed information, see:
- Full Report: `docs/PHASE_2_COMPLETION_REPORT.md`
- Specification: `.github/agents/PHASE_2_QUANTUM_EMERGENCE.md`
- Theory: `README.md` §3
