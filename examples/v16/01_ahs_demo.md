# IRH v16.0 Phase 1A: Algorithmic Holonomic States Demo

This demonstration shows the basic usage of the `AlgorithmicHolonomicState` class,
the fundamental ontological primitive of Intrinsic Resonance Holography v16.0.

## Overview

**Algorithmic Holonomic States (AHS)** are intrinsically complex-valued information processes
embodying both informational content (binary string) and a holonomic phase degree of freedom.

From the v16.0 manuscript §1, Axiom 0:
> "Reality consists solely of a finite, ordered set of distinguishable Algorithmic Holonomic States (AHS),
> each an intrinsically complex-valued information process."

## Basic Usage

### Creating an AHS

```python
import numpy as np
from irh.core.v16.ahs import AlgorithmicHolonomicState

# Create an AHS with binary information and phase
ahs = AlgorithmicHolonomicState(
    binary_string="10101010",  # Informational content
    holonomic_phase=np.pi/4     # Phase φ ∈ [0, 2π)
)

print(f"AHS: {ahs}")
print(f"Detailed: {repr(ahs)}")
```

Output:
```
AHS: AHS[8bits, φ=0.785rad]
Detailed: AHS(info=10101010, φ=0.7854, K_t=8.0)
```

### Properties

```python
# Information content (in bits)
print(f"Information content: {ahs.information_content} bits")

# Complex amplitude (e^{iφ})
amplitude = ahs.complex_amplitude
print(f"Complex amplitude: {amplitude}")
print(f"Magnitude: {abs(amplitude):.6f}")
print(f"Phase: {np.angle(amplitude):.6f} rad")

# Kolmogorov complexity estimate
kt = ahs.compute_complexity()
print(f"K_t complexity: {kt:.1f} bits")
```

Output:
```
Information content: 8 bits
Complex amplitude: (0.7071067811865476+0.7071067811865475j)
Magnitude: 1.000000
Phase: 0.785398 rad
K_t complexity: 96.0 bits
```

### Phase Normalization

Phases are automatically normalized to [0, 2π):

```python
# Create with phase > 2π
ahs_unnormalized = AlgorithmicHolonomicState("101", 3 * np.pi)
print(f"Phase 3π normalized to: {ahs_unnormalized.holonomic_phase:.4f} rad")
# Output: Phase 3π normalized to: 3.1416 rad (π)

# Create with negative phase
ahs_negative = AlgorithmicHolonomicState("101", -np.pi/2)
print(f"Phase -π/2 normalized to: {ahs_negative.holonomic_phase:.4f} rad")
# Output: Phase -π/2 normalized to: 4.7124 rad (3π/2)
```

### Equality and Hashing

AHS can be compared and used in sets/dictionaries:

```python
ahs1 = AlgorithmicHolonomicState("101", 0.5)
ahs2 = AlgorithmicHolonomicState("101", 0.5)
ahs3 = AlgorithmicHolonomicState("110", 0.5)

print(f"ahs1 == ahs2: {ahs1 == ahs2}")  # True
print(f"ahs1 == ahs3: {ahs1 == ahs3}")  # False

# Use in sets
ahs_set = {ahs1, ahs2, ahs3}
print(f"Unique AHS in set: {len(ahs_set)}")  # 2
```

### Validation

The class validates inputs rigorously:

```python
# Invalid binary string
try:
    AlgorithmicHolonomicState("012", 0.0)
except ValueError as e:
    print(f"Error: {e}")
# Output: Error: binary_string must contain only '0' and '1'

# Empty string
try:
    AlgorithmicHolonomicState("", 0.0)
except ValueError as e:
    print(f"Error: {e}")
# Output: Error: binary_string cannot be empty

# Wrong type for phase
try:
    AlgorithmicHolonomicState("101", "not a number")
except TypeError as e:
    print(f"Error: {e}")
# Output: Error: holonomic_phase must be numeric
```

## Creating Networks of AHS

The `create_ahs_network` function generates networks of random AHS:

```python
from irh.core.v16.ahs import create_ahs_network

# Create 10 AHS with reproducible seed
states = create_ahs_network(N=10, seed=42)

print(f"Created {len(states)} AHS:")
for i, state in enumerate(states[:5]):  # Show first 5
    print(f"  [{i}] {state}")
```

Output:
```
Created 10 AHS:
  [0] AHS[10bits, φ=4.782rad]
  [1] AHS[20bits, φ=0.401rad]
  [2] AHS[19bits, φ=0.969rad]
  [3] AHS[18bits, φ=4.209rad]
  [4] AHS[12bits, φ=2.435rad]
```

### Reproducibility

Networks are reproducible with the same seed:

```python
states1 = create_ahs_network(N=5, seed=123)
states2 = create_ahs_network(N=5, seed=123)

print(f"Networks identical: {all(s1 == s2 for s1, s2 in zip(states1, states2))}")
# Output: Networks identical: True
```

### Phase Distribution

Check phase distribution across a large network:

```python
import matplotlib.pyplot as plt

# Create large network
states = create_ahs_network(N=1000, phase_distribution="uniform", seed=42)
phases = [s.holonomic_phase for s in states]

# Plot histogram
plt.figure(figsize=(10, 4))
plt.hist(phases, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Holonomic Phase φ (radians)')
plt.ylabel('Count')
plt.title('Phase Distribution of 1000 AHS')
plt.axvline(np.pi, color='r', linestyle='--', label='π')
plt.axvline(2*np.pi, color='r', linestyle='--', label='2π')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Phase range: [{min(phases):.3f}, {max(phases):.3f}]")
print(f"Phase mean: {np.mean(phases):.3f}")
print(f"Phase std: {np.std(phases):.3f}")
```

## Next Steps

This completes **Phase 1A** of the v16.0 implementation roadmap.

**Next phases:**
- **Phase 1B**: Implement Algorithmic Coherence Weights (ACW) with basic NCD computation
- **Phase 1C**: Documentation integration (Axiom 0 & 1 summaries)
- **Phase 2**: Core Axioms 2-4 implementation

For complete details, see `AGENT_HANDOFF_V16.md`.

## References

- Main Manuscript §1: Axiom 0 (Algorithmic Holonomic Substrate)
- [IRH-MATH-2025-01] §1: Rigorous derivation of complex numbers from non-commutative transformations
- [IRH-COMP-2025-02] §2.1: Multi-fidelity K_t approximation
