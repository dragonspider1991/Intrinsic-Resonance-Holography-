# Intrinsic Resonance Holography v13.0: Main Driver Script

This document describes the `main.py` driver script that orchestrates the complete cosmic simulation process for IRH v13.0.

## Overview

The `main.py` script implements the **Tier 1 Empirical Test** for Intrinsic Resonance Holography v13.0, which derives fundamental physical constants from algorithmic information dynamics.

## Features

### 1. Adaptive Resonance Optimization (ARO)
- Initializes a Cymatic Resonance Network with N nodes
- Optimizes the network topology using hybrid simulated annealing
- Maximizes the Spectral Zeta Regularized Harmony Functional
- Converges to the Cosmic Fixed Point

### 2. Topological Analysis
- **Fine-Structure Constant (α⁻¹)**: Derived from frustration density via phase holonomies
  - Target: 137.036 ± 0.004
  - Formula: α⁻¹ = 2π/ρ_frust (Theorem 1.2)

- **Gauge Group (β₁)**: First Betti number of emergent network topology
  - Target: 12 (corresponding to SU(3)×SU(2)×U(1))
  
- **Fermion Generations**: Derived from flux matrix nullity
  - Target: 3

### 3. Dimensional Analysis
- **Spectral Dimension (d_spec)**: Computed via heat kernel trace method
  - Target: 4.00 (4D spacetime)
  - Method: P(t) = Tr(exp(-tM)) ~ t^(-d/2)

- **Dimensional Coherence Index (χ_D)**: Composite metric measuring geometric stability
  - Components: Holographic consistency, Residue, Categorical coherence
  - Range: [0, 1], maximized at d=4

## Usage

### Basic Usage

Run the simulation with default parameters (N=100, iterations=2000):

```bash
python3 main.py
```

### Expected Output

```
============================================================
INTRINSIC RESONANCE HOLOGRAPHY v13.0: COSMIC BOOTSTRAP
Nodes (N): 100 | Iterations: 2000 | Seed: 42
============================================================

[Phase 1] Initiating Adaptive Resonance Optimization (ARO)...
[ARO] Initialized random network: N=100, edges=898
[ARO] Starting optimization: 2000 iterations
...
[ARO] Optimization complete. Final S_H = 495.39967

[Phase 2] Measuring Topological Invariants...

[Phase 3] Verifying Dimensional Coherence...

============================================================
FINAL EXPERIMENTAL REPORT
============================================================
Parameter                 | Prediction (v13.0)   | Measured Value      
----------------------------------------------------------------------
Inv. Fine-Structure       | 137.036 ± 0.004      | X.XXXX
Spectral Dimension        | 4.00 (Exact)         | X.XXXX
Fermion Generations       | 3 (Exact)            | X
Gauge Group (Beta_1)      | 12 (SM)              | XX
----------------------------------------------------------------------
Dimensional Coherence Index (chi_D): X.XXXX (Max ~1.0 at d=4)
============================================================
```

### Customizing Parameters

You can modify the simulation parameters by editing the `run_cosmic_simulation()` call in `main.py`:

```python
if __name__ == "__main__":
    run_cosmic_simulation(
        N=500,           # Number of nodes (higher N → better accuracy)
        iterations=5000, # Optimization iterations
        seed=42          # Random seed for reproducibility
    )
```

### Performance Notes

The accuracy of emergent constants depends on system size:

- **N = 100, iterations = 2000**: Quick demo (< 1 minute)
  - Results show general trends but may deviate from predictions
  
- **N = 1000, iterations = 10000**: Moderate accuracy (~ 10 minutes)
  - Improved convergence to theoretical predictions
  
- **N ≥ 10000, iterations ≥ 100000**: High accuracy (hours)
  - Required to match α⁻¹ = 137.036 within experimental error
  - As stated in paper: "For full accuracy matching the paper, N should be >= 10^4"

## Architecture

### Module Structure

```
src/
├── core/
│   ├── harmony.py           # HarmonyEngine class (Spectral Zeta Regularization)
│   └── aro_optimizer.py     # AROOptimizer class (Hybrid optimization)
├── topology/
│   └── invariants.py        # TopologyAnalyzer class (α, β₁, generations)
└── metrics/
    └── dimensions.py        # DimensionalityAnalyzer class (d_spec, χ_D)
```

### Key Classes

#### HarmonyEngine
```python
from src.core.harmony import HarmonyEngine

# Compute Information Transfer Matrix
M = HarmonyEngine.compute_information_transfer_matrix(W)

# Calculate Harmony Functional
S_H = HarmonyEngine.calculate_harmony(W, N)
```

#### AROOptimizer
```python
from src.core.aro_optimizer import AROOptimizer

# Initialize and optimize
optimizer = AROOptimizer(N=100, connection_probability=0.2, rng_seed=42)
final_W = optimizer.optimize(iterations=2000, temp=1.0, cooling_rate=0.99)
```

#### TopologyAnalyzer
```python
from src.topology.invariants import TopologyAnalyzer

# Analyze topological invariants
analyzer = TopologyAnalyzer(W, threshold=1e-5)
alpha_inv = analyzer.derive_alpha_inv()
beta_1 = analyzer.calculate_betti_numbers()
n_gen = analyzer.calculate_generation_count()
```

#### DimensionalityAnalyzer
```python
from src.metrics.dimensions import DimensionalityAnalyzer

# Analyze dimensional properties
analyzer = DimensionalityAnalyzer(M)
d_spec = analyzer.calculate_spectral_dimension(t_start=1e-2, t_end=1.0)
chi_D = analyzer.calculate_dimensional_coherence(d_spec)
```

## Testing

Run the component tests to verify functionality:

```bash
python3 test_main.py
```

This validates:
- HarmonyEngine static methods
- AROOptimizer initialization and optimization
- TopologyAnalyzer calculations
- DimensionalityAnalyzer metrics

## References

- **IRH v13.0 Paper**: "Intrinsic Resonance Holography: Deriving Physical Laws from Algorithmic Information"
- **Theorem 1.2**: Emergence of Phase Structure and Fine-Structure Constant
- **Theorem 3.1**: Emergent 4D Spacetime
- **Theorem 4.1**: Uniqueness of Harmony Functional
- **Theorem 5.1**: Network Homology and Gauge Group

## Notes

1. **Zero Free Parameters**: All constants emerge from network structure, no tuning required
2. **Reproducibility**: Use same seed for deterministic results
3. **Scalability**: Sparse matrix operations enable efficient large-N simulations
4. **Validation**: Compare measured values against v13.0 predictions
5. **Convergence**: "CONVERGENCE SUCCESSFUL" indicates Cosmic Fixed Point stability

## Legacy CLI

The original CLI interface has been preserved as `main_cli_backup.py` for backward compatibility.
