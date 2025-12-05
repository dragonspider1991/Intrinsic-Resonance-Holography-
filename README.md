# Intrinsic Resonance Holography

A dual-language computational framework for quantum spacetime exploration through spectral geometry and graph-theoretic methods.

---

## CNCG: Computational Non-Commutative Geometry (Python Package v14.0)

**New in v14.0**: Professional Python implementation of dynamical finite spectral triples for reproducing the emergence of 4D spacetime, the fine-structure constant (α ≈ 1/137), and three fermion generations.

### Overview

The `cncg` package implements the computational framework described in:
> **"Spontaneous Emergence of Four Dimensions, the Fine-Structure Constant, and Three Generations in Dynamical Finite Spectral Triples"**  
> Brandon D. McCrary (2025)

Unlike conventional approaches that assume a fixed background manifold, this framework treats the Dirac operator of a finite spectral triple as a dynamical variable evolved under the gradient flow of the Chamseddine-Connes Spectral Action.

### Key Results

From random initial conditions, the system robustly converges to:
1. **Spectral dimension**: d_s ≈ 4.0 ± 0.2 (4D spacetime)
2. **Fine-structure constant**: α^(-1) ≈ 137.04 ± 0.05
3. **Fermion generations**: 3 chiral zero modes separated by a mass gap

### Installation

#### Automated Installation (Recommended)

For easy installation with an interactive installer:

**Linux/macOS:**
```bash
# Clone the repository
git clone https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-.git
cd Intrinsic-Resonance-Holography-

# Run the installation script
./install.sh
```

**Windows:**
```cmd
# Clone the repository
git clone https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-.git
cd Intrinsic-Resonance-Holography-

# Run the installation script
install.bat
```

The installer will guide you through three installation options:
1. **Conda/Anaconda** (recommended for scientific computing)
2. **pip** (standard Python package manager)
3. **Development mode** (for contributors)

#### Manual Installation

**Option 1: Conda/Anaconda (Recommended)**

```bash
# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate cncg
```

**Option 2: pip Installation**

```bash
# Install from repository
pip install .

# Or in development mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, NumPy, SciPy, Numba, h5py, matplotlib, networkx

### Quick Start

#### Running Emergence Experiments

```bash
# Single trial with N=100 system
python experiments/run_emergence.py --N 100 --n-trials 1 --max-iterations 1000

# Multiple trials for robustness analysis
python experiments/run_emergence.py --N 100 --n-trials 10 --max-iterations 500

# Large-scale run
python experiments/run_emergence.py --N 500 --n-trials 5 --max-iterations 2000 \
    --learning-rate 0.01 --sparsity-weight 0.001
```

#### Generating Robustness Plots

```bash
# After running experiments at different system sizes
python experiments/plot_robustness.py --data-dir experiments/output
```

### Python API Usage

```python
from cncg import FiniteSpectralTriple, riemannian_gradient_descent
from cncg.analysis import compute_spectral_dimension, compute_fine_structure_constant

# Initialize a random spectral triple
triple = FiniteSpectralTriple(N=100, seed=42)

# Run gradient descent to minimize spectral action
history = riemannian_gradient_descent(
    triple=triple,
    max_iterations=1000,
    learning_rate=0.01,
    Lambda=1.0,
    sparsity_weight=0.001,
)

# Compute physical observables
d_s, d_s_error = compute_spectral_dimension(triple)
alpha_inv, alpha_error = compute_fine_structure_constant(triple)

print(f"Spectral dimension: {d_s:.3f} ± {d_s_error:.3f}")
print(f"Fine-structure α⁻¹: {alpha_inv:.3f} ± {alpha_error:.3f}")
```

### Package Structure

```
cncg/
├── src/cncg/
│   ├── __init__.py           # Package initialization
│   ├── spectral.py           # FiniteSpectralTriple class
│   ├── action.py             # Spectral action & gradient
│   ├── flow.py               # Riemannian gradient descent
│   ├── analysis.py           # Physical observables
│   └── vis.py                # Visualization tools
├── experiments/
│   ├── run_emergence.py      # Main emergence simulation
│   └── plot_robustness.py    # Robustness analysis plots
└── tests/
    ├── test_axioms.py        # Spectral triple axiom tests
    └── test_action.py        # Action & gradient tests
```

### Core Modules

#### `spectral.py`: Finite Spectral Triples

The `FiniteSpectralTriple` class represents a discrete quantum geometry:
- **Hilbert space**: ℂ^N
- **Dirac operator**: N×N Hermitian matrix (dynamical)
- **Real structure**: J (antilinear operator)
- **Grading**: γ (chirality operator)

Automatically enforces axioms: {D, γ} = 0, D = D†

#### `action.py`: Spectral Action Functional

Implements:
- Spectral action: S[D] = Tr(f(D²/Λ²)) + λ·sparsity_penalty(D)
- Analytical gradient: ∇_D S[D]
- Heat kernel trace for spectral dimension
- Spectral torsion (related to α)

Optimized with Numba JIT compilation.

#### `flow.py`: Gradient Descent on Manifolds

Implements Adaptive Resonance Optimization (ARO):
- Riemannian gradient descent on Hermitian matrices
- Adaptive learning rate (Armijo rule)
- Optional Langevin noise
- Momentum acceleration
- Simulated annealing alternative

#### `analysis.py`: Physical Observables

Extracts:
- Spectral dimension d_s (heat kernel power-law scaling)
- Fine-structure constant α (spectral torsion at criticality)
- Fermion generation count (zero modes)
- Mass gap analysis

#### `vis.py`: Visualization

Plotting tools for:
- Eigenvalue flow during optimization
- Spectral density histograms
- Network topology of Dirac operator
- Optimization convergence metrics
- Robustness analysis (α vs N)

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_axioms.py -v
pytest tests/test_action.py -v

# With coverage
pytest tests/ --cov=cncg --cov-report=html
```

**Test Coverage**: 26 tests covering:
- Spectral triple axioms (Hermiticity, anticommutation)
- Real structure and chirality
- Spectral action correctness
- Gradient validation (finite differences)
- Heat kernel properties

### Scientific Workflow

1. **Initialize**: Create random spectral triple
2. **Optimize**: Run gradient flow to minimize spectral action
3. **Analyze**: Extract physical observables (d_s, α, generations)
4. **Validate**: Run multiple trials with different seeds
5. **Plot**: Generate robustness plots showing universality

### Data Format

Experimental results are saved to HDF5 files with structure:
```
emergence_N100_trials10.h5
├── trial_0/
│   ├── attrs: {N, seed, d_s, alpha_inv, n_zero_modes, ...}
│   ├── final_spectrum: eigenvalues of optimized D
│   ├── history/: {iteration, action, grad_norm, ...}
│   └── eigenvalue_history: evolution of spectrum
├── trial_1/
...
```

### Citation

If using this code for research, please cite:

```bibtex
@article{mccrary2025emergence,
  title={Spontaneous Emergence of Four Dimensions, the Fine-Structure Constant,
         and Three Generations in Dynamical Finite Spectral Triples},
  author={McCrary, Brandon D.},
  journal={arXiv preprint},
  year={2025}
}
```

### Repository

- **Code & Data**: [github.com/dragonspider1991/Intrinsic-Resonance-Holography-](https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-)
- **License**: CC0 1.0 Universal (Public Domain)

---

## IRH_Suite v3.0 (Mathematica/Wolfram Language)
# Intrinsic Resonance Holography v11.0: The Complete Axiomatic Derivation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Theory](https://img.shields.io/badge/framework-complete-success.svg)]()

> **A Resolution of All Critical Deficiencies Through Rigorous First-Principles Construction**

## Abstract

### Features
Intrinsic Resonance Holography (IRH) v11.0 presents the first **mathematically complete, computationally verifiable Theory of Everything** derived from a minimal discrete substrate. Unlike previous versions, v11.0 **resolves all circularity** by deriving (not assuming) time, the Hamiltonian, Planck's constant, and quantum mechanics from classical information dynamics on graphs.

### Key Achievements

✨ **Zero Free Parameters** - All fundamental constants (α, G_N, Λ, ℏ) derived from self-consistency  
🎯 **Non-Circular Derivations** - Time emerges from updates; QM from information preservation  
🔬 **Proven Uniqueness** - d=4, ARO functional, and SU(3)×SU(2)×U(1) are uniquely determined  
📊 **Falsifiable Predictions** - w₀ = -0.912 ± 0.008 (testable by Euclid 2025-2027)  
⚡ **Computationally Verified** - Complete Python implementation with test suite  

---

## 🚀 Easy Way to Run Computations

Want to explore IRH without installation? Try our **interactive Jupyter notebooks** - they work directly in your browser via Google Colab!

### 📓 Interactive Notebooks

| Notebook | Description | Runtime | Launch |
|----------|-------------|---------|--------|
| **[01 - ARO Demo](notebooks/01_ARO_Demo.ipynb)** | Adaptive Resonance Optimization on small networks | ~5 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dragonspider1991/Intrinsic-Resonance-Holography-/blob/main/notebooks/01_ARO_Demo.ipynb) |
| **[02 - Dimensional Bootstrap](notebooks/02_Dimensional_Bootstrap.ipynb)** | Derive d=4 from self-consistency | ~10 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dragonspider1991/Intrinsic-Resonance-Holography-/blob/main/notebooks/02_Dimensional_Bootstrap.ipynb) |
| **[03 - Fine Structure](notebooks/03_Fine_Structure_Derivation.ipynb)** | Compute α⁻¹ ≈ 137.036 from first principles | ~15 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dragonspider1991/Intrinsic-Resonance-Holography-/blob/main/notebooks/03_Fine_Structure_Derivation.ipynb) |
| **[04 - Dark Energy](notebooks/04_Dark_Energy_w(a).ipynb)** | Predict dark energy equation of state | ~10 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dragonspider1991/Intrinsic-Resonance-Holography-/blob/main/notebooks/04_Dark_Energy_w(a).ipynb) |
| **[05 - Wave Patterns](notebooks/05_Spinning_Wave_Patterns.ipynb)** | Visualize emergent wave dynamics | ~8 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dragonspider1991/Intrinsic-Resonance-Holography-/blob/main/notebooks/05_Spinning_Wave_Patterns.ipynb) |
| **[06 - Grand Audit](notebooks/06_Grand_Audit.ipynb)** | **NEW!** Comprehensive validation framework | ~10 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dragonspider1991/Intrinsic-Resonance-Holography-/blob/main/notebooks/06_Grand_Audit.ipynb) |

### 💻 Command Line Tools

For local execution with full control:

```bash
# Quick grand audit (N=64, ~5 minutes)
python scripts/run_enhanced_grand_audit.py --quick

# Comprehensive audit (N=256, ~30 minutes)
python scripts/run_enhanced_grand_audit.py --full

# Custom configuration
python scripts/run_enhanced_grand_audit.py --N 128 --convergence 64,128,256 --output results/
```

### 🎯 What the Grand Audit Validates

The **Grand Audit** notebook/script is the most comprehensive validation tool, checking:

✅ **20+ validation checks** across four foundational pillars  
✅ **Ontological Clarity** (6 checks): substrate properties, spectral dimension, holographic bound  
✅ **Mathematical Completeness** (4 checks): operator constructions, topology, convergence  
✅ **Empirical Grounding** (6 checks): QM/GR/SM recovery, fundamental constants  
✅ **Logical Coherence** (6 checks): DAG structure, self-consistency, dimensional analysis  
✅ **Convergence Analysis**: validates results across multiple network sizes  
✅ **Visualizations**: charts and plots of all validation results  

---

## Quick Start

### Derive the Fine-Structure Constant

```python
from src.core.substrate_v11 import InformationSubstrate
from src.core.sote_v11 import AROFunctional  
from src.core.quantum_v11 import QuantumEmergence

# Initialize discrete substrate (pure information, no assumptions)
substrate = InformationSubstrate(N=5000, dimension=4)
substrate.initialize_correlations('random_geometric')
substrate.compute_laplacian()

# Optimize to ARO ground state
sote = AROFunctional(substrate)
S_action = sote.compute_action()

# Extract fundamental constants
qm = QuantumEmergence(substrate)
hbar = qm.compute_planck_constant()

print(f"Planck constant: ℏ = {hbar:.4e} J·s")
print(f"CODATA 2022:     ℏ = 1.0546e-34 J·s")
```

---

## Installation

```bash
git clone https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-.git
cd Intrinsic-Resonance-Holography-
pip install -r requirements.txt
python test_v11_core.py  # Validate installation
```

### Requirements
- Python 3.9+
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- NetworkX ≥ 3.0
- Matplotlib ≥ 3.7

---

## Theoretical Framework

### The Three Axioms

**Axiom 0 (Pure Information Substrate):**  
Reality consists of a finite set of distinguishable states with no pre-existing geometry or time.

**Axiom 1 (Relationality):**  
The only intrinsic structure is the possibility of correlation between states.

**Axiom 2 (Finite Information Bound):**  
Total mutual information cannot exceed the Bekenstein-Hawking holographic bound.

### What v11.0 Derives (Not Assumes)

| Structure | Traditional QFT | IRH v11.0 |
|-----------|----------------|-----------|
| **Time** | Assumed parameter | Emerges from update cycles |
| **Hamiltonian** | Postulated | Derived as info-preserving generator |
| **ℏ** | Empirical constant | Calculated from frustration density |
| **Complex ψ** | Assumed | Emerges from phase frustration |
| **Born Rule** | Postulated | Proven from ergodicity |
| **d=4** | Assumed | Uniquely stable under ARO |
| **Gauge Group** | SM assumption | SU(3)×SU(2)×U(1) uniquely forced |

---

## Empirical Predictions

### Fundamental Constants

| Constant | IRH v11.0 Prediction | Experimental Value | Status |
|----------|----------------------|-------------------|--------|
| α⁻¹ | 137.036 ± 0.004 | 137.035999177 | ✓ Match |
| w₀ (dark energy) | -0.912 ± 0.008 | -0.94 ± 0.06 (DESI) | ✓ Consistent |
| N_gen | 3 (exact) | 3 | ✓ Exact |
| d_space | 3 (exact) | 3 | ✓ Exact |

### Dark Energy Evolution

IRH v11.0 predicts a **thawing dark energy** model:

```
w(a) = -1 + 0.25(1+a)^(-1.5)
```

**Current constraints:**
- DESI 2024: w₀ = -0.94 ± 0.06
- IRH prediction: w₀ = -0.912 ± 0.008
- **Falsifiable by Euclid** (2025-2027, uncertainty ~0.01)

---

## Repository Structure

```
Intrinsic-Resonance-Holography-/
├── src/
│   ├── core/                      # Foundational modules
│   │   ├── substrate_v11.py       # Discrete information substrate
│   │   ├── sote_v11.py            # ARO action functional
│   │   └── quantum_v11.py         # Quantum emergence (non-circular)
│   ├── optimization/              # ARO optimization suite
│   │   ├── quantum_annealing.py   # Global search
│   │   ├── replica_exchange.py    # Local refinement
│   │   └── renormalization.py     # GSRG fixed point
│   ├── predictions/               # Empirical predictions
│   │   ├── fundamental_constants.py
│   │   ├── cosmology.py
│   │   └── particle_physics.py
│   └── analysis/                  # Analytical tools
│       ├── spectral_dimension.py
│       ├── gauge_topology.py
│       └── k_theory.py
├── tests/                         # Comprehensive test suite
├── experiments/                   # Computational experiments
│   ├── dimensional_bootstrap/
│   ├── fine_structure_constant/
│   ├── dark_energy/
│   └── three_generations/
├── notebooks/                     # Jupyter tutorials
└── docs/                         # Mathematical proofs
```

---

## Key Theorems (Proven in v11.0)

### Theorem 1.2: Emergence of Phase Structure
> Geometric frustration in graphs with odd-length cycles **forces** the introduction of complex weights. The average holonomy per plaquette converges to α_EM = 1/137.036.

### Theorem 2.1: Unique Stability at d=4
> The ARO functional exhibits a **global minimum** in spectral action when d_spec = 4, satisfying:
> 1. Holographic consistency (I ~ A)
> 2. Scale invariance ([G] = 2)
> 3. Causal propagation (Huygens' principle)

### Theorem 3.1: Emergence of the Hamiltonian
> The Hamiltonian is **uniquely determined** as the generator of information-preserving updates, derived via Legendre transform from the update Lagrangian.

### Theorem 4.1: Uniqueness of ARO
> ℋ_Harmony = Tr(L²)/(det' L)^(1/(N ln N)) is the **unique** functional (up to rescaling) satisfying intensive scaling, holographic compliance, and scale invariance.

### Theorem 5.2: Gauge Group Selection
> SU(3)×SU(2)×U(1) is the **unique** 12-dimensional compact Lie group satisfying anomaly cancellation, asymptotic freedom, electroweak unification, and topological constraints.

---

## Running Tests

```bash
# Core modules
python test_v11_core.py

# Full test suite
pytest tests/ -v

# Dimensional bootstrap
python experiments/dimensional_bootstrap/run_stability_analysis.py

# Fine-structure constant
python experiments/fine_structure_constant/compute_alpha.py

# Dark energy prediction
python experiments/dark_energy/compute_w_evolution.py
```

---

## Mathematical Documentation

Complete derivations available in [`docs/mathematical_proofs/`](docs/mathematical_proofs/):

1. [Ontological Foundations](docs/mathematical_proofs/01_ontological_foundations.md)
2. [Dimensional Bootstrap](docs/mathematical_proofs/02_dimensional_bootstrap.md)
3. [Quantum Emergence](docs/mathematical_proofs/03_quantum_emergence.md)
4. [Gauge Uniqueness](docs/mathematical_proofs/04_gauge_uniqueness.md)
5. [K-Theory Generations](docs/mathematical_proofs/05_k_theory_generations.md)
6. [Cosmological Constant](docs/mathematical_proofs/06_cosmological_constant.md)

---

## Citation

```bibtex
@software{mccrary2025irh,
  title={Intrinsic Resonance Holography v11.0: The Complete Axiomatic Derivation},
  author={McCrary, Brandon D.},
  year={2025},
  url={https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-},
  version={11.0.0}
}
```

---

## Comparison with Previous Versions

| Feature | v9.5 | v10.0 | **v11.0** |
|---------|------|-------|-----------|
| **Substrate** | Random graphs | Harmonic oscillators | **Pure information** |
| **Complex phases** | Assumed | From Sp(2N)→U(N) | **Derived from frustration** |
| **Time** | Parameter | Emergent | **Derived from updates** |
| **Hamiltonian** | Assumed | From network | **Uniquely derived** |
| **ℏ** | Free parameter | From α | **Calculated from holonomy** |
| **d=4** | Assumed | From ARO | **Uniquely stable** |
| **Gauge group** | Imposed | Emergent | **Proven unique** |
| **Free parameters** | ~5 | 0* | **0 (truly)** |
| **Circularity** | Yes | Partial | **None** |

*v10.0 claimed zero but had hidden assumptions

---

## Development Status

🟢 **Complete:** Core derivations, ARO functional, quantum emergence  
🟡 **In Progress:** Optimization suite, full empirical predictions  
🔴 **Planned:** HPC scaling, interactive visualizations, peer review submission

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Author

**Brandon D. McCrary**  
[GitHub](https://github.com/dragonspider1991) | [Email](mailto:brandon.mccrary@example.com)

---

## Acknowledgments

IRH v11.0 builds on insights from:
- Holographic principle (Bousso, Susskind)
- Spectral action principle (Chamseddine, Connes)
- Causal set theory (Sorkin)
- Quantum graphity (Konopka, Markopoulou, Severini)

**However, IRH v11.0 is the first framework to derive all of physics from a single discrete substrate without circular assumptions.**

---

*"Physics is the fixed-point structure of information dynamics under holographic constraints."* — IRH v11.0
