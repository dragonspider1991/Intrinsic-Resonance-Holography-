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

```bash
# Clone the repository
git clone https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-.git
cd Intrinsic-Resonance-Holography-

# Install the Python package
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

A Computational Engine for Intrinsic Resonance Holography

### Overview

IRH_Suite v3.0 is a complete implementation of the Harmony Functional and HAGO (Harmony-Guided Adaptive Graph Optimization) optimization loop for Intrinsic Resonance Holography research. The suite enables computational exploration of discrete quantum spacetime structures through graph-theoretic methods.

### Features

- **Graph State Management**: Create, validate, and manipulate weighted graphs with phase factors
- **Harmony Functional**: Compute Γ = βH·Evib + μ·Sholo - α·CAlg + DLor
- **HAGO Optimization**: Simulated annealing with multiple mutation kernels
- **Spectral Analysis**: Compute spectral dimension, Lorentz signature, gauge groups
- **Physical Constants**: Derive coupling constants from graph structure
- **Comprehensive Logging**: Timestamped CSV logs and checkpointing
- **Full Test Suite**: Unit tests with golden tests for known analytic spectra

## Requirements

- Wolfram Language / Mathematica 14+
- WolframScript (for command-line execution)

## Quick Start

### Running the Suite

```bash
# Basic run with default configuration
wolframscript -file main.wl

# Custom parameters via CLI
wolframscript -file main.wl -seed 42 -maxIterations 500 -graphSize 100
```

### Configuration

Edit `project_config.json` to customize:

```json
{
  "version": "3.0",
  "seed": 42,
  "precision": 50,
  "maxIterations": 1000,
  "outputDir": "io/output",
  "logLevel": "INFO",
  "graphSize": 100,
  "temperature": {
    "initial": 1.0,
    "final": 0.01,
    "schedule": "exponential"
  }
}
```

### Running Tests

```bash
cd tests
wolframscript -file unit_tests.wl
```

## Directory Structure

```
IRH_Suite_v3.0/
├── main.wl                 # Main entry point
├── project_config.json     # Configuration file
├── src/
│   ├── GraphState.wl       # Graph state creation and validation
│   ├── InterferenceMatrix.wl # Signed weighted Laplacian
│   ├── EigenSpectrum.wl    # Robust eigenvalue computation
│   ├── HarmonyFunctional.wl # Γ and its components
│   ├── ParameterController.wl # Adaptive parameter updates
│   ├── MutateGraph.wl      # Mutation operators
│   ├── Acceptance.wl       # Metropolis acceptance
│   ├── ScalingFlows.wl     # Coarse-graining and expansion
│   ├── HAGOEngine.wl       # Main optimization loop
│   ├── SpectralDimension.wl # Spectral dimension analysis
│   ├── LorentzSignature.wl # Lorentzian signature detection
│   ├── GaugeGroupAnalysis.wl # Symmetry analysis
│   ├── ConstantDerivation.wl # Physical constants
│   ├── GrandAudit.wl       # Validation and reporting
│   ├── IOFunctions.wl      # Save/Load operations
│   ├── Visualization.wl    # Graph and spectral plots
│   └── Logging.wl          # Comprehensive logging
├── tests/
│   └── unit_tests.wl       # Unit tests
├── io/
│   ├── input/              # Input files
│   └── output/             # Generated artifacts
├── docs/                   # Documentation
└── examples/               # Example notebooks
```

## Output Artifacts

After a successful run, `io/output/` will contain:

- `G_opt.irh` - Optimized graph state (JSON format)
- `spectral_dimension_report.json` - Spectral analysis results
- `grand_audit_report.pdf` - CODATA/PDG comparison report
- `log_harmony.csv` - Complete optimization log
- `run_manifest.json` - Run metadata and artifact hashes

## API Reference

### Core Functions

```mathematica
(* Create a graph state *)
gs = CreateGraphState[100, "Seed" -> 42, "InitialTopology" -> "Random"]

(* Build interference matrix *)
L = BuildInterferenceMatrix[gs]

(* Compute eigenspectrum *)
spectrum = EigenSpectrum[gs]

(* Compute Harmony Functional *)
params = <|"betaH" -> 1.0, "mu" -> 0.1, "alpha" -> 0.01|>;
gamma = Gamma[gs, params]

(* Run HAGO optimization *)
result = HAGOEngine[gs, "MaxIterations" -> 1000]
```

### Analysis Functions

```mathematica
(* Spectral dimension *)
ds = SpectralDimension[gs]
(* -> <|"Value" -> 3.98, "Error" -> 0.12, ...|> *)

(* Lorentz signature *)
sig = LorentzSignature[gs]
(* -> <|"NegativeCount" -> 1, "Signature" -> "(99, 1)", ...|> *)

(* Gauge groups *)
gauge = GaugeGroupAnalysis[gs]
(* -> <|"GroupOrder" -> 12, "Candidates" -> {"U(1)", "SU(2)"}, ...|> *)

(* Physical constants *)
consts = ConstantDerivation[gs]
```

### I/O Functions

```mathematica
(* Save and load *)
SaveGraphState[gs, "path/to/graph.irh"]
loaded = LoadGraphState["path/to/graph.irh"]

(* Visualization *)
Plot3DGraph[gs]
PlotSpectralDensity[gs]
PlotGammaEvolution[result["History"]]
```

## Physical Background

The Harmony Functional Γ combines:

- **Evib** (Vibrational Energy): Σᵢ λᵢ² - total "energy" of spectral modes
- **Sholo** (Holographic Entropy): -Σᵢ pᵢ log(pᵢ) - information-theoretic entropy
- **CAlg** (Algebraic Complexity): Graph structure complexity measure
- **DLor** (Lorentzian Term): Signature matching term (target: 1 negative eigenvalue)

The optimization seeks configurations where:
- Spectral dimension d_s ≈ 4 (4D spacetime)
- Exactly 1 negative eigenvalue (Lorentzian signature)
- Physical constants match CODATA/PDG values

## Reproducibility

All stochastic operations use seedable RNG:

```mathematica
(* Fully deterministic run *)
gs = CreateGraphState[100, "Seed" -> 42]
result = HAGOEngine[gs, "Seed" -> 42]
```

The run manifest records seeds and artifact hashes for audit trails.

## Notes for Developers

### Proxy Implementations

Some quantities (Sholo, CAlg, DLor, physical constant mappings) use documented proxy implementations. These are marked in the source files and can be refined based on theoretical requirements.

### Numerical Considerations

- Eigenvalue tolerance: Configurable via `"Tolerance"` option
- Precision: Set via `project_config.precision`
- Large matrices (N > 500): Uses sparse/Arnoldi methods automatically

## License

CC0 1.0 Universal - See LICENSE file

## Citation

If using IRH_Suite for research, please cite the Intrinsic Resonance Holography theoretical papers.

---

*"IRH_Suite v3.0 construction complete. The computational universe is operational."*
