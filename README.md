# IRH_Suite v9.2

A Computational Engine for Intrinsic Resonance Holography

## Overview

IRH_Suite v9.2 is a complete implementation of the Harmony Functional and HAGO (Harmony-Guided Adaptive Graph Optimization) optimization loop for Intrinsic Resonance Holography research. The suite enables computational exploration of discrete quantum spacetime structures through graph-theoretic methods.

**New in v9.2**: Full Python implementation alongside the original Wolfram Language codebase, with comprehensive validation framework based on the Meta-Theoretical Validation Protocol.

## Features

- **Graph State Management**: Create, validate, and manipulate complex-weighted hypergraphs
- **Harmony Functional**: Compute Γ = βH·Evib + μ·Sholo - α·CAlg + DLor
- **HAGO Optimization**: Simulated annealing with multiple mutation kernels
- **Spectral Analysis**: Compute spectral dimension, Lorentz signature, gauge groups
- **Physical Constants**: Derive coupling constants from graph structure
- **GTEC Functional**: Graph Topological Emergent Complexity measure
- **NCGG Operators**: Non-Commutative Graph Geometry with CCR verification
- **Physics Recovery**: QM entanglement, GR field equations, SM beta functions
- **Predictions**: α⁻¹, neutrino masses, CKM matrix, dark energy EoS
- **Comprehensive Logging**: Timestamped CSV logs and checkpointing
- **Full Test Suite**: Unit tests with golden tests for known analytic spectra

## Requirements

### Python (Recommended)
- Python 3.10+
- NumPy, SciPy, NetworkX, SymPy

### Wolfram Language (Legacy)
- Wolfram Language / Mathematica 14+
- WolframScript (for command-line execution)

## Quick Start

### Python Usage

```bash
# Install dependencies
cd python
pip install -r requirements.txt

# Run tests
PYTHONPATH=$PYTHONPATH:$(pwd)/src pytest tests/ -v

# Interactive usage
python -c "
from irh import HyperGraph
from irh.spectral_dimension import SpectralDimension
from irh.grand_audit import grand_audit

# Create graph
G = HyperGraph(N=64, seed=42)
print(f'Graph: {G.N} nodes, {G.edge_count} edges')

# Compute spectral dimension
ds = SpectralDimension(G)
print(f'Spectral dimension: {ds.value:.2f} ± {ds.error:.2f}')

# Run grand audit
report = grand_audit(G)
print(f'Audit: {report.pass_count}/{report.total_checks} checks passed')
"
```

### Wolfram Language Usage

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
  "version": "9.2",
  "seed": 42,
  "precision": 50,
  "precision_target": 1e-10,
  "N_max": 4096,
  "maxIterations": 1000,
  "outputDir": "io/output",
  "logLevel": "INFO",
  "graphSize": 100,
  "holographic_lambda": 1.0,
  "temperature": {
    "initial": 1.0,
    "final": 0.01,
    "schedule": "exponential"
  }
}
```

### Running Tests

```bash
# Python tests
cd python
PYTHONPATH=$PYTHONPATH:$(pwd)/src pytest tests/ -v

# Wolfram tests
cd tests
wolframscript -file unit_tests.wl
```

## Directory Structure

```
IRH_Suite_v9.2/
├── main.wl                 # Wolfram main entry point
├── project_config.json     # Configuration file
├── changelog.md            # Version history
├── src/                    # Wolfram Language source
│   ├── GraphState.wl       # Graph state creation and validation
│   ├── InterferenceMatrix.wl # Signed weighted Laplacian
│   ├── EigenSpectrum.wl    # Robust eigenvalue computation
│   ├── HarmonyFunctional.wl # Γ and its components
│   ├── ScalingFlows.wl     # Coarse-graining and expansion
│   ├── HAGOEngine.wl       # Main optimization loop
│   ├── SpectralDimension.wl # Spectral dimension analysis
│   └── ...
├── python/                 # Python implementation (v9.2)
│   ├── requirements.txt    # Python dependencies
│   ├── setup.py            # Package setup
│   ├── src/irh/            # Python source modules
│   │   ├── __init__.py
│   │   ├── graph_state.py      # HyperGraph substrate
│   │   ├── spectral_dimension.py # Heat kernel analysis
│   │   ├── scaling_flows.py    # GSRG, metric emergence
│   │   ├── gtec.py             # GTEC functional
│   │   ├── ncgg.py             # NCGG operators
│   │   ├── dhga_gsrg.py        # Homotopy, EFE derivation
│   │   ├── asymptotics.py      # Asymptotic validators
│   │   ├── grand_audit.py      # Comprehensive validation
│   │   ├── dag_validator.py    # DAG enforcement
│   │   ├── recovery/           # Physics recovery suite
│   │   └── predictions/        # Constant predictions
│   └── tests/              # Python test suite
├── tests/
│   └── unit_tests.wl       # Wolfram unit tests
├── io/
│   ├── input/              # Input files
│   └── output/             # Generated artifacts
├── docs/                   # Documentation
├── notebooks/              # Jupyter notebooks
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

### Python API

```python
from irh import HyperGraph
from irh.spectral_dimension import SpectralDimension
from irh.scaling_flows import LorentzSignature, MetricEmergence
from irh.gtec import gtec
from irh.ncgg import NCGG
from irh.grand_audit import grand_audit
from irh.predictions.constants import predict_alpha_inverse

# Create a hypergraph
G = HyperGraph(N=64, seed=42, topology="Random")

# Compute spectral dimension
ds = SpectralDimension(G)
print(f"d_s = {ds.value:.2f} ± {ds.error:.2f}")

# Check Lorentz signature
sig = LorentzSignature(G)
print(f"Signature: {sig.signature}, Physical: {sig.is_physical}")

# GTEC complexity
gtec_result = gtec(G)
print(f"Complexity: {gtec_result.complexity:.4f}")

# NCGG operators and CCR
ncgg = NCGG(G)
ccr = ncgg.verify_all_ccr(max_modes=3)
print(f"CCR verified: {ccr['all_passed']}")

# Physical constant predictions
alpha = predict_alpha_inverse(G)
print(f"α⁻¹ = {alpha.value:.3f} (target: 137.036)")

# Grand audit
report = grand_audit(G)
print(f"Audit: {report.pass_count}/{report.total_checks} passed")
```

### Wolfram API

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

## Validation Framework

IRH v9.2 implements the Meta-Theoretical Validation Protocol with four pillars:

### Pillar A: Ontological Clarity
- Hypergraph substrate validation
- Spectral dimension d_s ≈ 4
- Lorentzian signature (1 negative eigenvalue)
- Holographic bound enforcement

### Pillar B: Mathematical Completeness
- GTEC complexity (C_E > 0)
- NCGG operators and CCR verification
- DHGA topology (target β₁ = 12)
- HGO optimization convergence

### Pillar C: Empirical Grounding
- QM: Entanglement entropy
- GR: Einstein field equations
- SM: Beta functions (QCD b₀ = -7)
- Predictions: α⁻¹ = 137.036

### Pillar D: Logical Coherence
- DAG structure (no circular derivations)
- No ad hoc parameters
- Asymptotic limit consistency

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

*"IRH_Suite v9.2 construction complete. The computational universe is operational."*
