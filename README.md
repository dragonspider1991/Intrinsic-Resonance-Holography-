# IRH_Suite v3.0

A Computational Engine for Intrinsic Resonance Holography

## Overview

IRH_Suite v3.0 is a complete implementation of the Harmony Functional and HAGO (Harmony-Guided Adaptive Graph Optimization) optimization loop for Intrinsic Resonance Holography research. The suite enables computational exploration of discrete quantum spacetime structures through graph-theoretic methods.

## Features

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
