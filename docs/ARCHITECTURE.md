# IRH_Suite v3.0 Architecture

## Overview

This document describes the architecture of IRH_Suite v3.0, a computational engine for Intrinsic Resonance Holography.

## Module Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.wl                                   │
│                    (Entry Point & Orchestration)                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      HAGOEngine.wl                                │
│                 (Optimization Orchestration)                      │
└─────────────────────────────────────────────────────────────────┘
          │              │              │              │
          ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ MutateGraph  │ │ Acceptance   │ │ Parameter    │ │ ScalingFlows │
│     .wl      │ │     .wl      │ │ Controller   │ │     .wl      │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
          │              │              │
          └──────────────┼──────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   HarmonyFunctional.wl                           │
│           (Γ = βH·Evib + μ·Sholo - α·CAlg + DLor)               │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EigenSpectrum.wl                              │
│              (Robust Eigenvalue Computation)                     │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 InterferenceMatrix.wl                            │
│              (Signed Weighted Laplacian)                         │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GraphState.wl                                 │
│            (Core Data Structure & Validation)                    │
└─────────────────────────────────────────────────────────────────┘
```

## Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Optimized GraphState                          │
└─────────────────────────────────────────────────────────────────┘
          │              │              │              │
          ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Spectral   │ │   Lorentz    │ │ GaugeGroup   │ │  Constant    │
│  Dimension   │ │  Signature   │ │  Analysis    │ │  Derivation  │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
          │              │              │              │
          └──────────────┼──────────────┼──────────────┘
                         ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GrandAudit.wl                               │
│            (CODATA/PDG Validation & Reporting)                   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
project_config.json ──► Configuration
         │
         ▼
    CreateGraphState[N] ──► Initial GraphState
         │
         ▼
    HAGOEngine[...] ◄────────────────────────────┐
         │                                        │
         ├──► MutateGraph ──► Candidate GraphState│
         │                           │            │
         │                           ▼            │
         │              BuildInterferenceMatrix   │
         │                           │            │
         │                           ▼            │
         │                    EigenSpectrum       │
         │                           │            │
         │                           ▼            │
         │                   Gamma[GraphState]    │
         │                           │            │
         │                           ▼            │
         │                    AcceptChange?       │
         │                      │       │         │
         │                   Accept   Reject      │
         │                      │       │         │
         │                      └───────┘         │
         │                           │            │
         │                           ▼            │
         │                  UpdateParameters      │
         │                           │            │
         │                           ▼            │
         │                  AnnealTemperature     │
         │                           │            │
         └──────── (Converged?) ◄────┴────────────┘
                           │
                           ▼
                  Optimized GraphState
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   SpectralDimension  LorentzSignature  GaugeGroupAnalysis
         │                 │                 │
         └─────────────────┼─────────────────┘
                           ▼
                      GrandAudit
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
      G_opt.irh    audit_report.pdf   log_harmony.csv
```

## Module Specifications

### GraphState.wl
- `CreateGraphState[N, opts]`: Create initial graph
- `ValidateGraphState[gs]`: Validate structure
- `GraphStateQ[expr]`: Predicate
- Supports: Random, Complete, Cycle, Lattice topologies

### InterferenceMatrix.wl
- `BuildInterferenceMatrix[gs]`: Construct L = D - W·exp(iΦ)
- Hermitian for real phases, complex for general

### EigenSpectrum.wl
- `EigenSpectrum[gs, opts]`: Full spectrum with degeneracy detection
- `CleanEigenvalues[vals, tol]`: Numerical cleanup
- Methods: Dense (N≤500), Sparse/Arnoldi (N>500)

### HarmonyFunctional.wl
- `Gamma[gs, params]`: Full functional
- `Evib[gs]`: Vibrational energy
- `Sholo[gs]`: Holographic entropy
- `CAlg[gs]`: Algebraic complexity
- `DLor[gs]`: Lorentzian signature term

### MutateGraph.wl
- `MutateGraph[gs, opts]`: Apply mutations
- Kernels: EdgeRewiring, WeightPerturbation, PhaseRotation, Mixed

### Acceptance.wl
- `AcceptChange[ΔΓ, T, opts]`: Metropolis + momentum
- Supports maximizing and minimizing modes

### HAGOEngine.wl
- `HAGOEngine[initGraph, opts]`: Full optimization loop
- Features: Checkpointing, logging, convergence detection

### Analysis Modules
- `SpectralDimension[gs]`: Heat kernel analysis
- `LorentzSignature[gs]`: Eigenvalue signature
- `GaugeGroupAnalysis[gs]`: Automorphism structure
- `ConstantDerivation[gs]`: Physical constants

### I/O and Support
- `SaveGraphState/LoadGraphState`: .irh JSON format
- `Logging`: Timestamped CSV, configurable levels
- `Visualization`: 3D graphs, spectral density, evolution plots

## Configuration Schema

```json
{
  "version": "string",
  "seed": "integer",
  "precision": "integer",
  "maxIterations": "integer",
  "outputDir": "string",
  "logLevel": "DEBUG|INFO|WARNING|ERROR",
  "graphSize": "integer",
  "checkpointInterval": "integer",
  "temperature": {
    "initial": "float",
    "final": "float",
    "schedule": "exponential|linear|cosine"
  },
  "optimizer": {
    "mutationProbability": "float",
    "edgeRewiringWeight": "float",
    "weightPerturbationWeight": "float",
    "phaseRotationWeight": "float"
  },
  "controller": {
    "strategy": "Fixed|Linear|Adaptive|Adam",
    "betaH": "float",
    "mu": "float",
    "alpha": "float",
    "learningRate": "float"
  },
  "analysis": {
    "spectralDimensionFitRange": "[float, float]",
    "eigenvalueTolerance": "float",
    "automorphismMaxIterations": "integer"
  }
}
```

## Testing Architecture

Tests use a simple assertion framework:

```mathematica
TestAssert[name, condition, message]
TestNumericClose[name, value, expected, tolerance]
```

Golden tests verify against analytically known spectra:
- Cycle graph Cn: λk = 2(1 - cos(2πk/n))
- Complete graph Kn: λ = 0 (×1), λ = n (×n-1)
