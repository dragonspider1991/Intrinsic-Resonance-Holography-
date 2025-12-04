# API Reference

## IRH_Suite v3.0 - Complete API Documentation

This document provides a comprehensive reference for all exported functions in IRH_Suite v3.0.

---

## GraphState Module

### CreateGraphState

Creates a new graph state for IRH computations.

```mathematica
CreateGraphState[n, opts]
```

**Parameters:**
- `n` (Integer): Number of nodes (2 to 10000)
- `opts` (Options): Configuration options

**Options:**
- `"Seed"` → Integer|Automatic: Random seed for reproducibility
- `"Precision"` → Integer: Working precision (default: MachinePrecision)
- `"InitialTopology"` → String: "Random"|"Complete"|"Cycle"|"Lattice"
- `"EdgeProbability"` → Real: Probability for random edges (0 to 1)
- `"WeightDistribution"` → String: "Uniform"|"Gaussian"
- `"PhaseDistribution"` → String: "Uniform"

**Returns:** Association with keys:
- `"Type"`: "GraphState"
- `"AdjacencyMatrix"`: n×n matrix
- `"Weights"`: n×n symmetric matrix
- `"Phases"`: n×n anti-symmetric matrix
- `"NodeCount"`: Integer
- `"EdgeCount"`: Integer
- `"EdgeList"`: List of {i, j} pairs
- `"Metadata"`: Creation metadata

**Example:**
```mathematica
gs = CreateGraphState[100, "Seed" -> 42, "InitialTopology" -> "Random"]
```

### ValidateGraphState

Validates a GraphState association.

```mathematica
ValidateGraphState[gs]
```

**Returns:** `True` if valid, or list of error messages.

### GraphStateQ

Predicate to test if expression is a valid GraphState.

```mathematica
GraphStateQ[expr]
```

**Returns:** `True` or `False`

---

## InterferenceMatrix Module

### BuildInterferenceMatrix

Constructs the signed weighted Laplacian.

```mathematica
BuildInterferenceMatrix[graphState]
```

**Returns:** n×n complex matrix L where:
- L_ij = -w_ij * exp(i*φ_ij) for i ≠ j (connected)
- L_ii = Σ_j w_ij (degree)

---

## EigenSpectrum Module

### EigenSpectrum

Computes eigenvalues and eigenvectors with robust handling.

```mathematica
EigenSpectrum[graphState, opts]
```

**Options:**
- `"Tolerance"` → Real: Numerical tolerance (default: 10^-10)
- `"Method"` → String: "Dense"|"Sparse"|"Arnoldi"|Automatic
- `"DegeneracyTolerance"` → Real: For detecting degeneracies
- `"ReturnVectors"` → Boolean: Include eigenvectors

**Returns:** Association with:
- `"Eigenvalues"`: Sorted list
- `"Eigenvectors"`: Matrix (if requested)
- `"Degeneracies"`: List of degenerate groups
- `"NumericalWarnings"`: Any issues detected

### CleanEigenvalues

Cleans numerical artifacts from eigenvalues.

```mathematica
CleanEigenvalues[eigenvalues, tolerance]
```

---

## HarmonyFunctional Module

### Gamma

Computes the Harmony Functional.

```mathematica
Gamma[graphState, params]
```

**params:** Association with `"betaH"`, `"mu"`, `"alpha"`

**Formula:** Γ = βH·Evib + μ·Sholo - α·CAlg + DLor

### Evib, Sholo, CAlg, DLor

Individual components of the Harmony Functional.

```mathematica
Evib[graphState]   (* Vibrational energy: Σ λ² *)
Sholo[graphState]  (* Holographic entropy: -Σ p log p *)
CAlg[graphState]   (* Algebraic complexity *)
DLor[graphState]   (* Lorentzian signature term *)
```

### GammaComponents

Returns all components as an Association.

```mathematica
GammaComponents[graphState, params]
```

---

## MutateGraph Module

### MutateGraph

Applies mutation operators.

```mathematica
MutateGraph[graphState, opts]
```

**Options:**
- `"MutationKernel"` → String: "EdgeRewiring"|"WeightPerturbation"|"PhaseRotation"|"Mixed"
- `"MutationStrength"` → Real: Scale factor (0 to 1)
- `"KernelWeights"` → List: Weights for Mixed mode

### EdgeRewiring, WeightPerturbation, PhaseRotation

Individual mutation operators.

---

## Acceptance Module

### AcceptChange

Determines whether to accept a proposed change.

```mathematica
AcceptChange[deltaGamma, temperature, opts]
```

**Options:**
- `"UseMomentum"` → Boolean: Use momentum-based acceptance
- `"Maximizing"` → Boolean: Optimization direction
- `"MinAcceptance"` → Real: Minimum acceptance probability

**Returns:** `True` to accept, `False` to reject

### AcceptanceRatio

Computes acceptance probability without making decision.

```mathematica
AcceptanceRatio[deltaGamma, temperature]
```

---

## ScalingFlows Module

### CoarseGrain

Reduces graph complexity by merging nodes.

```mathematica
CoarseGrain[graphState, opts]
```

**Options:**
- `"TargetSize"` → Integer|Automatic: Target node count
- `"Method"` → String: "Spectral"|"Random"|"Degree"
- `"PreserveConnectivity"` → Boolean

### Expand

Increases graph complexity by adding nodes.

```mathematica
Expand[graphState, opts]
```

**Options:**
- `"ExpansionFactor"` → Real: Scale factor
- `"Method"` → String: "Subdivision"|"Duplication"
- `"MaxSize"` → Integer: Maximum allowed size

---

## AROEngine Module

### AROEngine

Main optimization orchestration.

```mathematica
AROEngine[initGraph, opts]
```

**Options:**
- `"MaxIterations"` → Integer
- `"CheckpointInterval"` → Integer
- `"Temperature"` → Association with "initial", "final", "schedule"
- `"Optimizer"` → Association with mutation weights
- `"Controller"` → Association with parameter strategy
- `"OutputDir"` → String
- `"Seed"` → Integer|Automatic
- `"ConvergenceTolerance"` → Real
- `"Maximizing"` → Boolean

**Returns:** Association with:
- `"OptimizedGraph"`: Final GraphState
- `"FinalGamma"`: Best Γ value
- `"History"`: Complete iteration history
- `"TotalIterations"`: Count
- `"Converged"`: Boolean
- `"ExecutionTime"`: Seconds

---

## Analysis Modules

### SpectralDimension

Computes spectral dimension via heat kernel.

```mathematica
SpectralDimension[graphState, opts]
```

**Options:**
- `"FitRange"` → {tMin, tMax}: Diffusion time range
- `"NumPoints"` → Integer: Sample points

**Returns:** Association with "Value", "Error", "FitQuality"

### LorentzSignature

Analyzes eigenvalue signature.

```mathematica
LorentzSignature[graphState, opts]
```

**Options:**
- `"Tolerance"` → Real: Classification threshold
- `"TargetNegative"` → Integer: Expected negative count

**Returns:** Association with counts and signature string

### GaugeGroupAnalysis

Identifies gauge group structure.

```mathematica
GaugeGroupAnalysis[graphState, opts]
```

**Returns:** Association with:
- `"GroupOrder"`: Automorphism group order
- `"Candidates"`: List of candidate Lie groups
- `"Decomposition"`: Suggested factorization

### ConstantDerivation

Derives physical constants from structure.

```mathematica
ConstantDerivation[graphState]
```

**Returns:** Association with derived constants and CODATA comparisons

### GrandAudit

Comprehensive validation against physical constraints.

```mathematica
GrandAudit[graphState, results, opts]
```

**Options:**
- `"OutputDir"` → String
- `"GeneratePDF"` → Boolean
- `"GenerateCSV"` → Boolean

**Returns:** Association with pass/fail results

---

## I/O Module

### SaveGraphState / LoadGraphState

```mathematica
SaveGraphState[graphState, filepath]
LoadGraphState[filepath]
```

File format: `.irh` (JSON)

---

## Visualization Module

### Plot3DGraph

3D visualization of graph structure.

```mathematica
Plot3DGraph[graphState, opts]
```

### PlotSpectralDensity

Eigenvalue distribution histogram.

```mathematica
PlotSpectralDensity[graphState, opts]
```

### PlotGammaEvolution

Optimization trajectory plot.

```mathematica
PlotGammaEvolution[history, opts]
```

### PlotSignature

Eigenvalue signature visualization.

```mathematica
PlotSignature[graphState, opts]
```

---

## Logging Module

### IRHInitializeLog / IRHCloseLog

```mathematica
IRHInitializeLog[outputDir, logLevel]
IRHCloseLog[]
```

### IRHLog

```mathematica
IRHLog[level, message]
```

Levels: "DEBUG", "INFO", "WARNING", "ERROR"

### LogHarmony

```mathematica
LogHarmony[iteration, gamma, meta]
```

Logs to CSV file with timestamp.
