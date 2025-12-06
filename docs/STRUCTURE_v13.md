# IRH v13.0 Repository Structure

This document describes the reorganized repository structure for Intrinsic Resonance Holography v13.0.

## Directory Structure

```
Intrinsic-Resonance-Holography-/
├── main.py                         # CLI entry point for v13.0
├── setup.py                        # Package setup (kept from v14.0 CNCG)
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Modern Python project config
│
├── src/                           # Source code modules
│   ├── core/                      # ARO Engine and Harmony Functional
│   ├── topology/                  # Topological invariants (Betti, Frustration)
│   ├── cosmology/                 # Dark Energy/Horizon simulations
│   ├── metrics/                   # Dimensional Coherence Index & checks
│   ├── utils/                     # Math helpers (sparse matrices, zeta reg)
│   ├── cncg/                      # v14.0 CNCG package (preserved)
│   ├── irh_v10/                   # v10.0 legacy (preserved)
│   ├── predictions/               # Physical predictions
│   └── simulations/               # Simulation modules
│
├── tests/                         # Test suite
│   ├── unit/                      # pytest modules for individual functions
│   ├── integration/               # Cosmic Fixed Point Test & integration tests
│   └── *.py                       # Existing test files (preserved)
│
├── docs/                          # Documentation
│   ├── manuscripts/               # Theory manuscripts
│   │   └── IRH_v13_0_Theory.md   # v13.0 manuscript (placeholder)
│   ├── api/                       # Generated API documentation
│   ├── archive/                   # Archived legacy files
│   │   └── pre_v13/              # Files from before v13.0 reorganization
│   ├── derivations/               # Mathematical derivations
│   └── mathematical_proofs/       # Formal proofs
│
├── experiments/                   # Long-term simulation experiments
│   ├── dimensional_bootstrap/
│   ├── run_emergence.py
│   └── plot_robustness.py
│
├── notebooks/                     # Jupyter notebooks
├── examples/                      # Example scripts
├── scripts/                       # Utility scripts
├── webapp/                        # Web application
└── io/                           # Input/output utilities

## Archived Files (docs/archive/pre_v13/)

The following files were moved to the archive during the v13.0 reorganization:

### Python Files
- `orchestrator.py` - v11.0 orchestration script
- `setup_v11.py` - v11.0 setup configuration
- `test_v11_core.py` - v11.0 core tests
- `test_orchestrator.py` - Orchestrator tests
- `test_enhanced_audit.py` - Enhanced audit tests

### Text Files
- `Theory Validation Data Request.txt`
- `Validating Intrinsic Resonance Holography.txt`
- `fty.txt`
- `wolfram_notebook_prompt.txt`
- `requirements_v9.5_old.txt`

## CLI Usage

The new `main.py` provides a unified command-line interface:

```bash
# Run simulations
python main.py run --mode aro --N 1000 --output results/

# Run validation tests
python main.py validate --test cosmic-fixed-point

# Compute observables
python main.py compute --observable fine-structure
```

### Available Commands

- **run**: Run IRH simulations
  - Modes: aro, dimensional, cosmology
  - Configurable system size (--N)
  - Output directory specification

- **validate**: Run validation tests
  - Tests: cosmic-fixed-point, harmony, holographic

- **compute**: Compute physical observables
  - Observables: fine-structure, spectral-dimension, dark-energy, generations

## Module Organization

### src/core/
ARO Engine and Harmony Functional implementations per v13.0 specifications.

### src/topology/
Topological invariant calculations:
- Betti numbers
- Frustration density (ρ_frust → α^(-1))
- Minimal cycle basis algorithms

### src/cosmology/
Dark Energy and cosmological simulations:
- Horizon dynamics
- w(a) evolution predictions

### src/metrics/
Dimensional analysis:
- Spectral dimension (d_spec)
- Dimensional Coherence Index (χ_D = ℰ_H × ℰ_R × ℰ_C)
- Holographic bound checks

### src/utils/
Mathematical utilities:
- Sparse matrix operations (CSR format for large N)
- Spectral Zeta regularization
- Log-sum-exp for numerical stability

## Testing

### Unit Tests (tests/unit/)
Individual function tests using pytest.

### Integration Tests (tests/integration/)
System-level tests including the Cosmic Fixed Point Test.

## Next Steps

Phase 2 will implement:
1. Core engine refactoring with rigorous v13.0 mathematics
2. Harmony Functional with Spectral Zeta Regularization
3. ARO optimization with hybrid strategy
4. Topology and metrics modules
5. Integration of provided v13.0 scripts

## Version History

- **v13.0**: Current reorganization (this structure)
- **v14.0**: CNCG package (preserved in src/cncg/)
- **v11.0**: Legacy implementation (archived)
- **v10.0**: Legacy implementation (preserved in src/irh_v10/)
- **v9.5**: Early version (some files archived)

---

*Generated: 2025-12-06*
*Status: Phase 1 Complete, Phase 2 Pending*
