# Changelog

All notable changes to the IRH Suite will be documented in this file.

## [9.2.0] - 2025-11-30

### Added
- Complete Python port of IRH Suite
- `python/src/irh/` module with full implementation:
  - `graph_state.py`: HyperGraph substrate with complex weights
  - `spectral_dimension.py`: Heat kernel and d_s computation
  - `scaling_flows.py`: GSRG coarse-graining, metric emergence, Lorentz signature
  - `gtec.py`: Graph Topological Emergent Complexity functional
  - `ncgg.py`: Non-Commutative Graph Geometry operators
  - `dhga_gsrg.py`: Discrete Homotopy Group Analysis and EFE derivation
  - `asymptotics.py`: Low-energy and continuum limit validators
  - `recovery/`: Physics recovery suite (QM, GR, SM)
  - `predictions/`: Physical constant predictions (α⁻¹, neutrino masses, CKM, w_Λ)
  - `grand_audit.py`: Comprehensive validation framework
  - `dag_validator.py`: Derivational DAG enforcement

### Changed
- Updated `project_config.json` to version 9.2
- Enhanced `.gitignore` for Python artifacts
- Updated README with Python quickstart

### Infrastructure
- Added `.github/workflows/ci-cd.yml` for CI/CD
- Added `python/requirements.txt` with all dependencies
- Added `python/setup.py` for package installation
- Added comprehensive test suite in `python/tests/`

### Validation
- Implemented Meta-Theoretical Validation Protocol
- Four pillars: Ontological Clarity, Mathematical Completeness, Empirical Grounding, Logical Coherence
- Target: 95%+ alignment with IRH v9.2 specifications

---

## [3.0.0] - Previous

### Wolfram Language Implementation
- Original IRH_Suite in Mathematica/Wolfram Language
- Full HAGO optimization loop
- Spectral analysis and constant derivation
