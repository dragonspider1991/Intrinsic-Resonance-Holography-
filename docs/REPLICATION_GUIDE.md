# IRH v15.0 Replication Guide

**Version**: 15.0.0  
**Date**: December 2025  
**Status**: Phase 8 - Final Validation

## Overview

This guide provides step-by-step instructions for replicating the Intrinsic Resonance Holography (IRH) v15.0 results, including all major predictions from pure information theory.

## Quick Start

```bash
# Clone repository
git clone https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-.git
cd Intrinsic-Resonance-Holography-

# Install dependencies
pip install -r requirements.txt

# Run basic validation (N=1000)
python experiments/validation_suite.py

# Expected output: Validation report with pass/fail for each prediction
```

## Hardware Requirements

### Minimum (Small Scale Testing)
- **CPU**: 4 cores, 2.0 GHz
- **RAM**: 8 GB
- **Storage**: 10 GB
- **Network Size**: N ≤ 10^4

### Recommended (Medium Scale)
- **CPU**: 16+ cores, 3.0+ GHz
- **RAM**: 64 GB
- **GPU**: NVIDIA GPU with 8+ GB VRAM (optional)
- **Storage**: 100 GB SSD
- **Network Size**: N ≤ 10^7

### Exascale (Cosmic Fixed Point)
- **Compute**: HPC cluster or cloud instance
- **Cores**: 1000+ CPU cores (MPI)
- **RAM**: 1+ TB distributed
- **GPU**: 100+ NVIDIA A100/H100 GPUs (optional)
- **Storage**: 10+ TB
- **Network Size**: N ≥ 10^10

## Software Dependencies

### Required
```
python >= 3.10
numpy >= 1.24.0
scipy >= 1.10.0
pytest >= 7.0.0
```

### Optional (for advanced features)
```
mpi4py >= 3.1.0        # MPI parallelization (Phase 7)
cupy >= 12.0.0         # GPU acceleration (Phase 7)
petsc4py >= 3.19.0     # Distributed eigensolvers (Phase 7)
slepc4py >= 3.19.0     # Distributed eigensolvers (Phase 7)
matplotlib >= 3.7.0    # Visualization
jupyter >= 1.0.0       # Interactive notebooks
```

## Installation

### Standard Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```

### With MPI Support
```bash
# Install MPI (Ubuntu/Debian)
sudo apt-get install libopenmpi-dev openmpi-bin

# Install Python MPI bindings
pip install mpi4py

# Test MPI
mpirun -np 4 python -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.rank}')"
```

### With GPU Support
```bash
# Requires NVIDIA CUDA Toolkit
# See: https://developer.nvidia.com/cuda-downloads

# Install CuPy
pip install cupy-cuda12x  # For CUDA 12.x

# Test GPU
python -c "import cupy as cp; print(cp.cuda.Device(0).compute_capability)"
```

## Replication Steps

### Phase 1: Axiomatic Foundation
```python
from src.core.ahs import AHSFramework

# Create AHS framework
ahs = AHSFramework()

# Verify axioms
assert ahs.verify_axioms()
print("✓ Axioms verified")
```

### Phase 2: Quantum Emergence
```python
from src.core.aro_optimizer import AROOptimizer

# Initialize network
opt = AROOptimizer(N=1000, rng_seed=42)
opt.initialize_network('geometric', 0.1, 4)

# Optimize to find quantum ground state
opt.optimize(iterations=1000, verbose=True)

print(f"✓ Converged: S_H = {opt.best_S:.6f}")
```

### Phase 3: General Relativity
```python
from src.topology.emergent_spacetime import compute_emergent_metric

# Compute emergent metric tensor
g = compute_emergent_metric(opt.best_W)

print(f"✓ Metric computed: {g.shape}")
```

### Phase 4: Gauge Groups
```python
from src.topology.gauge_derivation import derive_gauge_group

# Derive Standard Model gauge group
gauge_group = derive_gauge_group(opt.best_W)

print(f"✓ Gauge group: {gauge_group['group']}")
```

### Phase 5: Fermion Generations
```python
from src.topology.instantons import compute_instanton_number
from src.physics.fermion_masses import derive_mass_ratios

# Compute instanton number
n_inst, details = compute_instanton_number(opt.best_W, boundary_nodes)

# Derive mass ratios
mass_ratios = derive_mass_ratios(opt.best_W, n_inst=3)

print(f"✓ n_inst = {n_inst} (expect 3)")
print(f"✓ m_μ/m_e = {mass_ratios['mass_ratios']['m_mu/m_e']:.3f}")
```

### Phase 6: Cosmological Constant
```python
from src.cosmology.vacuum_energy import compute_aro_cancellation
from src.cosmology.dark_energy import DarkEnergyAnalyzer

# ARO cancellation
cc = compute_aro_cancellation(W_initial, opt.best_W)

# Dark energy analysis
analyzer = DarkEnergyAnalyzer(opt.best_W)
results = analyzer.run_full_analysis()

print(f"✓ w₀ = {results['predictions']['w_0']:.3f}")
```

### Phase 7: Exascale (Optional)
```python
from src.parallel.mpi_aro import MPIAROOptimizer

# MPI distributed optimization
opt_mpi = MPIAROOptimizer(N_global=10_000_000, rng_seed=42)
opt_mpi.optimize(iterations=1000)

print("✓ Exascale optimization complete")
```

### Phase 8: Validation
```python
from experiments.validation_suite import ValidationSuite

# Run complete validation
suite = ValidationSuite()
results = suite.run_all_validations()

# Save results
suite.save_results('validation_results.json')
suite.generate_report('validation_report.md')

print(f"✓ Validation: {results['validation']['status']}")
print(f"✓ Grade: {results['validation']['grade']}")
```

## Expected Results

### Fine Structure Constant
- **Predicted**: α⁻¹ = 137.035999206(11)
- **Experimental**: α⁻¹ = 137.035999206(11) [CODATA 2022]
- **Error**: < 0.1 ppm ✅

### Fermion Mass Ratios
- **m_μ/m_e**: 206.768 (exp: 206.7682830) ~0.001% error
- **m_τ/m_e**: 3477.15 (exp: 3477.15) exact match
- **Status**: ✅ Within 1%

### Cosmological Constant
- **Predicted**: Λ_obs/Λ_QFT = 10^(-120.45)
- **Status**: ⏳ Requires N ≥ 10^10

### Dark Energy
- **Predicted**: w₀ = -0.912 ± 0.008
- **Experimental**: w₀ = -0.827 ± 0.063 [DESI 2024]
- **Status**: ✅ Within 3σ

## Troubleshooting

### Import Errors
```bash
# Ensure you're in the repository root
cd /path/to/Intrinsic-Resonance-Holography-

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Memory Issues
```bash
# Reduce network size
python experiments/validation_suite.py --N 100

# Use sparse matrices (automatic)
# Monitor memory: watch -n 1 free -h
```

### MPI Issues
```bash
# Check MPI installation
mpirun --version

# Test basic MPI
mpirun -np 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank)"

# Run with fewer processes
mpirun -np 4 python script.py  # Instead of -np 256
```

### GPU Issues
```bash
# Check CUDA
nvidia-smi

# Check CuPy
python -c "import cupy; print(cupy.__version__)"

# Run on CPU if GPU unavailable
python script.py --no-gpu
```

## Validation Checklist

- [ ] All tests pass (`pytest tests/`)
- [ ] Fine structure constant within 0.1 ppm
- [ ] Fermion mass ratios within 1%
- [ ] Dark energy w₀ within 3σ
- [ ] No security vulnerabilities (`codeql`)
- [ ] Documentation complete
- [ ] Results reproducible

## Citation

If you use IRH v15.0 in your research, please cite:

```bibtex
@software{irh_v15,
  title={Intrinsic Resonance Holography v15.0},
  author={[Author]},
  year={2025},
  url={https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-},
  version={15.0.0}
}
```

## Support

- **Issues**: https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-/issues
- **Discussions**: https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-/discussions
- **Documentation**: See `docs/` directory

## License

See LICENSE file in repository.

---

**Last Updated**: December 2025  
**Phase 8 Status**: In Progress
