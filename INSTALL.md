# CNCG Installation Guide

This guide covers all installation methods for the CNCG (Computational Non-Commutative Geometry) package.

## Quick Start

### Automated Installation Scripts

We provide automated installation scripts for all platforms:

#### Linux/macOS
```bash
./install.sh
```

#### Windows
```cmd
install.bat
```

These scripts will:
1. Detect your system configuration
2. Offer installation method choices
3. Install all dependencies
4. Verify the installation
5. Provide usage instructions

## Installation Methods

### 1. Conda/Anaconda (Recommended for Scientific Computing)

Conda is recommended because it handles complex scientific dependencies more reliably.

#### Prerequisites
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

#### Steps

**Create environment from file:**
```bash
conda env create -f environment.yml
conda activate cncg
```

**Manual environment creation:**
```bash
conda create -n cncg python=3.10
conda activate cncg
conda install -c conda-forge numpy scipy numba h5py matplotlib networkx pytest
pip install -e .
```

**Verify installation:**
```bash
conda activate cncg
python -c "import cncg; print(cncg.__version__)"
pytest tests/ -v
```

### 2. pip (Standard Python Package Manager)

#### Prerequisites
- Python 3.8 or higher
- pip (usually comes with Python)

#### Steps

**Standard installation:**
```bash
pip install .
```

**Development installation:**
```bash
pip install -e ".[dev]"
```

**Minimal dependencies:**
```bash
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import cncg; print(cncg.__version__)"
python -m pytest tests/ -v
```

### 3. Development Mode

For contributors and developers:

```bash
# Clone the repository
git clone https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-.git
cd Intrinsic-Resonance-Holography-

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run code quality checks
black src/ tests/
flake8 src/ tests/
mypy src/
```

## Troubleshooting

### Common Issues

**Issue: ModuleNotFoundError: No module named 'numba'**
```bash
# Solution: Install numba with conda (recommended)
conda install -c conda-forge numba

# Or with pip
pip install numba
```

**Issue: ImportError related to h5py**
```bash
# Solution: Reinstall h5py
pip uninstall h5py
pip install h5py --no-cache-dir
```

**Issue: Permission denied on install.sh**
```bash
# Solution: Make script executable
chmod +x install.sh
./install.sh
```

**Issue: Conda environment already exists**
```bash
# Solution: Remove and recreate
conda env remove -n cncg
conda env create -f environment.yml
```

### Platform-Specific Notes

#### Windows
- Use Anaconda Prompt or PowerShell
- Some NumPy/SciPy operations may require Microsoft Visual C++ Redistributable

#### macOS
- On Apple Silicon (M1/M2), use conda-forge channel for arm64 builds:
  ```bash
  CONDA_SUBDIR=osx-arm64 conda env create -f environment.yml
  ```

#### Linux
- Ensure development headers are installed:
  ```bash
  # Debian/Ubuntu
  sudo apt-get install python3-dev
  
  # RedHat/CentOS
  sudo yum install python3-devel
  ```

## Verifying Your Installation

After installation, verify everything works:

```bash
# Import the package
python -c "import cncg; print('CNCG version:', cncg.__version__)"

# Run a quick test
python -c "from cncg import FiniteSpectralTriple; t = FiniteSpectralTriple(N=10); print('Created spectral triple:', t)"

# Run the test suite
pytest tests/ -v

# Run a small emergence experiment
python experiments/run_emergence.py --N 30 --n-trials 1 --max-iterations 50
```

If all commands execute without errors, your installation is successful!

## Getting Help

- **Documentation**: See README.md and IMPLEMENTATION_SUMMARY.md
- **Issues**: https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-/issues
- **Tests**: All tests should pass. If not, check dependencies.

## Next Steps

After successful installation:

1. **Read the documentation**: `README.md`
2. **Run examples**: `python experiments/run_emergence.py --help`
3. **Explore the API**: `python -c "import cncg; help(cncg)"`
4. **Try a small experiment**: See Quick Start section in README.md

## Uninstalling

**Conda environment:**
```bash
conda deactivate
conda env remove -n cncg
```

**pip installation:**
```bash
pip uninstall cncg
```

**Development installation:**
```bash
pip uninstall cncg
# Remove the cloned repository directory
```
