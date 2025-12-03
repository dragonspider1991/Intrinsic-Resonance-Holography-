# IRH Orchestrator - Quick Start Guide

## Overview

The `orchestrator.py` script is a unified entry point for installing, configuring, and running the Intrinsic Resonance Holography (IRH) theoretical physics suite across different environments.

## Supported Environments

1. **Google Colab** - Automatic GDrive mounting, repository cloning, and dependency installation
2. **Local Bash Terminal** (Linux/Mac) - Virtual environment management and setup
3. **Windows** - Windows-specific virtual environment handling
4. **Wolfram Language/Mathematica** - Generation of .wls scripts and notebook prompts

## Quick Start

### Basic Usage (Interactive Mode)

```bash
# Using Python directly
python3 orchestrator.py

# Or using the setup.sh helper (Linux/Mac only)
./setup.sh
```

This will:
1. Detect your environment
2. Guide you through an interactive configuration wizard
3. Install dependencies (if needed)
4. Run the selected simulation modules

### Non-Interactive Mode

```bash
# Use default configuration without prompts
python3 orchestrator.py --non-interactive --skip-setup
```

### Wolfram Language Integration

```bash
# Generate Wolfram Language assets only
python3 orchestrator.py --wolfram-only

# This creates:
# - irh_wolfram_kernel.wls (Wolfram script)
# - wolfram_notebook_prompt.txt (LLM prompt for notebooks)
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--non-interactive` | Skip interactive prompts, use defaults/existing config |
| `--skip-setup` | Skip environment setup and dependency installation |
| `--reconfigure` | Force reconfiguration even if config.json exists |
| `--wolfram` | Generate Wolfram Language assets |
| `--wolfram-only` | Only generate Wolfram assets, skip Python execution |

## Configuration

The orchestrator saves your preferences to `config.json`. You can:

- **Edit manually**: Open `config.json` and modify values
- **Reconfigure interactively**: Run with `--reconfigure` flag
- **Use defaults**: Delete `config.json` to reset to defaults

### Configuration Options

```json
{
  "grid_size_N": 1000,           // Grid size (10-100000)
  "run_gtec": true,               // Run GTEC module
  "run_ncgg": true,               // Run NCGG module
  "run_cosmology": false,         // Run Cosmology calculations
  "output_verbosity": "brief",    // "brief" or "debug"
  "max_iterations": 1000,         // Max optimization iterations
  "precision": "high",            // "low", "medium", or "high"
  "use_gpu": false,               // Enable GPU acceleration
  "output_dir": "./outputs"       // Output directory
}
```

## Error Handling

If an error occurs, the orchestrator generates a detailed crash report:

- **File**: `crash_report_for_llm.txt`
- **Contents**: Full stack trace, system state, configuration, and suggested fixes
- **Use**: Share with an LLM for debugging assistance

## Module Descriptions

### GTEC (Graph Topological Emergent Complexity)
- Computes entanglement entropy for bipartite graph partitions
- Verifies dark energy cancellation mechanism
- Required for: Dark energy calculations

### NCGG (Non-Commutative Graph Geometry)
- Constructs position and momentum operators on graphs
- Computes quantum commutators
- Required for: Quantum emergence calculations

### Cosmology
- Derives dark energy equation of state w(a)
- Computes cosmological constants
- Optional advanced module

## Examples

### Example 1: First-time Setup (Interactive)

```bash
./setup.sh
# Follow the wizard prompts:
# - Grid Size N: 1000 [Enter]
# - Run GTEC? y [Enter]
# - Run NCGG? y [Enter]
# - Run Cosmology? n [Enter]
# - Verbosity: brief [Enter]
```

### Example 2: Automated Run (CI/CD)

```bash
python3 orchestrator.py --non-interactive --skip-setup
```

### Example 3: Reconfigure and Run

```bash
python3 orchestrator.py --reconfigure
```

### Example 4: Generate Wolfram Assets for External Use

```bash
python3 orchestrator.py --wolfram-only
# Then in Mathematica:
wolframscript -file irh_wolfram_kernel.wls
```

## Troubleshooting

### "No module named X"
**Solution**: Install dependencies manually
```bash
pip install -r requirements.txt
```

### "MemoryError"
**Solution**: Reduce grid size N in config.json
```json
{"grid_size_N": 500}
```

### Virtual Environment Issues
**Solution**: Create and activate manually
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows
python3 orchestrator.py
```

### Permission Denied
**Solution**: Make scripts executable
```bash
chmod +x orchestrator.py setup.sh
```

## Advanced Usage

### Custom Output Directory
Edit `config.json`:
```json
{"output_dir": "/path/to/custom/outputs"}
```

### High-Performance Computing
For large simulations:
```json
{
  "grid_size_N": 10000,
  "precision": "high",
  "use_gpu": true,
  "max_iterations": 5000
}
```

### Debug Mode
For detailed logging:
```json
{"output_verbosity": "debug"}
```

## Support

- **Repository**: https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-
- **Issues**: Use GitHub Issues for bug reports
- **Documentation**: See `/docs` directory

## License

CC0-1.0 (Public Domain)
