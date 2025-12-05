# IRH Interactive Notebooks

This directory contains Jupyter notebooks that demonstrate and validate Intrinsic Resonance Holography (IRH) computations.

## 🚀 Quick Start

All notebooks can be run directly in Google Colab - no installation required! Just click the "Open in Colab" badge at the top of each notebook.

### For Local Execution

1. Install dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

3. Open any notebook and run all cells

## 📓 Available Notebooks

### [01_ARO_Demo.ipynb](01_ARO_Demo.ipynb)
**Adaptive Resonance Optimization Demo**
- Runtime: ~5 minutes
- Demonstrates how random networks evolve toward 4D spacetime
- Shows convergence of the harmony functional
- Interactive visualizations of network evolution

### [02_Dimensional_Bootstrap.ipynb](02_Dimensional_Bootstrap.ipynb)
**Dimensional Bootstrap Analysis**
- Runtime: ~10 minutes
- Derives d=4 from self-consistency requirements
- Validates spectral dimension computation
- Tests stability across different topologies

### [03_Fine_Structure_Derivation.ipynb](03_Fine_Structure_Derivation.ipynb)
**Fine Structure Constant Derivation**
- Runtime: ~15 minutes
- Computes α⁻¹ ≈ 137.036 from first principles
- No free parameters or adjustable constants
- Comparison with CODATA 2022 experimental value

### [04_Dark_Energy_w(a).ipynb](04_Dark_Energy_w(a).ipynb)
**Dark Energy Equation of State**
- Runtime: ~10 minutes
- Predicts dark energy evolution w(a)
- Comparison with DESI 2024 observations
- Falsifiable predictions for Euclid mission

### [05_Spinning_Wave_Patterns.ipynb](05_Spinning_Wave_Patterns.ipynb)
**Emergent Wave Dynamics**
- Runtime: ~8 minutes
- Visualizes wave patterns on hypergraph substrate
- Demonstrates emergence of oscillatory behavior
- Beautiful animations of resonance modes

### [06_Grand_Audit.ipynb](06_Grand_Audit.ipynb) ⭐ **NEW!**
**Comprehensive Validation Framework**
- Runtime: ~10 minutes (quick mode), ~30-60 minutes (comprehensive)
- **Most comprehensive validation tool**
- 22+ validation checks across 4 foundational pillars
- Convergence testing across multiple network sizes
- Detailed visualizations and export capabilities

**What it validates:**
- ✅ Ontological Clarity (6 checks)
- ✅ Mathematical Completeness (4 checks)
- ✅ Empirical Grounding (6 checks)
- ✅ Logical Coherence (6 checks)
- ✅ Convergence analysis
- ✅ Comparison with experimental data

## 🎯 Recommended Learning Path

1. **Start here:** `01_ARO_Demo.ipynb` - Learn the basics
2. **Theory:** `02_Dimensional_Bootstrap.ipynb` - Understand the framework
3. **Predictions:** `03_Fine_Structure_Derivation.ipynb` - See it in action
4. **Validation:** `06_Grand_Audit.ipynb` - Comprehensive testing
5. **Explore:** `04_Dark_Energy_w(a).ipynb` and `05_Spinning_Wave_Patterns.ipynb`

## 💻 Command Line Alternative

Prefer command line? Use the standalone scripts:

```bash
# Quick grand audit
python ../scripts/run_enhanced_grand_audit.py --quick

# Full audit with visualizations
python ../scripts/run_enhanced_grand_audit.py --full --output results/
```

## 📊 Output Formats

Notebooks can export results in multiple formats:
- **JSON**: Structured data for further analysis
- **CSV**: Tabular data for spreadsheets
- **PNG**: High-resolution visualizations
- **TXT**: Human-readable summaries

## 🔧 Customization

All notebooks support customization:
- Adjust network size `N` for speed vs. accuracy tradeoff
- Modify random seed for different realizations
- Change visualization parameters
- Export custom subsets of results

## 📚 Documentation

For detailed documentation on the theoretical framework and implementation:
- [Main README](../README.md)
- [Mathematical Proofs](../docs/mathematical_proofs/)
- [Implementation Status](../IMPLEMENTATION_COMPLETE.md)

## 🐛 Troubleshooting

### "Module not found" errors
- **In Colab**: The setup cell installs dependencies automatically
- **Locally**: Run `pip install -r ../requirements.txt`

### Slow execution
- Reduce network size: `N = 32` instead of `N = 256`
- Use quick mode in Grand Audit
- Skip convergence analysis for faster results

### Memory issues
- Reduce `N` to a smaller value (32, 64, or 128)
- Close other applications
- Restart kernel and clear all outputs

## 🤝 Contributing

Found a bug or want to add a notebook? See our [contribution guidelines](../CONTRIBUTING.md).

## 📝 Citation

If you use these notebooks in your research, please cite:

```bibtex
@software{mccrary2025irh,
  title={Intrinsic Resonance Holography v11.0: The Complete Axiomatic Derivation},
  author={McCrary, Brandon D.},
  year={2025},
  url={https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-}
}
```

## 📧 Support

Questions? Issues? Contact us:
- [GitHub Issues](https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-/issues)
- Email: brandon.mccrary@example.com

---

**Happy exploring!** 🚀
