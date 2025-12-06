# Obsolete and Versioned Files

This directory contains obsolete, deprecated, and older versioned files that have been superseded by newer implementations but are kept for reference and historical purposes.

## Contents

### Root Level
- `README_v9.5_old.md` - Old v9.5 README (superseded by current README.md)
- `TERMINOLOGY_UPDATE_v10.0.md` - v10.0 terminology update (superseded by v13)
- `main_cli_backup.py` - Backup of old CLI script

### Source Code (`src/`)
- `irh_v10/` - Complete v10 implementation package (superseded by v13 in main src/)
- `v11_standalone/` - Standalone v11 modules (aro_v11.py, quantum_v11.py, sote_v11.py, substrate_v11.py)

### Documentation (`docs/`)
- `Final_Manuscript_v9.5.md` - v9.5 manuscript (superseded by current manuscripts)
- `STRUCTURE_v13.md` - v13 structure documentation (may still be useful for reference)
- `archive_pre_v13/` - Pre-v13 archive containing older requirements, tests, and documentation

### Examples (`examples/`)
- `v13/` - v13-specific examples (may be incorporated into main examples)

### Python Package (`python/`)
- Complete v9.2 Python package structure (superseded by current src/ implementation)
  - Contains its own src/irh/, tests/, setup.py, etc.

## Current Active Version

The repository is currently on **v13.0**. All active code is in the main directories:
- Source code: `src/`
- Documentation: `docs/`
- Examples: `examples/`
- Tests: `tests/`

These archived files are maintained for:
1. Historical reference
2. Tracking evolution of the framework
3. Potential recovery of specific implementations if needed
4. Understanding design decisions made in previous versions

**Note**: Files in this directory should not be used for new development. Always refer to the main repository directories for current implementations.
