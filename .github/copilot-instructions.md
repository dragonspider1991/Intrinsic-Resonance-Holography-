# Intrinsic Resonance Holography (IRH) — Copilot Instructions

These instructions onboard both the coding agent and the review agent. They are project-wide (not task-specific) and limited to the essentials—trust them first and only search further if something here is incomplete or shown to be wrong.

## 1) What this repo is
- Research codebase for **Intrinsic Resonance Holography**: derives fundamental constants and physical laws from Algorithmic Holonomic States and cGFT fixed points (v16 foundations, v17 analytical derivations, v18 plan).
- Primary active code: **Python 3.11/3.12** package in `python/src/irh` with tests in `python/tests`.
- Legacy/Mathematica + older Python scaffolding in root `src/` with legacy tests in root `tests/`.
- Web assets: `webapp/backend` (FastAPI) and `webapp/frontend` (React/Vite). Rich documentation under `docs/` and phase/status summaries in root.

## 2) Layout cheat sheet
- Root files: `README.md`, `CONTRIBUTING.md`, `pyproject.toml`, `requirements.txt`, `setup.py`, `setup.sh`, phase/status reports (`PHASE_*`, `IMPLEMENTATION_SUMMARY.md`, `RIGOR_ENHANCEMENTS_SUMMARY.md`).
- Active Python package: `python/src/irh/core/v16|v17|v18`, shared helpers in `python/src/irh`, tests in `python/tests/{v16,v17,v18,test_irh.py}`, fixtures in `python/tests/conftest.py`.
- Legacy code: `src/` (Python + `.wl`), legacy tests in `tests/` expect imports like `src.core.*`.
- Docs: `docs/manuscripts/IRHv16.md`, `IRHv16_Supplementary_Vol_1-5.md`, `IRHv17.md`, `IRHv18.md`, `docs/v16_IMPLEMENTATION_ROADMAP.md`, `docs/v18_IMPLEMENTATION_PLAN.md`, plus status docs.
- Web: `webapp/backend` (FastAPI app.py), `webapp/frontend` (Vite dev server).

## 3) Bootstrap (validated)
- Use Python **3.11 or 3.12**. From repo root:
  ```bash
  python -m pip install -e .[dev]
  ```
  (Validated 2025-12; installs numpy/scipy/networkx/qutip/matplotlib/sympy/mpmath + pytest/black/ruff/mypy.)
- Pathing is critical:
  - Working in `python/`: `export PYTHONPATH=$(pwd)/src`
  - Working in repo root (legacy tests/scripts): `export PYTHONPATH=$PWD`
- Optional: install pre-commit hooks (`pip install pre-commit && pre-commit install`) if modifying style-critical files.

## 4) Test commands (validated + gotchas)
- Active Python suite (preferred):
  ```bash
  cd python
  export PYTHONPATH=$(pwd)/src
  pytest tests/v16/test_ahs.py          # passes (~0.5s)
  pytest tests/                         # full python suite; use -k to scope
  ```
- Legacy/root suite (needs PYTHONPATH):
  ```bash
  cd /home/runner/work/Intrinsic-Resonance-Holography-/Intrinsic-Resonance-Holography-
  export PYTHONPATH=$PWD
  pytest tests/test_v16_core.py         # passes; without PYTHONPATH fails ModuleNotFound: src
  pytest tests/                         # may include Wolfram/legacy cases; expect failures if deps missing
  ```
- Baseline observed: running `pytest` from root **without** setting PYTHONPATH triggers import failures for `src.*`. `pytest` warns about unknown mark `slow` (harmless). Document if you intentionally skip legacy suites.

## 5) Lint, type-check, build
- Python package:
  ```bash
  cd python
  export PYTHONPATH=$(pwd)/src
  ruff check src/
  black --check src/ tests/ --line-length 100
  mypy src/irh/ --ignore-missing-imports
  ```
- Legacy root (only if touching root `src/`): `ruff check src/ --ignore E501`; `black --check src/ tests/`.
- Build (root pyproject uses setuptools): `python -m build && twine check dist/*`.
- CI mirrors these with some `|| true`; don’t rely on leniency locally.

## 6) Run/demo
- v16 demo: `python project_irh_v16.py` (root; set `PYTHONPATH=$PWD` if imports needed).
- Web:  
  - Backend: `cd webapp/backend && pip install -r requirements.txt && python app.py`  
  - Frontend: `cd webapp/frontend && npm install && npm run dev` (Vite defaults to http://localhost:5173).

## 7) Coding conventions (essentials)
- PEP 8, line length 100, full type hints, NumPy-style docstrings with manuscript equation references.
- Phase handling: wrap with `np.mod(angle, 2*np.pi)`; compare via `_wrapped_phase_difference` ([-π, π], π→-π) and `PHASE_TOLERANCE = 1e-10`.
- Input normalization: use `_to_bytes` (str/bytes/bytearray→bytes) shared across v16 ACW/AHS.
- Complex amplitudes: wrap `np.exp(...)` in `complex(...)` to avoid numpy subclass warnings.
- Keep changes minimal; place new code in `python/src/irh/...` with matching tests in `python/tests/...`.

## 8) CI awareness
- Workflows: `.github/workflows/ci.yml` (pytest on `tests/`, ruff on `src/`, mypy on `src/irh_v10`) and `ci-cd.yml` (black/mypy, v16 legacy tests, python package tests/coverage, docs check, benchmarks, Wolfram notice, release stub). Triggers on `main`, `feature/*`, `copilot/*`.
- To match CI locally: set the appropriate PYTHONPATH before pytest; prefer Python 3.12.

## 9) Review agent checklist
- Confirm `python -m pip install -e .[dev]` ran and PYTHONPATH is correctly set for the suite under review.
- Validate targeted pytest scope relevant to the change (python package first; legacy only if touched).
- Ensure formatting (black 100 cols) and lint (ruff) are clean on touched files; note CI leniency.
- Call out any deviation between these instructions and observed behavior.

## 10) Coding agent checklist
- Set PYTHONPATH **before** running tests (root vs `python/` differ).
- Use focused pytest (`-k`) for impacted modules; document if legacy suites are intentionally skipped.
- Avoid new dependencies unless required; if added, update `pyproject.toml`/`requirements.txt` and note security checks.
- Reference manuscript equations in new code/tests; keep edits surgical.
- Trust these instructions first; search only when details are missing or incorrect.
