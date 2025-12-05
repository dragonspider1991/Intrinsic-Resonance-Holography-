@echo off
REM Installation script for CNCG package on Windows
REM Supports both Conda/Anaconda and pip installation methods

setlocal enabledelayedexpansion

echo ==========================================
echo CNCG Package Installation Script
echo ==========================================
echo.

REM Check if we're in a git repository
if not exist ".git" (
    echo [WARNING] Not in a git repository. Make sure you're in the project root directory.
    echo.
)

echo Please select installation method:
echo   1) Conda/Anaconda (recommended for scientific computing)
echo   2) pip (standard Python package manager)
echo   3) Development mode with pip
echo.
set /p choice="Enter your choice [1-3]: "

if "%choice%"=="1" goto conda_install
if "%choice%"=="2" goto pip_install
if "%choice%"=="3" goto dev_install

echo [ERROR] Invalid choice. Please run the script again and select 1, 2, or 3.
exit /b 1

:conda_install
echo [INFO] Installing via Conda/Anaconda...

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Conda is not installed or not in PATH.
    echo [INFO] Please install Anaconda or Miniconda from:
    echo [INFO]   https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

echo [INFO] Creating conda environment 'cncg'...
conda env create -f environment.yml

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Conda environment created successfully!
    echo.
    echo [INFO] To activate the environment, run:
    echo     conda activate cncg
    echo.
    echo [INFO] To run experiments:
    echo     conda activate cncg
    echo     python experiments\run_emergence.py --N 100 --n-trials 1
) else (
    echo [ERROR] Failed to create conda environment.
    exit /b 1
)
goto end

:pip_install
echo [INFO] Installing via pip...

REM Check if pip is available
where pip >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] pip is not installed or not in PATH.
    exit /b 1
)

echo [INFO] Installing package and dependencies...
pip install .

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Package installed successfully!
    echo.
    echo [INFO] To run experiments:
    echo     python experiments\run_emergence.py --N 100 --n-trials 1
) else (
    echo [ERROR] Failed to install package.
    exit /b 1
)
goto end

:dev_install
echo [INFO] Installing in development mode via pip...

REM Check if pip is available
where pip >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] pip is not installed or not in PATH.
    exit /b 1
)

echo [INFO] Installing package in editable mode...
pip install -e .[dev]

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Package installed in development mode!
    echo.
    echo [INFO] Running tests to verify installation...
    
    where pytest >nul 2>nul
    if !errorlevel! equ 0 (
        pytest tests\ -v
        if !errorlevel! equ 0 (
            echo [SUCCESS] All tests passed!
        )
    ) else (
        echo [WARNING] pytest not found. Install dev dependencies with: pip install -e .[dev]
    )
    
    echo.
    echo [INFO] To run experiments:
    echo     python experiments\run_emergence.py --N 100 --n-trials 1
) else (
    echo [ERROR] Failed to install package.
    exit /b 1
)
goto end

:end
echo.
echo ==========================================
echo [SUCCESS] Installation complete!
echo ==========================================
echo.
echo [INFO] For more information, see:
echo   - README.md
echo   - IMPLEMENTATION_SUMMARY.md
echo   - Documentation: https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-
echo.

endlocal
