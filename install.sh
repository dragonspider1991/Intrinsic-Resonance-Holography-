#!/bin/bash
#
# Installation script for CNCG package
# Supports both Conda/Anaconda and pip installation methods
#

set -e  # Exit on error

echo "=========================================="
echo "CNCG Package Installation Script"
echo "=========================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is available
check_conda() {
    if command -v conda &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    print_warning "Not in a git repository. Make sure you're in the project root directory."
fi

echo "Please select installation method:"
echo "  1) Conda/Anaconda (recommended for scientific computing)"
echo "  2) pip (standard Python package manager)"
echo "  3) Development mode with pip"
echo ""
read -p "Enter your choice [1-3]: " choice

case $choice in
    1)
        print_info "Installing via Conda/Anaconda..."
        
        if ! check_conda; then
            print_error "Conda is not installed or not in PATH."
            print_info "Please install Anaconda or Miniconda from:"
            print_info "  https://docs.conda.io/en/latest/miniconda.html"
            exit 1
        fi
        
        print_info "Creating conda environment 'cncg'..."
        conda env create -f environment.yml
        
        print_success "Conda environment created successfully!"
        echo ""
        print_info "To activate the environment, run:"
        echo "    conda activate cncg"
        echo ""
        print_info "To run experiments:"
        echo "    conda activate cncg"
        echo "    python experiments/run_emergence.py --N 100 --n-trials 1"
        ;;
        
    2)
        print_info "Installing via pip..."
        
        if ! command -v pip &> /dev/null; then
            print_error "pip is not installed or not in PATH."
            exit 1
        fi
        
        print_info "Installing package and dependencies..."
        pip install .
        
        print_success "Package installed successfully!"
        echo ""
        print_info "To run experiments:"
        echo "    python experiments/run_emergence.py --N 100 --n-trials 1"
        ;;
        
    3)
        print_info "Installing in development mode via pip..."
        
        if ! command -v pip &> /dev/null; then
            print_error "pip is not installed or not in PATH."
            exit 1
        fi
        
        print_info "Installing package in editable mode..."
        pip install -e ".[dev]"
        
        print_success "Package installed in development mode!"
        echo ""
        print_info "Running tests to verify installation..."
        if command -v pytest &> /dev/null; then
            pytest tests/ -v
            print_success "All tests passed!"
        else
            print_warning "pytest not found. Install dev dependencies with: pip install -e '.[dev]'"
        fi
        echo ""
        print_info "To run experiments:"
        echo "    python experiments/run_emergence.py --N 100 --n-trials 1"
        ;;
        
    *)
        print_error "Invalid choice. Please run the script again and select 1, 2, or 3."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
print_success "Installation complete!"
echo "=========================================="
echo ""
print_info "For more information, see:"
echo "  - README.md"
echo "  - IMPLEMENTATION_SUMMARY.md"
echo "  - Documentation: https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-"
echo ""
