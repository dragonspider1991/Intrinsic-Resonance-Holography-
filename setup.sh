#!/bin/bash

################################################################################
# setup.sh - Helper Script for Bash Users
################################################################################
#
# This script serves as a simple entry point for pure Bash users.
# It calls the main orchestrator.py with appropriate Python interpreter.
#
# Usage:
#   ./setup.sh                  # Interactive mode
#   ./setup.sh --non-interactive # Non-interactive mode
#   ./setup.sh --help           # Show help
#
# Author: Generated for IRH v10.0 Project
# License: CC0-1.0 (Public Domain)
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "================================================================================"
echo "  IRH PHYSICS SUITE - SETUP & ORCHESTRATION"
echo "  Intrinsic Resonance Holography v10.0"
echo "================================================================================"
echo -e "${NC}"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed or not in PATH${NC}"
    echo "Please install Python 3.11 or higher and try again."
    echo ""
    echo "Installation instructions:"
    echo "  Ubuntu/Debian: sudo apt-get install python3 python3-pip python3-venv"
    echo "  Fedora/RHEL:   sudo dnf install python3 python3-pip"
    echo "  macOS:         brew install python3"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.11"

echo -e "${GREEN}✓ Python 3 found: version ${PYTHON_VERSION}${NC}"

# Version comparison (basic check)
if [[ $(echo -e "$PYTHON_VERSION\n$REQUIRED_VERSION" | sort -V | head -n1) != "$REQUIRED_VERSION" ]]; then
    echo -e "${YELLOW}⚠ Warning: Python ${PYTHON_VERSION} detected, but 3.11+ is recommended${NC}"
    echo "Some features may not work correctly with older Python versions."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if orchestrator.py exists
if [ ! -f "orchestrator.py" ]; then
    echo -e "${RED}Error: orchestrator.py not found in current directory${NC}"
    echo "Please run this script from the repository root."
    exit 1
fi

echo -e "${GREEN}✓ orchestrator.py found${NC}"

# Make orchestrator.py executable (if not already)
chmod +x orchestrator.py 2>/dev/null || true

# Check if we're in a git repository
if [ -d ".git" ]; then
    echo -e "${GREEN}✓ Git repository detected${NC}"
else
    echo -e "${YELLOW}⚠ Warning: Not in a git repository${NC}"
fi

echo ""
echo "--------------------------------------------------------------------------------"
echo "  Starting orchestrator..."
echo "--------------------------------------------------------------------------------"
echo ""

# Pass all arguments to orchestrator.py
python3 orchestrator.py "$@"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}"
    echo "================================================================================"
    echo "  ✓ Setup and orchestration completed successfully"
    echo "================================================================================"
    echo -e "${NC}"
else
    echo -e "${RED}"
    echo "================================================================================"
    echo "  ✗ Setup encountered errors (exit code: $EXIT_CODE)"
    echo "================================================================================"
    echo -e "${NC}"
    
    if [ -f "crash_report_for_llm.txt" ]; then
        echo ""
        echo -e "${YELLOW}A crash report has been generated: crash_report_for_llm.txt${NC}"
        echo "You can share this file with an LLM for debugging assistance."
    fi
fi

exit $EXIT_CODE
