#!/bin/bash
# IRH Backend Setup Script
# This script sets up and starts the IRH backend server

set -e  # Exit on error

echo "=================================="
echo "IRH Backend Setup & Startup"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$SCRIPT_DIR/backend"

echo "Repository root: $REPO_ROOT"
echo "Backend directory: $BACKEND_DIR"
echo ""

# Step 1: Check Python version
echo -e "${YELLOW}[1/5] Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo -e "${RED}Error: Python 3.8+ required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Step 2: Install IRH core package
echo -e "${YELLOW}[2/5] Installing IRH core package...${NC}"
cd "$REPO_ROOT"
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "Installing IRH package in development mode..."
    pip3 install -e . --quiet || {
        echo -e "${RED}Failed to install IRH package${NC}"
        echo "Trying alternative installation method..."
        cd python && pip3 install -e . --quiet
    }
    echo -e "${GREEN}✓ IRH package installed${NC}"
else
    echo -e "${YELLOW}⚠ No setup.py found, skipping core package installation${NC}"
fi
echo ""

# Step 3: Install backend dependencies
echo -e "${YELLOW}[3/5] Installing backend dependencies...${NC}"
cd "$BACKEND_DIR"
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt --quiet
    echo -e "${GREEN}✓ Backend dependencies installed${NC}"
else
    echo -e "${RED}Error: requirements.txt not found${NC}"
    exit 1
fi
echo ""

# Step 4: Verify installation
echo -e "${YELLOW}[4/5] Verifying installation...${NC}"
python3 -c "import fastapi; import uvicorn; print('FastAPI:', fastapi.__version__); print('Uvicorn:', uvicorn.__version__)" || {
    echo -e "${RED}Error: FastAPI/Uvicorn not installed correctly${NC}"
    exit 1
}
echo -e "${GREEN}✓ Backend dependencies verified${NC}"
echo ""

# Step 5: Start the server
echo -e "${YELLOW}[5/5] Starting backend server...${NC}"
echo ""
echo "=================================="
echo -e "${GREEN}Backend server starting...${NC}"
echo "=================================="
echo ""
echo "API Documentation: http://localhost:8000/api/docs"
echo "Backend Base URL:  http://localhost:8000"
echo "WebSocket:         ws://localhost:8000/ws/{job_id}"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "=================================="
echo ""

cd "$BACKEND_DIR"
python3 -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
