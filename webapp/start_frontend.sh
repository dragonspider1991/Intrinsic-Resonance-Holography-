#!/bin/bash
# IRH Frontend Startup Script
# This script starts the IRH frontend development server

set -e  # Exit on error

echo "=================================="
echo "IRH Frontend Startup"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo "Frontend directory: $FRONTEND_DIR"
echo ""

# Step 1: Check Node.js version
echo -e "${YELLOW}[1/3] Checking Node.js version...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js not found${NC}"
    echo "Please install Node.js 18+ from https://nodejs.org/"
    exit 1
fi

node_version=$(node --version)
echo "Found Node.js $node_version"
echo -e "${GREEN}✓ Node.js OK${NC}"
echo ""

# Step 2: Install dependencies if needed
echo -e "${YELLOW}[2/3] Checking dependencies...${NC}"
cd "$FRONTEND_DIR"
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Dependencies already installed${NC}"
fi
echo ""

# Step 3: Start dev server
echo -e "${YELLOW}[3/3] Starting frontend development server...${NC}"
echo ""
echo "=================================="
echo -e "${GREEN}Frontend server starting...${NC}"
echo "=================================="
echo ""
echo "Frontend URL:      http://localhost:5173"
echo "Backend API:       http://localhost:8000"
echo ""
echo "Make sure backend is running first!"
echo "Run: ./start_backend.sh (in another terminal)"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "=================================="
echo ""

npm run dev
