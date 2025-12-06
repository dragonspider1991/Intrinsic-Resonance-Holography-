# IRH Web Application - Complete Setup Guide

This guide will help you set up and run the complete IRH web application (backend + frontend) from scratch.

## What You'll Get

- **Backend API Server** running on http://localhost:8000
- **Frontend Web UI** running on http://localhost:5173
- **Interactive visualization** of IRH networks and physics predictions
- **Real-time updates** during simulations

## Prerequisites

Before starting, make sure you have:

### Required Software

1. **Python 3.8 or higher**
   - Check: `python3 --version`
   - Install from: https://www.python.org/downloads/

2. **Node.js 18 or higher**
   - Check: `node --version`
   - Install from: https://nodejs.org/

3. **pip (Python package manager)**
   - Usually comes with Python
   - Check: `pip3 --version`

4. **npm (Node package manager)**
   - Comes with Node.js
   - Check: `npm --version`

### System Requirements

- **OS**: Linux, macOS, or Windows (with WSL recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: 2GB free space
- **Internet**: Required for initial package downloads

## Quick Start (Recommended)

We've created automated setup scripts for you!

### Option 1: Using Setup Scripts (Easiest)

Open **TWO terminal windows** and run:

#### Terminal 1: Start Backend

```bash
cd webapp
./start_backend.sh
```

This script will:
1. ✓ Check Python version
2. ✓ Install IRH core package
3. ✓ Install backend dependencies
4. ✓ Verify installation
5. ✓ Start backend server on port 8000

**Expected output:**
```
==================================
Backend server starting...
==================================

API Documentation: http://localhost:8000/api/docs
Backend Base URL:  http://localhost:8000

INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

#### Terminal 2: Start Frontend

```bash
cd webapp
./start_frontend.sh
```

This script will:
1. ✓ Check Node.js version
2. ✓ Install frontend dependencies
3. ✓ Start development server on port 5173

**Expected output:**
```
==================================
Frontend server starting...
==================================

  VITE v7.2.6  ready in 500 ms

  ➜  Local:   http://localhost:5173/
```

#### Access the Application

Open your browser and go to:
**http://localhost:5173**

You should see the IRH web interface!

---

## Manual Setup (Step-by-Step)

If you prefer manual setup or the scripts don't work:

### Step 1: Install IRH Core Package

From the repository root:

```bash
# Option A: If you have setup.py in root
pip3 install -e .

# Option B: If setup.py is in python/
cd python
pip3 install -e .
cd ..
```

**Verify:**
```bash
python3 -c "import irh; print('IRH installed successfully')"
```

### Step 2: Install Backend Dependencies

```bash
cd webapp/backend
pip3 install -r requirements.txt
```

**This installs:**
- FastAPI (web framework)
- Uvicorn (ASGI server)
- Pydantic (data validation)
- WebSockets (real-time updates)
- Other required packages

**Verify:**
```bash
python3 -c "import fastapi, uvicorn; print('Backend dependencies OK')"
```

### Step 3: Install Frontend Dependencies

```bash
cd ../frontend  # or: cd webapp/frontend from root
npm install
```

**This installs:**
- React & TypeScript
- Material-UI components
- Three.js (3D graphics)
- Chart.js (charts)
- Axios (HTTP client)
- And ~260 other packages

**Note:** This may take 2-5 minutes depending on your internet speed.

### Step 4: Start Backend Server

```bash
cd webapp/backend
python3 -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Flags explained:**
- `--reload`: Auto-restart on code changes
- `--host 0.0.0.0`: Accept connections from anywhere
- `--port 8000`: Use port 8000

**Leave this terminal running!**

### Step 5: Start Frontend Server

Open a **NEW terminal** window:

```bash
cd webapp/frontend
npm run dev
```

**Leave this terminal running too!**

### Step 6: Open in Browser

Navigate to: **http://localhost:5173**

---

## Verifying Everything Works

### Check Backend

1. Open http://localhost:8000/api/docs
2. You should see the Swagger API documentation
3. Try the "GET /api/health" endpoint (if it exists)

### Check Frontend

1. Open http://localhost:5173
2. You should see:
   - Header: "IRH - Intrinsic Resonance Holography v10.0"
   - Left panel: Parameter controls
   - Center: Visualization area
   - Bottom: Results panel

### Test Integration

1. In the left panel, set:
   - Network Size: 16 (small for quick test)
   - Topology: Random
   - Keep other defaults

2. Click **"Run Simulation"**

3. You should see:
   - Progress bar moving
   - Status updates
   - Results appearing in tabs

If this works, **congratulations!** Everything is set up correctly.

---

## Troubleshooting

### Backend Issues

#### "ModuleNotFoundError: No module named 'irh'"

**Solution:**
```bash
# From repository root
pip3 install -e .

# If that fails, try:
cd python
pip3 install -e .
```

#### "Port 8000 already in use"

**Solution:**
```bash
# Find and kill the process using port 8000
# On Linux/Mac:
lsof -ti:8000 | xargs kill -9

# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

Or use a different port:
```bash
uvicorn app:app --reload --port 8001
```

Then update `webapp/frontend/.env`:
```env
VITE_API_URL=http://localhost:8001
VITE_WS_URL=ws://localhost:8001
```

#### "Import errors" when starting backend

Make sure you're in the correct directory:
```bash
cd webapp/backend
python3 app.py  # Should show error about uvicorn
python3 -m uvicorn app:app --reload  # Correct way
```

### Frontend Issues

#### "npm: command not found"

**Solution:** Install Node.js from https://nodejs.org/

#### "Cannot connect to backend"

**Checklist:**
1. ✓ Backend running? Check http://localhost:8000/api/docs
2. ✓ Correct URL in `.env`? Should be `http://localhost:8000`
3. ✓ CORS enabled? Backend should allow `localhost:5173`
4. ✓ Firewall not blocking? Try disabling temporarily

#### "Module not found" errors in npm

**Solution:**
```bash
cd webapp/frontend
rm -rf node_modules package-lock.json
npm install
```

#### Build errors

**Solution:**
```bash
npm run build
```
Fix any TypeScript errors shown.

### Common Errors

#### CORS Errors in Browser Console

**Error:** `Access-Control-Allow-Origin`

**Solution:** Backend CORS is configured for `localhost:5173`. If you changed ports, update backend `app.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your port here
    ...
)
```

#### WebSocket Connection Failed

**Solution:**
1. Check backend is running
2. Check browser console for errors
3. Try different browser (Chrome/Firefox recommended)
4. Disable browser extensions temporarily

#### 3D Visualization Blank

**Solution:**
1. Check WebGL support: https://get.webgl.org/
2. Update graphics drivers
3. Try 2D mode instead
4. Check browser console for Three.js errors

---

## Configuration

### Backend Configuration

**File:** `webapp/backend/app.py`

Change port:
```python
# At the bottom of the file, or run with:
uvicorn app:app --port 8001
```

Configure CORS:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    ...
)
```

### Frontend Configuration

**File:** `webapp/frontend/.env`

```env
# API endpoints
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

Change these if you use different ports.

---

## Using the Application

### Running a Simulation

1. **Configure Parameters** (left panel):
   - Network Size (N): 4 to 4096
   - Topology: Random, Complete, Cycle, Lattice
   - Edge Probability: 0.0 to 1.0 (Random only)
   - Random Seed: Optional

2. **Set Computations**:
   - ✓ Spectral Dimension
   - ✓ Physical Predictions
   - ☐ Grand Audit (expensive, use for small N only)

3. **Click "Run Simulation"**

4. **Watch Progress** - Real-time updates via WebSocket

5. **View Results**:
   - Visualization: 3D network or 2D charts
   - Network Tab: Basic info
   - Spectrum Tab: Eigenvalues
   - Predictions Tab: α⁻¹ and other constants
   - Grand Audit Tab: Validation results

### Tips

- Start with **small networks** (N=16-64) for faster results
- **Disable Grand Audit** for large networks (N>256)
- Use **3D mode** for small networks, **2D charts** for large ones
- **Toggle visualizations** to compare Network vs Spectrum
- **Set seed** for reproducible results

---

## Production Deployment

For production (not localhost):

### Build Frontend

```bash
cd webapp/frontend
npm run build
```

This creates `dist/` folder with optimized files.

### Serve with Nginx/Apache

Point web server to `dist/` folder and proxy API to backend.

### Use Production Settings

- Remove `--reload` from uvicorn
- Set `ENV=production`
- Configure proper CORS origins
- Use HTTPS (wss:// for WebSocket)
- Set up reverse proxy

---

## Getting Help

### Documentation

- **Frontend README**: `webapp/frontend/README.md`
- **API Integration**: `webapp/frontend/API_INTEGRATION.md`
- **Components**: `webapp/frontend/COMPONENTS.md`
- **Quick Start**: `webapp/QUICKSTART_FULLSTACK.md`

### Testing API Directly

Visit: http://localhost:8000/api/docs

Use Swagger UI to test endpoints without frontend.

### Check Logs

**Backend logs**: Printed in terminal where you started backend

**Frontend logs**: 
- Browser console (F12 → Console tab)
- Network tab to see API calls

### Common Questions

**Q: Do I need to keep both terminals running?**
A: Yes! Both backend and frontend must run simultaneously.

**Q: Can I run this on Windows?**
A: Yes, but WSL (Windows Subsystem for Linux) recommended for best experience.

**Q: How do I stop the servers?**
A: Press Ctrl+C in each terminal window.

**Q: Is internet required after installation?**
A: No, works offline after dependencies are installed.

**Q: Can I use this on a remote server?**
A: Yes, but you'll need to configure ports and firewall properly.

---

## Next Steps

Once everything is running:

1. **Explore different topologies** - Try Random, Complete, Cycle, Lattice
2. **Experiment with network sizes** - See how results change
3. **Compare predictions** - Check α⁻¹ against CODATA value
4. **Run Grand Audit** - Validate theoretical predictions (small N only)
5. **Export data** - Use browser DevTools to save JSON responses

---

## System Information

- **Backend Framework**: FastAPI 0.104+
- **Backend Server**: Uvicorn (ASGI)
- **Frontend Framework**: React 18 + TypeScript
- **Build Tool**: Vite 7
- **UI Library**: Material-UI v7
- **3D Graphics**: Three.js
- **Charts**: Chart.js
- **State Management**: Zustand

---

**Happy exploring! 🚀**

For issues or questions, check the documentation files in `webapp/frontend/`.
