# IRH Web Application - Quick Start Guide

Complete guide to running the IRH web application (backend + frontend).

## Prerequisites

- Python 3.11+
- Node.js 18+
- npm or yarn

## System Overview

The IRH web application consists of two parts:

1. **Backend** (FastAPI): REST API + WebSocket server
2. **Frontend** (React): Web UI

Both must be running for the application to work.

## Installation

### 1. Install Backend Dependencies

```bash
# From repository root
cd webapp/backend
pip install -r requirements.txt

# If requirements.txt doesn't exist, install manually:
pip install fastapi uvicorn pydantic websockets
```

### 2. Install IRH Python Package

```bash
# From repository root
pip install -e .
```

### 3. Install Frontend Dependencies

```bash
# From repository root
cd webapp/frontend
npm install
```

## Running the Application

### Method 1: Run Both Servers (Recommended)

Open **two terminal windows**:

#### Terminal 1: Backend Server

```bash
cd webapp/backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Verify backend is running: http://localhost:8000/api/docs

#### Terminal 2: Frontend Server

```bash
cd webapp/frontend
npm run dev
```

You should see:
```
  VITE v7.2.6  ready in XXX ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

Open browser: http://localhost:5173

### Method 2: Use Screen/Tmux (Linux/Mac)

```bash
# Start backend in background
screen -S backend
cd webapp/backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
# Press Ctrl+A then D to detach

# Start frontend in background
screen -S frontend
cd webapp/frontend
npm run dev
# Press Ctrl+A then D to detach

# List screens
screen -ls

# Reattach to a screen
screen -r backend
screen -r frontend
```

### Method 3: Production Mode

#### Build Frontend

```bash
cd webapp/frontend
npm run build
```

This creates `dist/` directory with optimized files.

#### Serve with Nginx/Apache

Configure web server to serve `dist/` directory and proxy API requests to backend.

Example Nginx config:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    # Serve frontend
    location / {
        root /path/to/webapp/frontend/dist;
        try_files $uri $uri/ /index.html;
    }
    
    # Proxy API requests to backend
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
    
    # Proxy WebSocket
    location /ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Using the Application

### 1. Configure Parameters

In the left panel:

- **Network Size**: Use slider to select N (4-4096)
- **Topology**: Choose Random, Complete, Cycle, or Lattice
- **Edge Probability**: Set for Random topology (0.0-1.0)
- **Random Seed**: Optional, for reproducible results
- **Optimization**: Expand accordion to configure
- **Computations**: Check boxes for what to compute

### 2. Run Simulation

Click **"Run Simulation"** button

Watch progress bar for updates

### 3. View Results

**Visualization Area** (center/top):
- Toggle between 3D and 2D modes
- Select Network, Spectrum, or Both
- Interact with 3D view (rotate, zoom, pan)

**Results Panel** (bottom):
- **Network Tab**: Basic network info
- **Spectrum Tab**: Eigenvalue statistics
- **Predictions Tab**: Physical constants
- **Grand Audit Tab**: Validation results (if enabled)

## Troubleshooting

### Backend Issues

**Error: "ModuleNotFoundError: No module named 'irh'"**

Solution:
```bash
cd /path/to/repository/root
pip install -e .
```

**Error: "Port 8000 already in use"**

Solution:
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn app:app --reload --port 8001
```

Then update frontend `.env`:
```env
VITE_API_URL=http://localhost:8001
```

**Error: "CORS error"**

Check backend `app.py` CORS configuration allows frontend origin.

### Frontend Issues

**Error: "Cannot connect to backend"**

1. Verify backend is running: http://localhost:8000/api/docs
2. Check `.env` has correct API URL
3. Check browser console for errors

**Error: "Module not found"**

Solution:
```bash
cd webapp/frontend
rm -rf node_modules package-lock.json
npm install
```

**3D Visualization not working**

1. Check WebGL support: https://get.webgl.org/
2. Try different browser (Chrome/Firefox recommended)
3. Check browser console for errors

**Build errors**

Solution:
```bash
cd webapp/frontend
npm run build
```

Fix any TypeScript errors shown.

### Network Issues

**WebSocket connection fails**

1. Check firewall not blocking WebSocket
2. Verify backend WebSocket endpoint: `ws://localhost:8000/ws/test`
3. Try disabling browser extensions
4. Check backend logs for errors

**Slow API responses**

1. Large network size causes slower computation
2. Grand Audit is expensive for large N
3. Reduce N or disable expensive computations

## Development Tips

### Backend Development

**Enable auto-reload**:
```bash
uvicorn app:app --reload
```

Changes to `.py` files automatically restart server.

**View logs**:
Backend logs print to console. Check for errors.

**Test API manually**:
Use Swagger UI at http://localhost:8000/api/docs

### Frontend Development

**Hot Module Replacement (HMR)**:
```bash
npm run dev
```

Changes to `.tsx/.ts` files automatically update browser.

**View console**:
Press F12 in browser, check Console and Network tabs.

**Check state**:
Install React DevTools browser extension to inspect Zustand store.

### Code Changes

**Backend changes**:
1. Edit Python files in `webapp/backend/`
2. Server auto-reloads (if `--reload` flag used)
3. Test in Swagger UI or frontend

**Frontend changes**:
1. Edit TypeScript files in `webapp/frontend/src/`
2. Browser auto-updates (HMR)
3. Check browser console for errors

## Environment Variables

### Frontend (.env)

```env
# API endpoints
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

### Backend (optional)

```bash
# Set environment
export ENV=development  # or production

# Custom port
export PORT=8000
```

## Performance Tips

### For Large Networks (N > 500)

1. Disable 3D visualization or use 2D mode
2. Disable Grand Audit
3. Reduce max iterations in optimization

### For Best Responsiveness

1. Use smaller network sizes for exploration
2. Enable only needed computations
3. Use 2D charts instead of 3D for large data

## Next Steps

1. **Explore presets**: Try different topologies
2. **Run experiments**: Compare results across parameters
3. **Analyze data**: Use Results Panel to examine predictions
4. **Export data**: Use browser DevTools to copy JSON responses

## Support

- **Backend API Docs**: http://localhost:8000/api/docs
- **Frontend README**: `webapp/frontend/README.md`
- **API Integration**: `webapp/frontend/API_INTEGRATION.md`
- **Component Docs**: `webapp/frontend/COMPONENTS.md`

## Stopping the Application

### Stop Servers

Press **Ctrl+C** in each terminal window to stop servers.

### Clean Up

```bash
# Kill all Python processes (use with caution)
pkill python

# Kill all Node processes (use with caution)
pkill node
```

### Screen/Tmux

```bash
# Kill screens
screen -X -S backend quit
screen -X -S frontend quit
```

---

**Enjoy exploring Intrinsic Resonance Holography!** 🚀
