# IRH Web Application - Quick Start Guide

## 🎯 What Is This?

This directory contains a **complete web application infrastructure** for the Intrinsic Resonance Holography (IRH) v10.0 test suite. It provides:

- ✅ **Backend API** (FastAPI) - 13 REST endpoints + WebSocket
- ✅ **3D/2D Visualization** - Data formats for Three.js and Chart.js
- ✅ **IRH Integration** - Full integration with IRH physics modules
- ✅ **Frontend Specification** - Complete prompt for Gemini AI to build the UI

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```bash
# From repository root
pip install -r requirements.txt
pip install -e .

# Install backend dependencies
pip install -r webapp/backend/requirements.txt
```

### 2. Start Backend Server
```bash
python webapp/start_server.py
```

### 3. Open API Documentation
Open browser: **http://localhost:8000/api/docs**

🎉 **Done!** The backend API is running and ready for frontend integration.

## 📚 Documentation Files

| File | Purpose | Size |
|------|---------|------|
| **README.md** | Overview and quick reference | 8.6 KB |
| **INSTALLATION.md** | Complete installation guide | 5.7 KB |
| **INFRASTRUCTURE_SUMMARY.md** | Comprehensive infrastructure docs | 14 KB |
| **GEMINI_FRONTEND_PROMPT.md** | Frontend specification for AI | 21 KB |

## 🔧 What's Included

### Backend API (`webapp/backend/`)
- `app.py` - FastAPI application (17 KB)
- `visualization.py` - 3D/2D data serializers (12 KB)
- `integration.py` - IRH module wrapper (11 KB)

### Key Features
- Parameter configuration (network size, topology, seed)
- Async simulation execution with progress tracking
- Physical predictions (α⁻¹, spectral dimension, etc.)
- 3D network visualization data (Three.js format)
- 2D chart data (Chart.js format)
- WebSocket real-time updates

### API Endpoints
```
POST /api/network/create              - Create network
POST /api/network/spectrum            - Compute eigenspectrum
POST /api/network/spectral-dimension  - Compute d_s
POST /api/predictions/alpha           - Predict α⁻¹
POST /api/simulation/run              - Run full simulation
GET  /api/jobs/{job_id}               - Job status
WS   /ws/{job_id}                     - Real-time updates
```

## 🎨 Frontend Implementation

The frontend is **not yet implemented**. To create it:

### Option 1: Use Google AI Studio Gemini 3.0
1. Open `GEMINI_FRONTEND_PROMPT.md` (21 KB complete specification)
2. Paste into Google AI Studio with Gemini 3.0
3. Request: "Implement this frontend application"
4. Gemini will generate complete React + TypeScript code

### Option 2: Manual Implementation
1. Review `GEMINI_FRONTEND_PROMPT.md` for requirements
2. Use React + TypeScript + Vite
3. Integrate Three.js (3D) and Chart.js (2D)
4. Connect to backend API at `http://localhost:8000`

## 🧪 Testing

### Test Backend API
```bash
# Run verification script
python webapp/verify_infrastructure.py

# Test with example client
python webapp/example_api_client.py
```

### Interactive API Testing
Open http://localhost:8000/api/docs and test endpoints directly in browser.

## 📊 Visualization Formats

### 3D Network (Three.js)
```json
{
  "nodes": [{"id": 0, "position": [x,y,z], "color": "#hex"}],
  "edges": [{"source": 0, "target": 1, "weight": 0.5}]
}
```

### 2D Charts (Chart.js)
```json
{
  "type": "line",
  "data": {"datasets": [...]},
  "options": {"responsive": true}
}
```

## 🏗️ Project Structure

```
webapp/
├── backend/              # FastAPI backend (complete)
├── frontend/             # React frontend (to be implemented)
├── config/               # Configuration
├── start_server.py       # Server startup
├── example_api_client.py # Test client
├── verify_infrastructure.py # Verification tool
└── [Documentation].md    # Comprehensive docs
```

## 📖 Learn More

- **Overview**: Read `README.md`
- **Installation**: Read `INSTALLATION.md`
- **Architecture**: Read `INFRASTRUCTURE_SUMMARY.md`
- **Frontend Spec**: Read `GEMINI_FRONTEND_PROMPT.md`

## 🎯 Next Steps

1. ✅ Backend complete - Running at http://localhost:8000
2. 📝 Frontend to implement - See `GEMINI_FRONTEND_PROMPT.md`
3. 🚀 Deploy - After frontend is complete

## 💡 Key Capabilities

| Capability | Status | Description |
|------------|--------|-------------|
| Parameter Input | ✅ | Network size, topology, seed, optimization |
| 3D Visualization Data | ✅ | Network topology for Three.js |
| 2D Visualization Data | ✅ | Charts and plots for Chart.js |
| Physical Predictions | ✅ | α⁻¹, spectral dimension, etc. |
| Real-time Updates | ✅ | WebSocket progress streaming |
| Grand Audit | ✅ | Comprehensive validation |
| Frontend UI | 📝 | To be implemented with Gemini |

## 🔗 Links

- **API Docs**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/api/health
- **IRH Main Repo**: ../README.md

## 📞 Support

For questions:
- Backend: Check `/api/docs` and `webapp/backend/app.py`
- Frontend: See `GEMINI_FRONTEND_PROMPT.md`
- IRH Theory: See main repository `README.md`

## 📜 License

CC0-1.0 Universal (Public Domain) - Same as IRH project

---

**Status**: Backend infrastructure complete ✅ | Frontend ready for implementation 📝
