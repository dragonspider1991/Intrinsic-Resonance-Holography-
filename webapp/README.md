# IRH Web Application

Interactive web interface for Intrinsic Resonance Holography v10.0 test suite.

## Overview

This web application provides a graphical user interface for exploring the IRH theory, running simulations, and visualizing results in both 2D and 3D.

**Status**: Backend infrastructure is complete. Frontend to be implemented using the Gemini prompt.

## Architecture

### Backend (✅ Complete)
- **Framework**: FastAPI (Python)
- **API Docs**: OpenAPI/Swagger at `/api/docs`
- **Real-time**: WebSocket support for live progress updates
- **Data Format**: JSON with proper numpy serialization
- **Visualization**: Pre-formatted data for Three.js and Chart.js

### Frontend (📝 To Be Implemented)
- **Framework**: React + TypeScript
- **Build Tool**: Vite
- **3D Graphics**: Three.js
- **2D Charts**: Chart.js
- **UI Library**: Material-UI or Tailwind CSS
- **Implementation**: See `GEMINI_FRONTEND_PROMPT.md` for complete specification

## Quick Start

### Backend Setup

1. **Install Python dependencies**:
   ```bash
   cd webapp/backend
   pip install -r requirements.txt
   ```

2. **Install IRH package** (if not already installed):
   ```bash
   cd ../../
   pip install -e .
   ```

3. **Run the backend server**:
   ```bash
   cd webapp/backend
   python app.py
   ```

4. **Verify backend is running**:
   Open http://localhost:8000/api/docs in your browser to see the API documentation.

### Frontend Setup (After Implementation)

1. **Install Node.js dependencies**:
   ```bash
   cd webapp/frontend
   npm install
   ```

2. **Run development server**:
   ```bash
   npm run dev
   ```

3. **Build for production**:
   ```bash
   npm run build
   ```

## API Endpoints

### Core Endpoints
- `POST /api/network/create` - Create a network with specified parameters
- `POST /api/network/spectrum` - Compute eigenspectrum
- `POST /api/network/spectral-dimension` - Compute spectral dimension
- `POST /api/predictions/alpha` - Predict fine structure constant
- `POST /api/simulation/run` - Run full simulation (async)
- `GET /api/jobs/{job_id}` - Get job status
- `WS /ws/{job_id}` - Real-time job updates

### Visualization Endpoints
- `POST /api/visualization/network-3d` - Get 3D network data
- `POST /api/visualization/spectrum-3d` - Get 3D spectrum data
- `POST /api/visualization/spectrum-chart` - Get 2D chart data

See `/api/docs` for complete API documentation with request/response schemas.

## Features

### Parameter Configuration
- Network size (N): 4 to 4096 nodes
- Topology: Random, Complete, Cycle, Lattice
- Random seed for reproducibility
- Edge probability (for random graphs)
- Optimization settings (iterations, temperature)

### Computations
- ✓ Eigenspectrum analysis
- ✓ Spectral dimension calculation
- ✓ Physical constant predictions (α⁻¹, etc.)
- ✓ Grand audit (comprehensive validation)
- ✓ Metric emergence
- ✓ Complexity metrics (GTEC, frustration)

### Visualizations

**3D (Three.js)**:
- Interactive network topology
- Node colors mapped to eigenvalues
- Edge thickness/opacity based on weights
- Orbit controls (rotate, zoom, pan)
- Spectral surface plots

**2D (Chart.js)**:
- Eigenvalue spectrum line chart
- Eigenvalue distribution histogram
- Adjacency matrix heatmap
- Comparative plots
- Publication-quality exports

### Real-time Updates
- WebSocket-based progress tracking
- Live progress bars
- Status messages during computation
- Automatic result display on completion

## Project Structure

```
webapp/
├── backend/                    # Backend API (FastAPI)
│   ├── app.py                 # Main API application
│   ├── visualization.py       # Data serialization for viz
│   ├── integration.py         # IRH module integration
│   └── requirements.txt       # Python dependencies
├── frontend/                   # Frontend (React + TypeScript)
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
├── config/                     # Configuration files
│   └── webapp_config.json     # App settings
├── static/                     # Static assets
└── README.md                  # This file
```

## Configuration

Edit `config/webapp_config.json` to customize:
- Backend host/port
- CORS settings
- Default parameters
- Visualization settings
- Feature flags

## Development

### Running Backend in Development Mode
```bash
cd webapp/backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### API Testing
Use the interactive API docs at http://localhost:8000/api/docs to test endpoints directly.

### Example API Calls

**Create a network**:
```bash
curl -X POST http://localhost:8000/api/network/create \
  -H "Content-Type: application/json" \
  -d '{"N": 64, "topology": "Random", "seed": 42, "edge_probability": 0.3}'
```

**Run a simulation**:
```bash
curl -X POST http://localhost:8000/api/simulation/run \
  -H "Content-Type: application/json" \
  -d '{
    "network_config": {"N": 128, "topology": "Lattice", "seed": 42},
    "compute_spectral_dimension": true,
    "compute_predictions": true,
    "run_grand_audit": false
  }'
```

## Data Format Examples

### Network 3D Data
```json
{
  "nodes": [
    {"id": 0, "position": [0.1, 0.2, 0.3], "color": "#3498db", "size": 1.0},
    ...
  ],
  "edges": [
    {"source": 0, "target": 1, "weight": 0.5, "color": "#888", "opacity": 0.7},
    ...
  ],
  "metadata": {
    "node_count": 64,
    "edge_count": 192,
    "avg_edge_weight": 0.35,
    "max_edge_weight": 0.89
  }
}
```

### Chart 2D Data (Chart.js format)
```json
{
  "type": "line",
  "data": {
    "datasets": [{
      "label": "Eigenvalue λ",
      "data": [{"x": 0, "y": 0.0}, {"x": 1, "y": 0.15}, ...],
      "borderColor": "#3282b8"
    }]
  },
  "options": {
    "responsive": true,
    "scales": {
      "x": {"title": {"display": true, "text": "Index"}},
      "y": {"title": {"display": true, "text": "Eigenvalue"}}
    }
  }
}
```

## Frontend Implementation Guide

See **`GEMINI_FRONTEND_PROMPT.md`** for:
- Complete technical specification
- UI/UX requirements
- Component architecture
- State management approach
- 3D visualization details (Three.js)
- 2D visualization details (Chart.js)
- WebSocket integration
- TypeScript types
- Success criteria

This prompt is designed for Google AI Studio's Gemini 3.0 with agentic vibe code to generate the complete frontend.

## Technology Stack

### Backend
- Python 3.11+
- FastAPI 0.104+
- Uvicorn (ASGI server)
- NumPy, SciPy, NetworkX (scientific computing)
- IRH v10.0 (physics simulation)

### Frontend (Planned)
- React 18+
- TypeScript 5+
- Vite (build tool)
- Material-UI or Tailwind CSS
- Three.js (3D graphics)
- Chart.js (2D charts)
- Axios (HTTP client)

## Performance Considerations

- **Backend**: Async/await for non-blocking I/O
- **Job Queue**: Background tasks for expensive computations
- **Caching**: Results cached by job_id
- **WebSocket**: Efficient real-time updates
- **3D Rendering**: Instanced meshes for N > 500
- **2D Charts**: Decimation for large datasets

## Security Notes

- CORS configured (adjust for production)
- Input validation via Pydantic models
- No authentication yet (add for production)
- Rate limiting recommended for production

## Troubleshooting

### Backend Issues

**"Module 'irh' not found"**:
- Ensure IRH package is installed: `pip install -e .` from repo root

**"Port 8000 already in use"**:
- Change port in `webapp_config.json` or use `uvicorn app:app --port 8001`

**CORS errors**:
- Adjust `allow_origins` in `app.py` or `webapp_config.json`

### Frontend Issues (After Implementation)

**"Cannot connect to backend"**:
- Verify backend is running on correct port
- Check CORS configuration
- Verify API base URL in frontend code

**3D visualization not rendering**:
- Check browser WebGL support
- Verify Three.js is properly imported
- Check browser console for errors

## Contributing

When contributing to the web application:
1. Follow existing code style
2. Add TypeScript types for all new code
3. Document API changes in OpenAPI schema
4. Test both 2D and 3D visualizations
5. Ensure responsive design works on all devices
6. Update this README with new features

## License

Same as IRH project: CC0-1.0 Universal (Public Domain)

## Contact

For issues related to:
- **Backend API**: Check OpenAPI docs or backend code
- **IRH Theory**: See main project README
- **Frontend**: See GEMINI_FRONTEND_PROMPT.md

---

**Next Steps**:
1. ✅ Backend infrastructure complete
2. 📝 Review GEMINI_FRONTEND_PROMPT.md
3. 🚀 Use Google AI Studio Gemini to implement frontend
4. 🎨 Customize UI/UX as desired
5. 🧪 Test integration
6. 📦 Deploy!
