# IRH Web Application - Installation Guide

## Prerequisites

Before running the web application, ensure you have:

1. **Python 3.11+** installed
2. **Node.js 18+** (for frontend, when implemented)
3. **Git** (for cloning repository)

## Installation Steps

### Step 1: Install IRH Core Package

From the repository root:

```bash
# Install core dependencies
pip install -r requirements.txt

# Install IRH package in development mode
pip install -e .
```

This installs the IRH physics simulation package that the backend depends on.

### Step 2: Install Backend Dependencies

```bash
# Navigate to backend directory
cd webapp/backend

# Install backend-specific dependencies
pip install -r requirements.txt
```

This installs:
- FastAPI (web framework)
- Uvicorn (ASGI server)
- Pydantic (data validation)
- WebSockets support
- Additional utilities

### Step 3: Verify Installation

Test that all modules can be imported:

```bash
# From repository root
python -c "from irh.graph_state import HyperGraph; print('✓ IRH modules OK')"
python -c "from fastapi import FastAPI; print('✓ FastAPI OK')"
python -c "from webapp.backend import visualization; print('✓ Webapp modules OK')"
```

### Step 4: Start the Backend Server

```bash
# From repository root
python webapp/start_server.py

# Or with auto-reload for development
python webapp/start_server.py --reload

# Or directly with uvicorn
cd webapp/backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The server will start on http://localhost:8000

### Step 5: Verify Backend is Running

Open your browser and navigate to:
- **API Documentation**: http://localhost:8000/api/docs
- **Alternative Docs**: http://localhost:8000/api/redoc
- **Health Check**: http://localhost:8000/api/health

You should see the interactive API documentation (Swagger UI).

### Step 6: Test the API

Run the example client:

```bash
# From repository root
python webapp/example_api_client.py
```

This will test all API endpoints and display results.

Or use curl:

```bash
# Health check
curl http://localhost:8000/api/health

# Create a network
curl -X POST http://localhost:8000/api/network/create \
  -H "Content-Type: application/json" \
  -d '{"N": 64, "topology": "Random", "seed": 42, "edge_probability": 0.3}'
```

## Frontend Setup (After Implementation)

Once the frontend is implemented (see GEMINI_FRONTEND_PROMPT.md):

```bash
# Navigate to frontend directory
cd webapp/frontend

# Install Node.js dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## Troubleshooting

### "Module 'fastapi' not found"

Install backend dependencies:
```bash
pip install -r webapp/backend/requirements.txt
```

### "Module 'irh' not found"

Install the IRH package:
```bash
# From repository root
pip install -e .
```

### "Port 8000 already in use"

Use a different port:
```bash
python webapp/start_server.py --port 8080
```

### Import errors with IRH modules

Ensure you're running from the repository root, or check PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/Intrinsic-Resonance-Holography-"
```

## Development Workflow

### Backend Development

1. **Make changes** to `webapp/backend/*.py`
2. **Run with auto-reload**: `python webapp/start_server.py --reload`
3. **Test immediately** at http://localhost:8000/api/docs
4. **Check logs** in terminal for errors

### Testing API Endpoints

Use the interactive Swagger UI at `/api/docs` to:
- See all available endpoints
- Test requests with example payloads
- View response schemas
- Download OpenAPI specification

### Adding New Endpoints

1. Add endpoint function in `webapp/backend/app.py`
2. Define request/response models using Pydantic
3. Document with docstrings
4. Test in Swagger UI
5. Update this documentation

## Production Deployment

### Backend Deployment

For production, use a proper WSGI/ASGI server:

```bash
# Install production dependencies
pip install gunicorn

# Run with gunicorn
gunicorn webapp.backend.app:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Security Considerations

- [ ] Configure CORS appropriately (not wildcard `*`)
- [ ] Add authentication/authorization
- [ ] Enable HTTPS/TLS
- [ ] Add rate limiting
- [ ] Validate all inputs
- [ ] Set up monitoring/logging
- [ ] Use environment variables for secrets

### Performance Optimization

- [ ] Use Redis for job queue and caching
- [ ] Add database for persistent storage
- [ ] Enable response caching
- [ ] Add CDN for static assets
- [ ] Monitor memory usage for large networks

## Directory Structure After Installation

```
Intrinsic-Resonance-Holography-/
├── python/                    # IRH core package
│   └── src/irh/              # Installed via `pip install -e .`
├── webapp/
│   ├── backend/              # FastAPI backend
│   │   ├── app.py           # Main API
│   │   ├── visualization.py # Data serializers
│   │   ├── integration.py   # IRH integration
│   │   └── requirements.txt # Backend deps
│   ├── frontend/            # React frontend (to be implemented)
│   ├── config/              # Configuration
│   ├── start_server.py      # Startup script
│   └── example_api_client.py # Test client
└── requirements.txt         # Core IRH dependencies
```

## Next Steps

1. ✅ Backend infrastructure complete
2. 📝 Review `GEMINI_FRONTEND_PROMPT.md` for frontend specification
3. 🚀 Implement frontend using Google AI Studio Gemini
4. 🧪 Test full stack integration
5. 📦 Deploy to production

## Support

For issues:
- Backend API: Check `/api/docs` and logs
- IRH modules: See main README.md
- Frontend: See GEMINI_FRONTEND_PROMPT.md

## License

Same as IRH project: CC0-1.0 Universal (Public Domain)
