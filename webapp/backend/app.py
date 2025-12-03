"""
IRH Web Application - Backend API
==================================

FastAPI-based REST API for Intrinsic Resonance Holography test suite.
Provides endpoints for parameter configuration, simulation execution,
and data retrieval for 2D/3D visualizations.

Author: IRH Development Team
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
import asyncio
import uuid
from datetime import datetime
import sys
import os

# Add parent directories to path to import IRH modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python/src'))

from irh.graph_state import HyperGraph
from irh.spectral_dimension import SpectralDimension, HeatKernelTrace
from irh.scaling_flows import MetricEmergence, LorentzSignature
from irh.predictions.constants import predict_alpha_inverse
from irh.grand_audit import grand_audit

# Import visualization and integration modules
from webapp.backend.visualization import (
    serialize_network_3d,
    serialize_spectrum_3d,
    serialize_chart_2d,
)
from webapp.backend.integration import IRHSimulation, run_full_simulation

# Initialize FastAPI app
app = FastAPI(
    title="IRH Web API",
    description="REST API for Intrinsic Resonance Holography Test Suite",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global job storage (in production, use Redis or database)
jobs_db: Dict[str, Dict[str, Any]] = {}
websocket_connections: Dict[str, WebSocket] = {}


# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class NetworkConfig(BaseModel):
    """Configuration for creating a Cymatic Resonance Network."""
    N: int = Field(default=64, ge=4, le=4096, description="Number of oscillators")
    topology: str = Field(default="Random", description="Network topology: Random, Complete, Cycle, Lattice")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    edge_probability: float = Field(default=0.3, ge=0.0, le=1.0, description="Edge probability for random topology")


class OptimizationConfig(BaseModel):
    """Configuration for Adaptive Resonance Optimization."""
    max_iterations: int = Field(default=500, ge=10, le=10000, description="Maximum optimization iterations")
    T_initial: float = Field(default=1.0, ge=0.1, le=10.0, description="Initial temperature for simulated annealing")
    T_final: float = Field(default=0.01, ge=0.001, le=1.0, description="Final temperature")
    verbose: bool = Field(default=False, description="Enable verbose output")


class SimulationRequest(BaseModel):
    """Request to run a full IRH simulation."""
    network_config: NetworkConfig
    optimization_config: Optional[OptimizationConfig] = None
    compute_spectral_dimension: bool = Field(default=True, description="Compute spectral dimension")
    compute_predictions: bool = Field(default=True, description="Compute physical constant predictions")
    run_grand_audit: bool = Field(default=False, description="Run full grand audit (expensive)")


class JobStatus(BaseModel):
    """Status of a background job."""
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float = Field(ge=0.0, le=100.0)
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


# ============================================================================
# Utility Functions
# ============================================================================

def create_job(job_type: str) -> str:
    """Create a new job and return its ID."""
    job_id = str(uuid.uuid4())
    jobs_db[job_id] = {
        "job_id": job_id,
        "type": job_type,
        "status": "pending",
        "progress": 0.0,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "result": None,
    }
    return job_id


def update_job(job_id: str, **kwargs):
    """Update job status."""
    if job_id in jobs_db:
        jobs_db[job_id].update(kwargs)


def serialize_numpy(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    elif isinstance(obj, dict):
        return {k: serialize_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_numpy(item) for item in obj]
    return obj


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "IRH Web API",
        "version": "1.0.0",
        "description": "Backend API for Intrinsic Resonance Holography",
        "endpoints": {
            "docs": "/api/docs",
            "network": "/api/network",
            "simulation": "/api/simulation",
            "jobs": "/api/jobs/{job_id}",
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/network/create")
async def create_network(config: NetworkConfig):
    """Create a new network with specified configuration."""
    try:
        # Create HyperGraph
        network = HyperGraph(
            N=config.N,
            seed=config.seed,
            topology=config.topology,
            edge_probability=config.edge_probability,
        )
        
        # Extract basic properties
        L = network.get_laplacian()
        eigenvalues = np.linalg.eigvalsh(L)
        
        result = {
            "N": network.N,
            "edge_count": network.edge_count,
            "topology": config.topology,
            "spectrum": {
                "eigenvalues": serialize_numpy(eigenvalues),
                "min": float(eigenvalues[1] if len(eigenvalues) > 1 else 0),  # Skip λ_0 = 0
                "max": float(eigenvalues[-1]),
            },
            "adjacency_matrix": serialize_numpy(network.adjacency_matrix),
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/network/spectrum")
async def compute_spectrum(config: NetworkConfig):
    """Compute the eigenspectrum of a network."""
    try:
        network = HyperGraph(
            N=config.N,
            seed=config.seed,
            topology=config.topology,
            edge_probability=config.edge_probability,
        )
        
        L = network.get_laplacian()
        eigenvalues = np.linalg.eigvalsh(L)
        
        result = {
            "eigenvalues": serialize_numpy(eigenvalues),
            "spectral_gap": float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0,
            "min_eigenvalue": float(eigenvalues[1] if len(eigenvalues) > 1 else 0),
            "max_eigenvalue": float(eigenvalues[-1]),
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/network/spectral-dimension")
async def compute_spectral_dimension(config: NetworkConfig):
    """Compute the spectral dimension of a network."""
    try:
        network = HyperGraph(
            N=config.N,
            seed=config.seed,
            topology=config.topology,
            edge_probability=config.edge_probability,
        )
        
        ds_result = SpectralDimension(network)
        
        result = {
            "spectral_dimension": float(ds_result.value) if not np.isnan(ds_result.value) else None,
            "error": float(ds_result.error) if hasattr(ds_result, 'error') else None,
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predictions/alpha")
async def predict_alpha(config: NetworkConfig):
    """Predict fine structure constant from network."""
    try:
        network = HyperGraph(
            N=config.N,
            seed=config.seed,
            topology=config.topology,
            edge_probability=config.edge_probability,
        )
        
        prediction = predict_alpha_inverse(network)
        
        result = {
            "alpha_inverse": float(prediction.value),
            "codata_value": 137.035999084,
            "difference": float(prediction.value - 137.035999084),
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulation/run")
async def run_simulation(request: SimulationRequest):
    """
    Run a full IRH simulation asynchronously.
    Returns a job_id for tracking progress.
    """
    job_id = create_job("simulation")
    
    # Start background task
    asyncio.create_task(execute_simulation(job_id, request))
    
    return {"job_id": job_id, "status": "pending"}


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JSONResponse(content=jobs_db[job_id])


@app.get("/api/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Get the full result of a completed job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")
    
    return JSONResponse(content=job["result"])


@app.post("/api/visualization/network-3d")
async def get_network_3d_visualization(config: NetworkConfig):
    """Get 3D visualization data for network."""
    try:
        network = HyperGraph(
            N=config.N,
            seed=config.seed,
            topology=config.topology,
            edge_probability=config.edge_probability,
        )
        
        L = network.get_laplacian()
        eigenvalues = np.linalg.eigvalsh(L)
        
        viz_data = serialize_network_3d(
            adjacency_matrix=network.adjacency_matrix,
            eigenvalues=eigenvalues,
        )
        
        return JSONResponse(content=viz_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/visualization/spectrum-3d")
async def get_spectrum_3d_visualization(config: NetworkConfig):
    """Get 3D visualization data for eigenspectrum."""
    try:
        network = HyperGraph(
            N=config.N,
            seed=config.seed,
            topology=config.topology,
            edge_probability=config.edge_probability,
        )
        
        L = network.get_laplacian()
        eigenvalues = np.linalg.eigvalsh(L)
        
        viz_data = serialize_spectrum_3d(eigenvalues, mode="scatter")
        
        return JSONResponse(content=viz_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/visualization/spectrum-chart")
async def get_spectrum_chart(config: NetworkConfig):
    """Get 2D chart data for eigenspectrum."""
    try:
        network = HyperGraph(
            N=config.N,
            seed=config.seed,
            topology=config.topology,
            edge_probability=config.edge_probability,
        )
        
        L = network.get_laplacian()
        eigenvalues = np.linalg.eigvalsh(L)
        
        viz_data = serialize_chart_2d(
            x_data=np.arange(len(eigenvalues)),
            y_data=eigenvalues,
            chart_type="line",
            title="Eigenvalue Spectrum",
            x_label="Index",
            y_label="Eigenvalue λ",
        )
        
        return JSONResponse(content=viz_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates."""
    await websocket.accept()
    websocket_connections[job_id] = websocket
    
    try:
        while True:
            # Send job updates
            if job_id in jobs_db:
                await websocket.send_json(jobs_db[job_id])
            
            # Check if job is complete
            if job_id in jobs_db and jobs_db[job_id]["status"] in ["completed", "failed"]:
                break
            
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        if job_id in websocket_connections:
            del websocket_connections[job_id]


# ============================================================================
# Background Task Execution
# ============================================================================

async def execute_simulation(job_id: str, request: SimulationRequest):
    """Execute a full IRH simulation in the background."""
    try:
        update_job(job_id, status="running", progress=10.0)
        
        # Create network
        network = HyperGraph(
            N=request.network_config.N,
            seed=request.network_config.seed,
            topology=request.network_config.topology,
            edge_probability=request.network_config.edge_probability,
        )
        
        update_job(job_id, progress=30.0)
        
        result = {
            "network": {
                "N": network.N,
                "edge_count": network.edge_count,
                "topology": request.network_config.topology,
            }
        }
        
        # Compute spectrum
        L = network.get_laplacian()
        eigenvalues = np.linalg.eigvalsh(L)
        result["spectrum"] = {
            "eigenvalues": serialize_numpy(eigenvalues),
            "min": float(eigenvalues[1] if len(eigenvalues) > 1 else 0),
            "max": float(eigenvalues[-1]),
        }
        
        update_job(job_id, progress=50.0)
        
        # Compute spectral dimension
        if request.compute_spectral_dimension:
            ds_result = SpectralDimension(network)
            result["spectral_dimension"] = {
                "value": float(ds_result.value) if not np.isnan(ds_result.value) else None,
                "error": float(ds_result.error) if hasattr(ds_result, 'error') else None,
            }
        
        update_job(job_id, progress=70.0)
        
        # Compute predictions
        if request.compute_predictions:
            prediction = predict_alpha_inverse(network)
            result["predictions"] = {
                "alpha_inverse": float(prediction.value),
                "codata_value": 137.035999084,
                "difference": float(prediction.value - 137.035999084),
            }
        
        update_job(job_id, progress=90.0)
        
        # Run grand audit (expensive)
        if request.run_grand_audit:
            audit_result = grand_audit(network)
            result["grand_audit"] = {
                "total_checks": audit_result.total_checks,
                "pass_count": audit_result.pass_count,
                "fail_count": audit_result.fail_count,
            }
        
        # Mark job as completed
        update_job(
            job_id,
            status="completed",
            progress=100.0,
            completed_at=datetime.now().isoformat(),
            result=result,
        )
        
    except Exception as e:
        update_job(
            job_id,
            status="failed",
            error=str(e),
            completed_at=datetime.now().isoformat(),
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
