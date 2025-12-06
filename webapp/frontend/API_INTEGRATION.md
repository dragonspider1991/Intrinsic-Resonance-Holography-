# IRH Frontend - Backend API Integration Guide

This document describes how the IRH web frontend integrates with the FastAPI backend.

## Overview

The frontend is a React + TypeScript application that communicates with the backend via:
- **HTTP REST API** for synchronous operations
- **WebSocket** for real-time progress updates

## API Client Architecture

### Base Configuration

**File**: `src/services/api.ts`

```typescript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

Environment variable can be set in `.env`:
```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

### HTTP Client (Axios)

All HTTP requests use Axios with:
- Base URL: `http://localhost:8000`
- Content-Type: `application/json`
- Timeout: 30 seconds

## API Endpoints

### 1. Network Creation

**Endpoint**: `POST /api/network/create`

**Request**:
```typescript
{
  N: number,              // 4-4096
  topology: string,       // "Random" | "Complete" | "Cycle" | "Lattice"
  seed?: number,          // Optional random seed
  edge_probability: number // 0.0-1.0
}
```

**Response**:
```typescript
{
  N: number,
  edge_count: number,
  topology: string,
  spectrum: {
    eigenvalues: number[],
    min: number,
    max: number
  },
  adjacency_matrix: number[][]
}
```

**Frontend Usage**:
```typescript
const network = await apiClient.createNetwork(networkConfig);
```

### 2. Spectrum Computation

**Endpoint**: `POST /api/network/spectrum`

**Request**: Same as network creation

**Response**:
```typescript
{
  eigenvalues: number[],
  spectral_gap: number,
  min_eigenvalue: number,
  max_eigenvalue: number
}
```

**Frontend Usage**:
```typescript
const spectrum = await apiClient.getSpectrum(networkConfig);
```

### 3. Spectral Dimension

**Endpoint**: `POST /api/network/spectral-dimension`

**Request**: Same as network creation

**Response**:
```typescript
{
  spectral_dimension: number | null,
  error: number | null
}
```

**Frontend Usage**:
```typescript
const sd = await apiClient.getSpectralDimension(networkConfig);
```

### 4. Physical Constant Prediction

**Endpoint**: `POST /api/predictions/alpha`

**Request**: Same as network creation

**Response**:
```typescript
{
  alpha_inverse: number,
  codata_value: 137.035999084,
  difference: number
}
```

**Frontend Usage**:
```typescript
const alpha = await apiClient.predictAlpha(networkConfig);
```

### 5. Full Simulation (Async)

**Endpoint**: `POST /api/simulation/run`

**Request**:
```typescript
{
  network_config: {
    N: number,
    topology: string,
    seed?: number,
    edge_probability: number
  },
  optimization_config?: {
    max_iterations: number,
    T_initial: number,
    T_final: number,
    verbose: boolean
  },
  compute_spectral_dimension: boolean,
  compute_predictions: boolean,
  run_grand_audit: boolean
}
```

**Response**:
```typescript
{
  job_id: string,
  status: "pending"
}
```

**Frontend Usage**:
```typescript
const response = await apiClient.runSimulation(request);
const jobId = response.job_id;
// Then connect WebSocket for updates
```

### 6. Job Status

**Endpoint**: `GET /api/jobs/{job_id}`

**Response**:
```typescript
{
  job_id: string,
  status: "pending" | "running" | "completed" | "failed",
  progress: number,       // 0-100
  created_at: string,
  completed_at?: string,
  error?: string,
  result?: {
    // Full simulation results
  }
}
```

**Frontend Usage**:
```typescript
const job = await apiClient.getJobStatus(jobId);
```

### 7. Network 3D Visualization

**Endpoint**: `POST /api/visualization/network-3d`

**Request**: Same as network creation

**Response**:
```typescript
{
  nodes: Array<{
    id: number,
    position: [number, number, number],
    color: string,
    size: number
  }>,
  edges: Array<{
    source: number,
    target: number,
    weight: number,
    color: string,
    opacity: number
  }>,
  metadata: {
    node_count: number,
    edge_count: number,
    avg_edge_weight: number,
    max_edge_weight: number
  }
}
```

**Frontend Usage**:
```typescript
const vizData = await apiClient.getNetwork3D(networkConfig);
// Render with Three.js
```

### 8. Spectrum 3D Visualization

**Endpoint**: `POST /api/visualization/spectrum-3d`

**Request**: Same as network creation

**Response**:
```typescript
{
  type: "scatter3d",
  points: Array<{
    x: number,
    y: number,
    z: number,
    color: string,
    size: number
  }>,
  metadata: {
    min_eigenvalue: number,
    max_eigenvalue: number,
    spectral_gap: number
  }
}
```

**Frontend Usage**:
```typescript
const spectrumViz = await apiClient.getSpectrum3D(networkConfig);
// Render with Three.js
```

### 9. Spectrum Chart (2D)

**Endpoint**: `POST /api/visualization/spectrum-chart`

**Request**: Same as network creation

**Response**:
```typescript
{
  type: "line",
  data: {
    labels?: any[],
    datasets: Array<{
      label: string,
      data: any[],
      borderColor?: string,
      backgroundColor?: string,
      borderWidth?: number,
      pointRadius?: number
    }>
  },
  options: {
    // Chart.js options
  }
}
```

**Frontend Usage**:
```typescript
const chartData = await apiClient.getSpectrumChart(networkConfig);
// Render with Chart.js
```

## WebSocket Integration

### Connection

**File**: `src/services/websocket.ts`

**URL**: `ws://localhost:8000/ws/{job_id}`

**Frontend Usage**:
```typescript
websocketClient.connect(
  jobId,
  (jobUpdate) => {
    // Handle progress updates
    setCurrentJob({
      status: jobUpdate.status,
      progress: jobUpdate.progress,
      error: jobUpdate.error,
    });
    
    // Handle completion
    if (jobUpdate.status === 'completed') {
      setResults(jobUpdate.result);
    }
  },
  (error) => {
    // Handle errors
    console.error('WebSocket error:', error);
  }
);
```

### Message Format

WebSocket sends periodic updates:
```typescript
{
  job_id: string,
  status: "pending" | "running" | "completed" | "failed",
  progress: number,
  created_at: string,
  completed_at?: string,
  error?: string,
  result?: object
}
```

### Connection Lifecycle

1. **Connect**: When simulation starts
2. **Receive Updates**: Every few seconds during simulation
3. **Auto-Close**: When job completes or fails
4. **Auto-Reconnect**: Up to 5 attempts if connection drops

## State Management

### Zustand Store

**File**: `src/store/appStore.ts`

**State Structure**:
```typescript
{
  networkConfig: { N, topology, seed, edge_probability },
  optimizationConfig: { max_iterations, T_initial, T_final, verbose },
  computeFlags: { spectral_dimension, predictions, grand_audit },
  currentJob: { job_id, status, progress, error },
  results: { network, spectrum, spectral_dimension, predictions, grand_audit },
  ui: { visualizationMode, visualizationType, activeTab, darkMode }
}
```

### Actions

- `setNetworkConfig(config)` - Update network parameters
- `setOptimizationConfig(config)` - Update optimization settings
- `setComputeFlags(flags)` - Toggle computation options
- `setCurrentJob(job)` - Update job status
- `setResults(results)` - Store simulation results
- `setUI(ui)` - Update UI state
- `resetParameters()` - Reset to defaults

## Simulation Workflow

### Hook: `useSimulation`

**File**: `src/hooks/useSimulation.ts`

**Functions**:
- `runSimulation()` - Start a new simulation
- `stopSimulation()` - Cancel running simulation

### Complete Flow

```typescript
// 1. User clicks "Run Simulation"
const { runSimulation } = useSimulation();
await runSimulation();

// 2. Hook prepares request
const request = {
  network_config: networkConfig,
  optimization_config: optimizationConfig,
  compute_spectral_dimension: computeFlags.spectral_dimension,
  compute_predictions: computeFlags.predictions,
  run_grand_audit: computeFlags.grand_audit
};

// 3. POST to /api/simulation/run
const response = await apiClient.runSimulation(request);
const jobId = response.job_id;

// 4. Connect WebSocket
websocketClient.connect(jobId, handleUpdate);

// 5. Receive progress updates
// - Update progress bar
// - Show status messages

// 6. On completion
// - Disconnect WebSocket
// - Store results in Zustand
// - Display in UI
```

## Error Handling

### HTTP Errors

```typescript
try {
  const result = await apiClient.someMethod();
} catch (error: any) {
  console.error('API error:', error);
  setCurrentJob({
    status: 'failed',
    error: error.response?.data?.detail || error.message
  });
}
```

### WebSocket Errors

```typescript
websocketClient.connect(
  jobId,
  handleUpdate,
  (error) => {
    setCurrentJob({
      status: 'failed',
      error: 'WebSocket connection failed'
    });
  }
);
```

## CORS Configuration

Backend must allow frontend origin:

```python
# In backend/app.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

For production, update to actual domain.

## Development vs Production

### Development

```env
# .env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

### Production

```env
# .env.production
VITE_API_URL=https://api.yourdomain.com
VITE_WS_URL=wss://api.yourdomain.com
```

## Testing Integration

### Manual Testing

1. Start backend:
   ```bash
   cd webapp
   python start_server.py
   ```

2. Start frontend:
   ```bash
   cd webapp/frontend
   npm run dev
   ```

3. Open http://localhost:5173

4. Test endpoints:
   - Create network
   - Run simulation
   - View visualizations
   - Check real-time updates

### Verify API

Visit http://localhost:8000/api/docs to test endpoints directly.

## Performance Considerations

### Debouncing

Parameter changes are debounced to avoid excessive API calls:

```typescript
// In ParameterPanel.tsx
const handleSliderChange = useMemo(
  () => debounce((value) => setNetworkConfig({ N: value }), 300),
  []
);
```

### Caching

Results are cached in Zustand store to avoid re-fetching.

### Visualization Limits

- 3D rendering optimized for N ≤ 500 nodes
- Larger networks use instanced meshes
- Chart decimation for large datasets

## Troubleshooting

### Common Issues

1. **"Network Error"**
   - Backend not running
   - CORS not configured
   - Wrong API URL in `.env`

2. **"WebSocket connection failed"**
   - Backend WebSocket endpoint not accessible
   - Firewall blocking WebSocket
   - Wrong WS URL in `.env`

3. **"Visualization not loading"**
   - Check browser console for errors
   - Verify API returns valid data format
   - Check WebGL support

### Debug Mode

Enable in browser console:
```javascript
localStorage.setItem('debug', 'true');
```

Check network tab in browser DevTools for API calls.

---

**Last Updated**: 2025-12-06
**Version**: 1.0.0
