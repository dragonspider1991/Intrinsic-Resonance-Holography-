# GEMINI 3.0 AGENTIC VIBE CODE PROMPT
# Technical Specification for IRH Web Application Frontend

## PROJECT OVERVIEW

You are tasked with creating a modern, interactive web application frontend for the **Intrinsic Resonance Holography (IRH) v10.0** test suite. This is a Theory of Everything framework that derives fundamental physics from coupled harmonic oscillators.

The backend REST API and data infrastructure is **already complete and operational**. Your task is to overlay a professional, feature-rich frontend UI on top of this existing infrastructure.

---

## TECHNICAL ARCHITECTURE

### Backend Infrastructure (ALREADY COMPLETE)
- **Framework**: FastAPI (Python)
- **Base URL**: `http://localhost:8000`
- **API Documentation**: Available at `/api/docs` (OpenAPI/Swagger)
- **Real-time Updates**: WebSocket support at `/ws/{job_id}`
- **Data Format**: JSON with proper serialization for numpy arrays

### Frontend Requirements (YOUR TASK)
- **Framework**: Use **React** with **TypeScript** (preferred) OR vanilla JavaScript with modern ES6+
- **Build Tool**: Vite (fast, modern, with HMR)
- **UI Framework**: Material-UI (MUI) or Tailwind CSS for modern, responsive design
- **3D Visualization**: Three.js for WebGL-based 3D rendering
- **2D Visualization**: Chart.js or D3.js for charts and plots
- **State Management**: React Context API or Zustand (lightweight)
- **HTTP Client**: Axios or Fetch API with async/await
- **WebSocket Client**: Native WebSocket API or Socket.io-client

---

## API ENDPOINTS REFERENCE

### Network Creation & Configuration
```typescript
POST /api/network/create
Body: {
  N: number (4-4096),
  topology: "Random" | "Complete" | "Cycle" | "Lattice",
  seed?: number,
  edge_probability: number (0.0-1.0)
}
Response: {
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

### Spectrum Computation
```typescript
POST /api/network/spectrum
Body: NetworkConfig (same as above)
Response: {
  eigenvalues: number[],
  spectral_gap: number,
  min_eigenvalue: number,
  max_eigenvalue: number
}
```

### Spectral Dimension
```typescript
POST /api/network/spectral-dimension
Body: NetworkConfig
Response: {
  spectral_dimension: number | null,
  error: number | null
}
```

### Physical Constant Prediction
```typescript
POST /api/predictions/alpha
Body: NetworkConfig
Response: {
  alpha_inverse: number,
  codata_value: 137.035999084,
  difference: number
}
```

### Full Simulation (Async)
```typescript
POST /api/simulation/run
Body: {
  network_config: NetworkConfig,
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
Response: {
  job_id: string,
  status: "pending"
}
```

### Job Status Tracking
```typescript
GET /api/jobs/{job_id}
Response: {
  job_id: string,
  status: "pending" | "running" | "completed" | "failed",
  progress: number (0-100),
  created_at: string,
  completed_at?: string,
  error?: string,
  result?: object
}
```

### Visualization Data
```typescript
POST /api/visualization/network-3d
Body: NetworkConfig
Response: {
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
  metadata: { node_count, edge_count, avg_edge_weight, max_edge_weight }
}

POST /api/visualization/spectrum-3d
Body: NetworkConfig
Response: {
  type: "scatter3d",
  points: Array<{ x, y, z, color, size }>,
  metadata: { min_eigenvalue, max_eigenvalue, spectral_gap }
}

POST /api/visualization/spectrum-chart
Body: NetworkConfig
Response: Chart.js compatible format with datasets and options
```

### WebSocket Real-time Updates
```typescript
WebSocket: ws://localhost:8000/ws/{job_id}
Receives: Periodic job status updates (same format as GET /api/jobs/{job_id})
```

---

## UI/UX REQUIREMENTS

### Layout Structure

```
┌─────────────────────────────────────────────────────────────┐
│  Header: "IRH - Intrinsic Resonance Holography v10.0"      │
│  [Zero Free Parameters. Explicit Mathematics.]              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │  PARAMETER      │  │                                 │  │
│  │  CONTROL PANEL  │  │    VISUALIZATION CANVAS         │  │
│  │                 │  │                                 │  │
│  │  Network Size   │  │    [3D/2D Toggle]               │  │
│  │  [Slider] N     │  │                                 │  │
│  │                 │  │    [Network | Spectrum | Both]  │  │
│  │  Topology       │  │                                 │  │
│  │  [Dropdown]     │  │    [Rendered visualization]     │  │
│  │                 │  │                                 │  │
│  │  Random Seed    │  │                                 │  │
│  │  [Input]        │  │                                 │  │
│  │                 │  │                                 │  │
│  │  [Run Sim]      │  │                                 │  │
│  │  [Stop]         │  │                                 │  │
│  │                 │  │                                 │  │
│  │  Progress Bar   │  └─────────────────────────────────┘  │
│  │  [========  ]   │                                       │
│  │  75%            │  ┌─────────────────────────────────┐  │
│  │                 │  │   RESULTS PANEL                 │  │
│  │  Results:       │  │                                 │  │
│  │  α⁻¹ = 137.04   │  │   [Tabs: Network | Spectrum |   │  │
│  │  d_s = 4.00     │  │          Predictions | Audit]   │  │
│  │                 │  │                                 │  │
│  └─────────────────┘  │   [Data tables and metrics]     │  │
│                       │                                 │  │
│                       └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Parameter Control Panel (Left Sidebar)
- **Network Size Slider**: 4 to 4096, default 64, logarithmic scale
- **Topology Dropdown**: Random, Complete, Cycle, Lattice
- **Edge Probability Slider**: 0.0 to 1.0, default 0.3 (only for Random)
- **Random Seed Input**: Optional integer, checkbox to enable/disable
- **Optimization Settings** (Collapsible):
  - Max Iterations: 10 to 10000, default 500
  - Initial Temperature: 0.1 to 10.0, default 1.0
  - Final Temperature: 0.001 to 1.0, default 0.01
- **Computation Checkboxes**:
  - ☑ Compute Spectral Dimension
  - ☑ Compute Physical Predictions
  - ☐ Run Grand Audit (expensive)
- **Action Buttons**:
  - Primary: "Run Simulation" (large, prominent)
  - Secondary: "Stop Simulation"
  - Tertiary: "Reset Parameters", "Load Preset"
- **Progress Indicator**:
  - Linear progress bar with percentage
  - Status text ("Creating network...", "Computing spectrum...", etc.)
  - Estimated time remaining

#### 2. Visualization Canvas (Center/Right Main Area)
- **Mode Toggle Buttons**:
  - 3D View (default)
  - 2D Charts View
- **Visualization Type Selector**:
  - Network Topology
  - Eigenvalue Spectrum
  - Split View (both)
- **3D Visualization** (Three.js):
  - Interactive orbit controls (rotate, zoom, pan)
  - Nodes rendered as spheres with colors mapped to eigenvalues
  - Edges rendered as lines with opacity based on weight
  - Lighting: ambient + directional for depth
  - Background: dark gradient or solid dark color
  - FPS counter (optional, debug mode)
  - Camera controls widget
- **2D Visualization** (Chart.js):
  - Line chart: Eigenvalue spectrum (index vs. value)
  - Histogram: Eigenvalue distribution
  - Heatmap: Adjacency matrix (for smaller N)
  - Interactive tooltips
  - Zoom/pan controls
  - Export buttons (PNG, SVG)
- **Legend**:
  - Color scale for eigenvalue mapping
  - Node size meaning
  - Edge opacity meaning

#### 3. Results Panel (Bottom or Right Tabs)
Tabbed interface with:

**Tab 1: Network Info**
- Node count: N
- Edge count
- Topology
- Connectivity metrics
- Table of basic stats

**Tab 2: Spectrum**
- Eigenvalue statistics:
  - Min (excluding λ₀=0)
  - Max
  - Spectral gap
  - Mean
  - Standard deviation
- Spectral dimension d_s
- Comparison to d_s = 4 target

**Tab 3: Physical Predictions**
- Fine structure constant α⁻¹
  - Predicted value
  - CODATA value: 137.035999084
  - Difference
  - Relative error
  - Status indicator (✓ within tolerance, ✗ outside)
- Neutrino masses (if computed)
- CKM matrix (if computed)
- Other predictions

**Tab 4: Grand Audit** (if run)
- Total checks
- Passed count
- Failed count
- Pass rate percentage
- Detailed results table

### Visual Design Requirements

#### Color Scheme (Dark Theme)
- Background: `#1a1a2e` (dark blue-black)
- Surface: `#16213e` (slightly lighter)
- Primary: `#0f4c75` (deep blue)
- Accent: `#3282b8` (bright blue)
- Success: `#00d9ff` (cyan)
- Warning: `#ffa62b` (orange)
- Error: `#ef4444` (red)
- Text Primary: `#eee` (light gray)
- Text Secondary: `#aaa` (medium gray)

#### Typography
- **Headings**: "Inter" or "Roboto", bold, sans-serif
- **Body**: "Inter" or "Roboto", regular, sans-serif
- **Monospace** (for numbers): "JetBrains Mono" or "Fira Code"

#### Responsive Design
- Desktop (≥1200px): Full three-column layout as shown
- Tablet (768px-1199px): Two-column, parameters + visualization, results in tabs below
- Mobile (<768px): Single column, stacked, with collapsible sections

#### Accessibility
- ARIA labels for all interactive elements
- Keyboard navigation support
- High contrast mode option
- Screen reader friendly
- Focus indicators

---

## 3D VISUALIZATION TECHNICAL DETAILS (Three.js)

### Scene Setup
```javascript
// Basic Three.js scene structure
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });

// Background
scene.background = new THREE.Color(0x1a1a2e);

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
scene.add(ambientLight, directionalLight);

// Orbit controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
```

### Network Rendering
- **Nodes**: `THREE.SphereGeometry` with `THREE.MeshStandardMaterial`
  - Radius based on degree or eigenvalue magnitude
  - Color from eigenvalue mapping (use Viridis colormap)
- **Edges**: `THREE.Line` or `THREE.LineSegments`
  - Color based on weight
  - Opacity proportional to weight
  - Use `LineBasicMaterial` or `LineDashedMaterial`

### Performance Optimization
- Use `THREE.InstancedMesh` for large node counts (N > 500)
- Level of detail (LOD) for distant nodes
- Frustum culling
- Limit edge rendering for dense graphs (threshold)

### Animation
- Smooth camera transitions
- Optional auto-rotate when idle
- Fade in/out on data updates

---

## 2D VISUALIZATION TECHNICAL DETAILS (Chart.js)

### Chart Configurations

**Eigenvalue Spectrum Line Chart**
```javascript
{
  type: 'line',
  data: {
    labels: eigenvalueIndices,
    datasets: [{
      label: 'Eigenvalue λ',
      data: eigenvalues,
      borderColor: '#3282b8',
      backgroundColor: 'rgba(50, 130, 184, 0.1)',
      borderWidth: 2,
      pointRadius: 2,
    }]
  },
  options: {
    responsive: true,
    plugins: {
      title: { display: true, text: 'Eigenvalue Spectrum' },
      legend: { display: true }
    },
    scales: {
      x: { title: { display: true, text: 'Index' } },
      y: { title: { display: true, text: 'Eigenvalue' } }
    }
  }
}
```

**Eigenvalue Histogram**
```javascript
{
  type: 'bar',
  data: {
    labels: binCenters,
    datasets: [{
      label: 'Frequency',
      data: binCounts,
      backgroundColor: '#3282b8',
    }]
  },
  options: {
    responsive: true,
    plugins: { title: { display: true, text: 'Eigenvalue Distribution' } },
    scales: {
      x: { title: { display: true, text: 'Eigenvalue' } },
      y: { title: { display: true, text: 'Count' } }
    }
  }
}
```

---

## STATE MANAGEMENT

### Application State
```typescript
interface AppState {
  // Parameters
  networkConfig: {
    N: number;
    topology: string;
    seed: number | null;
    edge_probability: number;
  };
  optimizationConfig: {
    max_iterations: number;
    T_initial: number;
    T_final: number;
    verbose: boolean;
  };
  computeFlags: {
    spectral_dimension: boolean;
    predictions: boolean;
    grand_audit: boolean;
  };
  
  // Simulation status
  currentJob: {
    job_id: string | null;
    status: 'idle' | 'pending' | 'running' | 'completed' | 'failed';
    progress: number;
    error: string | null;
  };
  
  // Results
  results: {
    network?: object;
    spectrum?: object;
    spectral_dimension?: object;
    predictions?: object;
    grand_audit?: object;
    visualization_3d?: object;
    visualization_2d?: object;
  };
  
  // UI state
  ui: {
    visualizationMode: '3d' | '2d';
    visualizationType: 'network' | 'spectrum' | 'both';
    activeTab: number;
    darkMode: boolean;
  };
}
```

---

## INTERACTION FLOWS

### Flow 1: Run Simple Simulation
1. User adjusts parameters (N, topology, seed)
2. User clicks "Run Simulation"
3. Frontend POSTs to `/api/simulation/run`
4. Backend returns `job_id`
5. Frontend connects WebSocket to `/ws/{job_id}`
6. Progress updates stream in, UI updates progress bar
7. On completion, frontend GETs `/api/jobs/{job_id}/result`
8. Results displayed in tabs
9. Visualization rendered in canvas

### Flow 2: Interactive Parameter Exploration
1. User changes slider (e.g., network size)
2. Debounced API call to `/api/network/create`
3. Quick preview of network properties shown
4. User can visualize network structure instantly
5. Full simulation optional for detailed analysis

### Flow 3: Real-time Visualization Update
1. WebSocket receives progress update
2. Extract current data (e.g., partial spectrum)
3. Update 2D chart incrementally
4. Smooth animation (no jarring redraws)

---

## ADVANCED FEATURES (OPTIONAL BUT RECOMMENDED)

### Parameter Presets
- "Quick Demo" (N=64, Random)
- "High Precision" (N=256, Lattice)
- "Complete Graph" (N=32, Complete)
- "Custom" (user-defined)

### Export Functionality
- **Data Export**: JSON, CSV of results
- **Image Export**: PNG of visualizations
- **Report Export**: PDF with all results and charts

### Comparison Mode
- Run multiple simulations with different parameters
- Side-by-side comparison
- Overlay spectra on same chart

### Animation/Playback
- For time-evolution simulations (future feature)
- Scrub timeline
- Play/pause controls

---

## GOOGLE AI STUDIO INTEGRATION NOTES

### OLM Features to Leverage
- **Code Completion**: Use Gemini's code completion for boilerplate
- **Error Detection**: Real-time error checking during development
- **Documentation Generation**: Auto-generate JSDoc comments
- **Refactoring Suggestions**: Optimize code structure
- **Accessibility Checks**: Ensure WCAG compliance

### Suggested Development Workflow
1. **Phase 1**: Set up React + Vite + TypeScript project structure
2. **Phase 2**: Create basic layout with placeholder components
3. **Phase 3**: Implement parameter control panel with state management
4. **Phase 4**: Integrate API client and test endpoints
5. **Phase 5**: Build 3D visualization with Three.js
6. **Phase 6**: Build 2D charts with Chart.js
7. **Phase 7**: Implement WebSocket real-time updates
8. **Phase 8**: Add results panel with tabs
9. **Phase 9**: Polish UI/UX, add animations
10. **Phase 10**: Test, debug, optimize performance

---

## DELIVERABLES

### Required Files Structure
```
frontend/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── index.html
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── components/
│   │   ├── ParameterPanel.tsx
│   │   ├── VisualizationCanvas.tsx
│   │   ├── Visualization3D.tsx
│   │   ├── Visualization2D.tsx
│   │   ├── ResultsPanel.tsx
│   │   ├── ProgressIndicator.tsx
│   │   └── ...
│   ├── services/
│   │   ├── api.ts
│   │   └── websocket.ts
│   ├── hooks/
│   │   └── useSimulation.ts
│   ├── store/
│   │   └── appStore.ts
│   ├── types/
│   │   └── index.ts
│   ├── utils/
│   │   └── helpers.ts
│   └── styles/
│       └── global.css
├── public/
│   └── favicon.ico
└── README.md
```

### Documentation
- **README.md**: How to run the frontend
- **API_INTEGRATION.md**: Details on backend API usage
- **COMPONENTS.md**: Component hierarchy and props
- Inline code comments (especially for Three.js and complex logic)

---

## SUCCESS CRITERIA

Your frontend implementation will be considered successful if:

1. ✅ All API endpoints are correctly integrated
2. ✅ 3D network visualization renders smoothly (60 FPS for N ≤ 500)
3. ✅ 2D charts display accurate data with proper labels
4. ✅ Real-time progress updates via WebSocket work correctly
5. ✅ UI is responsive across desktop, tablet, and mobile
6. ✅ Dark theme is aesthetically pleasing and consistent
7. ✅ All parameters can be adjusted and persist in state
8. ✅ Results display correctly in tabbed interface
9. ✅ No console errors or warnings
10. ✅ Code is clean, typed (if TypeScript), and well-documented

---

## ADDITIONAL CONTEXT

### About IRH Theory
Intrinsic Resonance Holography is a "Theory of Everything" that derives all physics from a network of coupled harmonic oscillators. Key concepts:
- **Network**: Graph of N nodes (oscillators) with weighted edges (couplings)
- **Eigenspectrum**: Eigenvalues of the graph Laplacian determine physical properties
- **Spectral Dimension**: Should converge to d_s = 4 (our 4D spacetime)
- **α⁻¹**: Fine structure constant, predicted to be ≈137.036 from network topology
- **Zero Free Parameters**: All constants derived from topology, no tuning

### User Persona
- **Primary Users**: Physics researchers, computational scientists
- **Secondary Users**: Students, educators, enthusiasts
- **Technical Level**: Comfortable with scientific software, expect precision
- **Needs**: Explore parameter space, validate predictions, generate publication-quality visualizations

---

## FINAL NOTES

This prompt provides a **complete technical specification** for the frontend. You have:
- Full API documentation with TypeScript types
- Exact layout requirements
- Visual design specifications
- Component structure
- State management approach
- 3D and 2D visualization details
- Interaction flows
- Success criteria

**Your task**: Implement a production-ready, beautiful, performant frontend that brings this theoretical physics framework to life in the browser.

Good luck! 🚀

---

**Contact for Questions**: If you need clarification during implementation, refer to the OpenAPI docs at `http://localhost:8000/api/docs` or examine the backend code in `webapp/backend/app.py`.
