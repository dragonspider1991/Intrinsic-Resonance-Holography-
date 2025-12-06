# Component Architecture

This document describes the component hierarchy and architecture of the IRH frontend.

## Component Tree

```
App
└── ThemeProvider
    └── Layout
        ├── AppBar (Header)
        └── Container (Main Content)
            ├── ParameterPanel
            │   ├── Network Size Slider
            │   ├── Topology Selector
            │   ├── Edge Probability Slider
            │   ├── Random Seed Input
            │   ├── Optimization Settings (Accordion)
            │   ├── Computation Checkboxes
            │   ├── Action Buttons
            │   └── Progress Indicator
            ├── VisualizationCanvas
            │   ├── Mode Toggle (3D/2D)
            │   ├── Type Selector (Network/Spectrum/Both)
            │   └── Visualization (Conditional)
            │       ├── Visualization3D (Three.js)
            │       └── Visualization2D (Chart.js)
            └── ResultsPanel
                └── Tabs
                    ├── NetworkTab
                    ├── SpectrumTab
                    ├── PredictionsTab
                    └── GrandAuditTab
```

## Component Details

### 1. App.tsx

**Purpose**: Root component that sets up theme and layout

**Props**: None

**State**: None (uses Zustand store)

**Features**:
- Applies MUI theme
- Sets up global CSS baseline
- Renders Layout with main content areas

**File**: `src/App.tsx`

---

### 2. Layout.tsx

**Purpose**: Provides consistent page structure

**Props**:
```typescript
{
  children: React.ReactNode
}
```

**Features**:
- Fixed header with app title
- Main content area with container
- Responsive padding and sizing

**File**: `src/components/Layout.tsx`

---

### 3. ParameterPanel.tsx

**Purpose**: Control panel for configuring simulation parameters

**Props**: None (uses Zustand store)

**Store Access**:
- `networkConfig` - Read/Write
- `optimizationConfig` - Read/Write
- `computeFlags` - Read/Write
- `currentJob` - Read only
- Actions: `setNetworkConfig`, `setOptimizationConfig`, `setComputeFlags`, `resetParameters`

**Features**:
- Network size slider (logarithmic scale)
- Topology dropdown
- Edge probability slider (conditional on Random topology)
- Random seed toggle and input
- Collapsible optimization settings
- Computation checkboxes
- Run/Stop/Reset buttons
- Progress bar during simulation
- Error display

**Interactions**:
- Calls `useSimulation().runSimulation()` on Run button
- Calls `useSimulation().stopSimulation()` on Stop button
- Calls `resetParameters()` on Reset button

**File**: `src/components/ParameterPanel.tsx`

---

### 4. VisualizationCanvas.tsx

**Purpose**: Container for 2D and 3D visualizations

**Props**: None (uses Zustand store)

**Store Access**:
- `ui.visualizationMode` - Read/Write
- `ui.visualizationType` - Read/Write
- Action: `setUI`

**Features**:
- Toggle between 3D and 2D modes
- Select visualization type (Network, Spectrum, Both)
- Renders appropriate child component

**Children**:
- `Visualization3D` - When mode is '3d'
- `Visualization2D` - When mode is '2d'

**File**: `src/components/VisualizationCanvas.tsx`

---

### 5. Visualization3D.tsx

**Purpose**: 3D visualization using Three.js

**Props**: None (uses Zustand store)

**Store Access**:
- `networkConfig` - Read only
- `ui.visualizationType` - Read only

**Features**:
- Three.js scene setup (camera, renderer, lights)
- Orbit controls for interaction
- Network node rendering (spheres)
- Network edge rendering (lines)
- Spectrum point rendering (spheres)
- Loading state
- Error handling

**API Calls**:
- `apiClient.getNetwork3D(networkConfig)` - For network visualization
- `apiClient.getSpectrum3D(networkConfig)` - For spectrum visualization

**Performance**:
- Uses `requestAnimationFrame` for smooth rendering
- Cleans up scene objects on re-render
- Disposes WebGL resources on unmount

**File**: `src/components/Visualization3D.tsx`

---

### 6. Visualization2D.tsx

**Purpose**: 2D charts using Chart.js

**Props**: None (uses Zustand store)

**Store Access**:
- `networkConfig` - Read only

**Features**:
- Line chart rendering
- Bar chart rendering
- Loading state
- Error handling

**API Calls**:
- `apiClient.getSpectrumChart(networkConfig)` - Get chart data

**Chart Types**:
- Line: Eigenvalue spectrum
- Bar: Eigenvalue distribution histogram

**File**: `src/components/Visualization2D.tsx`

---

### 7. ResultsPanel.tsx

**Purpose**: Tabbed panel displaying simulation results

**Props**: None (uses Zustand store)

**Store Access**:
- `ui.activeTab` - Read/Write
- `results` - Read only
- Action: `setUI`

**Features**:
- 4 tabs: Network, Spectrum, Predictions, Grand Audit
- Conditional rendering based on available data
- Empty state messages

**Tabs**:

#### Network Tab
- Node count
- Edge count
- Topology
- Min/max eigenvalues

#### Spectrum Tab
- Eigenvalue statistics
- Spectral gap
- Spectral dimension (if computed)
- Error estimate

#### Predictions Tab
- Fine structure constant (α⁻¹)
- Predicted vs CODATA value
- Difference
- Status indicator (pass/fail)

#### Grand Audit Tab
- Total checks
- Passed/failed counts
- Pass rate percentage
- Detailed results table

**File**: `src/components/ResultsPanel.tsx`

---

## Hooks

### useSimulation

**Purpose**: Manages simulation lifecycle

**File**: `src/hooks/useSimulation.ts`

**Returns**:
```typescript
{
  runSimulation: () => Promise<string>,
  stopSimulation: () => void
}
```

**Functionality**:
1. Prepares simulation request from store state
2. Calls API to start simulation
3. Connects WebSocket for updates
4. Updates store with progress
5. Stores results on completion

**Usage**:
```typescript
const { runSimulation, stopSimulation } = useSimulation();

// Start simulation
await runSimulation();

// Stop simulation
stopSimulation();
```

---

## Services

### API Client

**Purpose**: Centralized HTTP client for backend API

**File**: `src/services/api.ts`

**Class**: `APIClient`

**Methods**:
- `createNetwork(config)` - Create network
- `getSpectrum(config)` - Get spectrum
- `getSpectralDimension(config)` - Get spectral dimension
- `predictAlpha(config)` - Predict alpha
- `runSimulation(request)` - Start simulation
- `getJobStatus(jobId)` - Get job status
- `getNetwork3D(config)` - Get 3D network data
- `getSpectrum3D(config)` - Get 3D spectrum data
- `getSpectrumChart(config)` - Get 2D chart data

**Export**: `apiClient` singleton instance

---

### WebSocket Client

**Purpose**: Manages WebSocket connections for real-time updates

**File**: `src/services/websocket.ts`

**Class**: `WebSocketClient`

**Methods**:
- `connect(jobId, onMessage, onError)` - Connect to job updates
- `disconnect()` - Close connection
- `isConnected()` - Check connection status

**Features**:
- Auto-reconnect (up to 5 attempts)
- Auto-close on job completion
- Error handling

**Export**: `websocketClient` singleton instance

---

## Store

### App Store (Zustand)

**Purpose**: Global application state

**File**: `src/store/appStore.ts`

**State Shape**:
```typescript
{
  networkConfig: NetworkConfig,
  optimizationConfig: OptimizationConfig,
  computeFlags: ComputeFlags,
  currentJob: {
    job_id: string | null,
    status: JobStatus,
    progress: number,
    error: string | null
  },
  results: SimulationResults,
  ui: {
    visualizationMode: '3d' | '2d',
    visualizationType: 'network' | 'spectrum' | 'both',
    activeTab: number,
    darkMode: boolean
  }
}
```

**Actions**:
- `setNetworkConfig(config)`
- `setOptimizationConfig(config)`
- `setComputeFlags(flags)`
- `setCurrentJob(job)`
- `setResults(results)`
- `setUI(ui)`
- `resetParameters()`

**Usage**:
```typescript
import { useAppStore } from '../store/appStore';

const Component = () => {
  const { networkConfig, setNetworkConfig } = useAppStore();
  
  // Read state
  console.log(networkConfig.N);
  
  // Update state
  setNetworkConfig({ N: 128 });
};
```

---

## Types

### TypeScript Definitions

**File**: `src/types/index.ts`

**Exports**:
- `NetworkConfig` - Network configuration
- `OptimizationConfig` - Optimization settings
- `ComputeFlags` - Computation toggles
- `NetworkResponse` - Network API response
- `SpectrumResponse` - Spectrum API response
- `SpectralDimensionResponse` - Spectral dimension response
- `AlphaPredictionResponse` - Alpha prediction response
- `SimulationRequest` - Full simulation request
- `JobResponse` - Job status response
- `Node3D` - 3D node data
- `Edge3D` - 3D edge data
- `Network3DVisualization` - Network 3D data
- `Spectrum3DVisualization` - Spectrum 3D data
- `ChartData` - Chart.js data format
- `GrandAuditResult` - Grand audit results
- `SimulationResults` - All results
- `AppState` - Zustand store state

---

## Styling

### Theme

**File**: `src/utils/theme.ts`

**MUI Theme Configuration**:
- Dark mode palette
- Custom colors (primary, secondary, success, warning, error)
- Typography (Inter font family)
- Component overrides (buttons, papers, cards)

**Color Scheme**:
```typescript
{
  background: {
    default: '#1a1a2e',
    paper: '#16213e'
  },
  primary: '#3282b8',
  secondary: '#00d9ff',
  success: '#00d9ff',
  warning: '#ffa62b',
  error: '#ef4444'
}
```

### Global Styles

**File**: `src/index.css`

- CSS reset
- Font configuration
- Scrollbar styling
- Root element sizing

---

## Data Flow

### Simulation Flow

```
User Action (ParameterPanel)
  ↓
useSimulation hook
  ↓
API Client (POST /api/simulation/run)
  ↓
Backend returns job_id
  ↓
WebSocket connects (/ws/{job_id})
  ↓
Progress updates received
  ↓
Store updated (setCurrentJob, setResults)
  ↓
UI re-renders (ParameterPanel, ResultsPanel)
```

### Visualization Flow

```
User changes parameters
  ↓
Store updated (setNetworkConfig)
  ↓
useEffect triggers in Visualization3D/2D
  ↓
API Client fetches viz data
  ↓
Three.js/Chart.js renders
  ↓
User sees updated visualization
```

---

## Best Practices

### Component Design

1. **Functional Components**: All components use hooks
2. **No Props Drilling**: Use Zustand store for shared state
3. **Separation of Concerns**: 
   - Components for UI
   - Services for API/WebSocket
   - Hooks for business logic
   - Store for state management

### State Management

1. **Single Source of Truth**: Zustand store
2. **Immutable Updates**: Always spread previous state
3. **Selective Updates**: Only update changed values

### Performance

1. **Memoization**: Use `useMemo` and `useCallback` where needed
2. **Cleanup**: Dispose Three.js resources, close WebSocket
3. **Debouncing**: Debounce rapid parameter changes

### Error Handling

1. **Try-Catch**: Wrap all API calls
2. **User Feedback**: Show errors in UI
3. **Logging**: Console.error for debugging

---

**Last Updated**: 2025-12-06
**Version**: 1.0.0
