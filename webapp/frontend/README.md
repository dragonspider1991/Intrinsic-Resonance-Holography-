# IRH Web Application Frontend

Interactive web interface for Intrinsic Resonance Holography v10.0 test suite.

## Features

- **Parameter Control Panel**: Configure network size, topology, optimization settings, and computation flags
- **3D Visualization**: Interactive network topology and eigenvalue spectrum using Three.js
- **2D Charts**: Eigenvalue spectrum charts and distributions using Chart.js
- **Real-time Updates**: WebSocket-based progress tracking during simulations
- **Results Panel**: Tabbed interface displaying network info, spectrum data, physical predictions, and grand audit results
- **Dark Theme**: Professional dark theme optimized for data visualization
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Technology Stack

- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite 7
- **UI Library**: Material-UI (MUI) v7
- **3D Graphics**: Three.js
- **Charts**: Chart.js with react-chartjs-2
- **State Management**: Zustand
- **HTTP Client**: Axios
- **WebSocket**: Native WebSocket API

## Prerequisites

- Node.js 18+ and npm
- Backend API running on http://localhost:8000

## Quick Start

### Installation

```bash
# Install dependencies
npm install
```

### Development

```bash
# Start development server with hot reload
npm run dev
```

The application will be available at http://localhost:5173

### Production Build

```bash
# Build for production
npm run build

# Preview production build locally
npm run preview
```

## Configuration

Edit `.env` file to configure API endpoints:

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## Project Structure

```
src/
├── components/          # React components
│   ├── Layout.tsx      # Main layout wrapper
│   ├── ParameterPanel.tsx
│   ├── VisualizationCanvas.tsx
│   ├── Visualization3D.tsx
│   ├── Visualization2D.tsx
│   └── ResultsPanel.tsx
├── services/           # API and WebSocket clients
│   ├── api.ts
│   └── websocket.ts
├── hooks/              # Custom React hooks
│   └── useSimulation.ts
├── store/              # Zustand state management
│   └── appStore.ts
├── types/              # TypeScript type definitions
│   └── index.ts
├── utils/              # Utilities and theme
│   └── theme.ts
├── App.tsx             # Main application component
├── main.tsx            # Application entry point
└── index.css           # Global styles
```

## Usage

### Running a Simulation

1. Configure parameters in the left panel:
   - Network size (N): 4 to 4096 nodes
   - Topology: Random, Complete, Cycle, or Lattice
   - Edge probability (for Random topology)
   - Optional: Set random seed for reproducibility

2. Adjust optimization settings (collapsible section):
   - Max iterations
   - Initial and final temperature

3. Select computations:
   - ✓ Compute Spectral Dimension
   - ✓ Compute Physical Predictions
   - ☐ Run Grand Audit (expensive)

4. Click "Run Simulation"

5. Monitor progress in real-time via the progress bar

6. View results in the visualization canvas and results panel

### Viewing Visualizations

**3D Mode** (default):
- Rotate: Left-click and drag
- Zoom: Scroll wheel
- Pan: Right-click and drag
- Toggle between Network, Spectrum, or Both

**2D Mode**:
- View eigenvalue spectrum as line chart
- Interactive tooltips on hover

### Interpreting Results

**Network Tab**:
- Node and edge counts
- Topology information
- Eigenvalue statistics

**Spectrum Tab**:
- Min/max eigenvalues
- Spectral gap
- Spectral dimension (d_s)

**Predictions Tab**:
- Fine structure constant (α⁻¹)
- Comparison with CODATA value
- Difference and tolerance status

**Grand Audit Tab** (if enabled):
- Total checks, passed, failed
- Pass rate percentage
- Detailed results table

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

WebGL 2.0 required for 3D visualizations.

## License

Same as IRH project: CC0-1.0 Universal (Public Domain)

---

**Built with** ❤️ **for the IRH community**
