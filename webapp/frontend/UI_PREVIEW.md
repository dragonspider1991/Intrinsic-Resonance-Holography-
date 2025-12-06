# IRH Frontend UI Layout Preview

## Application Layout

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  IRH - Intrinsic Resonance Holography v10.0                                 │
│  Zero Free Parameters. Explicit Mathematics.                                 │
└──────────────────────────────────────────────────────────────────────────────┘

┌────────────────┬──────────────────────────────────────────────────────────────┐
│                │                                                              │
│  PARAMETERS    │                   VISUALIZATION CANVAS                       │
│  ═══════════   │  ┌──────────────────────────────────────────────────────┐   │
│                │  │ View: [3D] [2D]    Display: [Network] [Spectrum]     │   │
│ Network Size   │  └──────────────────────────────────────────────────────┘   │
│ ────────●──    │                                                              │
│ N = 64         │          ╭─────────────────────────────╮                     │
│                │         ╱                               ╲                    │
│ Topology       │        │    ● ─── ●       ●             │                   │
│ [Random   ▼]   │        │    │      │╲     │╲            │                   │
│                │        │    ●      ● ●────● ●           │                   │
│ Edge Prob.     │        │   ╱│╲    ╱       ╱│            │                   │
│ ──────●────    │        │  ● ● ●  ●       ● ●            │                   │
│ 0.3            │         ╲   Interactive 3D Network     ╱                    │
│                │          ╰─────────────────────────────╯                     │
│ □ Random Seed  │                  (Rotate, Zoom, Pan)                        │
│                │                                                              │
│ ▶ Optimization │  Color Scale:  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                        │
│   Settings     │                Low ←── Eigenvalue ──→ High                  │
│                │                                                              │
│ Computations:  │──────────────────────────────────────────────────────────────│
│ ☑ Spectral     │                                                              │
│   Dimension    │                      RESULTS PANEL                           │
│ ☑ Physical     │  ┌────────────────────────────────────────────────────────┐ │
│   Predictions  │  │ [Network] [Spectrum] [Predictions] [Grand Audit]       │ │
│ □ Grand Audit  │  ├────────────────────────────────────────────────────────┤ │
│                │  │                                                          │ │
│ ┌────────────┐ │  │  Network Information:                                  │ │
│ │ RUN SIM    │ │  │  ┌──────────────────────────────────────────┐          │ │
│ │    ▶       │ │  │  │  Node Count:           64                │          │ │
│ └────────────┘ │  │  │  Edge Count:           192               │          │ │
│ ┌────────────┐ │  │  │  Topology:             Random            │          │ │
│ │ STOP  ■    │ │  │  │  Min Eigenvalue:       0.000134          │          │ │
│ └────────────┘ │  │  │  Max Eigenvalue:       9.873421          │          │ │
│ ┌────────────┐ │  │  └──────────────────────────────────────────┘          │ │
│ │ RESET  ⟲   │ │  │                                                          │ │
│ └────────────┘ │  │  Spectral Dimension: d_s = 4.0023 ± 0.0012              │ │
│                │  │                                                          │ │
│ Progress:      │  └──────────────────────────────────────────────────────────┘ │
│ ██████████ 75% │                                                              │
│ Computing...   │                                                              │
│                │                                                              │
└────────────────┴──────────────────────────────────────────────────────────────┘
```

## Color Scheme (Dark Theme)

### Background Colors
- **Main Background**: `#1a1a2e` - Dark blue-black
- **Panels/Cards**: `#16213e` - Slightly lighter blue
- **Borders**: Subtle dividers

### Interactive Elements
- **Primary Buttons**: `#3282b8` - Deep blue
  - Hover: `#4a9cd4` - Brighter blue
- **Secondary Buttons**: `#00d9ff` - Cyan
- **Disabled**: Grayed out
- **Progress Bar**: Cyan gradient

### Status Colors
- **Success/Valid**: `#00d9ff` - Cyan
- **Warning**: `#ffa62b` - Orange
- **Error**: `#ef4444` - Red

### Text
- **Primary Text**: `#eeeeee` - Light gray (high contrast)
- **Secondary Text**: `#aaaaaa` - Medium gray
- **Numbers**: Monospace font (JetBrains Mono)

### 3D Visualization
- **Background**: Dark gradient
- **Nodes**: Colored by eigenvalue (viridis colormap)
  - Blue → Cyan → Green → Yellow → Red
- **Edges**: Gray with opacity based on weight
- **Lighting**: Ambient + directional

## Component Details

### Parameter Panel (Left Side)

**Network Configuration:**
- Slider: Network Size (4-4096, logarithmic scale)
  - Shows markers at: 4, 16, 64, 256, 1024, 4096
  - Current value displayed below
- Dropdown: Topology selection
  - Options: Random, Complete, Cycle, Lattice
- Slider: Edge Probability (0.0-1.0)
  - Only visible for Random topology
- Checkbox + Input: Random seed (optional)

**Optimization Settings (Collapsible):**
- Max Iterations: Number input
- Initial Temperature: Decimal input
- Final Temperature: Decimal input

**Computations:**
- ☑ Compute Spectral Dimension (checked by default)
- ☑ Compute Physical Predictions (checked by default)
- ☐ Run Grand Audit (unchecked - expensive)

**Action Buttons:**
- Large primary button: "RUN SIMULATION" with play icon
- Secondary button: "STOP SIMULATION" with stop icon
- Tertiary button: "RESET PARAMETERS" with reset icon

**Progress Indicator:**
- Linear progress bar (0-100%)
- Percentage text
- Status message ("Creating network...", "Computing spectrum...", etc.)

### Visualization Canvas (Center/Top)

**Control Bar:**
- Toggle buttons: [3D] [2D]
- Type selector: [Network] [Spectrum] [Both]

**3D View (Three.js):**
- Interactive canvas
- Orbit controls:
  - Left-click drag: Rotate
  - Right-click drag: Pan
  - Scroll: Zoom
- Nodes: Spheres colored by eigenvalue
- Edges: Lines with opacity
- Legend: Color scale bar

**2D View (Chart.js):**
- Eigenvalue spectrum line chart
  - X-axis: Index
  - Y-axis: Eigenvalue
- Interactive tooltips on hover
- Responsive chart that fills container

### Results Panel (Bottom)

**Tabs:**
1. **Network Tab:**
   - Data table showing:
     - Node count
     - Edge count
     - Topology
     - Min/max eigenvalues

2. **Spectrum Tab:**
   - Eigenvalue statistics
   - Spectral gap
   - Spectral dimension (if computed)
   - Error estimate

3. **Predictions Tab:**
   - Fine structure constant (α⁻¹)
   - Predicted value (monospace)
   - CODATA value: 137.035999084
   - Difference
   - Status chip: ✓ "Within tolerance" or ✗ "Outside tolerance"

4. **Grand Audit Tab:**
   - Summary statistics:
     - Total checks
     - Passed count (green)
     - Failed count (red)
     - Pass rate percentage
   - Scrollable table of detailed results
   - Each row: Check name | Status badge

## Responsive Breakpoints

### Desktop (≥1200px)
- 3-column layout as shown above
- Parameter panel: ~25% width
- Main content: ~75% width
  - Visualization: 60% of height
  - Results: 40% of height

### Tablet (768px-1199px)
- 2-column layout:
  - Top: Parameter panel + Visualization side by side
  - Bottom: Results panel full width
- Panels adjust width proportionally

### Mobile (<768px)
- Single column stack:
  - Parameter panel (collapsible)
  - Visualization canvas
  - Results panel
- Touch-friendly controls
- Scrollable sections

## Interactions

### User Flow Example:

1. **User opens application**
   - See default parameters (N=64, Random topology)
   - See placeholder/empty visualization
   - Results panel shows "No data" message

2. **User adjusts parameters**
   - Move slider to N=128
   - Select "Lattice" topology
   - Enable "Grand Audit"

3. **User clicks "Run Simulation"**
   - Button becomes disabled
   - Progress bar appears at 0%
   - WebSocket connects

4. **Progress updates (real-time)**
   - Bar animates: 10%, 25%, 50%, 75%...
   - Status text updates:
     - "Creating network..."
     - "Computing spectrum..."
     - "Calculating predictions..."
   - Takes 5-30 seconds depending on N

5. **Completion**
   - Progress bar reaches 100%
   - Button re-enables
   - 3D visualization loads and displays network
   - Results panel populates with data
   - User can rotate/explore 3D view

6. **User explores results**
   - Click tabs to see different data
   - Toggle 2D/3D views
   - Switch between Network and Spectrum visualizations
   - Read predicted α⁻¹ value

7. **User runs another simulation**
   - Adjust parameters
   - Click "Run Simulation" again
   - Previous results replaced with new data

## Animation & Polish

- **Smooth transitions**: 0.3s ease-in-out
- **Button hover states**: Color brightens
- **Progress bar**: Animated fill
- **Tab switching**: Fade transition
- **3D camera**: Damped rotation
- **Loading spinners**: Material-UI circular progress
- **Tooltips**: Hover over controls for help text
- **Disabled states**: Visual feedback (grayed out)
- **Error messages**: Red background with alert icon

## Accessibility Features

- Keyboard navigation support
- ARIA labels on all interactive elements
- Focus indicators
- High contrast dark theme
- Screen reader friendly text
- Semantic HTML structure

---

**This is what you'll see when you open http://localhost:5173 !** 🎨
