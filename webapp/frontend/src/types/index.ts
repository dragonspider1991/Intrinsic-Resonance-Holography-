/**
 * TypeScript Type Definitions for IRH Web Application
 * Intrinsic Resonance Holography v10.0
 */

// Network Configuration
export interface NetworkConfig {
  N: number;
  topology: 'Random' | 'Complete' | 'Cycle' | 'Lattice';
  seed?: number;
  edge_probability: number;
}

// Optimization Configuration
export interface OptimizationConfig {
  max_iterations: number;
  T_initial: number;
  T_final: number;
  verbose: boolean;
}

// Computation Flags
export interface ComputeFlags {
  spectral_dimension: boolean;
  predictions: boolean;
  grand_audit: boolean;
}

// Network Response
export interface NetworkResponse {
  N: number;
  edge_count: number;
  topology: string;
  spectrum: {
    eigenvalues: number[];
    min: number;
    max: number;
  };
  adjacency_matrix: number[][];
}

// Spectrum Response
export interface SpectrumResponse {
  eigenvalues: number[];
  spectral_gap: number;
  min_eigenvalue: number;
  max_eigenvalue: number;
}

// Spectral Dimension Response
export interface SpectralDimensionResponse {
  spectral_dimension: number | null;
  error: number | null;
}

// Physical Constant Prediction
export interface AlphaPredictionResponse {
  alpha_inverse: number;
  codata_value: number;
  difference: number;
}

// Simulation Request
export interface SimulationRequest {
  network_config: NetworkConfig;
  optimization_config?: OptimizationConfig;
  compute_spectral_dimension: boolean;
  compute_predictions: boolean;
  run_grand_audit: boolean;
}

// Job Status
export type JobStatus = 'idle' | 'pending' | 'running' | 'completed' | 'failed';

// Job Response
export interface JobResponse {
  job_id: string;
  status: JobStatus;
  progress: number;
  created_at: string;
  completed_at?: string;
  error?: string;
  result?: any;
}

// 3D Visualization - Node
export interface Node3D {
  id: number;
  position: [number, number, number];
  color: string;
  size: number;
}

// 3D Visualization - Edge
export interface Edge3D {
  source: number;
  target: number;
  weight: number;
  color: string;
  opacity: number;
}

// Network 3D Visualization
export interface Network3DVisualization {
  nodes: Node3D[];
  edges: Edge3D[];
  metadata: {
    node_count: number;
    edge_count: number;
    avg_edge_weight: number;
    max_edge_weight: number;
  };
}

// Spectrum 3D Visualization
export interface Spectrum3DVisualization {
  type: 'scatter3d';
  points: Array<{
    x: number;
    y: number;
    z: number;
    color: string;
    size: number;
  }>;
  metadata: {
    min_eigenvalue: number;
    max_eigenvalue: number;
    spectral_gap: number;
  };
}

// Chart.js compatible format
export interface ChartData {
  type: string;
  data: {
    labels?: any[];
    datasets: Array<{
      label: string;
      data: any[];
      borderColor?: string;
      backgroundColor?: string;
      borderWidth?: number;
      pointRadius?: number;
    }>;
  };
  options: any;
}

// Grand Audit Result
export interface GrandAuditResult {
  total_checks: number;
  passed: number;
  failed: number;
  pass_rate: number;
  results: Array<{
    check: string;
    passed: boolean;
    value?: any;
    expected?: any;
    error?: string;
  }>;
}

// Simulation Results
export interface SimulationResults {
  network?: NetworkResponse;
  spectrum?: SpectrumResponse;
  spectral_dimension?: SpectralDimensionResponse;
  predictions?: {
    alpha?: AlphaPredictionResponse;
  };
  grand_audit?: GrandAuditResult;
  visualization_3d?: Network3DVisualization;
  visualization_2d?: ChartData;
}

// Application State
export interface AppState {
  // Parameters
  networkConfig: NetworkConfig;
  optimizationConfig: OptimizationConfig;
  computeFlags: ComputeFlags;

  // Simulation status
  currentJob: {
    job_id: string | null;
    status: JobStatus;
    progress: number;
    error: string | null;
  };

  // Results
  results: SimulationResults;

  // UI state
  ui: {
    visualizationMode: '3d' | '2d';
    visualizationType: 'network' | 'spectrum' | 'both';
    activeTab: number;
    darkMode: boolean;
  };

  // Actions
  setNetworkConfig: (config: Partial<NetworkConfig>) => void;
  setOptimizationConfig: (config: Partial<OptimizationConfig>) => void;
  setComputeFlags: (flags: Partial<ComputeFlags>) => void;
  setCurrentJob: (job: Partial<AppState['currentJob']>) => void;
  setResults: (results: Partial<SimulationResults>) => void;
  setUI: (ui: Partial<AppState['ui']>) => void;
  resetParameters: () => void;
}
