/**
 * API Client for IRH Backend
 * Handles all HTTP requests to the FastAPI backend
 */

import axios from 'axios';
import type { AxiosInstance } from 'axios';
import type {
  NetworkConfig,
  NetworkResponse,
  SpectrumResponse,
  SpectralDimensionResponse,
  AlphaPredictionResponse,
  SimulationRequest,
  JobResponse,
  Network3DVisualization,
  Spectrum3DVisualization,
  ChartData,
} from '../types';

// API Base URL - can be configured via environment variable
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class APIClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000, // 30 seconds
    });
  }

  // Network endpoints
  async createNetwork(config: NetworkConfig): Promise<NetworkResponse> {
    const response = await this.client.post('/api/network/create', config);
    return response.data;
  }

  async getSpectrum(config: NetworkConfig): Promise<SpectrumResponse> {
    const response = await this.client.post('/api/network/spectrum', config);
    return response.data;
  }

  async getSpectralDimension(config: NetworkConfig): Promise<SpectralDimensionResponse> {
    const response = await this.client.post('/api/network/spectral-dimension', config);
    return response.data;
  }

  // Prediction endpoints
  async predictAlpha(config: NetworkConfig): Promise<AlphaPredictionResponse> {
    const response = await this.client.post('/api/predictions/alpha', config);
    return response.data;
  }

  // Simulation endpoints
  async runSimulation(request: SimulationRequest): Promise<{ job_id: string; status: string }> {
    const response = await this.client.post('/api/simulation/run', request);
    return response.data;
  }

  async getJobStatus(jobId: string): Promise<JobResponse> {
    const response = await this.client.get(`/api/jobs/${jobId}`);
    return response.data;
  }

  // Visualization endpoints
  async getNetwork3D(config: NetworkConfig): Promise<Network3DVisualization> {
    const response = await this.client.post('/api/visualization/network-3d', config);
    return response.data;
  }

  async getSpectrum3D(config: NetworkConfig): Promise<Spectrum3DVisualization> {
    const response = await this.client.post('/api/visualization/spectrum-3d', config);
    return response.data;
  }

  async getSpectrumChart(config: NetworkConfig): Promise<ChartData> {
    const response = await this.client.post('/api/visualization/spectrum-chart', config);
    return response.data;
  }
}

export const apiClient = new APIClient();
export default apiClient;
