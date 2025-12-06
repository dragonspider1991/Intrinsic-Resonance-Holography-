/**
 * Application State Store using Zustand
 * Manages global state for IRH Web Application
 */

import { create } from 'zustand';
import type { AppState, NetworkConfig, OptimizationConfig, ComputeFlags } from '../types';

// Default values
const defaultNetworkConfig: NetworkConfig = {
  N: 64,
  topology: 'Random',
  seed: undefined,
  edge_probability: 0.3,
};

const defaultOptimizationConfig: OptimizationConfig = {
  max_iterations: 500,
  T_initial: 1.0,
  T_final: 0.01,
  verbose: false,
};

const defaultComputeFlags: ComputeFlags = {
  spectral_dimension: true,
  predictions: true,
  grand_audit: false,
};

export const useAppStore = create<AppState>((set) => ({
  // Initial state
  networkConfig: defaultNetworkConfig,
  optimizationConfig: defaultOptimizationConfig,
  computeFlags: defaultComputeFlags,

  currentJob: {
    job_id: null,
    status: 'idle',
    progress: 0,
    error: null,
  },

  results: {},

  ui: {
    visualizationMode: '3d',
    visualizationType: 'network',
    activeTab: 0,
    darkMode: true,
  },

  // Actions
  setNetworkConfig: (config) =>
    set((state) => ({
      networkConfig: { ...state.networkConfig, ...config },
    })),

  setOptimizationConfig: (config) =>
    set((state) => ({
      optimizationConfig: { ...state.optimizationConfig, ...config },
    })),

  setComputeFlags: (flags) =>
    set((state) => ({
      computeFlags: { ...state.computeFlags, ...flags },
    })),

  setCurrentJob: (job) =>
    set((state) => ({
      currentJob: { ...state.currentJob, ...job },
    })),

  setResults: (results) =>
    set((state) => ({
      results: { ...state.results, ...results },
    })),

  setUI: (ui) =>
    set((state) => ({
      ui: { ...state.ui, ...ui },
    })),

  resetParameters: () =>
    set({
      networkConfig: defaultNetworkConfig,
      optimizationConfig: defaultOptimizationConfig,
      computeFlags: defaultComputeFlags,
    }),
}));
