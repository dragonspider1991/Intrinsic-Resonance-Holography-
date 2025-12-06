/**
 * Custom Hook for Simulation Management
 * Manages simulation lifecycle, WebSocket connections, and state updates
 */

import { useCallback } from 'react';
import { useAppStore } from '../store/appStore';
import { apiClient } from '../services/api';
import { websocketClient } from '../services/websocket';
import type { SimulationRequest } from '../types';

export const useSimulation = () => {
  const {
    networkConfig,
    optimizationConfig,
    computeFlags,
    setCurrentJob,
    setResults,
  } = useAppStore();

  const runSimulation = useCallback(async () => {
    try {
      // Reset previous results
      setResults({});
      setCurrentJob({
        job_id: null,
        status: 'pending',
        progress: 0,
        error: null,
      });

      // Prepare simulation request
      const request: SimulationRequest = {
        network_config: networkConfig,
        optimization_config: optimizationConfig,
        compute_spectral_dimension: computeFlags.spectral_dimension,
        compute_predictions: computeFlags.predictions,
        run_grand_audit: computeFlags.grand_audit,
      };

      // Start simulation
      const response = await apiClient.runSimulation(request);
      
      setCurrentJob({
        job_id: response.job_id,
        status: 'running',
      });

      // Connect WebSocket for real-time updates
      websocketClient.connect(
        response.job_id,
        (jobUpdate) => {
          setCurrentJob({
            status: jobUpdate.status,
            progress: jobUpdate.progress,
            error: jobUpdate.error,
          });

          // If completed, fetch and store results
          if (jobUpdate.status === 'completed' && jobUpdate.result) {
            setResults(jobUpdate.result);
          }
        },
        (error) => {
          console.error('WebSocket error:', error);
          setCurrentJob({
            status: 'failed',
            error: 'WebSocket connection failed',
          });
        }
      );

      return response.job_id;
    } catch (error: any) {
      console.error('Error running simulation:', error);
      setCurrentJob({
        status: 'failed',
        error: error.response?.data?.detail || error.message || 'Unknown error',
      });
      throw error;
    }
  }, [networkConfig, optimizationConfig, computeFlags, setCurrentJob, setResults]);

  const stopSimulation = useCallback(() => {
    websocketClient.disconnect();
    setCurrentJob({
      status: 'idle',
      progress: 0,
      error: null,
    });
  }, [setCurrentJob]);

  return {
    runSimulation,
    stopSimulation,
  };
};
