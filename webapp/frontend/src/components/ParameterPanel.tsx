/**
 * Parameter Control Panel Component
 * Allows users to configure network and simulation parameters
 */

import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Slider,
  Button,
  Checkbox,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  LinearProgress,
  Divider,
  Stack,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import { useAppStore } from '../store/appStore';
import { useSimulation } from '../hooks/useSimulation';

export const ParameterPanel: React.FC = () => {
  const {
    networkConfig,
    optimizationConfig,
    computeFlags,
    currentJob,
    setNetworkConfig,
    setOptimizationConfig,
    setComputeFlags,
    resetParameters,
  } = useAppStore();

  const { runSimulation, stopSimulation } = useSimulation();
  const [useSeed, setUseSeed] = useState(false);

  const handleRunSimulation = async () => {
    try {
      await runSimulation();
    } catch (error) {
      console.error('Failed to run simulation:', error);
    }
  };

  const handleStopSimulation = () => {
    stopSimulation();
  };

  const handleReset = () => {
    resetParameters();
    setUseSeed(false);
  };

  const isRunning = currentJob.status === 'running' || currentJob.status === 'pending';

  return (
    <Paper elevation={3} sx={{ p: 3, height: '100%', overflow: 'auto' }}>
      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, color: 'primary.main' }}>
        Parameter Control Panel
      </Typography>

      <Divider sx={{ my: 2 }} />

      {/* Network Size */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="body2" gutterBottom sx={{ color: 'text.secondary' }}>
          Network Size (N)
        </Typography>
        <Slider
          value={networkConfig.N}
          onChange={(_, value) => setNetworkConfig({ N: value as number })}
          min={4}
          max={4096}
          scale={(x) => 2 ** Math.round(Math.log2(x))}
          step={null}
          marks={[
            { value: 4, label: '4' },
            { value: 16, label: '16' },
            { value: 64, label: '64' },
            { value: 256, label: '256' },
            { value: 1024, label: '1024' },
            { value: 4096, label: '4096' },
          ]}
          valueLabelDisplay="auto"
          disabled={isRunning}
        />
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          Current: {networkConfig.N} nodes
        </Typography>
      </Box>

      {/* Topology */}
      <FormControl fullWidth sx={{ mb: 3 }}>
        <InputLabel>Topology</InputLabel>
        <Select
          value={networkConfig.topology}
          onChange={(e) => setNetworkConfig({ topology: e.target.value as any })}
          label="Topology"
          disabled={isRunning}
        >
          <MenuItem value="Random">Random</MenuItem>
          <MenuItem value="Complete">Complete</MenuItem>
          <MenuItem value="Cycle">Cycle</MenuItem>
          <MenuItem value="Lattice">Lattice</MenuItem>
        </Select>
      </FormControl>

      {/* Edge Probability (only for Random topology) */}
      {networkConfig.topology === 'Random' && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="body2" gutterBottom sx={{ color: 'text.secondary' }}>
            Edge Probability
          </Typography>
          <Slider
            value={networkConfig.edge_probability}
            onChange={(_, value) => setNetworkConfig({ edge_probability: value as number })}
            min={0.0}
            max={1.0}
            step={0.01}
            marks={[
              { value: 0.0, label: '0.0' },
              { value: 0.5, label: '0.5' },
              { value: 1.0, label: '1.0' },
            ]}
            valueLabelDisplay="auto"
            disabled={isRunning}
          />
        </Box>
      )}

      {/* Random Seed */}
      <Box sx={{ mb: 3 }}>
        <FormControlLabel
          control={
            <Checkbox
              checked={useSeed}
              onChange={(e) => {
                setUseSeed(e.target.checked);
                if (!e.target.checked) {
                  setNetworkConfig({ seed: undefined });
                }
              }}
              disabled={isRunning}
            />
          }
          label="Use Random Seed"
        />
        {useSeed && (
          <TextField
            fullWidth
            type="number"
            label="Seed"
            value={networkConfig.seed || ''}
            onChange={(e) => setNetworkConfig({ seed: parseInt(e.target.value) || undefined })}
            disabled={isRunning}
            size="small"
            sx={{ mt: 1 }}
          />
        )}
      </Box>

      {/* Optimization Settings (Collapsible) */}
      <Accordion sx={{ mb: 3 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            Optimization Settings
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Stack spacing={2}>
            <TextField
              fullWidth
              type="number"
              label="Max Iterations"
              value={optimizationConfig.max_iterations}
              onChange={(e) => setOptimizationConfig({ max_iterations: parseInt(e.target.value) })}
              disabled={isRunning}
              size="small"
            />
            <TextField
              fullWidth
              type="number"
              label="Initial Temperature"
              value={optimizationConfig.T_initial}
              onChange={(e) => setOptimizationConfig({ T_initial: parseFloat(e.target.value) })}
              disabled={isRunning}
              size="small"
              inputProps={{ step: 0.1 }}
            />
            <TextField
              fullWidth
              type="number"
              label="Final Temperature"
              value={optimizationConfig.T_final}
              onChange={(e) => setOptimizationConfig({ T_final: parseFloat(e.target.value) })}
              disabled={isRunning}
              size="small"
              inputProps={{ step: 0.001 }}
            />
          </Stack>
        </AccordionDetails>
      </Accordion>

      {/* Computation Checkboxes */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="body2" gutterBottom sx={{ fontWeight: 600 }}>
          Computations
        </Typography>
        <FormControlLabel
          control={
            <Checkbox
              checked={computeFlags.spectral_dimension}
              onChange={(e) => setComputeFlags({ spectral_dimension: e.target.checked })}
              disabled={isRunning}
            />
          }
          label="Compute Spectral Dimension"
        />
        <FormControlLabel
          control={
            <Checkbox
              checked={computeFlags.predictions}
              onChange={(e) => setComputeFlags({ predictions: e.target.checked })}
              disabled={isRunning}
            />
          }
          label="Compute Physical Predictions"
        />
        <FormControlLabel
          control={
            <Checkbox
              checked={computeFlags.grand_audit}
              onChange={(e) => setComputeFlags({ grand_audit: e.target.checked })}
              disabled={isRunning}
            />
          }
          label="Run Grand Audit (expensive)"
        />
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* Action Buttons */}
      <Stack spacing={2}>
        <Button
          variant="contained"
          color="primary"
          size="large"
          startIcon={<PlayArrowIcon />}
          onClick={handleRunSimulation}
          disabled={isRunning}
          fullWidth
        >
          Run Simulation
        </Button>
        <Button
          variant="outlined"
          color="error"
          startIcon={<StopIcon />}
          onClick={handleStopSimulation}
          disabled={!isRunning}
          fullWidth
        >
          Stop Simulation
        </Button>
        <Button
          variant="outlined"
          startIcon={<RestartAltIcon />}
          onClick={handleReset}
          disabled={isRunning}
          fullWidth
        >
          Reset Parameters
        </Button>
      </Stack>

      {/* Progress Indicator */}
      {isRunning && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="body2" gutterBottom sx={{ color: 'text.secondary' }}>
            {currentJob.status === 'pending' ? 'Starting simulation...' : 'Running simulation...'}
          </Typography>
          <LinearProgress
            variant="determinate"
            value={currentJob.progress}
            sx={{ mb: 1, height: 8, borderRadius: 4 }}
          />
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            {currentJob.progress.toFixed(0)}%
          </Typography>
        </Box>
      )}

      {/* Error Display */}
      {currentJob.error && (
        <Box sx={{ mt: 2, p: 2, bgcolor: 'error.dark', borderRadius: 1 }}>
          <Typography variant="body2" color="error.contrastText">
            Error: {currentJob.error}
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default ParameterPanel;
