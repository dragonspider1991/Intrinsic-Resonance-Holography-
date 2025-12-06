/**
 * Visualization Canvas Component
 * Container for 2D and 3D visualizations with mode toggles
 */

import React from 'react';
import {
  Box,
  Paper,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
  Stack,
} from '@mui/material';
import ViewInArIcon from '@mui/icons-material/ViewInAr';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import { useAppStore } from '../store/appStore';
import Visualization3D from './Visualization3D';
import Visualization2D from './Visualization2D';

export const VisualizationCanvas: React.FC = () => {
  const { ui, setUI } = useAppStore();

  const handleModeChange = (_: React.MouseEvent<HTMLElement>, newMode: '3d' | '2d' | null) => {
    if (newMode !== null) {
      setUI({ visualizationMode: newMode });
    }
  };

  const handleTypeChange = (
    _: React.MouseEvent<HTMLElement>,
    newType: 'network' | 'spectrum' | 'both' | null
  ) => {
    if (newType !== null) {
      setUI({ visualizationType: newType });
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Controls */}
      <Stack direction="row" spacing={2} sx={{ mb: 2 }} alignItems="center">
        <Typography variant="body2" sx={{ fontWeight: 600, color: 'text.secondary' }}>
          View Mode:
        </Typography>
        <ToggleButtonGroup
          value={ui.visualizationMode}
          exclusive
          onChange={handleModeChange}
          size="small"
        >
          <ToggleButton value="3d">
            <ViewInArIcon sx={{ mr: 0.5 }} />
            3D
          </ToggleButton>
          <ToggleButton value="2d">
            <ShowChartIcon sx={{ mr: 0.5 }} />
            2D
          </ToggleButton>
        </ToggleButtonGroup>

        <Typography variant="body2" sx={{ fontWeight: 600, color: 'text.secondary', ml: 3 }}>
          Display:
        </Typography>
        <ToggleButtonGroup
          value={ui.visualizationType}
          exclusive
          onChange={handleTypeChange}
          size="small"
        >
          <ToggleButton value="network">Network</ToggleButton>
          <ToggleButton value="spectrum">Spectrum</ToggleButton>
          <ToggleButton value="both">Both</ToggleButton>
        </ToggleButtonGroup>
      </Stack>

      {/* Visualization Area */}
      <Box sx={{ flexGrow: 1, minHeight: 0 }}>
        {ui.visualizationMode === '3d' ? <Visualization3D /> : <Visualization2D />}
      </Box>
    </Paper>
  );
};

export default VisualizationCanvas;
