/**
 * 2D Visualization Component
 * Renders charts using Chart.js
 */

import React, { useEffect, useState } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import { useAppStore } from '../store/appStore';
import { apiClient } from '../services/api';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

export const Visualization2D: React.FC = () => {
  const { networkConfig } = useAppStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chartData, setChartData] = useState<any>(null);

  useEffect(() => {
    const loadChart = async () => {
      setLoading(true);
      setError(null);

      try {
        const data = await apiClient.getSpectrumChart(networkConfig);
        setChartData(data);
        setLoading(false);
      } catch (err: any) {
        console.error('Error loading chart:', err);
        setError(err.message || 'Failed to load chart');
        setLoading(false);
      }
    };

    loadChart();
  }, [networkConfig]);

  if (loading) {
    return (
      <Box
        sx={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 2,
        }}
      >
        <CircularProgress />
        <Typography variant="body2" color="text.secondary">
          Loading chart...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box
        sx={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Typography variant="body1" color="error">
          {error}
        </Typography>
      </Box>
    );
  }

  if (!chartData) {
    return (
      <Box
        sx={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Typography variant="body2" color="text.secondary">
          No data available
        </Typography>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        width: '100%',
        height: '100%',
        p: 2,
        bgcolor: 'background.paper',
        borderRadius: 2,
      }}
    >
      {chartData.type === 'line' ? (
        <Line data={chartData.data} options={chartData.options} />
      ) : chartData.type === 'bar' ? (
        <Bar data={chartData.data} options={chartData.options} />
      ) : null}
    </Box>
  );
};

export default Visualization2D;
