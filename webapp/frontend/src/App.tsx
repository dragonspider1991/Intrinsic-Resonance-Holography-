/**
 * Main Application Component
 * IRH - Intrinsic Resonance Holography v10.0
 */

import { ThemeProvider, CssBaseline, Box } from '@mui/material';
import { theme } from './utils/theme';
import Layout from './components/Layout';
import ParameterPanel from './components/ParameterPanel';
import VisualizationCanvas from './components/VisualizationCanvas';
import ResultsPanel from './components/ResultsPanel';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Layout>
        <Box sx={{ display: 'flex', gap: 3, height: 'calc(100vh - 120px)' }}>
          {/* Parameter Control Panel - Left Column */}
          <Box sx={{ width: { xs: '100%', md: '25%' }, minWidth: '300px', maxWidth: { xs: '100%', md: '400px' } }}>
            <ParameterPanel />
          </Box>

          {/* Main Content - Right Columns */}
          <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 3, minWidth: 0 }}>
            {/* Visualization Canvas - Top */}
            <Box sx={{ height: { xs: '500px', md: '60%' }, minHeight: '400px' }}>
              <VisualizationCanvas />
            </Box>

            {/* Results Panel - Bottom */}
            <Box sx={{ height: { xs: 'auto', md: '40%' }, minHeight: '300px' }}>
              <ResultsPanel />
            </Box>
          </Box>
        </Box>
      </Layout>
    </ThemeProvider>
  );
}

export default App;
