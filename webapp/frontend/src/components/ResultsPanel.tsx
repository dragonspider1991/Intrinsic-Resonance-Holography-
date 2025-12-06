/**
 * Results Panel Component
 * Displays simulation results in tabbed interface
 */

import React from 'react';
import {
  Box,
  Paper,
  Tabs,
  Tab,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableRow,
  Chip,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import { useAppStore } from '../store/appStore';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`results-tabpanel-${index}`}
      aria-labelledby={`results-tab-${index}`}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
};

export const ResultsPanel: React.FC = () => {
  const { ui, setUI, results } = useAppStore();

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setUI({ activeTab: newValue });
  };

  return (
    <Paper elevation={3} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={ui.activeTab} onChange={handleTabChange} aria-label="results tabs">
          <Tab label="Network" />
          <Tab label="Spectrum" />
          <Tab label="Predictions" />
          <Tab label="Grand Audit" />
        </Tabs>
      </Box>

      {/* Tab 1: Network Info */}
      <TabPanel value={ui.activeTab} index={0}>
        {results.network ? (
          <TableContainer>
            <Table size="small">
              <TableBody>
                <TableRow>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      Node Count
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {results.network.N}
                    </Typography>
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      Edge Count
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {results.network.edge_count}
                    </Typography>
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      Topology
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">{results.network.topology}</Typography>
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      Min Eigenvalue
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {results.network.spectrum.min.toFixed(6)}
                    </Typography>
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      Max Eigenvalue
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {results.network.spectrum.max.toFixed(6)}
                    </Typography>
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Typography variant="body2" color="text.secondary">
            No network data available. Run a simulation to see results.
          </Typography>
        )}
      </TabPanel>

      {/* Tab 2: Spectrum */}
      <TabPanel value={ui.activeTab} index={1}>
        {results.spectrum ? (
          <>
            <TableContainer>
              <Table size="small">
                <TableBody>
                  <TableRow>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        Min Eigenvalue
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                        {results.spectrum.min_eigenvalue.toFixed(6)}
                      </Typography>
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        Max Eigenvalue
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                        {results.spectrum.max_eigenvalue.toFixed(6)}
                      </Typography>
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        Spectral Gap
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                        {results.spectrum.spectral_gap.toFixed(6)}
                      </Typography>
                    </TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
            {results.spectral_dimension && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                  Spectral Dimension
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableBody>
                      <TableRow>
                        <TableCell>
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            d_s
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                            {results.spectral_dimension.spectral_dimension?.toFixed(4) || 'N/A'}
                          </Typography>
                        </TableCell>
                      </TableRow>
                      {results.spectral_dimension.error !== null && (
                        <TableRow>
                          <TableCell>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              Error
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                              {results.spectral_dimension.error.toFixed(6)}
                            </Typography>
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
            )}
          </>
        ) : (
          <Typography variant="body2" color="text.secondary">
            No spectrum data available. Run a simulation to see results.
          </Typography>
        )}
      </TabPanel>

      {/* Tab 3: Predictions */}
      <TabPanel value={ui.activeTab} index={2}>
        {results.predictions?.alpha ? (
          <TableContainer>
            <Table size="small">
              <TableBody>
                <TableRow>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      Fine Structure Constant (α⁻¹)
                    </Typography>
                  </TableCell>
                  <TableCell></TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <Typography variant="body2" sx={{ pl: 2 }}>
                      Predicted Value
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {results.predictions.alpha.alpha_inverse.toFixed(6)}
                    </Typography>
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <Typography variant="body2" sx={{ pl: 2 }}>
                      CODATA Value
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {results.predictions.alpha.codata_value.toFixed(6)}
                    </Typography>
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <Typography variant="body2" sx={{ pl: 2 }}>
                      Difference
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                        {results.predictions.alpha.difference.toFixed(6)}
                      </Typography>
                      {Math.abs(results.predictions.alpha.difference) < 0.1 ? (
                        <Chip
                          icon={<CheckCircleIcon />}
                          label="Within tolerance"
                          color="success"
                          size="small"
                        />
                      ) : (
                        <Chip
                          icon={<ErrorIcon />}
                          label="Outside tolerance"
                          color="error"
                          size="small"
                        />
                      )}
                    </Box>
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Typography variant="body2" color="text.secondary">
            No prediction data available. Enable "Compute Physical Predictions" and run a
            simulation.
          </Typography>
        )}
      </TabPanel>

      {/* Tab 4: Grand Audit */}
      <TabPanel value={ui.activeTab} index={3}>
        {results.grand_audit ? (
          <>
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                Grand Audit Results
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableBody>
                    <TableRow>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          Total Checks
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          {results.grand_audit.total_checks}
                        </Typography>
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          Passed
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace', color: 'success.main' }}>
                          {results.grand_audit.passed}
                        </Typography>
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          Failed
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace', color: 'error.main' }}>
                          {results.grand_audit.failed}
                        </Typography>
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          Pass Rate
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          {(results.grand_audit.pass_rate * 100).toFixed(1)}%
                        </Typography>
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
            {results.grand_audit.results && results.grand_audit.results.length > 0 && (
              <Box>
                <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                  Detailed Results
                </Typography>
                <TableContainer sx={{ maxHeight: 300 }}>
                  <Table size="small" stickyHeader>
                    <TableBody>
                      {results.grand_audit.results.map((result, idx) => (
                        <TableRow key={idx}>
                          <TableCell>{result.check}</TableCell>
                          <TableCell align="right">
                            {result.passed ? (
                              <Chip label="Passed" color="success" size="small" />
                            ) : (
                              <Chip label="Failed" color="error" size="small" />
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
            )}
          </>
        ) : (
          <Typography variant="body2" color="text.secondary">
            No grand audit data available. Enable "Run Grand Audit" and run a simulation.
          </Typography>
        )}
      </TabPanel>
    </Paper>
  );
};

export default ResultsPanel;
