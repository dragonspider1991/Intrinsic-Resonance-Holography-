"""
Integration Module for IRH Python Package
==========================================

Provides high-level integration functions that wrap IRH modules
for easy consumption by the web API.

Handles:
- Network creation and initialization
- Running simulations with progress tracking
- Computing various physical predictions
- Data formatting for visualization
"""

import numpy as np
from typing import Dict, Any, Optional, List, Callable
import sys
import os

# Import IRH modules
# NOTE: For production, ensure IRH package is installed via: pip install -e .
# This sys.path modification is a fallback for development.
try:
    from irh.graph_state import HyperGraph
except ImportError:
    # Fallback: add IRH package to path if not installed
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python/src'))
    from irh.graph_state import HyperGraph
from irh.spectral_dimension import SpectralDimension, HeatKernelTrace
from irh.scaling_flows import MetricEmergence, LorentzSignature, GSRGDecimate
from irh.predictions.constants import (
    predict_alpha_inverse,
    predict_neutrino_masses,
    predict_ckm_matrix,
)
from irh.grand_audit import grand_audit
from irh.gtec import gtec
from irh.ncgg import NCGG, frustration

from webapp.backend.visualization import (
    serialize_network_3d,
    serialize_spectrum_3d,
    serialize_chart_2d,
    serialize_histogram_2d,
    serialize_heatmap_2d,
)


class IRHSimulation:
    """
    High-level interface for running IRH simulations.
    
    Encapsulates network creation, computation, and result serialization
    for web API consumption.
    """
    
    def __init__(self, network_config: Dict[str, Any]):
        """
        Initialize simulation with network configuration.
        
        Args:
            network_config: Dictionary with N, topology, seed, etc.
        """
        self.config = network_config
        self.network = None
        self.results = {}
        
    def create_network(self) -> Dict[str, Any]:
        """Create network and return basic properties."""
        self.network = HyperGraph(
            N=self.config.get('N', 64),
            seed=self.config.get('seed'),
            topology=self.config.get('topology', 'Random'),
            edge_probability=self.config.get('edge_probability', 0.3),
        )
        
        return {
            'N': self.network.N,
            'edge_count': self.network.edge_count,
            'topology': self.config.get('topology'),
        }
    
    def compute_spectrum(self) -> Dict[str, Any]:
        """Compute and serialize eigenspectrum."""
        if self.network is None:
            self.create_network()
        
        L = self.network.get_laplacian()
        eigenvalues = np.linalg.eigvalsh(L)
        
        # Store for later use
        self.results['eigenvalues'] = eigenvalues
        self.results['laplacian'] = L
        
        return {
            'eigenvalues': eigenvalues.tolist(),
            'min': float(eigenvalues[1] if len(eigenvalues) > 1 else 0),
            'max': float(eigenvalues[-1]),
            'spectral_gap': float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0,
        }
    
    def compute_spectral_dimension(self) -> Dict[str, Any]:
        """Compute spectral dimension."""
        if self.network is None:
            self.create_network()
        
        ds_result = SpectralDimension(self.network)
        
        return {
            'value': float(ds_result.value) if not np.isnan(ds_result.value) else None,
            'error': float(ds_result.error) if hasattr(ds_result, 'error') else None,
            'target': 4.0,
            'match': abs(ds_result.value - 4.0) < 0.1 if not np.isnan(ds_result.value) else False,
        }
    
    def compute_metric_emergence(self) -> Dict[str, Any]:
        """Compute emergent metric properties."""
        if self.network is None:
            self.create_network()
        
        metric_result = MetricEmergence(self.network)
        lorentz_result = LorentzSignature(self.network)
        
        return {
            'metric_tensor_available': metric_result.metric_tensor is not None,
            'signature': getattr(metric_result, 'signature', None),
            'lorentz_signature': {
                'negative_count': lorentz_result.negative_count,
                'positive_count': lorentz_result.positive_count,
                'expected': '(-,+,+,+)',
            }
        }
    
    def compute_predictions(self) -> Dict[str, Any]:
        """Compute physical constant predictions."""
        if self.network is None:
            self.create_network()
        
        predictions = {}
        
        # Fine structure constant
        try:
            alpha_result = predict_alpha_inverse(self.network)
            predictions['alpha_inverse'] = {
                'value': float(alpha_result.value),
                'codata': 137.035999084,
                'difference': float(alpha_result.value - 137.035999084),
                'relative_error': float(abs(alpha_result.value - 137.035999084) / 137.035999084),
            }
        except Exception as e:
            predictions['alpha_inverse'] = {'error': str(e)}
        
        # Neutrino masses
        try:
            neutrino_result = predict_neutrino_masses(self.network)
            predictions['neutrino_masses'] = {
                'sum': float(neutrino_result.sum_masses),
                'bound': 0.12,  # eV (Planck bound)
                'within_bound': neutrino_result.sum_masses < 0.12,
            }
        except Exception as e:
            predictions['neutrino_masses'] = {'error': str(e)}
        
        # CKM matrix
        try:
            ckm_result = predict_ckm_matrix(self.network)
            predictions['ckm_matrix'] = {
                'matrix': ckm_result.matrix.tolist(),
                'unitarity_check': float(np.linalg.norm(
                    ckm_result.matrix @ ckm_result.matrix.conj().T - np.eye(3)
                )),
            }
        except Exception as e:
            predictions['ckm_matrix'] = {'error': str(e)}
        
        return predictions
    
    def compute_complexity_metrics(self) -> Dict[str, Any]:
        """Compute GTEC and complexity metrics."""
        if self.network is None:
            self.create_network()
        
        gtec_result = gtec(self.network)
        frustration_result = frustration(self.network)
        
        return {
            'gtec': {
                'complexity': float(gtec_result.complexity),
                'shannon_global': float(gtec_result.shannon_global),
            },
            'frustration': {
                'total': float(frustration_result.total_frustration),
            }
        }
    
    def run_grand_audit(self) -> Dict[str, Any]:
        """Run comprehensive grand audit."""
        if self.network is None:
            self.create_network()
        
        audit_result = grand_audit(self.network)
        
        return {
            'total_checks': audit_result.total_checks,
            'pass_count': audit_result.pass_count,
            'fail_count': audit_result.fail_count,
            'pass_rate': audit_result.pass_count / audit_result.total_checks if audit_result.total_checks > 0 else 0.0,
        }
    
    def get_visualization_data_3d(self) -> Dict[str, Any]:
        """Get data formatted for 3D visualization."""
        if self.network is None:
            self.create_network()
        
        adj_matrix = self.network.adjacency_matrix
        eigenvalues = self.results.get('eigenvalues')
        
        if eigenvalues is None:
            L = self.network.get_laplacian()
            eigenvalues = np.linalg.eigvalsh(L)
        
        network_3d = serialize_network_3d(adj_matrix, eigenvalues=eigenvalues)
        spectrum_3d = serialize_spectrum_3d(eigenvalues, mode="scatter")
        
        return {
            'network': network_3d,
            'spectrum': spectrum_3d,
        }
    
    def get_visualization_data_2d(self) -> Dict[str, Any]:
        """Get data formatted for 2D charts."""
        if 'eigenvalues' not in self.results:
            self.compute_spectrum()
        
        eigenvalues = self.results['eigenvalues']
        
        # Eigenvalue spectrum plot
        spectrum_chart = serialize_chart_2d(
            x_data=np.arange(len(eigenvalues)),
            y_data=eigenvalues,
            chart_type="line",
            title="Eigenvalue Spectrum",
            x_label="Index",
            y_label="Eigenvalue λ",
        )
        
        # Eigenvalue histogram
        histogram = serialize_histogram_2d(
            data=eigenvalues,
            bins=50,
            title="Eigenvalue Distribution",
            x_label="Eigenvalue",
            y_label="Frequency",
        )
        
        # Adjacency matrix heatmap
        heatmap = serialize_heatmap_2d(
            matrix=self.network.adjacency_matrix,
            title="Adjacency Matrix",
            colorscale="Viridis",
        )
        
        return {
            'spectrum_chart': spectrum_chart,
            'histogram': histogram,
            'adjacency_heatmap': heatmap,
        }
    
    def get_full_results(self) -> Dict[str, Any]:
        """Get all computed results."""
        return {
            'network': {
                'N': self.network.N if self.network else None,
                'edge_count': self.network.edge_count if self.network else None,
                'topology': self.config.get('topology'),
            },
            'results': self.results,
        }


def run_full_simulation(
    network_config: Dict[str, Any],
    compute_all: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Run a full IRH simulation with all computations.
    
    Args:
        network_config: Network configuration dictionary
        compute_all: Whether to compute all metrics (expensive)
        progress_callback: Optional callback for progress updates (progress%, message)
        
    Returns:
        Complete results dictionary
    """
    sim = IRHSimulation(network_config)
    results = {}
    
    def update_progress(pct: float, msg: str):
        if progress_callback:
            progress_callback(pct, msg)
    
    # Create network
    update_progress(10, "Creating network...")
    results['network'] = sim.create_network()
    
    # Compute spectrum
    update_progress(30, "Computing eigenspectrum...")
    results['spectrum'] = sim.compute_spectrum()
    
    # Spectral dimension
    update_progress(50, "Computing spectral dimension...")
    results['spectral_dimension'] = sim.compute_spectral_dimension()
    
    if compute_all:
        # Metric emergence
        update_progress(60, "Computing metric emergence...")
        results['metric_emergence'] = sim.compute_metric_emergence()
        
        # Physical predictions
        update_progress(70, "Computing physical predictions...")
        results['predictions'] = sim.compute_predictions()
        
        # Complexity metrics
        update_progress(80, "Computing complexity metrics...")
        results['complexity'] = sim.compute_complexity_metrics()
        
        # Grand audit
        update_progress(90, "Running grand audit...")
        results['grand_audit'] = sim.run_grand_audit()
    
    # Visualization data
    update_progress(95, "Preparing visualization data...")
    results['visualization_3d'] = sim.get_visualization_data_3d()
    results['visualization_2d'] = sim.get_visualization_data_2d()
    
    update_progress(100, "Complete!")
    
    return results
