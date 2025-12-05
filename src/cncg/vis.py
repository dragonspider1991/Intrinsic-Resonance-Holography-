"""
Visualization Tools

Plotting functions for spectral triples and optimization results.
"""

from typing import Optional, List, Dict
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import networkx as nx
from .spectral import FiniteSpectralTriple
from .analysis import spectral_density


def plot_eigenvalue_flow(
    history: List[NDArray[np.float64]],
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot the evolution of eigenvalues during optimization.
    
    Parameters
    ----------
    history : List[NDArray]
        List of eigenvalue arrays at different iterations
    figsize : tuple, default=(12, 6)
        Figure size
    save_path : Optional[str], default=None
        Path to save figure
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Each eigenvalue gets its own line
    n_eigenvalues = len(history[0])
    iterations = np.arange(len(history))
    
    # Transpose to get one array per eigenvalue
    eigenvalue_trajectories = np.array(history).T
    
    # Plot each eigenvalue's trajectory
    for i, traj in enumerate(eigenvalue_trajectories):
        ax.plot(iterations, traj, alpha=0.3, linewidth=0.5, color='blue')
    
    # Highlight zero modes
    for traj in eigenvalue_trajectories:
        if np.abs(traj[-1]) < 1e-4:
            ax.plot(iterations, traj, alpha=0.8, linewidth=1.5, color='red',
                   label='Zero mode' if i == 0 else '')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title('Eigenvalue Flow During Optimization', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Remove duplicate labels
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_spectral_density(
    triple: FiniteSpectralTriple,
    bins: int = 100,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot the density of states.
    
    Parameters
    ----------
    triple : FiniteSpectralTriple
        The spectral triple
    bins : int, default=100
        Number of histogram bins
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : Optional[str], default=None
        Path to save figure
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    spectrum = triple.spectrum()
    
    ax.hist(spectrum, bins=bins, density=True, alpha=0.7, color='steelblue',
            edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Eigenvalue', fontsize=12)
    ax.set_ylabel('Density of States', fontsize=12)
    ax.set_title(f'Spectral Density (N={triple.N})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Zero')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_network_topology(
    triple: FiniteSpectralTriple,
    threshold: float = 0.1,
    max_nodes: int = 100,
    figsize: tuple = (10, 10),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot the network topology of the Dirac operator.
    
    Each matrix element D_ij with |D_ij| > threshold is an edge.
    
    Parameters
    ----------
    triple : FiniteSpectralTriple
        The spectral triple
    threshold : float, default=0.1
        Minimum weight for edge inclusion
    max_nodes : int, default=100
        Maximum number of nodes to display
    figsize : tuple, default=(10, 10)
        Figure size
    save_path : Optional[str], default=None
        Path to save figure
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    N = min(triple.N, max_nodes)
    D = triple.D[:N, :N]
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(N))
    
    # Add edges for significant matrix elements
    for i in range(N):
        for j in range(i+1, N):
            weight = np.abs(D[i, j])
            if weight > threshold:
                G.add_edge(i, j, weight=weight)
    
    # Layout
    pos = nx.spring_layout(G, k=1/np.sqrt(N), iterations=50, seed=42)
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightblue',
                          alpha=0.8, ax=ax)
    
    # Edge colors based on weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5,
                          edge_color=weights, edge_cmap=plt.cm.viridis,
                          ax=ax)
    
    ax.set_title(f'Network Topology (threshold={threshold:.2f})', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_optimization_history(
    history: Dict[str, List],
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot optimization metrics over time.
    
    Parameters
    ----------
    history : Dict[str, List]
        Dictionary with "iteration", "action", "grad_norm", "learning_rate"
    figsize : tuple, default=(12, 8)
        Figure size
    save_path : Optional[str], default=None
        Path to save figure
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    iterations = history["iteration"]
    
    # Action
    axes[0].plot(iterations, history["action"], 'b-', linewidth=2)
    axes[0].set_ylabel('Spectral Action S', fontsize=12)
    axes[0].set_title('Optimization Progress', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Gradient norm
    axes[1].semilogy(iterations, history["grad_norm"], 'r-', linewidth=2)
    axes[1].set_ylabel('Gradient Norm |∇S|', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[2].semilogy(iterations, history["learning_rate"], 'g-', linewidth=2)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_xlabel('Iteration', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_robustness_analysis(
    N_values: List[int],
    alpha_inv_means: List[float],
    alpha_inv_stds: List[float],
    target_alpha_inv: float = 137.036,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot α^(-1) vs system size N to demonstrate robustness.
    
    This reproduces Figure 1 from the manuscript.
    
    Parameters
    ----------
    N_values : List[int]
        System sizes
    alpha_inv_means : List[float]
        Mean α^(-1) values
    alpha_inv_stds : List[float]
        Standard deviations
    target_alpha_inv : float, default=137.036
        Target value (experimental)
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : Optional[str], default=None
        Path to save figure
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot with error bars
    ax.errorbar(N_values, alpha_inv_means, yerr=alpha_inv_stds,
               fmt='o-', capsize=5, capthick=2, linewidth=2,
               markersize=8, label='Computed α⁻¹')
    
    # Target line
    ax.axhline(y=target_alpha_inv, color='red', linestyle='--',
              linewidth=2, label=f'Experimental α⁻¹ = {target_alpha_inv}')
    
    # Shaded error band
    ax.fill_between(N_values,
                    np.array(alpha_inv_means) - np.array(alpha_inv_stds),
                    np.array(alpha_inv_means) + np.array(alpha_inv_stds),
                    alpha=0.3)
    
    ax.set_xlabel('System Size N', fontsize=12)
    ax.set_ylabel('α⁻¹', fontsize=12)
    ax.set_title('Robustness of Fine-Structure Constant Emergence', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_heat_kernel_scaling(
    t_values: NDArray[np.float64],
    K_values: NDArray[np.float64],
    d_s: float,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot heat kernel trace vs time with power-law fit.
    
    Parameters
    ----------
    t_values : NDArray
        Time values
    K_values : NDArray
        Heat kernel trace values
    d_s : float
        Fitted spectral dimension
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : Optional[str], default=None
        Path to save figure
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Log-log plot
    ax.loglog(t_values, K_values, 'o', markersize=6, alpha=0.7,
             label='Computed K(t)')
    
    # Fitted power law: K(t) ~ t^(-d_s/2)
    t_fit = np.logspace(np.log10(t_values.min()), np.log10(t_values.max()), 100)
    # Find normalization from data
    C = np.exp(np.mean(np.log(K_values) + (d_s/2) * np.log(t_values)))
    K_fit = C * t_fit ** (-d_s / 2)
    
    ax.loglog(t_fit, K_fit, 'r--', linewidth=2,
             label=f'Fit: K(t) ∝ t^(-{d_s/2:.2f}), d_s = {d_s:.2f}')
    
    ax.set_xlabel('Heat Kernel Time t', fontsize=12)
    ax.set_ylabel('Tr(e^(-tD²))', fontsize=12)
    ax.set_title('Spectral Dimension from Heat Kernel Scaling', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
