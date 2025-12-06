"""
Visualization Tools for IRH v10.0

Plotting utilities for networks, spectra, and physical predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import networkx as nx


def plot_eigenvalue_spectrum(
    eigenvalues: np.ndarray,
    title: str = "Eigenvalue Spectrum",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot eigenvalue spectrum with histogram.
    
    Args:
        eigenvalues: Eigenvalues to plot
        title: Plot title
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter non-zero
    lambdas_nz = eigenvalues[eigenvalues > 1e-10]
    
    # Histogram
    ax.hist(lambdas_nz, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Eigenvalue λ', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_heat_kernel(
    t_values: np.ndarray,
    K_values: np.ndarray,
    d_s: float,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot heat kernel trace K(t) vs time.
    
    Args:
        t_values: Time values
        K_values: Heat kernel values
        d_s: Spectral dimension (for annotation)
        ax: Matplotlib axes
    
    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(t_values, K_values, 'o-', linewidth=2, markersize=6, label='K(t)')
    
    # Reference line
    t_ref = np.logspace(np.log10(t_values.min()), np.log10(t_values.max()), 50)
    K_ref = K_values[0] * (t_ref / t_values[0])**(-d_s/2)
    ax.loglog(t_ref, K_ref, '--', color='red', linewidth=2, 
              label=f'd_s = {d_s:.2f}', alpha=0.7)
    
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('Heat Kernel K(t)', fontsize=12)
    ax.set_title('Heat Kernel Trace', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_dark_energy_evolution(
    a_range: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot dark energy equation of state w(a).
    
    Args:
        a_range: Scale factor values (default: 0.1 to 3.0)
        ax: Matplotlib axes
    
    Returns:
        ax: Matplotlib axes
    """
    from ..cosmology import w_dark_energy
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if a_range is None:
        a_range = np.linspace(0.1, 3.0, 200)
    
    w_vals = w_dark_energy(a_range)
    
    ax.plot(a_range, w_vals, linewidth=3, color='darkblue', label='IRH v10.0')
    ax.axhline(-1, linestyle='--', color='red', linewidth=2, label='ΛCDM (w=-1)', alpha=0.7)
    ax.axvline(1, linestyle=':', color='gray', linewidth=1.5, alpha=0.5)
    
    ax.set_xlabel('Scale Factor a', fontsize=12)
    ax.set_ylabel('Dark Energy EoS w(a)', fontsize=12)
    ax.set_title('Dark Energy Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_network_graph(
    adjacency: np.ndarray,
    layout: str = "spring",
    node_size: int = 50,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Visualize network structure.
    
    Args:
        adjacency: Adjacency or coupling matrix
        layout: Layout algorithm ("spring", "circular", "kamada_kawai")
        node_size: Size of nodes
        ax: Matplotlib axes
    
    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create NetworkX graph
    G = nx.from_numpy_array(adjacency)
    
    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue', 
                           edgecolors='black', ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    
    ax.set_title('Cymatic Resonance Network', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    return ax


def plot_harmony_evolution(
    harmony_history: list,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot ARO harmony evolution.
    
    Args:
        harmony_history: History of Harmony Functional values
        ax: Matplotlib axes
    
    Returns:
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(harmony_history, linewidth=2, color='steelblue')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Harmony Functional ℋ', fontsize=12)
    ax.set_title('ARO Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return ax
