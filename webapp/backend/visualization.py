"""
Visualization Data Serializers
================================

Converts IRH computational results into formats optimized for
3D (Three.js/WebGL) and 2D (Chart.js/D3.js) visualization.

Provides data structures for:
- 3D network topology visualization
- 3D spectral/eigenvalue surface plots
- 2D line charts, scatter plots, histograms
- Animation frames for time-evolution
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import networkx as nx


def serialize_network_3d(adjacency_matrix: np.ndarray, 
                         positions: Optional[np.ndarray] = None,
                         eigenvalues: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Serialize network for 3D visualization (e.g., Three.js, plotly).
    
    Args:
        adjacency_matrix: N x N adjacency matrix
        positions: Optional N x 3 position array (if None, use spring layout)
        eigenvalues: Optional eigenvalue array for coloring nodes
        
    Returns:
        Dictionary with nodes, edges, and metadata for 3D rendering
    """
    N = adjacency_matrix.shape[0]
    
    # Generate 3D positions if not provided
    if positions is None:
        # Use NetworkX spring layout in 3D
        G = nx.from_numpy_array(np.abs(adjacency_matrix))
        pos_dict = nx.spring_layout(G, dim=3, seed=42)
        positions = np.array([pos_dict[i] for i in range(N)])
    
    # Normalize positions to [-1, 1] cube
    if positions.max() > positions.min():
        positions = 2 * (positions - positions.min()) / (positions.max() - positions.min()) - 1
    
    # Build node list
    nodes = []
    for i in range(N):
        node = {
            "id": int(i),
            "position": positions[i].tolist(),
            "color": get_node_color(i, eigenvalues) if eigenvalues is not None else "#3498db",
            "size": 1.0,
        }
        nodes.append(node)
    
    # Build edge list
    edges = []
    edge_weights = []
    for i in range(N):
        for j in range(i + 1, N):  # Symmetric, only upper triangle
            weight = float(np.abs(adjacency_matrix[i, j]))
            if weight > 1e-10:  # Threshold for visualization
                edges.append({
                    "source": int(i),
                    "target": int(j),
                    "weight": weight,
                    "color": get_edge_color(weight),
                    "opacity": min(1.0, weight / adjacency_matrix.max()) if adjacency_matrix.max() > 0 else 0.5,
                })
                edge_weights.append(weight)
    
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "node_count": N,
            "edge_count": len(edges),
            "avg_edge_weight": float(np.mean(edge_weights)) if edge_weights else 0.0,
            "max_edge_weight": float(np.max(edge_weights)) if edge_weights else 0.0,
        }
    }


def serialize_spectrum_3d(eigenvalues: np.ndarray,
                          eigenvectors: Optional[np.ndarray] = None,
                          mode: str = "scatter") -> Dict[str, Any]:
    """
    Serialize spectral data for 3D visualization.
    
    Args:
        eigenvalues: 1D array of eigenvalues
        eigenvectors: Optional N x N eigenvector matrix
        mode: "scatter", "surface", or "volume"
        
    Returns:
        Dictionary for 3D spectral visualization
    """
    N = len(eigenvalues)
    
    if mode == "scatter":
        # Simple 3D scatter of eigenvalues (index, value, multiplicity)
        points = []
        for i, val in enumerate(eigenvalues):
            points.append({
                "x": float(i),
                "y": float(val),
                "z": 0.0,
                "color": get_spectrum_color(val, eigenvalues),
                "size": 2.0,
            })
        
        return {
            "type": "scatter3d",
            "points": points,
            "metadata": {
                "min_eigenvalue": float(eigenvalues.min()),
                "max_eigenvalue": float(eigenvalues.max()),
                "spectral_gap": float(eigenvalues[1] - eigenvalues[0]) if N > 1 else 0.0,
            }
        }
    
    elif mode == "surface" and eigenvectors is not None:
        # Create surface plot of first few eigenvector components
        grid_size = min(int(np.sqrt(N)), 50)
        surfaces = []
        
        for k in range(min(3, N)):  # First 3 eigenvectors
            evec = eigenvectors[:, k]
            # Reshape to grid (approximate)
            grid_data = np.zeros((grid_size, grid_size))
            for i in range(min(N, grid_size * grid_size)):
                row = i // grid_size
                col = i % grid_size
                if row < grid_size:
                    grid_data[row, col] = evec[i] if i < len(evec) else 0.0
            
            surfaces.append({
                "eigenvalue_index": k,
                "eigenvalue": float(eigenvalues[k]),
                "grid": grid_data.tolist(),
                "grid_size": [grid_size, grid_size],
            })
        
        return {
            "type": "surface3d",
            "surfaces": surfaces,
        }
    
    else:
        # Default to simple list
        return {
            "type": "list",
            "eigenvalues": eigenvalues.tolist(),
        }


def serialize_chart_2d(x_data: np.ndarray,
                       y_data: np.ndarray,
                       chart_type: str = "line",
                       title: str = "",
                       x_label: str = "X",
                       y_label: str = "Y") -> Dict[str, Any]:
    """
    Serialize data for 2D chart visualization (Chart.js format).
    
    Args:
        x_data: X-axis data
        y_data: Y-axis data
        chart_type: "line", "scatter", "bar", "histogram"
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        
    Returns:
        Dictionary in Chart.js format
    """
    datasets = [{
        "label": y_label,
        "data": [{"x": float(x), "y": float(y)} for x, y in zip(x_data, y_data)],
        "borderColor": "#3498db",
        "backgroundColor": "rgba(52, 152, 219, 0.2)",
        "borderWidth": 2,
        "pointRadius": 3 if chart_type == "scatter" else 0,
    }]
    
    return {
        "type": chart_type,
        "data": {
            "datasets": datasets,
        },
        "options": {
            "responsive": True,
            "plugins": {
                "title": {
                    "display": bool(title),
                    "text": title,
                },
                "legend": {
                    "display": True,
                }
            },
            "scales": {
                "x": {
                    "type": "linear",
                    "title": {
                        "display": True,
                        "text": x_label,
                    }
                },
                "y": {
                    "title": {
                        "display": True,
                        "text": y_label,
                    }
                }
            }
        }
    }


def serialize_histogram_2d(data: np.ndarray,
                           bins: int = 50,
                           title: str = "Histogram",
                           x_label: str = "Value",
                           y_label: str = "Frequency") -> Dict[str, Any]:
    """
    Create histogram data for 2D visualization.
    
    Args:
        data: 1D array to histogram
        bins: Number of bins
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        
    Returns:
        Histogram data in Chart.js format
    """
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return serialize_chart_2d(
        x_data=bin_centers,
        y_data=counts,
        chart_type="bar",
        title=title,
        x_label=x_label,
        y_label=y_label,
    )


def serialize_heatmap_2d(matrix: np.ndarray,
                         title: str = "Heatmap",
                         colorscale: str = "Viridis") -> Dict[str, Any]:
    """
    Serialize matrix for 2D heatmap visualization.
    
    Args:
        matrix: 2D array to visualize
        title: Chart title
        colorscale: Color scale name
        
    Returns:
        Heatmap data for visualization
    """
    return {
        "type": "heatmap",
        "data": matrix.tolist(),
        "metadata": {
            "shape": list(matrix.shape),
            "min": float(matrix.min()),
            "max": float(matrix.max()),
            "mean": float(matrix.mean()),
            "std": float(matrix.std()),
        },
        "options": {
            "title": title,
            "colorscale": colorscale,
        }
    }


def serialize_animation_frames(network_history: List[np.ndarray],
                               eigenvalue_history: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
    """
    Serialize time-evolution data for animation.
    
    Args:
        network_history: List of adjacency matrices over time
        eigenvalue_history: Optional list of eigenvalue arrays over time
        
    Returns:
        Animation frame data
    """
    frames = []
    
    for t, adj_matrix in enumerate(network_history):
        frame_data = serialize_network_3d(
            adj_matrix,
            eigenvalues=eigenvalue_history[t] if eigenvalue_history else None
        )
        frames.append({
            "time": t,
            "data": frame_data,
        })
    
    return {
        "type": "animation",
        "frames": frames,
        "frame_count": len(frames),
        "fps": 30,
    }


# ============================================================================
# Utility Functions for Colors
# ============================================================================

def get_node_color(index: int, eigenvalues: Optional[np.ndarray] = None) -> str:
    """Get color for node based on eigenvalue magnitude."""
    if eigenvalues is None:
        return "#3498db"
    
    # Normalize eigenvalue to [0, 1]
    val = eigenvalues[index] if index < len(eigenvalues) else 0.0
    eig_min, eig_max = eigenvalues.min(), eigenvalues.max()
    
    if eig_max > eig_min:
        normalized = (val - eig_min) / (eig_max - eig_min)
    else:
        normalized = 0.5
    
    # Use blue-to-red colormap
    return rgb_to_hex(colormap_viridis(normalized))


def get_edge_color(weight: float) -> str:
    """Get color for edge based on weight."""
    # Simple gray scale
    intensity = min(1.0, weight)
    gray = int(255 * (1 - intensity * 0.7))
    return f"#{gray:02x}{gray:02x}{gray:02x}"


def get_spectrum_color(eigenvalue: float, all_eigenvalues: np.ndarray) -> str:
    """Get color for eigenvalue in spectrum plot."""
    eig_min, eig_max = all_eigenvalues.min(), all_eigenvalues.max()
    
    if eig_max > eig_min:
        normalized = (eigenvalue - eig_min) / (eig_max - eig_min)
    else:
        normalized = 0.5
    
    return rgb_to_hex(colormap_viridis(normalized))


def colormap_viridis(t: float) -> Tuple[int, int, int]:
    """
    Simplified Viridis colormap.
    
    Args:
        t: Value in [0, 1]
        
    Returns:
        (r, g, b) tuple with values in [0, 255]
    """
    # Simplified viridis colors at key points
    colors = [
        (68, 1, 84),      # Purple (t=0)
        (59, 82, 139),    # Blue
        (33, 145, 140),   # Teal
        (94, 201, 98),    # Green
        (253, 231, 37),   # Yellow (t=1)
    ]
    
    # Linear interpolation
    t = max(0.0, min(1.0, t))
    idx = t * (len(colors) - 1)
    i = int(idx)
    frac = idx - i
    
    if i >= len(colors) - 1:
        return colors[-1]
    
    c1 = colors[i]
    c2 = colors[i + 1]
    
    r = int(c1[0] + frac * (c2[0] - c1[0]))
    g = int(c1[1] + frac * (c2[1] - c1[1]))
    b = int(c1[2] + frac * (c2[2] - c1[2]))
    
    return (r, g, b)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert (r, g, b) to hex color string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
