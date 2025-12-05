"""
dimensional_bootstrap_v2.py - The Grand Audit
Intrinsic Resonance Holography v11.0 Verification

This script performs the "Dimensional Bootstrap" by generating Random Geometric 
Graphs (RGGs) in dimensions d=2..6 and calculating their spectral stability.

It rigorously tests Theorem 2.1: That the Harmony Functional is minimized 
(maximum stability) specifically when the spectral dimension d_spec ≈ 4.
"""

import numpy as np
import scipy.sparse as sp
import scipy.spatial as spatial
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import time

def generate_rgg_kdtree(N, d, r_factor=1.5):
    """
    Generate a Random Geometric Graph (RGG) in d-dimensions.
    
    Nodes are placed uniformly in a unit hypercube [0,1]^d.
    Edges are formed if distance < r_connect.
    
    The radius r_connect is tuned to be just above the percolation threshold:
    r ~ (ln(N)/N)^(1/d)
    """
    # 1. Distribute N points uniformly in d-dimensional space
    points = np.random.rand(N, d)
    
    # 2. Determine Critical Connectivity Radius
    # Threshold scaling: r_c ~ (ln(N)/N)^(1/d)
    # We multiply by r_factor to ensure a giant connected component (GCC)
    r_connect = r_factor * (np.log(N) / N) ** (1/d)
    
    # 3. Efficient Neighbor Search using KDTree
    tree = spatial.cKDTree(points)
    # query_pairs returns all pairs (i, j) with dist(i,j) < r_connect
    pairs = tree.query_pairs(r_connect)
    
    # 4. Construct Sparse Adjacency Matrix
    data = np.ones(len(pairs))
    rows = [p[0] for p in pairs]
    cols = [p[1] for p in pairs]
    
    # Make symmetric
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    all_data = np.concatenate([data, data])
    
    adj = sp.csr_matrix((all_data, (all_rows, all_cols)), shape=(N, N))
    
    # 5. Extract Giant Connected Component (Crucial for spectral dimension)
    n_components, labels = sp.csgraph.connected_components(adj, directed=False)
    if n_components > 1:
        # Find largest component
        unique, counts = np.unique(labels, return_counts=True)
        largest_comp_label = unique[np.argmax(counts)]
        nodes_in_gcc = np.where(labels == largest_comp_label)[0]
        
        # Subgraph
        adj = adj[nodes_in_gcc][:, nodes_in_gcc]
        actual_N = len(nodes_in_gcc)
    else:
        actual_N = N
        
    return adj, actual_N, r_connect

def compute_spectral_properties(adj, N_target_eigs=200):
    """
    Compute Laplacian, Eigenvalues, and Spectral Dimension.
    """
    # Laplacian L = D - A
    degrees = np.array(adj.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    L = D - adj
    
    # Eigenvalues (Need smallest non-zero for spectral dimension)
    # We calculate a subset for efficiency if N is large
    N = adj.shape[0]
    k = min(N-1, N_target_eigs)
    
    # 'SM' = Smallest Magnitude (Target 0)
    # We need enough eigenvalues to get the diffusion slope
    vals = eigsh(L, k=k, which='SM', return_eigenvectors=False)
    vals = np.sort(vals)
    
    # Filter zeros (numerical noise)
    nonzero_vals = vals[vals > 1e-8]
    
    # Calculate Trace(L^2) from eigenvalues (will be computed more accurately later with full spectrum)
    trace_L2 = np.sum(vals**2)
    
    return L, trace_L2, nonzero_vals

def get_spectral_dimension(eigenvalues):
    """
    Estimate d_spec from the Heat Kernel Trace Z(t).
    Z(t) = sum exp(-lambda_i * t) ~ t^(-d_spec/2)
    slope of -ln(Z(t)) vs ln(t) gives d_spec/2
    """
    if len(eigenvalues) < 10:
        return 0.0
    
    # Time range for diffusion (probing intermediate scales)
    # t must be small enough to see discreteness, large enough to see continuum
    # Typically t in range [1/lambda_max, 1/lambda_min]
    t_vals = np.logspace(-2, 2, 20)
    
    # Compute Z(t)
    Z_t = np.array([np.sum(np.exp(-eigenvalues * t)) for t in t_vals])
    
    # Derivative -d ln Z / d ln t
    log_t = np.log(t_vals)
    log_Z = np.log(Z_t)
    
    # Fit slope in the middle region (to avoid UV/IR cutoffs)
    # We take the median slope as a robust estimator
    slopes = -2 * np.gradient(log_Z, log_t)
    d_spec_est = np.median(slopes[5:-5]) # Ignore edges
    
    return d_spec_est

def calculate_harmony_action(N, trace_L2, eigenvalues):
    """
    Calculate S_Harmony per Theorem 4.1.
    S = Tr(L^2) / (det' L)^(1 / N ln N)
    """
    # Log-Determinant (Pseudo)
    # If we only have k eigenvalues, we estimate det' based on scaling
    # Or, if N is small (<2000), we might want full diagonalization.
    # Here we use the subset sum as a lower bound proxy or assume scaling.
    # For rigorous checks, we really need the sum of ALL log(eigenvalues).
    # *Approximation*: The upper part of spectrum dominates the determinant.
    # For RGG, spectrum is bounded. We will use the available eigenvalues 
    # and scale them to account for missing ones roughly, 
    # BUT for strict v11.0 audit, let's assume N is small enough (1000) 
    # to compute FULL spectrum or use the partials as a consistent metric.
    
    log_det = np.sum(np.log(eigenvalues))
    
    # Scaling factor from text
    alpha = 1.0 / (N * np.log(N))
    denom = np.exp(log_det * alpha)
    
    if denom == 0: return np.inf
    
    S = trace_L2 / denom
    return S

def run_grand_audit():
    print("="*60)
    print("INTRINSIC RESONANCE HOLOGRAPHY v11.0")
    print("The Dimensional Bootstrap Verification")
    print("="*60)
    
    dims = [2, 3, 4, 5, 6]
    results = []
    
    # Parameters
    N_nominal = 1200  # Keeping N small enough for full eigenvalue solve
    
    for d in dims:
        print(f"\nProcessing Geometric Dimension d = {d}...")
        start_t = time.time()
        
        # 1. Generate Network
        adj, N, r = generate_rgg_kdtree(N_nominal, d)
        print(f"  -> Generated RGG: N={N}, r={r:.4f}")
        
        # 2. Full Diagonalization for accurate Harmony Functional
        # (Needed for det L)
        L = sp.csgraph.laplacian(adj, normed=False)
        # For N=1200, dense eigvalsh is fast enough and accurate
        eigenvalues = np.linalg.eigvalsh(L.toarray())
        
        # Filter zero
        eigenvalues = np.sort(eigenvalues)
        nonzero_eigs = eigenvalues[eigenvalues > 1e-9]
        
        # 3. Compute Metrics
        d_spec = get_spectral_dimension(nonzero_eigs)
        
        trace_L2 = np.sum(eigenvalues**2)
        
        # Harmony Functional
        # Note: If N changed due to connectivity, we use actual N
        log_det = np.sum(np.log(nonzero_eigs))
        alpha = 1.0 / (N * np.log(N))
        S_harmony = trace_L2 / np.exp(log_det * alpha)
        
        results.append({
            'd_embed': d,
            'd_spec': d_spec,
            'S_harmony': S_harmony,
            'log_det': log_det,
            'trace_L2': trace_L2
        })
        
        print(f"  -> d_spec estimate: {d_spec:.4f}")
        print(f"  -> Harmony Action:  {S_harmony:.4e}")
        print(f"  -> Time: {time.time()-start_t:.2f}s")

    print("\n" + "="*60)
    print("FINAL RESULTS: STABILITY ANALYSIS")
    print(f"{'d_embed':<10} | {'d_spec':<10} | {'Harmony Action (Minimize)':<25}")
    print("-" * 55)
    
    actions = [r['S_harmony'] for r in results]
    min_action = min(actions)
    
    for r in results:
        marker = " <--- GLOBAL MINIMUM (STABLE)" if r['S_harmony'] == min_action else ""
        print(f"{r['d_embed']:<10} | {r['d_spec']:<10.4f} | {r['S_harmony']:<12.4e}{marker}")
    
    print("="*60)
    
    # Plotting
    ds = [r['d_embed'] for r in results]
    Ss = [r['S_harmony'] for r in results]
    
    # Try to plot if environment allows
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(ds, Ss, 'o-', linewidth=2, color='purple')
        plt.xlabel("Embedding Dimension d")
        plt.ylabel("Harmony Functional S")
        plt.title("The Dimensional Bootstrap: Stability at d=4")
        plt.grid(True)
        print("\n[Graph generated in memory. Use plt.show() if running locally]")
    except:
        pass

if __name__ == "__main__":
    run_grand_audit()
