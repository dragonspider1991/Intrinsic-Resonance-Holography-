import numpy as np

def ncgg_covariant_derivative(f, W, adj_list, embedding, k, v):
    """
    Constructs the discrete gauge-covariant derivative D_k f(v).
    Uses spectral embedding to identify directional neighbors via projection.
    
    Formalism v9.4 Section IV.A
    
    Args:
        f (array): Scalar field on nodes.
        W (matrix): Complex edge weights.
        adj_list (list of lists): Adjacency list.
        embedding (matrix): Spectral embedding coordinates [N, d_s].
        k (int): Direction index.
        v (int): Node index.
    """
    neighbors = adj_list[v]
    # Basis vector for direction k
    vec_k = np.zeros(embedding.shape[1])
    vec_k[k] = 1.0
    
    sum_val = 0.0 + 0.0j
    count = 0
    
    for u in neighbors:
        # Check alignment via projection
        edge_vec = embedding[u] - embedding[v]
        # Directional filter (alignment > 0.5 rad equivalent)
        if np.dot(edge_vec, vec_k) > 0.5: 
            # Parallel Transport: phase of W_vu
            w_vu = W[v, u]
            # Handle potential zero weight
            if np.abs(w_vu) > 1e-12:
                phase = w_vu / np.abs(w_vu)
                sum_val += f[u] * phase - f[v]
                count += 1
            
    if count == 0: return 0.0 + 0.0j
    
    # Scale by average connection strength
    avg_weight = np.abs(W[v, neighbors[0]]) # Simplified for kernel demo
    return (avg_weight / count) * sum_val
