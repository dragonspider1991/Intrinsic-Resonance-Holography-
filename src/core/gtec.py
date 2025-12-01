import numpy as np

def gtec_entanglement_energy(eigenvalues, coupling_mu, L_G, hbar_G):
    """
    Explicitly calculates the negative energy contribution from vacuum entanglement.
    
    Formalism v9.4: E_GTEC = - mu * S_ent
    
    Args:
        eigenvalues (np.array): Normalized spectrum of the entanglement Hamiltonian.
        coupling_mu (float): Derived coupling constant (approx 1/(N ln N)).
        L_G (float): Emergent graph scale (dimensionless).
        hbar_G (float): Emergent action quantum (dimensionless).
        
    Returns: 
        float: Negative energy value (E_gtec).
    """
    # Filter zeros to ensure log stability
    spectrum = eigenvalues[eigenvalues > 0]
    
    # Von Neumann Entropy (bits)
    S_ent = -np.sum(spectrum * np.log2(spectrum))
    
    # Thermodynamic relation: Energy = - mu * Entropy
    # The negative sign is crucial for Dark Energy cancellation.
    E_gtec = - (L_G / hbar_G) * coupling_mu * S_ent
    
    return E_gtec
