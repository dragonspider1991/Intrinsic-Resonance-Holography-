"""
IRH v17.0 Streamlit Dashboard

Interactive web interface for exploring IRH v17.0 predictions.

Run with: streamlit run streamlit_v17.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys
import os

# Add path for IRH modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python/src'))

# Import v17 modules
from irh.core.v17.beta_functions import (
    beta_lambda,
    beta_gamma,
    beta_mu,
    FIXED_POINT_LAMBDA,
    FIXED_POINT_GAMMA,
    FIXED_POINT_MU,
    compute_stability_matrix,
)
from irh.core.v17.constants import (
    compute_C_H,
    compute_w0,
    compute_topological_invariants,
    compute_fermion_masses,
)
from irh.core.v17.spectral_dimension import (
    compute_spectral_dimension_flow,
    D_SPEC_UV,
    D_SPEC_ONE_LOOP,
    D_SPEC_IR,
)


# Page configuration
st.set_page_config(
    page_title="IRH v17.0 Dashboard",
    page_icon="🌌",
    layout="wide",
)

# Title
st.title("🌌 Intrinsic Resonance Holography v17.0")
st.markdown("""
**The Unified Theory of Emergent Reality**

All constants of Nature are now **derived, not discovered**.
""")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Overview", "RG Flow", "Spectral Dimension", "Physical Constants", "Fermion Masses"]
)

# ==============================================================================
# Overview Page
# ==============================================================================
if page == "Overview":
    st.header("The Cosmic Fixed Point")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("λ̃*", f"{FIXED_POINT_LAMBDA:.4f}", "48π²/9")
    with col2:
        st.metric("γ̃*", f"{FIXED_POINT_GAMMA:.4f}", "32π²/3")
    with col3:
        st.metric("μ̃*", f"{FIXED_POINT_MU:.4f}", "16π²")
    
    st.markdown("""
    ### Key Predictions
    
    From the unique infrared fixed point of the cGFT:
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("C_H", f"{compute_C_H():.6f}", "Universal constant")
    with col2:
        w0 = compute_w0(include_graviton_corrections=True)
        st.metric("w₀", f"{w0}", "Dark energy")
    with col3:
        topo = compute_topological_invariants()
        st.metric("β₁*", topo["beta_1"], "Gauge generators")
    with col4:
        st.metric("n_inst*", topo["n_inst"], "Fermion generations")
    
    st.markdown("""
    ### The Asymptotic Safety Signature
    
    The spectral dimension flows from:
    - **UV**: d_spec ≈ 2 (dimensional reduction)
    - **Intermediate**: d_spec ≈ 42/11 ≈ 3.818 (one-loop)
    - **IR**: d_spec → 4 exactly (graviton corrections)
    """)

# ==============================================================================
# RG Flow Page
# ==============================================================================
elif page == "RG Flow":
    st.header("Renormalization Group Flow")
    
    st.markdown("""
    Explore how couplings flow from UV to the Cosmic Fixed Point.
    
    **β-functions (Eq.1.13):**
    - β_λ = -6λ̃ + (9/8π²)λ̃²
    - β_γ = -2γ̃ + (3/4π²)λ̃γ̃
    - β_μ = -4μ̃ + (1/2π²)λ̃μ̃
    """)
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lambda_init = st.slider("Initial λ̃", 10.0, 100.0, 30.0)
    with col2:
        gamma_init = st.slider("Initial γ̃", 20.0, 200.0, 80.0)
    with col3:
        mu_init = st.slider("Initial μ̃", 40.0, 250.0, 120.0)
    
    t_max = st.slider("RG time", 1.0, 10.0, 5.0)
    
    # Compute flow
    def rg_rhs(t, y):
        lam, gam, mu = y
        return [
            beta_lambda(lam),
            beta_gamma(lam, gam),
            beta_mu(lam, gam, mu),
        ]
    
    t_eval = np.linspace(0, t_max, 500)
    sol = solve_ivp(
        rg_rhs,
        (0, t_max),
        [lambda_init, gamma_init, mu_init],
        t_eval=t_eval,
        method='RK45',
    )
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(sol.t, sol.y[0], 'b-', linewidth=2)
    axes[0].axhline(FIXED_POINT_LAMBDA, color='red', linestyle='--', label=f'λ̃* = {FIXED_POINT_LAMBDA:.2f}')
    axes[0].set_xlabel('RG time t')
    axes[0].set_ylabel('λ̃(t)')
    axes[0].set_title('Flow of λ̃')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(sol.t, sol.y[1], 'g-', linewidth=2)
    axes[1].axhline(FIXED_POINT_GAMMA, color='red', linestyle='--', label=f'γ̃* = {FIXED_POINT_GAMMA:.2f}')
    axes[1].set_xlabel('RG time t')
    axes[1].set_ylabel('γ̃(t)')
    axes[1].set_title('Flow of γ̃')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(sol.t, sol.y[2], 'm-', linewidth=2)
    axes[2].axhline(FIXED_POINT_MU, color='red', linestyle='--', label=f'μ̃* = {FIXED_POINT_MU:.2f}')
    axes[2].set_xlabel('RG time t')
    axes[2].set_ylabel('μ̃(t)')
    axes[2].set_title('Flow of μ̃')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Final values
    st.markdown("### Flow Endpoints")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("λ̃(t_max)", f"{sol.y[0][-1]:.4f}", f"→ {FIXED_POINT_LAMBDA:.2f}")
    with col2:
        st.metric("γ̃(t_max)", f"{sol.y[1][-1]:.4f}", f"→ {FIXED_POINT_GAMMA:.2f}")
    with col3:
        st.metric("μ̃(t_max)", f"{sol.y[2][-1]:.4f}", f"→ {FIXED_POINT_MU:.2f}")

# ==============================================================================
# Spectral Dimension Page
# ==============================================================================
elif page == "Spectral Dimension":
    st.header("Spectral Dimension Flow")
    
    st.markdown("""
    The spectral dimension d_spec(k) demonstrates asymptotic safety:
    
    **Flow equation (Eq.2.8):**
    $$\\partial_t d_{\\text{spec}}(k) = \\eta(k)[d_{\\text{spec}}(k) - 4] + \\Delta_{\\text{grav}}(k)$$
    """)
    
    # Controls
    col1, col2 = st.columns(2)
    with col1:
        t_final = st.slider("IR scale (t_final)", -20.0, -2.0, -10.0)
    with col2:
        include_grav = st.checkbox("Include graviton corrections", True)
    
    # Compute flow
    result = compute_spectral_dimension_flow(
        t_final=t_final,
        d_spec_initial=D_SPEC_UV,
        num_points=500,
        include_graviton_corrections=include_grav,
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(result.t, result.d_spec, 'b-', linewidth=2, label='d_spec(k)')
    ax.axhline(D_SPEC_UV, color='orange', linestyle=':', alpha=0.7, label=f'UV: {D_SPEC_UV}')
    ax.axhline(D_SPEC_ONE_LOOP, color='purple', linestyle=':', alpha=0.7, label=f'One-loop: 42/11 ≈ {D_SPEC_ONE_LOOP:.3f}')
    ax.axhline(D_SPEC_IR, color='red', linestyle=':', alpha=0.7, label=f'IR: {D_SPEC_IR}')
    
    ax.set_xlabel('RG time t = log(k/Λ_UV)', fontsize=12)
    ax.set_ylabel('Spectral Dimension d_spec(k)', fontsize=12)
    ax.set_title('Spectral Dimension Flow (Eq.2.8-2.9)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 5)
    
    st.pyplot(fig)
    
    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("UV (t=0)", f"{result.d_spec[0]:.4f}", "Dimensional reduction")
    with col2:
        mid_idx = len(result.t) // 4
        st.metric("Intermediate", f"{result.d_spec[mid_idx]:.4f}", "~42/11")
    with col3:
        st.metric("IR (t→-∞)", f"{result.d_spec[-1]:.4f}", "→ 4 exactly")

# ==============================================================================
# Physical Constants Page
# ==============================================================================
elif page == "Physical Constants":
    st.header("Physical Constants from the Cosmic Fixed Point")
    
    st.markdown("""
    All fundamental constants are **analytically derived** from the fixed-point couplings.
    """)
    
    # C_H
    st.subheader("Universal Constant C_H (Eq.1.15-1.16)")
    c_h = compute_C_H()
    st.latex(r"C_H = \frac{3\tilde\lambda_*}{2\tilde\gamma_*} = \frac{3}{4} = " + f"{c_h:.12f}")
    
    # w₀
    st.subheader("Dark Energy Equation of State w₀ (Eq.2.22-2.23)")
    w0_ol = compute_w0(include_graviton_corrections=False)
    w0_full = compute_w0(include_graviton_corrections=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("One-loop", f"{w0_ol:.6f}", "-5/6")
    with col2:
        st.metric("With graviton corrections", f"{w0_full}", "Certified")
    
    st.latex(r"w_0 = -1 + \frac{\tilde\mu_*}{96\pi^2} = -\frac{5}{6} \approx -0.8333...")
    
    # Topology
    st.subheader("Topological Invariants (Eq.3.1-3.2)")
    topo = compute_topological_invariants()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("β₁* (First Betti Number)", topo["beta_1"])
        st.markdown("→ SU(3)×SU(2)×U(1) generators: 8+3+1 = 12")
    with col2:
        st.metric("n_inst* (Instanton Number)", topo["n_inst"])
        st.markdown("→ 3 fermion generations")

# ==============================================================================
# Fermion Masses Page
# ==============================================================================
elif page == "Fermion Masses":
    st.header("Fermion Masses from Topological Complexity")
    
    st.markdown("""
    Fermion masses are determined by topological complexity integers $\mathcal{K}_f$ (Eq.3.3-3.8):
    
    $$m_f = \\sqrt{2}\,\\mathcal{K}_f\,\\tilde\\lambda_*^{1/2}\\left(\\frac{\\tilde\\mu_*}{\\tilde\\lambda_*}\\right)^{1/2}\\ell_0^{-1}$$
    """)
    
    data = compute_fermion_masses()
    K = data["K_values"]
    masses = data["masses_GeV"]
    
    # Create table
    import pandas as pd
    
    df = pd.DataFrame({
        "Fermion": list(K.keys()),
        "K_f": list(K.values()),
        "Predicted Mass (GeV)": list(masses.values()),
    })
    
    st.dataframe(df, use_container_width=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fermions = list(masses.keys())
    mass_values = [masses[f] for f in fermions]
    
    ax.bar(fermions, mass_values, color='steelblue')
    ax.set_yscale('log')
    ax.set_xlabel('Fermion', fontsize=12)
    ax.set_ylabel('Mass (GeV)', fontsize=12)
    ax.set_title('Fermion Mass Spectrum from IRH v17.0', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
**IRH v17.0** | [Manuscript](docs/manuscripts/IRHv17.md) | 
*All constants of Nature are now derived, not discovered.*
""")
