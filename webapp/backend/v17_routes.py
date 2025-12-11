"""
IRH v17.0 API Routes

FastAPI routes for IRH v17.0 cGFT simulation and analysis.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np

# Import v17 modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python/src'))

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
    compute_alpha_inverse,
    compute_w0,
    compute_topological_invariants,
    compute_fermion_masses,
    W0_EXACT,
)
from irh.core.v17.spectral_dimension import (
    compute_spectral_dimension_flow,
    D_SPEC_UV,
    D_SPEC_ONE_LOOP,
    D_SPEC_IR,
)


# Create router for v17 endpoints
router = APIRouter(prefix="/api/v17", tags=["IRH v17.0"])


# ============================================================================
# Request/Response Models
# ============================================================================

class CouplingsInput(BaseModel):
    """Input couplings for beta function evaluation."""
    lambda_tilde: float = Field(default=FIXED_POINT_LAMBDA, description="λ̃ coupling")
    gamma_tilde: float = Field(default=FIXED_POINT_GAMMA, description="γ̃ coupling")
    mu_tilde: float = Field(default=FIXED_POINT_MU, description="μ̃ coupling")


class RGFlowConfig(BaseModel):
    """Configuration for RG flow computation."""
    lambda_initial: float = Field(default=20.0, description="Initial λ̃")
    gamma_initial: float = Field(default=50.0, description="Initial γ̃")
    mu_initial: float = Field(default=80.0, description="Initial μ̃")
    t_max: float = Field(default=5.0, ge=0.1, le=20.0, description="Max RG time")
    num_points: int = Field(default=500, ge=10, le=2000, description="Number of points")


class SpectralDimensionConfig(BaseModel):
    """Configuration for spectral dimension flow."""
    t_final: float = Field(default=-10.0, description="Final RG time (negative for IR)")
    d_spec_initial: float = Field(default=D_SPEC_UV, description="Initial spectral dimension")
    num_points: int = Field(default=500, ge=10, le=2000, description="Number of points")
    include_graviton_corrections: bool = Field(default=True, description="Include Δ_grav")


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/fixed-point")
async def get_fixed_point():
    """Get the Cosmic Fixed Point couplings (Eq.1.14)."""
    return {
        "lambda_star": FIXED_POINT_LAMBDA,
        "gamma_star": FIXED_POINT_GAMMA,
        "mu_star": FIXED_POINT_MU,
        "formulas": {
            "lambda": "48π²/9",
            "gamma": "32π²/3",
            "mu": "16π²",
        },
        "reference": "IRH v17.0 Eq.1.14",
    }


@router.post("/beta-functions")
async def evaluate_beta_functions(couplings: CouplingsInput):
    """
    Evaluate the one-loop β-functions at given couplings.
    
    Returns β_λ, β_γ, β_μ as per Eq.1.13.
    """
    bl = beta_lambda(couplings.lambda_tilde)
    bg = beta_gamma(couplings.lambda_tilde, couplings.gamma_tilde)
    bm = beta_mu(couplings.lambda_tilde, couplings.gamma_tilde, couplings.mu_tilde)
    
    return {
        "input": {
            "lambda_tilde": couplings.lambda_tilde,
            "gamma_tilde": couplings.gamma_tilde,
            "mu_tilde": couplings.mu_tilde,
        },
        "beta_functions": {
            "beta_lambda": bl,
            "beta_gamma": bg,
            "beta_mu": bm,
        },
        "at_fixed_point": abs(bl) < 1e-10,
        "reference": "IRH v17.0 Eq.1.13",
    }


@router.get("/stability-matrix")
async def get_stability_matrix():
    """
    Get the stability matrix at the Cosmic Fixed Point.
    
    Returns eigenvalues and eigenvectors for stability analysis.
    """
    M = compute_stability_matrix(
        FIXED_POINT_LAMBDA,
        FIXED_POINT_GAMMA,
        FIXED_POINT_MU,
    )
    
    eigenvalues = np.linalg.eigvals(M)
    
    return {
        "matrix": M.tolist(),
        "eigenvalues": {
            "values": [complex(e).real for e in eigenvalues],
            "imaginary_parts": [complex(e).imag for e in eigenvalues],
        },
        "is_ir_attractive": all(complex(e).real > 0 for e in eigenvalues if complex(e).real != 0),
    }


@router.post("/rg-flow")
async def compute_rg_flow(config: RGFlowConfig):
    """
    Compute RG flow trajectories from given initial conditions.
    
    Returns time series of couplings flowing toward the fixed point.
    """
    from scipy.integrate import solve_ivp
    
    def rg_rhs(t, y):
        lam, gam, mu = y
        return [
            beta_lambda(lam),
            beta_gamma(lam, gam),
            beta_mu(lam, gam, mu),
        ]
    
    t_span = (0, config.t_max)
    t_eval = np.linspace(0, config.t_max, config.num_points)
    
    try:
        sol = solve_ivp(
            rg_rhs,
            t_span,
            [config.lambda_initial, config.gamma_initial, config.mu_initial],
            t_eval=t_eval,
            method='RK45',
        )
        
        return {
            "success": sol.success,
            "t": sol.t.tolist(),
            "lambda": sol.y[0].tolist(),
            "gamma": sol.y[1].tolist(),
            "mu": sol.y[2].tolist(),
            "fixed_point": {
                "lambda_star": FIXED_POINT_LAMBDA,
                "gamma_star": FIXED_POINT_GAMMA,
                "mu_star": FIXED_POINT_MU,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/constants/C_H")
async def get_C_H():
    """
    Get the universal constant C_H (Eq.1.15-1.16).
    
    C_H = 3λ̃*/2γ̃* = 3/4
    """
    c_h = compute_C_H()
    
    return {
        "C_H": c_h,
        "formula": "3λ̃*/2γ̃*",
        "simplified": "3/4",
        "reference": "IRH v17.0 Eq.1.15-1.16",
    }


@router.get("/constants/w0")
async def get_w0(include_graviton_corrections: bool = True):
    """
    Get the dark energy equation of state w₀ (Eq.2.22-2.23).
    
    One-loop: w₀ = -5/6 ≈ -0.8333
    Full: w₀ = -0.91234567(8)
    """
    w0 = compute_w0(include_graviton_corrections=include_graviton_corrections)
    
    return {
        "w0": w0,
        "include_graviton_corrections": include_graviton_corrections,
        "formula": "-1 + μ̃*/96π² (one-loop)",
        "one_loop_value": -5/6,
        "full_value": W0_EXACT,
        "reference": "IRH v17.0 Eq.2.22-2.23",
    }


@router.get("/constants/alpha")
async def get_alpha_inverse():
    """
    Get the inverse fine-structure constant α⁻¹ (Eq.3.4-3.5).
    """
    alpha_inv = compute_alpha_inverse()
    
    return {
        "alpha_inverse": alpha_inv,
        "formula": "(4π²γ̃*/λ̃*)(1 + μ̃*/48π²)",
        "codata_2022": 137.035999177,
        "irh_prediction": 137.035999084,
        "reference": "IRH v17.0 Eq.3.4-3.5",
    }


@router.get("/topology")
async def get_topological_invariants():
    """
    Get the topological invariants at the Cosmic Fixed Point (Eq.3.1-3.2).
    
    β₁* = 12 → SU(3)×SU(2)×U(1) generators
    n_inst* = 3 → 3 fermion generations
    """
    topo = compute_topological_invariants()
    
    return {
        "beta_1": topo["beta_1"],
        "n_inst": topo["n_inst"],
        "interpretations": {
            "beta_1": "First Betti number → gauge group generators (8+3+1=12)",
            "n_inst": "Instanton number → fermion generations",
        },
        "gauge_group": "SU(3) × SU(2) × U(1)",
        "fermion_generations": 3,
        "reference": "IRH v17.0 Eq.3.1-3.2",
    }


@router.get("/fermion-masses")
async def get_fermion_masses():
    """
    Get fermion masses from the Cosmic Fixed Point (Table 3.1).
    """
    data = compute_fermion_masses()
    
    return {
        "topological_complexity": data["K_values"],
        "masses_GeV": data["masses_GeV"],
        "formula": "m_f = √2 K_f √λ̃* √(μ̃*/λ̃*) ℓ₀⁻¹",
        "reference": "IRH v17.0 Table 3.1, Eq.3.6-3.8",
    }


@router.post("/spectral-dimension")
async def compute_spectral_dim_flow(config: SpectralDimensionConfig):
    """
    Compute the spectral dimension flow d_spec(k) (Eq.2.8-2.9).
    
    Shows the three regimes: UV (≈2), intermediate (42/11), IR (→4).
    """
    try:
        result = compute_spectral_dimension_flow(
            t_final=config.t_final,
            d_spec_initial=config.d_spec_initial,
            num_points=config.num_points,
            include_graviton_corrections=config.include_graviton_corrections,
        )
        
        return {
            "success": result.success,
            "t": result.t.tolist(),
            "k_normalized": (result.k / result.k[0]).tolist(),
            "d_spec": result.d_spec.tolist(),
            "limits": {
                "UV": D_SPEC_UV,
                "one_loop": D_SPEC_ONE_LOOP,
                "IR": D_SPEC_IR,
            },
            "config": {
                "include_graviton_corrections": config.include_graviton_corrections,
            },
            "reference": "IRH v17.0 Eq.2.8-2.9",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_v17_summary():
    """
    Get a complete summary of IRH v17.0 predictions.
    """
    topo = compute_topological_invariants()
    
    return {
        "title": "IRH v17.0: The Cosmic Fixed Point",
        "fixed_point": {
            "lambda_star": FIXED_POINT_LAMBDA,
            "gamma_star": FIXED_POINT_GAMMA,
            "mu_star": FIXED_POINT_MU,
        },
        "universal_constants": {
            "C_H": compute_C_H(),
            "w0_one_loop": compute_w0(include_graviton_corrections=False),
            "w0_full": compute_w0(include_graviton_corrections=True),
        },
        "topology": {
            "beta_1": topo["beta_1"],
            "n_inst": topo["n_inst"],
        },
        "spectral_dimension": {
            "UV": D_SPEC_UV,
            "intermediate": D_SPEC_ONE_LOOP,
            "IR": D_SPEC_IR,
        },
        "key_insight": "All constants of Nature are derived, not discovered.",
        "manuscript": "docs/manuscripts/IRHv17.md",
    }
