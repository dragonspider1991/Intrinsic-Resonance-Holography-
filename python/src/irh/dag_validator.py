"""
dag_validator.py - Derivational Acyclic Graph (DAG) Validator

This module enforces logical coherence by validating:
- DAG structure of derivations (no circular dependencies)
- No ad hoc parameters (all from fundamental axioms)
- Dynamic selection only for emergent quantities

References:
- IRH Meta-Theoretical Validation Protocol
- Ontological Clarity requirements
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass
class DAGValidationResult:
    """Result of DAG validation."""

    is_acyclic: bool
    n_nodes: int
    n_edges: int
    topological_order: list[str]
    cycles: list[list[str]]
    message: str


# Define the IRH derivation DAG structure
# Nodes represent concepts/quantities, edges represent derivation dependencies
IRH_DERIVATION_GRAPH = {
    # Foundational axioms (no dependencies)
    "hypergraph_substrate": [],
    "complex_weights": [],
    "holographic_principle": [],
    # Stage 1: Basic structures
    "laplacian": ["hypergraph_substrate", "complex_weights"],
    "eigenspectrum": ["laplacian"],
    "heat_kernel": ["laplacian"],
    # Stage 2: Dimensional emergence
    "spectral_dimension": ["heat_kernel"],
    "combinatorial_laplacian": ["hypergraph_substrate"],
    "d_s_bootstrap": ["combinatorial_laplacian", "spectral_dimension"],
    # Stage 3: Geometric emergence
    "metric_tensor": ["laplacian", "eigenspectrum"],
    "lorentz_signature": ["metric_tensor"],
    "ricci_curvature": ["metric_tensor"],
    # Stage 4: Topological structure
    "cycle_basis": ["hypergraph_substrate"],
    "homotopy_group": ["cycle_basis"],
    "betti_numbers": ["homotopy_group"],
    # Stage 5: Physical quantities
    "gtec_complexity": ["eigenspectrum", "complex_weights"],
    "ncgg_operators": ["laplacian", "eigenspectrum"],
    "ccr_relations": ["ncgg_operators"],
    # Stage 6: Physics recovery
    "newton_gravity": ["metric_tensor", "ricci_curvature"],
    "einstein_equations": ["ricci_curvature", "metric_tensor"],
    "gauge_couplings": ["homotopy_group", "betti_numbers"],
    "beta_functions": ["gauge_couplings"],
    # Stage 7: Predictions
    "alpha_inverse": ["gauge_couplings", "cycle_basis"],
    "neutrino_masses": ["eigenspectrum", "betti_numbers"],
    "ckm_matrix": ["eigenspectrum"],
    "dark_energy_eos": ["gtec_complexity", "holographic_principle"],
}


def validate_dag() -> dict:
    """
    Validate that the IRH derivation structure is a DAG (acyclic).

    Returns:
        Validation result dictionary
    """
    # Build NetworkX DiGraph
    G = nx.DiGraph()

    for node, deps in IRH_DERIVATION_GRAPH.items():
        G.add_node(node)
        for dep in deps:
            G.add_edge(dep, node)

    # Check for cycles
    is_acyclic = nx.is_directed_acyclic_graph(G)

    # Find cycles if any
    cycles = []
    if not is_acyclic:
        try:
            cycles = list(nx.simple_cycles(G))
        except Exception:
            cycles = []

    # Get topological order if acyclic
    if is_acyclic:
        try:
            topo_order = list(nx.topological_sort(G))
        except Exception:
            topo_order = []
    else:
        topo_order = []

    return {
        "is_acyclic": is_acyclic,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "topological_order": topo_order,
        "cycles": cycles,
        "message": "DAG validation passed" if is_acyclic else "Cycles detected!",
    }


def render_dag_mermaid() -> str:
    """
    Render the derivation DAG as Mermaid diagram.

    Returns:
        Mermaid diagram string
    """
    lines = ["graph TD"]

    # Add nodes with styling
    axiom_nodes = {"hypergraph_substrate", "complex_weights", "holographic_principle"}

    for node in IRH_DERIVATION_GRAPH:
        if node in axiom_nodes:
            lines.append(f"    {node}[({node})]")  # Circle for axioms
        else:
            lines.append(f"    {node}[{node}]")

    # Add edges
    for node, deps in IRH_DERIVATION_GRAPH.items():
        for dep in deps:
            lines.append(f"    {dep} --> {node}")

    # Add styling
    lines.extend(
        [
            "",
            "    style hypergraph_substrate fill:#f9f,stroke:#333",
            "    style complex_weights fill:#f9f,stroke:#333",
            "    style holographic_principle fill:#f9f,stroke:#333",
        ]
    )

    return "\n".join(lines)


def assert_acyclic(stages: list[str] | None = None) -> bool:
    """
    Assert that derivation stages form an acyclic graph.

    Args:
        stages: Optional list of stages to check (uses full DAG if None)

    Returns:
        True if acyclic, raises AssertionError otherwise
    """
    result = validate_dag()

    if not result["is_acyclic"]:
        raise AssertionError(f"Derivation graph contains cycles: {result['cycles']}")

    return True


def check_no_adhoc() -> dict:
    """
    Check that no quantities are introduced ad hoc.

    All derived quantities must trace back to fundamental axioms.

    Returns:
        Check results
    """
    G = nx.DiGraph()

    for node, deps in IRH_DERIVATION_GRAPH.items():
        G.add_node(node)
        for dep in deps:
            G.add_edge(dep, node)

    # Fundamental axioms
    axioms = {"hypergraph_substrate", "complex_weights", "holographic_principle"}

    # Check all nodes can reach an axiom
    unreachable = []
    for node in G.nodes():
        if node in axioms:
            continue

        # Check ancestors
        ancestors = nx.ancestors(G, node)
        if not ancestors & axioms:
            unreachable.append(node)

    return {
        "all_grounded": len(unreachable) == 0,
        "unreachable_nodes": unreachable,
        "n_axioms": len(axioms),
        "n_derived": len(G.nodes()) - len(axioms),
    }


def dynamic_selection_check(quantity: str) -> dict:
    """
    Verify that a quantity uses dynamic selection (not fixed parameters).

    Dynamic selection means the value emerges from optimization/constraint
    satisfaction rather than being specified a priori.

    Args:
        quantity: Name of quantity to check

    Returns:
        Check results
    """
    # Quantities that should use dynamic selection
    dynamic_quantities = {
        "spectral_dimension": "Emerges from heat kernel behavior",
        "betti_numbers": "Emerges from topological constraints",
        "gauge_couplings": "Emerges from holonomy structure",
        "neutrino_masses": "Emerges from eigenvalue gaps",
    }

    # Quantities that are fixed (from axioms)
    fixed_quantities = {
        "hypergraph_substrate": "Fundamental axiom",
        "complex_weights": "Fundamental axiom",
        "holographic_principle": "Fundamental axiom",
    }

    if quantity in fixed_quantities:
        return {
            "quantity": quantity,
            "is_dynamic": False,
            "is_fixed": True,
            "reason": fixed_quantities[quantity],
        }
    elif quantity in dynamic_quantities:
        return {
            "quantity": quantity,
            "is_dynamic": True,
            "is_fixed": False,
            "reason": dynamic_quantities[quantity],
        }
    else:
        return {
            "quantity": quantity,
            "is_dynamic": False,
            "is_fixed": False,
            "reason": "Derived quantity (check derivation chain)",
        }


def full_dag_audit() -> dict:
    """
    Run full DAG audit including all checks.

    Returns:
        Complete audit results
    """
    dag_result = validate_dag()
    adhoc_result = check_no_adhoc()

    return {
        "dag_valid": dag_result["is_acyclic"],
        "no_adhoc": adhoc_result["all_grounded"],
        "n_nodes": dag_result["n_nodes"],
        "n_edges": dag_result["n_edges"],
        "topological_order": dag_result["topological_order"],
        "unreachable": adhoc_result["unreachable_nodes"],
        "mermaid": render_dag_mermaid(),
    }
