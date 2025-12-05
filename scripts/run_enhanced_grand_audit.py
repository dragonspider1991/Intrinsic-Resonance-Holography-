#!/usr/bin/env python3
"""
Enhanced Grand Audit Script for IRH v9.2

This script provides a comprehensive standalone version of the grand audit
that can be run directly from the command line without requiring a notebook.

Usage:
    python scripts/run_enhanced_grand_audit.py --N 64 --quick
    python scripts/run_enhanced_grand_audit.py --N 256 --full --output results/

Features:
    - Comprehensive validation across all four pillars
    - Convergence testing across multiple network sizes
    - Detailed visualizations and exports
    - JSON and CSV output formats
    - Progress tracking and timing
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Add IRH package to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "python" / "src"))

from irh.graph_state import HyperGraph
from irh.grand_audit import grand_audit, ci_convergence_test, GrandAuditReport
from irh.spectral_dimension import SpectralDimension
from irh.predictions.constants import predict_alpha_inverse


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def print_section(title: str) -> None:
    """Print a formatted subsection."""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")


def display_audit_results(report: GrandAuditReport) -> None:
    """Display comprehensive audit results organized by pillar."""
    print_header("GRAND AUDIT RESULTS")
    
    print(f"\nTimestamp: {report.timestamp}")
    print(f"Version: {report.version}")
    print(f"\n📈 Overall Score: {report.pass_count}/{report.total_checks} checks passed ({report.summary['pass_rate']*100:.1f}%)")
    
    # Group results by pillar
    pillars = {
        "Ontological": [],
        "Mathematical": [],
        "Empirical": [],
        "Logical": []
    }
    
    for result in report.results:
        for pillar_name in pillars.keys():
            if pillar_name in result.name:
                pillars[pillar_name].append(result)
                break
    
    # Display results by pillar
    for pillar_name, checks in pillars.items():
        print_section(f"{pillar_name.upper()} PILLAR")
        
        for check in checks:
            status = "✅ PASS" if check.passed else "❌ FAIL"
            print(f"\n{status} - {check.name}")
            print(f"  Value: {check.value}")
            print(f"  Target: {check.target}")
            if check.tolerance is not None:
                print(f"  Tolerance: ±{check.tolerance}")
            print(f"  Message: {check.message}")
    
    # Summary by pillar
    print_section("SUMMARY BY PILLAR")
    print(f"  Ontological:  {report.summary['ontological']} checks passed")
    print(f"  Mathematical: {report.summary['mathematical']} checks passed")
    print(f"  Empirical:    {report.summary['empirical']} checks passed")
    print(f"  Logical:      {report.summary['logical']} checks passed")


def display_convergence_results(convergence_results: dict) -> None:
    """Display convergence analysis results."""
    print_header("CONVERGENCE ANALYSIS")
    
    print(f"\nTesting N values: {[r['N'] for r in convergence_results['results']]}")
    
    print("\n📊 CONVERGENCE TABLE")
    print("─" * 80)
    print(f"{'N':>8} | {'d_s':>10} | {'Error':>10} | {'Fit Quality':>12}")
    print("─" * 80)
    
    for result in convergence_results['results']:
        print(f"{result['N']:>8} | {result['d_s']:>10.4f} | "
              f"{result['d_s_error']:>10.4f} | {result['fit_quality']:>12.4f}")
    
    print("─" * 80)
    print(f"\nConverging to d=4: {'✅ YES' if convergence_results['converging'] else '❌ NO'}")
    print(f"Number of values tested: {convergence_results['n_values_tested']}")


def create_visualizations(
    report: GrandAuditReport,
    convergence_results: dict,
    output_dir: Path
) -> None:
    """Create comprehensive visualizations."""
    print_section("CREATING VISUALIZATIONS")
    
    # Visualization 1: Pillar Scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    pillar_names = ['Ontological', 'Mathematical', 'Empirical', 'Logical']
    pillar_scores = [
        report.summary['ontological'],
        report.summary['mathematical'],
        report.summary['empirical'],
        report.summary['logical']
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax1.bar(pillar_names, pillar_scores, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=2)
    ax1.set_ylabel('Checks Passed', fontsize=12, fontweight='bold')
    ax1.set_title('Validation Results by Pillar', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(pillar_scores) + 1)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Pie chart
    passed = report.pass_count
    failed = report.total_checks - report.pass_count
    sizes = [passed, failed]
    labels = [f'Passed\n({passed})', f'Failed\n({failed})']
    colors_pie = ['#2ca02c', '#d62728']
    explode = (0.05, 0)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
            autopct='%1.1f%%', shadow=True, startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title(f'Overall Validation Score\n({report.pass_count}/{report.total_checks})',
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    viz_path = output_dir / "pillar_scores.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {viz_path}")
    plt.close()
    
    # Visualization 2: Convergence
    if len(convergence_results['results']) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        n_values = [r['N'] for r in convergence_results['results']]
        ds_values = [r['d_s'] for r in convergence_results['results']]
        ds_errors = [r['d_s_error'] for r in convergence_results['results']]
        
        ax1.errorbar(n_values, ds_values, yerr=ds_errors,
                    fmt='o-', linewidth=2, markersize=10,
                    capsize=5, capthick=2, color='#1f77b4', label='d_s')
        ax1.axhline(y=4.0, color='red', linestyle='--', linewidth=2, label='Target (d=4)')
        ax1.fill_between(n_values, 3.5, 4.5, alpha=0.2, color='green', label='±0.5 tolerance')
        ax1.set_xlabel('Network Size (N)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Spectral Dimension (d_s)', fontsize=12, fontweight='bold')
        ax1.set_title('Spectral Dimension Convergence', fontsize=14, fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        distances = [abs(ds - 4.0) for ds in ds_values]
        ax2.semilogy(n_values, distances, 'o-', linewidth=2, markersize=10, color='#ff7f0e')
        ax2.set_xlabel('Network Size (N)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('|d_s - 4.0|', fontsize=12, fontweight='bold')
        ax2.set_title('Convergence Error (log scale)', fontsize=14, fontweight='bold')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        conv_path = output_dir / "convergence.png"
        plt.savefig(conv_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {conv_path}")
        plt.close()


def export_results(
    report: GrandAuditReport,
    convergence_results: dict,
    graph: HyperGraph,
    audit_duration: float,
    output_dir: Path
) -> None:
    """Export results to JSON and CSV formats."""
    print_section("EXPORTING RESULTS")
    
    # Create export data
    export_data = {
        "metadata": {
            "timestamp": report.timestamp,
            "version": report.version,
            "network_size": graph.N,
            "seed": graph.metadata.seed,
            "audit_duration_seconds": audit_duration
        },
        "summary": {
            "total_checks": report.total_checks,
            "passed_checks": report.pass_count,
            "pass_rate": report.summary['pass_rate'],
            "ontological_passed": report.summary['ontological'],
            "mathematical_passed": report.summary['mathematical'],
            "empirical_passed": report.summary['empirical'],
            "logical_passed": report.summary['logical']
        },
        "results": [
            {
                "name": r.name,
                "passed": r.passed,
                "value": str(r.value),
                "target": str(r.target),
                "tolerance": r.tolerance,
                "message": r.message
            }
            for r in report.results
        ],
        "convergence": convergence_results
    }
    
    # Save JSON
    json_path = output_dir / f"grand_audit_N{graph.N}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"  ✓ Saved JSON: {json_path}")
    
    # Save summary text report
    txt_path = output_dir / f"grand_audit_summary_N{graph.N}.txt"
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("IRH v9.2 GRAND AUDIT SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nTimestamp: {report.timestamp}\n")
        f.write(f"Network size: N={graph.N}\n")
        f.write(f"Audit duration: {audit_duration:.1f} seconds\n")
        f.write(f"\nOverall Score: {report.pass_count}/{report.total_checks} ({report.summary['pass_rate']*100:.1f}%)\n")
        f.write("\nPillar Scores:\n")
        f.write(f"  Ontological:  {report.summary['ontological']}\n")
        f.write(f"  Mathematical: {report.summary['mathematical']}\n")
        f.write(f"  Empirical:    {report.summary['empirical']}\n")
        f.write(f"  Logical:      {report.summary['logical']}\n")
    print(f"  ✓ Saved summary: {txt_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Enhanced Grand Audit for IRH v9.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick test (N=64):
    python scripts/run_enhanced_grand_audit.py --quick
  
  Comprehensive audit (N=256):
    python scripts/run_enhanced_grand_audit.py --N 256 --full
  
  Custom configuration:
    python scripts/run_enhanced_grand_audit.py --N 128 --convergence 64,128,256 --output results/
        """
    )
    
    parser.add_argument("--N", type=int, default=64,
                       help="Network size (default: 64)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test mode (N=64, convergence=[32,64,128])")
    parser.add_argument("--full", action="store_true",
                       help="Full audit mode (N=256, convergence=[64,128,256,512])")
    parser.add_argument("--convergence", type=str, default=None,
                       help="Comma-separated list of N values for convergence test (e.g., '64,128,256')")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output", type=str, default="audit_results",
                       help="Output directory for results (default: audit_results)")
    parser.add_argument("--no-viz", action="store_true",
                       help="Skip visualization generation")
    
    args = parser.parse_args()
    
    # Configure based on mode
    if args.quick:
        N = 64
        convergence_n_values = [32, 64, 128]
        print("🚀 Quick test mode enabled")
    elif args.full:
        N = 256
        convergence_n_values = [64, 128, 256, 512]
        print("🔬 Full audit mode enabled")
    else:
        N = args.N
        if args.convergence:
            convergence_n_values = [int(x.strip()) for x in args.convergence.split(',')]
        else:
            convergence_n_values = [N // 2, N, N * 2] if N >= 32 else [32, 64, 128]
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_header("IRH v9.2 ENHANCED GRAND AUDIT")
    print(f"\nConfiguration:")
    print(f"  Network size: N = {N}")
    print(f"  Random seed: {args.seed}")
    print(f"  Convergence test N values: {convergence_n_values}")
    print(f"  Output directory: {output_dir}")
    
    # Step 1: Create hypergraph
    print_section("Step 1: Creating Hypergraph Substrate")
    graph = HyperGraph(N=N, seed=args.seed, topology="Random", edge_probability=0.3)
    print(f"  ✓ Created HyperGraph with N={graph.N} nodes")
    print(f"    Number of edges: {graph.edge_count}")
    print(f"    Topology: {graph.metadata.topology}")
    
    # Step 2: Run grand audit
    print_section("Step 2: Running Grand Audit")
    audit_start = datetime.now()
    report = grand_audit(graph, output_dir=str(output_dir))
    audit_duration = (datetime.now() - audit_start).total_seconds()
    print(f"  ✓ Audit completed in {audit_duration:.1f} seconds")
    
    # Step 3: Display results
    display_audit_results(report)
    
    # Step 4: Convergence analysis
    print_section("Step 4: Running Convergence Analysis")
    convergence_results = ci_convergence_test(n_values=convergence_n_values)
    display_convergence_results(convergence_results)
    
    # Step 5: Create visualizations
    if not args.no_viz:
        create_visualizations(report, convergence_results, output_dir)
    
    # Step 6: Export results
    export_results(report, convergence_results, graph, audit_duration, output_dir)
    
    # Final summary
    print_header("AUDIT COMPLETE")
    print(f"\n  Overall Score: {report.pass_count}/{report.total_checks} ({report.summary['pass_rate']*100:.1f}%)")
    print(f"  Duration: {audit_duration:.1f} seconds")
    print(f"  Results saved to: {output_dir}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
