#!/usr/bin/env python3
"""
Test script for the enhanced grand audit functionality.

This script performs a quick validation that:
1. The enhanced grand_audit module can be imported
2. All new validation checks are registered
3. The audit can run on a small test graph
4. Results are properly structured
"""

import sys
from pathlib import Path

# Add IRH to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root / "python" / "src"))

from irh.graph_state import HyperGraph
from irh.grand_audit import grand_audit, GrandAuditReport


def test_enhanced_audit():
    """Test the enhanced grand audit functionality."""
    print("=" * 80)
    print("TESTING ENHANCED GRAND AUDIT")
    print("=" * 80)
    
    # Create a small test graph
    print("\n1. Creating test graph (N=32)...")
    graph = HyperGraph(N=32, seed=42, topology="Random", edge_probability=0.3)
    print(f"   ✓ Graph created: {graph.N} nodes, {graph.edge_count} edges")
    
    # Run the audit
    print("\n2. Running enhanced grand audit...")
    try:
        report = grand_audit(graph)
        print(f"   ✓ Audit completed successfully")
    except Exception as e:
        print(f"   ✗ Audit failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check report structure
    print("\n3. Validating report structure...")
    assert isinstance(report, GrandAuditReport), "Report is not a GrandAuditReport"
    assert report.total_checks > 0, "No checks were run"
    assert len(report.results) == report.total_checks, "Mismatch in results count"
    print(f"   ✓ Report structure valid")
    print(f"   Total checks: {report.total_checks}")
    print(f"   Checks passed: {report.pass_count}")
    
    # Verify enhanced checks are present
    print("\n4. Verifying enhanced validation checks...")
    check_names = [r.name for r in report.results]
    
    # Count checks by pillar
    ontological_count = sum(1 for name in check_names if "Ontological" in name)
    mathematical_count = sum(1 for name in check_names if "Mathematical" in name)
    empirical_count = sum(1 for name in check_names if "Empirical" in name)
    logical_count = sum(1 for name in check_names if "Logical" in name)
    
    print(f"   Ontological checks: {ontological_count} (expected: ≥6)")
    print(f"   Mathematical checks: {mathematical_count} (expected: ≥4)")
    print(f"   Empirical checks: {empirical_count} (expected: ≥6)")
    print(f"   Logical checks: {logical_count} (expected: ≥6)")
    
    # Verify minimum check counts (enhanced version should have more)
    assert ontological_count >= 6, f"Expected ≥6 ontological checks, got {ontological_count}"
    assert mathematical_count >= 4, f"Expected ≥4 mathematical checks, got {mathematical_count}"
    assert empirical_count >= 6, f"Expected ≥6 empirical checks, got {empirical_count}"
    assert logical_count >= 6, f"Expected ≥6 logical checks, got {logical_count}"
    
    print(f"   ✓ All enhanced checks present")
    
    # Display sample results
    print("\n5. Sample validation results:")
    for pillar in ["Ontological", "Mathematical", "Empirical", "Logical"]:
        pillar_checks = [r for r in report.results if pillar in r.name]
        if pillar_checks:
            print(f"\n   {pillar} Pillar:")
            for check in pillar_checks[:2]:  # Show first 2 checks from each pillar
                status = "✅" if check.passed else "❌"
                print(f"     {status} {check.name}")
                print(f"        Value: {check.value}, Target: {check.target}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Overall pass rate: {report.pass_count}/{report.total_checks} ({report.summary['pass_rate']*100:.1f}%)")
    print(f"Ontological:  {report.summary['ontological']} passed")
    print(f"Mathematical: {report.summary['mathematical']} passed")
    print(f"Empirical:    {report.summary['empirical']} passed")
    print(f"Logical:      {report.summary['logical']} passed")
    print("=" * 80)
    print("\n✓ ALL TESTS PASSED")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = test_enhanced_audit()
    sys.exit(0 if success else 1)
