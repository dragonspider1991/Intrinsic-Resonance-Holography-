"""
Test suite for Phase 8: Final Validation

Tests the comprehensive validation suite and final documentation infrastructure.
"""
import pytest
import os
import json
from experiments.validation_suite import ValidationSuite


class TestValidationSuite:
    """Test validation suite functionality."""
    
    def test_validation_suite_initialization(self):
        """Test ValidationSuite can be initialized."""
        suite = ValidationSuite()
        
        assert suite is not None
        assert 'config' in suite.results
        assert 'timestamp' in suite.results['config']
        assert suite.results['config']['version'] == '15.0.0'
    
    def test_validate_fine_structure_constant(self):
        """Test fine structure constant validation."""
        suite = ValidationSuite()
        
        check = suite.validate_fine_structure_constant()
        
        # Should return validation check result
        assert isinstance(check, dict)
        assert 'quantity' in check
        assert 'predicted' in check
        assert 'experimental' in check
        assert 'pass' in check
        
        # Should pass (exact match expected)
        assert check['pass'] is True
    
    def test_validate_fermion_mass_ratios(self):
        """Test fermion mass ratio validation."""
        suite = ValidationSuite()
        
        check = suite.validate_fermion_mass_ratios()
        
        # Should return validation check
        assert isinstance(check, dict)
        assert 'quantity' in check
        assert 'predicted' in check
        assert 'experimental' in check
        
        # Should have reasonable values
        assert 200 < check['predicted'] < 210
        assert 200 < check['experimental'] < 210
    
    def test_validate_cosmological_constant(self):
        """Test cosmological constant validation."""
        suite = ValidationSuite()
        
        check = suite.validate_cosmological_constant()
        
        # Should return validation check
        assert isinstance(check, dict)
        assert 'quantity' in check
        assert 'predicted' in check
        
        # Should have target value
        assert abs(check['predicted'] - (-120.45)) < 0.1
    
    def test_validate_dark_energy_eos(self):
        """Test dark energy equation of state validation."""
        suite = ValidationSuite()
        
        check = suite.validate_dark_energy_eos()
        
        # Should return validation check
        assert isinstance(check, dict)
        assert 'quantity' in check
        assert 'predicted' in check
        assert 'experimental' in check
        
        # w₀ should be close to -1
        assert -1.2 < check['predicted'] < -0.7
        assert -1.2 < check['experimental'] < -0.7
    
    def test_run_all_validations(self):
        """Test complete validation suite run."""
        suite = ValidationSuite()
        
        results = suite.run_all_validations()
        
        # Should return complete results
        assert isinstance(results, dict)
        assert 'validation' in results
        
        val = results['validation']
        assert 'n_total' in val
        assert 'n_pass' in val
        assert 'pass_rate' in val
        assert 'status' in val
        assert 'grade' in val
        
        # Should have run multiple checks
        assert val['n_total'] >= 4
    
    def test_validation_checks_recorded(self):
        """Test that validation checks are recorded."""
        suite = ValidationSuite()
        suite.run_all_validations()
        
        # Should have recorded checks
        assert len(suite.checks) > 0
        
        # Each check should have required fields
        for check in suite.checks:
            assert 'quantity' in check
            assert 'predicted' in check
    
    def test_save_results(self, tmp_path):
        """Test saving validation results to JSON."""
        suite = ValidationSuite()
        suite.run_all_validations()
        
        output_file = tmp_path / "validation_results.json"
        suite.save_results(str(output_file))
        
        # File should exist
        assert output_file.exists()
        
        # Should be valid JSON
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert 'validation' in data
        assert 'config' in data
    
    def test_generate_report(self, tmp_path):
        """Test generating validation report."""
        suite = ValidationSuite()
        suite.run_all_validations()
        
        output_file = tmp_path / "validation_report.md"
        suite.generate_report(str(output_file))
        
        # File should exist
        assert output_file.exists()
        
        # Should be markdown
        with open(output_file, 'r') as f:
            content = f.read()
        
        assert '# IRH v15.0 Validation Report' in content
        assert 'Status' in content
        assert 'Grade' in content
    
    def test_compute_grade(self):
        """Test grade computation from pass rate."""
        suite = ValidationSuite()
        
        # Test various pass rates
        assert suite._compute_grade(1.0) == 'A+'
        assert suite._compute_grade(0.95) == 'A'
        assert suite._compute_grade(0.90) == 'A-'
        assert suite._compute_grade(0.85) == 'B'
        assert suite._compute_grade(0.80) == 'B-'
        assert suite._compute_grade(0.70) == 'C or lower'


class TestDocumentation:
    """Test that documentation files exist and are valid."""
    
    def test_replication_guide_exists(self):
        """Test that replication guide exists."""
        guide_path = 'docs/REPLICATION_GUIDE.md'
        
        assert os.path.exists(guide_path)
        
        # Should have content
        with open(guide_path, 'r') as f:
            content = f.read()
        
        assert len(content) > 1000
        assert 'Replication Guide' in content
        assert 'Hardware Requirements' in content
        assert 'Installation' in content
    
    def test_replication_guide_structure(self):
        """Test replication guide has proper structure."""
        guide_path = 'docs/REPLICATION_GUIDE.md'
        
        with open(guide_path, 'r') as f:
            content = f.read()
        
        # Should have key sections
        assert '## Overview' in content
        assert '## Quick Start' in content
        assert '## Hardware Requirements' in content
        assert '## Software Dependencies' in content
        assert '## Installation' in content
        assert '## Replication Steps' in content
        assert '## Expected Results' in content
        assert '## Troubleshooting' in content


class TestIntegration:
    """Integration tests for Phase 8."""
    
    def test_full_validation_pipeline(self, tmp_path):
        """Test complete validation pipeline."""
        # Initialize suite
        suite = ValidationSuite()
        
        # Run validations
        results = suite.run_all_validations()
        
        # Save outputs
        json_file = tmp_path / "results.json"
        md_file = tmp_path / "report.md"
        
        suite.save_results(str(json_file))
        suite.generate_report(str(md_file))
        
        # Verify outputs exist
        assert json_file.exists()
        assert md_file.exists()
        
        # Verify results structure
        assert results['validation']['n_total'] > 0
        assert results['validation']['status'] in ['PASS', 'FAIL']
        assert 'grade' in results['validation']


@pytest.mark.slow
class TestLargeScaleValidation:
    """Tests for larger-scale validation (marked as slow)."""
    
    def test_validation_with_larger_dataset(self):
        """Test validation can handle larger checks."""
        suite = ValidationSuite()
        
        # Run base validations
        suite.run_all_validations()
        
        # Should handle multiple checks
        assert len(suite.checks) >= 4
        
        # Each check should be valid
        for check in suite.checks:
            assert isinstance(check, dict)
            assert 'quantity' in check
