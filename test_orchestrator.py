#!/usr/bin/env python3
"""
test_orchestrator.py - Comprehensive tests for orchestrator.py

This script validates the orchestrator functionality without requiring
external dependencies or interactive input.
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# Add repository to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import (
    EnvironmentDetector,
    ErrorAnalyzer,
    ConfigurationWizard,
    WolframIntegration,
    DEFAULT_CONFIG
)

def test_environment_detector():
    """Test environment detection functionality."""
    print("\n" + "="*80)
    print("TEST: Environment Detector")
    print("="*80)
    
    detector = EnvironmentDetector()
    
    print(f"✓ Is Colab: {detector.is_colab()}")
    print(f"✓ Is Windows: {detector.is_windows()}")
    print(f"✓ Is Linux: {detector.is_linux()}")
    print(f"✓ Is Mac: {detector.is_mac()}")
    print(f"✓ Has wolframscript: {detector.has_wolframscript()}")
    print(f"✓ Environment Name: {detector.get_environment_name()}")
    
    # Validate that exactly one OS is detected
    os_count = sum([
        detector.is_windows(),
        detector.is_linux(),
        detector.is_mac()
    ])
    assert os_count == 1, "Exactly one OS should be detected"
    
    print("\n✓ Environment Detector: PASSED")
    return True

def test_error_analyzer():
    """Test error analysis and crash report generation."""
    print("\n" + "="*80)
    print("TEST: Error Analyzer")
    print("="*80)
    
    analyzer = ErrorAnalyzer("Test Context")
    
    # Test with a simulated error
    try:
        raise ModuleNotFoundError("No module named 'test_module'")
    except Exception as e:
        # Test crash report generation
        report = analyzer.generate_crash_report(
            e, 
            "Test Module",
            {"test_param": 123}
        )
        
        # Validate report contents
        assert "IRH PHYSICS SUITE - CRASH REPORT" in report
        assert "ModuleNotFoundError" in report
        assert "test_module" in report
        assert "SUGGESTED FIXES" in report
        assert "Test Module" in report
        
        print("✓ Crash report generated successfully")
        
        # Test saving
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_file = f.name
        
        analyzer.save_crash_report(report, temp_file)
        assert os.path.exists(temp_file), "Crash report file should exist"
        
        # Cleanup
        os.unlink(temp_file)
        print("✓ Crash report saved and validated")
    
    # Test suggestion generation
    suggestions = analyzer.generate_suggested_fix(
        "MemoryError",
        "Unable to allocate array"
    )
    assert "Lower the grid size N" in suggestions
    print("✓ Suggestion generation works")
    
    print("\n✓ Error Analyzer: PASSED")
    return True

def test_configuration_wizard():
    """Test configuration wizard functionality."""
    print("\n" + "="*80)
    print("TEST: Configuration Wizard")
    print("="*80)
    
    # Test non-interactive mode
    wizard = ConfigurationWizard()
    config = wizard.run(skip_interactive=True)
    
    # Validate default configuration
    assert config["grid_size_N"] == DEFAULT_CONFIG["grid_size_N"]
    assert config["run_gtec"] == DEFAULT_CONFIG["run_gtec"]
    assert config["run_ncgg"] == DEFAULT_CONFIG["run_ncgg"]
    assert "output_verbosity" in config
    
    print("✓ Non-interactive wizard works")
    
    # Test saving
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_file = f.name
    
    wizard.save_config(temp_file)
    assert os.path.exists(temp_file), "Config file should exist"
    
    # Test loading
    loaded_config = ConfigurationWizard.load_config(temp_file)
    assert loaded_config == config, "Loaded config should match saved config"
    
    print("✓ Save/load configuration works")
    
    # Cleanup
    os.unlink(temp_file)
    
    # Test with existing config
    existing = {"grid_size_N": 500, "run_gtec": False}
    wizard2 = ConfigurationWizard(existing)
    config2 = wizard2.run(skip_interactive=True)
    assert config2["grid_size_N"] == 500
    assert config2["run_gtec"] == False
    
    print("✓ Existing config handling works")
    
    print("\n✓ Configuration Wizard: PASSED")
    return True

def test_wolfram_integration():
    """Test Wolfram Language asset generation."""
    print("\n" + "="*80)
    print("TEST: Wolfram Integration")
    print("="*80)
    
    # Change to temp directory
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)
    
    try:
        # Generate assets
        WolframIntegration.generate_wolfram_assets()
        
        # Validate .wls file
        assert os.path.exists("irh_wolfram_kernel.wls"), "WLS file should exist"
        with open("irh_wolfram_kernel.wls", 'r') as f:
            wls_content = f.read()
            assert "(* ::Package:: *)" in wls_content
            assert "IRH ARO Kernel" in wls_content
            assert "Eigenvalues" in wls_content or "Eigensystem" in wls_content
        
        print("✓ Wolfram script (.wls) generated")
        
        # Validate notebook prompt file
        assert os.path.exists("wolfram_notebook_prompt.txt"), "Prompt file should exist"
        with open("wolfram_notebook_prompt.txt", 'r') as f:
            prompt_content = f.read()
            assert "PROMPT FOR LLM-ENABLED WOLFRAM NOTEBOOK" in prompt_content
            assert "ARO" in prompt_content
            assert "Mathematica" in prompt_content
        
        print("✓ Notebook prompt generated")
        
    finally:
        # Cleanup
        os.chdir(original_dir)
        import shutil
        shutil.rmtree(temp_dir)
    
    print("\n✓ Wolfram Integration: PASSED")
    return True

def test_config_validation():
    """Test configuration validation and edge cases."""
    print("\n" + "="*80)
    print("TEST: Configuration Validation")
    print("="*80)
    
    # Test all config keys exist
    wizard = ConfigurationWizard()
    config = wizard.run(skip_interactive=True)
    
    required_keys = [
        "grid_size_N",
        "run_gtec",
        "run_ncgg",
        "run_cosmology",
        "output_verbosity",
        "max_iterations",
        "precision",
        "use_gpu",
        "output_dir"
    ]
    
    for key in required_keys:
        assert key in config, f"Config should have key: {key}"
    
    print("✓ All required config keys present")
    
    # Test value types
    assert isinstance(config["grid_size_N"], int)
    assert isinstance(config["run_gtec"], bool)
    assert isinstance(config["output_verbosity"], str)
    
    print("✓ Config value types are correct")
    
    print("\n✓ Configuration Validation: PASSED")
    return True

def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ORCHESTRATOR TEST SUITE")
    print("="*80)
    
    tests = [
        test_environment_detector,
        test_error_analyzer,
        test_configuration_wizard,
        test_wolfram_integration,
        test_config_validation,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"\n✗ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
