"""
Basic functionality tests for the GMOO wrapper.

These tests verify core functionality like DLL loading, model creation,
and basic operations.
"""

import pytest
import numpy as np
from numpy.linalg import norm
import os

def test_dll_loading(loaded_dll):
    """Test that the DLL loads correctly."""
    assert loaded_dll is not None, "DLL should be loaded"

def test_simple_model_creation(simple_model):
    """Test that a simple model can be created."""
    assert simple_model is not None, "Model should be created"
    assert simple_model.nVars.value == 3, "Should have 3 input variables"
    assert simple_model.nObjs.value == 3, "Should have 3 output variables"

def test_model_function(simple_model):
    """Test that the model function returns expected values."""
    test_input = [5.0, 5.0, 5.0]
    expected_output = np.array([625.0, 225.0, 625.0])
    actual_output = simple_model.model_function(test_input)
    
    np.testing.assert_array_almost_equal(
        actual_output, expected_output, 
        decimal=6, 
        err_msg="Model function does not return expected values"
    )

def test_development_initialization(simple_model):
    """Test that we can initialize the development process."""
    # Initialize development setup
    simple_model.development.load_vsme_name()
    simple_model.development.initialize_variables()
    simple_model.development.load_variable_types()
    simple_model.development.load_variable_limits()
    
    # If we get here without exceptions, the test passes
    assert True, "Development initialization should succeed"

def test_design_agents_and_cases(simple_model):
    """Test designing agents and cases for the model."""
    # Initialize development setup
    simple_model.development.load_vsme_name()
    simple_model.development.initialize_variables()
    simple_model.development.load_variable_types()
    simple_model.development.load_variable_limits()
    
    # Design agents and cases
    simple_model.development.design_agents()
    simple_model.development.design_cases()
    
    # Get and check the case count
    case_count = simple_model.development.get_case_count()
    assert case_count > 0, "Expected at least one case to be designed"
    
    # Test retrieving case variables for a few cases
    for i in range(1, min(3, case_count + 1)):
        case_vars = simple_model.development.poke_case_variables(i)
        
        # Check case variables against the expected limits
        for j, var in enumerate(case_vars):
            assert simple_model.aVarLimMin[j] <= var <= simple_model.aVarLimMax[j], \
                f"Case {i}, variable {j} ({var}) out of range [{simple_model.aVarLimMin[j]}, {simple_model.aVarLimMax[j]}]"
