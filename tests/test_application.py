"""
Application mode tests for the GMOO wrapper.

These tests verify the inverse optimization functionality including
loading models, setting objectives, and finding solutions.
"""

import pytest
import numpy as np
import os
import logging
from numpy.linalg import norm
from gmoo_sdk.dll_interface import GMOOAPI

from conftest import develop_model, perform_inverse_optimization

# Get logger
logger = logging.getLogger("gmoo_test")

def test_simple_inverse_optimization(simple_model):
    """Test basic inverse optimization on a simple model."""
    # Develop the model first
    gmoo_file = develop_model(simple_model)
    
    # Define a target output
    truth_case = np.array([5.0, 5.0, 5.0])
    target_outputs = simple_model.model_function(truth_case)
    
    # Run inverse optimization
    result = perform_inverse_optimization(
        model=simple_model,
        target_outputs=target_outputs,
        max_iterations=40
    )
    
    # Verify that we found a solution
    assert result['inputs'] is not None, "Should find a solution"
    
    # Calculate error metrics
    output_error = norm(result['outputs'] - target_outputs, ord=2)
    relative_error = output_error / norm(target_outputs, ord=2)
    
    # Calculate input error to truth case (this is informational)
    input_error = norm(result['inputs'] - truth_case, ord=2)
    
    # Check that the solution is reasonably good
    assert relative_error < 0.2, f"Solution error {relative_error} is too large"
    
    # Clean up - already handled by fixture

@pytest.mark.parametrize('objective_types', [
    [1, 1, 1],  # All percentage error
    [2, 2, 2],  # All absolute error
    [1, 2, 1],  # Mixed error types
])
def test_objective_types(simple_model, objective_types):
    """Test different objective types for inverse optimization."""
    # Develop the model first
    gmoo_file = develop_model(simple_model)
    
    # Define a target output
    truth_case = np.array([5.0, 5.0, 5.0])
    target_outputs = simple_model.model_function(truth_case)
    
    # Run inverse optimization with specified objective types
    result = perform_inverse_optimization(
        model=simple_model,
        target_outputs=target_outputs,
        objective_types=objective_types,
        max_iterations=40
    )
    
    # Verify that we found a solution
    assert result['inputs'] is not None, "Should find a solution"
    
    # Calculate error metrics
    output_error = norm(result['outputs'] - target_outputs, ord=2)
    relative_error = output_error / norm(target_outputs, ord=2)
    
    # The acceptance threshold depends on the objective types
    if all(ot == 1 for ot in objective_types):
        # Percentage error tends to work best for this test case
        max_allowed_error = 0.2
    else:
        # Allow higher error for absolute and mixed types
        max_allowed_error = 0.3
        
    # Check that the solution is reasonably good
    assert relative_error < max_allowed_error, f"Solution error {relative_error} is too large"

def test_rescoping(complex_model, request):
    """Test the rescoping capability to narrow the search space."""
    # Develop the model first
    gmoo_file = develop_model(complex_model)
    
    # Define a target near the Rosenbrock minimum
    truth_case = np.array([1.1, 0.9, 1.05, 0.95])
    target_outputs = complex_model.model_function(truth_case)
    
    # Run initial inverse optimization
    result1 = perform_inverse_optimization(
        model=complex_model,
        target_outputs=target_outputs,
        max_iterations=30
    )
    
    # Get error metrics for the first solution
    output_error1 = norm(result1['outputs'] - target_outputs, ord=2)
    relative_error1 = output_error1 / norm(target_outputs, ord=2)
    
    # Use the solution to rescope the problem
    # Create a tighter search space around the best solution
    # Add some margin to ensure the optimal solution is within the bounds
    min_values = result1['inputs'] - 0.2
    max_values = result1['inputs'] + 0.2
    
    # Ensure the bounds are within the original problem bounds
    min_values = np.maximum(min_values, complex_model.aVarLimMin)
    max_values = np.minimum(max_values, complex_model.aVarLimMax)
    
    # Create a new model with the rescoped search space
    rescoped_model_params = {
        'filename': 'test_rescoped',
        'var_mins': min_values.tolist(),
        'var_maxs': max_values.tolist(),
        'num_input_vars': len(min_values),
        'num_output_vars': len(target_outputs),
        'model_function': complex_model.model_function,
    }
    
    # Create rescoped model - Directly instead of using fixture
    loaded_dll = request.getfixturevalue('loaded_dll')
    rescoped_model = GMOOAPI(
        vsme_windll=loaded_dll,
        vsme_input_filename='test_rescoped',
        var_mins=min_values.tolist(),
        var_maxs=max_values.tolist(),
        num_output_vars=len(target_outputs),
        model_function=complex_model.model_function,
        save_file_dir=os.path.abspath('.')
    )
    
    # Develop the rescoped model
    rescoped_gmoo_file = develop_model(rescoped_model)
    
    # Run inverse optimization on the rescoped model
    result2 = perform_inverse_optimization(
        model=rescoped_model,
        target_outputs=target_outputs,
        max_iterations=20  # Fewer iterations needed for rescoped problem
    )
    
    # Get error metrics for the second solution
    output_error2 = norm(result2['outputs'] - target_outputs, ord=2)
    relative_error2 = output_error2 / norm(target_outputs, ord=2)
    
    # Either the rescoped solution should be better, or both should be good
    assert (relative_error2 < relative_error1 or 
            relative_error2 < 0.15), "Rescoped solution should improve or be good"
    
    # Enhanced cleanup for both models
    logger.info("=== Enhanced cleanup for test_rescoping ===")
    
    # First ensure all resources are released
    try:
        # Explicitly log the test name and model filenames for debugging
        model_name = complex_model.vsme_input_filename
        logger.info(f"Running test_rescoping with model: {model_name}")
        
        # Unload both models with explicit error handling
        logger.info("Unloading complex_model development")
        complex_model.development.unload_vsme()
        
        logger.info("Unloading complex_model application")
        complex_model.application.unload_vsme()
        
        logger.info("Unloading rescoped_model development")
        rescoped_model.development.unload_vsme()
        
        logger.info("Unloading rescoped_model application")
        rescoped_model.application.unload_vsme()
        
        # Force garbage collection
        import gc
        logger.info("Forcing garbage collection")
        gc.collect()
        
        # Get all possible file paths
        complex_model_file = f"{complex_model.vsme_input_filename}.gmoo"
        test_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            complex_model_file,                                       # Relative to current directory
            os.path.abspath(complex_model_file),                      # Full path
            os.path.basename(complex_model_file),                     # Just the filename
            os.path.join(test_dir, os.path.basename(complex_model_file)),  # In the tests directory
            os.path.join('.', os.path.basename(complex_model_file))  # Explicitly in current directory
        ]
        
        # Log all the paths we're checking
        for i, path in enumerate(possible_paths):
            logger.info(f"Path {i}: {path}, exists: {os.path.exists(path)}")
        
        # Try to remove the file using all possible paths
        for filepath in possible_paths:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logger.info(f"Successfully removed {filepath}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to remove {filepath}: {e}")
                    
        # Also try to specifically target test_complex_5.gmoo if it exists
        special_file = "test_complex_5.gmoo"
        if os.path.exists(special_file):
            try:
                os.remove(special_file)
                logger.info(f"Specifically removed {special_file}")
            except Exception as e:
                logger.warning(f"Failed to remove specific file {special_file}: {e}")
    except Exception as e:
        logger.error(f"Error during enhanced cleanup: {e}")
        
    # test_rescoped.gmoo is already handled in develop_model
    
    # Clean up - already handled by fixtures
