"""
Integrated test for GMOO functionality with proper memory management.

This test demonstrates the full workflow with careful unloading of models
and DLL state management to avoid memory issues when run with other tests.
"""

import os
import ctypes
import numpy as np
import pytest
from numpy.linalg import norm
import time

from gmoo_sdk.dll_interface import GMOOAPI

@pytest.fixture(scope="module")
def vsme_dll():
    """
    Load the GMOO DLL with error handling.
    This fixture has module scope, so the DLL is loaded only once for all tests.
    """
    try:
        dll_path = os.environ.get('MOOLIB')
        if not dll_path:
            pytest.skip("MOOLIB environment variable is not set. Skipping test.")
            
        if not os.path.exists(dll_path):
            pytest.skip(f"DLL file {dll_path} does not exist. Skipping test.")
            
        # Handle Intel MPI paths if needed
        intel_redist = os.environ.get('I_MPI_ROOT')
        if intel_redist and os.path.exists(intel_redist):
            os.add_dll_directory(intel_redist)
        else:
            default_path = r'C:\Program Files (x86)\Common Files\Intel\Shared Libraries\intel64'
            try:
                if os.path.exists(default_path):
                    os.add_dll_directory(default_path)
            except:
                pass  # Ignore if it doesn't exist
            
        dll = ctypes.CDLL(dll_path)
        yield dll
        
        # No cleanup needed for the DLL itself
    except Exception as e:
        pytest.skip(f"Failed to load the DLL: {e}")

def simple_function(input_arr):
    """A simple test function."""
    input_arr = np.array(input_arr, ndmin=1)
    v01 = input_arr[0]
    v02 = input_arr[1]
    v03 = input_arr[2]
    
    o01 = v01 * v01 * v03 * v03
    o02 = (v01 - 2.0) * (v01 - 2.0) * v03 * v03
    o03 = v01 * v02 * v03 * v03

    return np.array([o01, o02, o03])

def test_model_basics(vsme_dll):
    """Test basic model creation and function evaluation."""
    # Create a unique filename to avoid conflicts with other tests
    filename = f"basic_test_{int(time.time())}"
    
    try:
        # Create model
        model = GMOOAPI(
            vsme_windll=vsme_dll,
            vsme_input_filename=filename,
            var_mins=[0.0, 0.0, 0.0],
            var_maxs=[10.0, 10.0, 10.0],
            num_output_vars=3,
            model_function=simple_function,
            save_file_dir=os.path.abspath('.')
        )
        
        # Verify model parameters
        assert model.nVars.value == 3, "Should have 3 input variables"
        assert model.nObjs.value == 3, "Should have 3 output variables"
        
        # Test model function
        test_input = [5.0, 5.0, 5.0]
        expected_output = np.array([625.0, 225.0, 625.0])
        actual_output = model.model_function(test_input)
        np.testing.assert_array_almost_equal(
            actual_output, expected_output, 
            decimal=6, 
            err_msg="Model function does not return expected values"
        )
        
        print("Model created and tested successfully")
    
    finally:
        # Ensure model is unloaded, even if the test fails
        if 'model' in locals():
            try:
                # Unload development mode if it was used
                if hasattr(model, 'development'):
                    model.development.unload_vsme()
                
                # Unload application mode if it was used
                if hasattr(model, 'application'):
                    model.application.unload_vsme()
            except:
                pass

def test_model_development(vsme_dll):
    """Test the model development process."""
    # Create a unique filename to avoid conflicts with other tests
    filename = f"dev_test_{int(time.time())}"
    gmoo_file = None
    
    try:
        # Create model
        model = GMOOAPI(
            vsme_windll=vsme_dll,
            vsme_input_filename=filename,
            var_mins=[0.0, 0.0, 0.0],
            var_maxs=[10.0, 10.0, 10.0],
            num_output_vars=3,
            model_function=simple_function,
            save_file_dir=os.path.abspath('.')
        )
        
        # Initialize development setup
        model.development.load_vsme_name()
        model.development.initialize_variables()
        model.development.load_variable_types()
        model.development.load_variable_limits()
        
        # Design agents and cases
        model.development.design_agents()
        model.development.design_cases()
        
        # Get case count
        case_count = model.development.get_case_count()
        assert case_count > 0, "Should have designed at least one case"
        
        print(f"Designed {case_count} cases")
        
        # Initialize the backup file
        model.development.init_backup_file()
        
        # Generate and evaluate cases without using the backup file
        input_cases = []
        output_cases = []
        
        for kk in range(1, case_count + 1):
            case_vars = model.development.poke_case_variables(kk)
            input_cases.append(case_vars)
            evaluation = model.model_function(case_vars)
            output_cases.append(evaluation)
        
        print(f"Evaluated {len(output_cases)} cases")
        
        # Initialize outcomes without reading backup
        model.development.initialize_outcomes()
        
        # Load each case result
        for kk in range(1, len(output_cases) + 1):
            model.development.load_case_results(kk, output_cases[kk-1])
        
        # Develop the VSME model
        model.development.develop_vsme()
        
        # Export VSME to file
        gmoo_file = model.development.export_vsme()
        assert os.path.exists(gmoo_file), f"GMOO file {gmoo_file} should exist"
        
        print(f"Model developed and exported to {gmoo_file}")
        
        # Make sure to unload the model before the function ends
        model.development.unload_vsme()
    
    finally:
        # Cleanup - make sure model is unloaded and file is removed
        if 'model' in locals():
            try:
                # Unload development mode
                model.development.unload_vsme()
            except:
                pass
                
        # Remove the gmoo file if it exists
        if gmoo_file and os.path.exists(gmoo_file):
            try:
                os.remove(gmoo_file)
                print(f"Cleaned up {gmoo_file}")
            except:
                print(f"Could not remove {gmoo_file}")

def test_inverse_optimization(vsme_dll):
    """Test the inverse optimization process."""
    # Create unique filenames
    dev_filename = f"inverse_dev_{int(time.time())}"
    gmoo_file = None
    
    try:
        # Create and develop model first
        model = GMOOAPI(
            vsme_windll=vsme_dll,
            vsme_input_filename=dev_filename,
            var_mins=[0.0, 0.0, 0.0],
            var_maxs=[10.0, 10.0, 10.0],
            num_output_vars=3,
            model_function=simple_function,
            save_file_dir=os.path.abspath('.')
        )
        
        # Initialize development and design cases
        model.development.load_vsme_name()
        model.development.initialize_variables()
        model.development.load_variable_types()
        model.development.load_variable_limits()
        model.development.design_agents()
        model.development.design_cases()
        
        # Get case count
        case_count = model.development.get_case_count()
        
        # Generate and evaluate cases
        input_cases = []
        output_cases = []
        
        for kk in range(1, case_count + 1):
            case_vars = model.development.poke_case_variables(kk)
            input_cases.append(case_vars)
            evaluation = model.model_function(case_vars)
            output_cases.append(evaluation)
        
        # Initialize outcomes directly
        model.development.initialize_outcomes()
        
        # Load each case result
        for kk in range(1, len(output_cases) + 1):
            model.development.load_case_results(kk, output_cases[kk-1])
        
        # Develop and export the model
        model.development.develop_vsme()
        gmoo_file = model.development.export_vsme()
        
        # Unload development mode before using application mode
        model.development.unload_vsme()
        
        # Truth case for validation
        truth_case = np.array([5.0, 5.0, 5.0])
        target_outputs = simple_function(truth_case)
        
        # Load the VSME model in application mode
        model.application.load_model()
        
        # Assign target objectives (percentage error)
        objective_types = [1, 1, 1]  # Type 1 = percentage error
        model.application.assign_objectives_target(target_outputs, objective_types)
        
        # Set uncertainty
        objectives_uncertainty = [3.0, 3.0, 3.0]
        model.application.load_objective_uncertainty(
            objectives_uncertainty, 
            objectives_uncertainty
        )
        
        # Initial guess
        initial_guess = np.mean([model.aVarLimMin, model.aVarLimMax], axis=0)
        next_input_vars = initial_guess
        next_output_vars = model.model_function(next_input_vars)
        
        # Optimization parameters
        best_l1 = float("inf")
        best_l1_case = None
        best_l1_output = None
        iterations = 30
        
        print("Starting inverse optimization...")
        
        # Optimization loop
        for ii in range(1, iterations + 1):
            next_vars, l1norm, l2norm = model.application.perform_inverse_iteration(
                target_outputs=target_outputs,
                current_inputs=next_input_vars,
                current_outputs=next_output_vars,
                objective_types=objective_types,
                objective_uncertainty_minus=objectives_uncertainty,
                objective_uncertainty_plus=objectives_uncertainty
            )
            
            next_input_vars = next_vars
            next_output_vars = model.model_function(next_input_vars)
            
            l1current = norm(next_output_vars - target_outputs, ord=1)
            if l1current < best_l1:
                best_l1 = l1current
                best_l1_case = next_input_vars.copy()
                best_l1_output = next_output_vars.copy()
                
                if l1current < 1.0:
                    print(f"Converged in {ii} iterations")
                    break
            
            if ii % 10 == 0:
                print(f"Iteration {ii}: L1 norm = {l1norm:.4f}")
        
        # Check results
        assert best_l1_case is not None, "Should have found a solution"
        
        # Calculate errors
        output_error = norm(best_l1_output - target_outputs, ord=2)
        relative_error = output_error / norm(target_outputs, ord=2)
        input_error = norm(best_l1_case - truth_case, ord=2)
        
        print(f"Best solution: {best_l1_case}")
        print(f"Output error: {output_error:.6f}")
        print(f"Relative error: {relative_error:.6f}")
        
        # Use a reasonable success criterion
        assert relative_error < 0.2, "Solution should be reasonably accurate"
        
        # Unload the model
        model.application.unload_vsme()
    
    finally:
        # Cleanup - ensure model is unloaded and file is removed
        if 'model' in locals():
            try:
                if hasattr(model, 'development'):
                    model.development.unload_vsme()
                if hasattr(model, 'application'):
                    model.application.unload_vsme()
            except:
                pass
                
        # Clean up file
        if gmoo_file and os.path.exists(gmoo_file):
            try:
                os.remove(gmoo_file)
            except:
                pass

if __name__ == "__main__":
    # Direct execution via python tests/test_memory_safe.py
    pytest.main(["-v", __file__])
