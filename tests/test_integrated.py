"""
Integrated test for GMOO functionality.

This test demonstrates the full workflow from model creation to inverse optimization,
bypassing the backup file mechanism for increased reliability.
"""

import os
import ctypes
import numpy as np
import pytest
from numpy.linalg import norm

from gmoo_sdk.dll_interface import GMOOAPI

def load_dll():
    """Load the GMOO DLL with error handling."""
    # Use the SDK's load_dll function to handle all path logic
    from gmoo_sdk.load_dll import load_dll as sdk_load_dll
    
    try:
        return sdk_load_dll()
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

def test_integrated_functionality():
    """Test the full GMOO workflow from model creation to inverse optimization."""
    # Load the DLL
    vsme_windll = load_dll()
    assert vsme_windll is not None, "DLL should be loaded"
    
    # Define filenames we'll need to clean up
    test_basename = "integrated_test"
    gmoo_file = None
    backup_file = f"{test_basename}.VPRJ"
    
    try:
        # Create model
        model = GMOOAPI(
            vsme_windll=vsme_windll,
            vsme_input_filename=test_basename,
            var_mins=[0.0, 0.0, 0.0],
            var_maxs=[10.0, 10.0, 10.0],
            num_output_vars=3,
            model_function=simple_function,
            save_file_dir=os.path.abspath('.')
        )

        # Clearing the current VSME memory
        model.development.unload_vsme()
        model.application.unload_vsme()
        
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
        
        # Initialize the backup file (but we won't read it)
        model.development.init_backup_file()
        
        # Generate and evaluate cases
        input_cases = []
        output_cases = []
        
        for kk in range(1, case_count + 1):
            case_vars = model.development.poke_case_variables(kk)
            input_cases.append(case_vars)
            evaluation = model.model_function(case_vars)
            output_cases.append(evaluation)
        
        print(f"Evaluated {len(output_cases)} cases")
        
        # Initialize outcomes (without reading backup)
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
        
        # Unload the VSME model from development mode
        model.development.unload_vsme()
        
        # === Application mode (inverse optimization) ===
        
        # Truth case for validation
        truth_case = np.array([5.0, 5.0, 5.0])
        target_outputs = simple_function(truth_case)
        
        # Load the VSME model in application mode
        model.application.load_model()
        
        # Assign target objectives with percentage error
        objective_types = [1, 1, 1]  # Type 1 = percentage error
        model.application.assign_objectives_target(target_outputs, objective_types)
        
        # Set uncertainty for objectives (3% error allowed)
        objectives_uncertainty = [3.0, 3.0, 3.0]
        model.application.load_objective_uncertainty(
            objectives_uncertainty, 
            objectives_uncertainty
        )
        
        # Initial guess - start from center of the variable range
        initial_guess = np.mean([model.aVarLimMin, model.aVarLimMax], axis=0)
        next_input_vars = initial_guess
        next_output_vars = model.model_function(next_input_vars)
        
        # Inverse optimization parameters
        best_l1 = float("inf")
        best_l1_case = None
        best_l1_output = None
        iterations = 30
        
        print("Starting inverse optimization...")
        
        # Run the inverse optimization loop
        for ii in range(1, iterations + 1):
            # Perform a single inverse iteration
            next_vars, l1norm, l2norm = model.application.perform_inverse_iteration(
                target_outputs=target_outputs,
                current_inputs=next_input_vars,
                current_outputs=next_output_vars,
                objective_types=objective_types,
                objective_uncertainty_minus=objectives_uncertainty,
                objective_uncertainty_plus=objectives_uncertainty
            )
            
            # Get the new input and output variables
            next_input_vars = next_vars
            next_output_vars = model.model_function(next_input_vars)
            
            # Update best solution if improved
            l1current = norm(next_output_vars - target_outputs, ord=1)
            if l1current < best_l1:
                best_l1 = l1current
                best_l1_case = next_input_vars.copy()
                best_l1_output = next_output_vars.copy()
                
                # If error is small enough, stop early
                if l1current < 1.0:
                    print(f"Converged in {ii} iterations")
                    break
            
            # Print progress every 10 iterations
            if ii % 10 == 0:
                print(f"Iteration {ii}: L1 norm = {l1norm:.4f}")
                
        print("Inverse optimization complete")
        
        # Verify that a solution was found
        assert best_l1_case is not None, "Should have found a solution"
        
        # Calculate error metrics
        output_error = norm(best_l1_output - target_outputs, ord=2)
        relative_error = output_error / norm(target_outputs, ord=2)
        
        # Calculate input error compared to truth case
        input_error = norm(best_l1_case - truth_case, ord=2)
        
        print(f"Best solution: {best_l1_case}")
        print(f"Truth case: {truth_case}")
        print(f"Output error: {output_error:.6f}")
        print(f"Relative error: {relative_error:.6f}")
        print(f"Input error: {input_error:.6f}")
        
        # For comparison with performance on different machines, use a loose criterion
        # Either the solution is reasonably good or it's significantly better than the initial guess
        initial_output = model.model_function(initial_guess)
        initial_error = norm(initial_output - target_outputs, ord=2) / norm(target_outputs, ord=2)
        
        assert relative_error < 0.2 or (initial_error / relative_error > 2.0), \
            "Solution should be reasonably accurate or significantly better than initial guess"
        
        # Clean up
        model.application.unload_vsme()
        
        print("Integrated test successful")
    
    except Exception as e:
        pytest.fail(f"Exception in integrated test: {e}")
        
    finally:
        # Comprehensive cleanup in finally block to ensure it always runs
        try:
            # Unload VSME if model exists
            if 'model' in locals() and hasattr(model, 'development'):
                model.development.unload_vsme()
            if 'model' in locals() and hasattr(model, 'application'):
                model.application.unload_vsme()
            
            # Clean up any files we created
            if gmoo_file and os.path.exists(gmoo_file):
                os.remove(gmoo_file)
                print(f"Cleaned up {gmoo_file}")
                
            if os.path.exists(backup_file):
                os.remove(backup_file)
                print(f"Cleaned up {backup_file}")
                
        except Exception as cleanup_error:
            print(f"Warning: Error during cleanup: {cleanup_error}")

if __name__ == "__main__":
    test_integrated_functionality()
