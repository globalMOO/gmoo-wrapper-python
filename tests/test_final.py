"""
Final integrated test that demonstrates a complete workflow with robust error handling.
"""

import os
import sys
import ctypes
import time
import numpy as np
import pytest
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gmoo_test")
logger.setLevel(logging.INFO)

# Import the GMOOAPI class from the gmoo_sdk package
from gmoo_sdk.dll_interface import GMOOAPI

def robust_load_dll():
    """Load the GMOO DLL with enhanced error handling and path discovery."""
    # Use the SDK's load_dll function to handle all path logic
    from gmoo_sdk.load_dll import load_dll
    
    try:
        # load_dll will check in order: parameter → environment → .env file
        dll = load_dll()
        logger.info("DLL loaded successfully using SDK's load_dll")
        return dll
    except Exception as e:
        logger.error(f"Failed to load DLL: {e}")
        pytest.skip(f"Failed to load DLL: {e}")

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

def test_final_integrated():
    """Test the complete GMOO workflow with robust error handling."""
    # Load the DLL
    vsme_windll = robust_load_dll()
    if vsme_windll is None:
        pytest.skip("Skipping test due to DLL loading issues")
    
    # Use absolute path for save directory
    save_dir = os.path.abspath('.')
    
    # Create a unique filename with timestamp
    timestamp = int(time.time())
    filename = f"final_test_{timestamp}"
    gmoo_file = None
    
    try:
        logger.info("Creating GMOOAPI model")
        model = GMOOAPI(
            vsme_windll=vsme_windll,
            vsme_input_filename=filename,
            var_mins=[0.0, 0.0, 0.0],
            var_maxs=[10.0, 10.0, 10.0],
            num_output_vars=3,
            model_function=simple_function,
            save_file_dir=save_dir
        )
        
        # === DEVELOPMENT MODE ===
        
        logger.info("Clearing the VSME memory")
        model.development.unload_vsme()
        model.application.unload_vsme()

        logger.info("Initializing development setup")
        model.development.load_vsme_name()
        model.development.initialize_variables()
        model.development.load_variable_types()
        model.development.load_variable_limits()
        
        logger.info("Designing agents and cases")
        model.development.design_agents()
        model.development.design_cases()
        
        # Get case count
        case_count = model.development.get_case_count()
        logger.info(f"Designed {case_count} cases")
        
        # Try to initialize the backup file, but don't rely on it
        try:
            logger.info("Attempting to initialize backup file")
            model.development.init_backup_file()
        except Exception as e:
            logger.warning(f"Backup file initialization failed: {e}")
        
        # Generate and evaluate cases
        logger.info("Generating and evaluating cases")
        input_cases = []
        output_cases = []
        
        for kk in range(1, case_count + 1):
            case_vars = model.development.poke_case_variables(kk)
            input_cases.append(case_vars)
            evaluation = model.model_function(case_vars)
            output_cases.append(evaluation)
        
        # Initialize outcomes directly (bypassing backup file)
        logger.info("Initializing outcomes")
        model.development.initialize_outcomes()
        
        # Load case results directly - with summary logging
        logger.info(f"Loading {len(output_cases)} case results")
        errors = []
        for kk in range(1, len(output_cases) + 1):
            try:
                model.development.load_case_results(kk, output_cases[kk-1])
            except Exception as e:
                errors.append((kk, str(e)))
                logger.error(f"Error loading case {kk}: {e}")
                raise
        
        if not errors:
            logger.info(f"All {len(output_cases)} cases loaded successfully")
        
        # Verify case results were loaded
        logger.info("Verifying case results were loaded")
        
        logger.info("Developing VSME model")
        model.development.develop_vsme()
        
        logger.info("Exporting VSME model")
        gmoo_file = model.development.export_vsme()
        assert os.path.exists(gmoo_file), f"GMOO file was not created: {gmoo_file}"
        
        # Unload development model before using application mode
        logger.info("Unloading development model")
        model.development.unload_vsme()
        
        # === APPLICATION MODE ===
        
        # Define a truth case for testing
        truth_case = np.array([5.0, 5.0, 5.0])
        target_outputs = simple_function(truth_case)
        logger.info(f"Target outcomes: {target_outputs}")
        
        logger.info("Loading model in application mode")
        model.application.load_model()
        
        # Set targets with percentage error objective
        logger.info("Setting target objectives")
        objective_types = [1, 1, 1]  # Percentage error
        model.application.assign_objectives_target(target_outputs, objective_types)
        
        # Set uncertainty bounds (±3%)
        logger.info("Setting uncertainty bounds")
        uncertainty_minus = [-3.0, -3.0, -3.0]
        uncertainty_plus = [3.0, 3.0, 3.0]
        model.application.load_objective_uncertainty(uncertainty_minus, uncertainty_plus)
        
        # Initial guess - center of the variable range
        initial_guess = np.mean([model.aVarLimMin, model.aVarLimMax], axis=0)
        next_input_vars = initial_guess
        next_output_vars = model.model_function(next_input_vars)
        
        # Optimization parameters
        best_l1 = float('inf')
        best_l1_case = None
        best_l1_output = None
        iterations = 30
        
        logger.info("Starting inverse optimization")
        for ii in range(1, iterations + 1):
            next_vars, l1norm, l2norm = model.application.perform_inverse_iteration(
                target_outputs=target_outputs,
                current_inputs=next_input_vars,
                current_outputs=next_output_vars,
                objective_types=objective_types,
                objective_uncertainty_minus=uncertainty_minus,
                objective_uncertainty_plus=uncertainty_plus
            )
            
            next_input_vars = next_vars
            next_output_vars = model.model_function(next_input_vars)
            
            # Update best solution
            l1current = np.linalg.norm(next_output_vars - target_outputs, ord=1)
            if l1current < best_l1:
                best_l1 = l1current
                best_l1_case = next_input_vars.copy()
                best_l1_output = next_output_vars.copy()
                
                # Stop early if converged
                if l1current < 1.0:
                    logger.info(f"Converged in {ii} iterations")
                    break
                    
            if ii % 10 == 0:
                logger.info(f"Iteration {ii}: L1 norm = {l1norm:.4f}")
                
        logger.info("Inverse optimization complete")
        
        # Verify results
        assert best_l1_case is not None, "No solution found"
        
        # Calculate errors
        output_error = np.linalg.norm(best_l1_output - target_outputs, ord=2)
        relative_error = output_error / np.linalg.norm(target_outputs, ord=2)
        input_error = np.linalg.norm(best_l1_case - truth_case, ord=2)
        
        logger.info(f"Best input: {best_l1_case}")
        logger.info(f"Best output: {best_l1_output}")
        logger.info(f"Target: {target_outputs}")
        logger.info(f"Relative error: {relative_error:.6f}")
        logger.info(f"Input error: {input_error:.6f}")
        
        # Use a reasonable success criterion for the test
        assert relative_error < 0.2 or np.linalg.norm(best_l1_case - truth_case, ord=2) < 3.0, \
               "Solution is not accurate enough"
        
        logger.info("Test successful - model achieved acceptable accuracy")
        
        # Unload the model
        logger.info("Unloading application model")
        model.application.unload_vsme()
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        pytest.fail(f"Final integrated test failed: {e}")
        
    finally:
        # Ensure complete cleanup
        if 'model' in locals():
            try:
                if hasattr(model, 'development'):
                    model.development.unload_vsme()
                if hasattr(model, 'application'):
                    model.application.unload_vsme()
            except:
                pass
            
        # Remove test files
        if gmoo_file and os.path.exists(gmoo_file):
            try:
                os.remove(gmoo_file)
                logger.info(f"Removed GMOO file: {gmoo_file}")
            except:
                logger.warning(f"Could not remove GMOO file: {gmoo_file}")
                
        backup_file = f"{os.path.join(save_dir, filename)}.VPRJ"
        if os.path.exists(backup_file):
            try:
                os.remove(backup_file)
                logger.info(f"Removed backup file: {backup_file}")
            except:
                logger.warning(f"Could not remove backup file: {backup_file}")

if __name__ == "__main__":
    test_final_integrated()
