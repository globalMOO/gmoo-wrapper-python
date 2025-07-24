import os
import ctypes
import numpy as np
import pytest
from numpy.linalg import norm

# Import the GMOOAPI class from the gmoo_sdk package
from gmoo_sdk.dll_interface import GMOOAPI

def simple_test_function(input_arr):
    """
    Implementation of a simple but highly nonlinear test problem.
    """
    # Ensure input_arr is a numpy array
    input_arr = np.array(input_arr, ndmin=1)

    v01 = input_arr[0]
    v02 = input_arr[1]
    v03 = input_arr[2]
    
    o01 = v01 * v01 * v03 * v03
    o02 = (v01 - 2.0) * (v01 - 2.0) * v03 * v03
    o03 = v01 * v02 * v03 * v03

    # Combine outputs into an array
    output_arr = np.array([o01, o02, o03])

    return output_arr

def setup_model():
    """Helper function to set up a basic model for testing."""
    # Use load_dll to handle all the path logic
    from gmoo_sdk.load_dll import load_dll
    
    # Try to load the DLL
    try:
        vsme_windll = load_dll()
        
        # Set up basic model inputs
        filename = "inverse_test"
        num_input_vars = 3
        num_output_vars = 3    
        var_mins = [0.0, 0.0, 0.0]
        var_maxs = [10.0, 10.0, 10.0]
        model_function = simple_test_function
        save_file_dir = "."
        
        # Create the GMOOAPI object
        model = GMOOAPI(
            vsme_windll=vsme_windll,
            vsme_input_filename=filename,
            var_mins=var_mins,
            var_maxs=var_maxs,
            num_output_vars=num_output_vars,
            model_function=model_function,
            save_file_dir=save_file_dir
        )
        
        return model
    except Exception as e:
        pytest.fail(f"Exception when setting up model: {e}")

def setup_and_develop_model(model):
    """Helper function to set up and develop the model."""
    try:
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
        
        # Initialize the backup file
        model.development.init_backup_file()
        
        # Generate input cases
        input_dev = []
        for kk in range(1, case_count + 1):
            case_vars = model.development.poke_case_variables(kk)
            input_dev.append(case_vars)
        
        # Run the model function on all development cases
        output_dev = []
        for case in input_dev:
            evaluation = model.model_function(case)
            output_dev.append(evaluation)
        
        # Load development cases into the endpoint and train
        model.development.read_backup_file()
        model.development.initialize_outcomes()
        
        # Load each case result
        for kk in range(1, len(output_dev) + 1):
            model.development.load_case_results(kk, output_dev[kk-1])
        
        # Develop the VSME model
        model.development.develop_vsme()
        
        # Export VSME to file
        gmoo_file = model.development.export_vsme()
        
        # Unload the VSME model from development mode
        model.development.unload_vsme()
        
        print(f"Model developed and exported to {gmoo_file}")
        return gmoo_file
    except Exception as e:
        pytest.fail(f"Exception during model development: {e}")

def test_inverse_optimization():
    """Test the inverse optimization functionality."""
    # Define filenames we'll need to clean up
    test_basename = "inverse_test"
    backup_file = f"{test_basename}.VPRJ"
    gmoo_file = None
    
    try:
        # Set up the model
        model = setup_model()
        
        # Set up and develop the model
        gmoo_file = setup_and_develop_model(model)
        assert os.path.exists(gmoo_file), f"Expected GMOO file {gmoo_file} to exist"
        
        # Truth case to test against
        truthcase_inputs = np.array([5.0, 5.0, 5.0])
        truth_outputs = model.model_function(truthcase_inputs)
        print(f"Truth case inputs: {truthcase_inputs}")
        print(f"Truth case outputs: {truth_outputs}")
        
        # Target objectives for inverse optimization
        # In a real case, we would know the outputs and want to find the inputs
        objectives_array = truth_outputs.copy()
        
        # Load the VSME model in application mode
        model.application.load_model()
        print("Model loaded in application mode")
        
        # Assign target objectives
        objective_types = [1] * len(objectives_array)  # Type 1 = percentage error
        model.application.assign_objectives_target(objectives_array, objective_types)
        
        # Set uncertainty for objectives (2% error allowed)
        objectives_uncertainty_minus = [2.0] * len(objectives_array)
        objectives_uncertainty_plus = [2.0] * len(objectives_array)
        model.application.load_objective_uncertainty(
            objectives_uncertainty_plus, 
            objectives_uncertainty_minus
        )
        
        # Initial guess - start from center of the variable range
        initial_guess = np.mean([model.aVarLimMin, model.aVarLimMax], axis=0)
        next_input_vars = initial_guess
        next_output_vars = model.model_function(next_input_vars)
        
        # Initializations for inverse loop
        best_l1 = float("inf")
        best_l1_case = None
        iterations = 20  # Limit for test purposes
        satisfaction_threshold = 0.1
        
        # Store learned cases for analysis
        learned_case_inputs = []
        learned_case_outputs = []
        
        print("Starting inverse optimization...")
        converged = False
        
        for ii in range(1, iterations + 1):
            # Perform a single inverse iteration
            next_vars, l1norm, l2norm = model.application.perform_inverse_iteration(
                target_outputs=objectives_array,
                current_inputs=next_input_vars,
                current_outputs=next_output_vars,
                objective_types=objective_types,
                objective_uncertainty_minus=objectives_uncertainty_minus,
                objective_uncertainty_plus=objectives_uncertainty_plus
            )
            
            # Get the new input and output variables
            next_input_vars = next_vars
            next_output_vars = model.model_function(next_input_vars)
            
            # Track learned cases
            learned_case_inputs.append(next_input_vars.copy())
            learned_case_outputs.append(next_output_vars.copy())
            
            # Check for satisfaction based on percentage errors
            percentage_errors = np.abs((next_output_vars - objectives_array) / objectives_array) * 100
            if np.all(percentage_errors <= objectives_uncertainty_plus):
                best_l1_case = next_input_vars.copy()
                best_l1_output = next_output_vars.copy()
                print(f"Converged in {ii} iterations based on percentage errors.")
                converged = True
                break
            else:
                l1current = norm(next_output_vars - objectives_array, ord=1)
                if l1current < best_l1:
                    best_l1 = l1current
                    best_l1_case = next_input_vars.copy()
                    best_l1_output = next_output_vars.copy()
            
            # Print progress every few iterations
            if ii % 5 == 0 or ii == 1:
                print(f"Iteration {ii}: L1 norm = {l1norm:.6f}, L2 norm = {l2norm:.6f}")
                print(f"Current inputs: {next_input_vars}")
                print(f"Current outputs: {next_output_vars}")
                print(f"Target outputs: {objectives_array}")
                print(f"Percentage errors: {percentage_errors}")
                print("-" * 50)
                
        if not converged and iterations > 0:
            print(f"Did not converge after {iterations} iterations. Best L1 norm: {best_l1}")
        
        # We should have found a solution
        assert best_l1_case is not None, "Failed to find any solution"
        
        print(f"Best case inputs: {best_l1_case}")
        print(f"Truth case inputs: {truthcase_inputs}")
        print(f"Best case outputs: {best_l1_output}")
        print(f"Target outputs: {objectives_array}")
        
        # Calculate error metrics
        input_error = norm(best_l1_case - truthcase_inputs, ord=2)
        output_error = norm(best_l1_output - objectives_array, ord=2)
        
        print(f"Input error (L2 norm): {input_error}")
        print(f"Output error (L2 norm): {output_error}")
        
        # Check that the solution is close to the truth case
        assert output_error < 0.1 * norm(objectives_array, ord=2), "Output error is too large"
        
        # Unload the VSME model
        model.application.unload_vsme()
        print("VSME model unloaded")
        
        print("Inverse optimization test successful")
        
    except Exception as e:
        pytest.fail(f"Exception during inverse optimization test: {e}")
        
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
                try:
                    os.remove(gmoo_file)
                    print(f"Cleaned up {gmoo_file}")
                except:
                    print(f"Could not remove {gmoo_file}")
                    
            if os.path.exists(backup_file):
                try:
                    os.remove(backup_file)
                    print(f"Cleaned up {backup_file}")
                except:
                    print(f"Could not remove {backup_file}")
                    
        except Exception as cleanup_error:
            print(f"Warning: Error during cleanup: {cleanup_error}")

if __name__ == "__main__":
    # If run directly, execute the test
    test_inverse_optimization()
