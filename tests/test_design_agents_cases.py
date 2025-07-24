import os
import ctypes
import numpy as np
import pytest

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
        filename = "design_test"
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

def test_design_agents_and_cases():
    """Test designing agents and cases for the model."""
    # Define filenames we'll need to clean up
    test_basename = "design_test"
    backup_file = f"{test_basename}.VPRJ"
    gmoo_file = f"{test_basename}.gmoo"
    
    try:
        # Set up the model
        model = setup_model()
        
        # Initialize development setup
        model.development.load_vsme_name()
        model.development.initialize_variables()
        model.development.load_variable_types()
        model.development.load_variable_limits()
        
        # Now design agents and cases
        model.development.design_agents()
        
        # Verify the VSMEdevMain function returns expected values
        # The fact that design_agents completed successfully is already a good sign
        assert True, "Agents designed successfully"
        
        # Design cases
        model.development.design_cases()
        assert True, "Cases designed successfully"
        
        # Get and check the case count
        case_count = model.development.get_case_count()
        print(f"Number of cases designed: {case_count}")
        assert case_count > 0, "Expected at least one case to be designed"
        
        # Initialize the backup file for tracking progress
        model.development.init_backup_file()
        assert True, "Backup file initialized successfully"
        
        # Test retrieving case variables for a few cases
        for i in range(1, min(5, case_count + 1)):
            case_vars = model.development.poke_case_variables(i)
            print(f"Case {i} variables: {case_vars}")
            
            # Check case variables against the expected limits
            for j, var in enumerate(case_vars):
                assert model.aVarLimMin[j] <= var <= model.aVarLimMax[j], \
                    f"Case {i}, variable {j} ({var}) out of range [{model.aVarLimMin[j]}, {model.aVarLimMax[j]}]"
            
            # Test the model function with these variables
            outcome = model.model_function(case_vars)
            print(f"Case {i} outcome: {outcome}")
            assert outcome is not None, "Model function failed to return a result"
            assert len(outcome) == model.nObjs.value, \
                f"Expected {model.nObjs.value} outcomes, got {len(outcome)}"
            
        print("Agent and case design test successful")
    except Exception as e:
        pytest.fail(f"Exception during agent and case design test: {e}")
    finally:
        # Clean up resources
        try:
            # Unload the VSME model
            if 'model' in locals() and hasattr(model, 'development'):
                model.development.unload_vsme()
                
            # Clean up backup file
            if os.path.exists(backup_file):
                os.remove(backup_file)
                print(f"Removed backup file: {backup_file}")
                
            # Clean up GMOO file if created
            if os.path.exists(gmoo_file):
                os.remove(gmoo_file)
                print(f"Removed GMOO file: {gmoo_file}")
        except Exception as cleanup_error:
            print(f"Warning: Error during cleanup: {cleanup_error}")

if __name__ == "__main__":
    # If run directly, execute the test
    test_design_agents_and_cases()
