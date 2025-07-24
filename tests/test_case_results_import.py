"""
Test specifically focused on diagnosing and fixing the case results import issue.
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
logger = logging.getLogger("case_results_test")
logger.setLevel(logging.INFO)

# Import the GMOOAPI class from the gmoo_sdk package
from gmoo_sdk.dll_interface import GMOOAPI

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

def load_dll():
    """Load the GMOO DLL with basic error handling."""
    # Use the SDK's load_dll function to handle all path logic
    from gmoo_sdk.load_dll import load_dll as sdk_load_dll
    
    try:
        return sdk_load_dll()
    except Exception as e:
        logger.error(f"Failed to load DLL: {e}")
        pytest.skip(f"Failed to load DLL: {e}")

def test_case_results_import():
    """Test focused specifically on the case results import process."""
    # Load the DLL
    vsme_windll = load_dll()
    if vsme_windll is None:
        return
    
    # Use simple relative path for save directory
    save_dir = "."
    
    # Create a unique filename with timestamp
    timestamp = int(time.time())
    filename = f"case_import_test_import2_{timestamp}"
    
    logger.info(f"Using filename: {filename}")
    logger.info(f"Save directory: {save_dir}")
    
    # Create model
    model = GMOOAPI(
        vsme_windll=vsme_windll,
        vsme_input_filename=filename,
        var_mins=[0.0, 0.0, 0.0],
        var_maxs=[10.0, 10.0, 10.0],
        num_output_vars=3,
        model_function=simple_function,
        save_file_dir=save_dir
    )

    # Clearing the current memory
    model.development.unload_vsme()
    model.application.unload_vsme()
    
    # Verify model parameters
    logger.info(f"nVars: {model.nVars.value}")
    logger.info(f"nObjs: {model.nObjs.value}")
    
    try:
        # Development setup
        logger.info("Initializing development setup")
        model.development.load_vsme_name()
        model.development.initialize_variables()
        model.development.load_variable_types()
        model.development.load_variable_limits()
        
        # Design agents and cases
        logger.info("Designing agents and cases")
        model.development.design_agents()
        model.development.design_cases()
        
        # Get case count
        case_count = model.development.get_case_count()
        logger.info(f"Designed {case_count} cases")
        
        # Skip backup file entirely for this test
        #model.development.init_backup_file()
        
        # Generate and evaluate cases
        logger.info("Generating and evaluating cases")
        input_cases = []
        output_cases = []
        
        for kk in range(1, case_count + 1):
            case_vars = model.development.poke_case_variables(kk)
            input_cases.append(case_vars)
            evaluation = model.model_function(case_vars)
            output_cases.append(evaluation)
            
            # Print a few cases for debugging
            if kk <= 3 or kk > case_count - 3:
                logger.info(f"Case {kk}: Input {case_vars} -> Output {evaluation}")
        
        # Direct initialization of outcomes
        logger.info("Initializing outcomes directly")
        model.development.initialize_outcomes()
        
        # Verify that outcomes are initialized
        logger.info(f"nObjs after initialization: {model.nObjs.value}")
        
        # Load each case result with extensive validation
        logger.info(f"Loading {len(output_cases)} case results with validation")
        import copy
        errors = []
        for kk in range(1, len(output_cases) + 1):
            # Validate output before loading
            output = output_cases[kk-1]
            assert output is not None, f"Output for case {kk} is None"
            assert len(output) == model.nObjs.value, f"Output dimension {len(output)} doesn't match model nObjs {model.nObjs.value}"
            assert all(isinstance(val, (int, float)) for val in output), f"Output contains non-numeric values: {output}"
            
            try:
                # Load with extra error handling
                model.development.load_case_results(kk, copy.deepcopy(output))
            except Exception as e:
                errors.append((kk, str(e)))
                logger.error(f"Error loading case {kk}: {e}")
                raise
        
        if not errors:
            logger.info(f"All {len(output_cases)} cases loaded successfully")
        else:
            logger.error(f"Failed to load {len(errors)} cases: {errors}")
        
        #model.development.read_backup_file()
        #model.development.initialize_outcomes()

        # Develop the VSME model with careful error handling
        logger.info("Developing VSME model")
        try:
            model.development.develop_vsme()
            logger.info("VSME model developed successfully")
        except Exception as e:
            logger.error(f"Failed to develop VSME model: {e}")
            # Add diagnostic info
            logger.error(f"nVars: {model.nVars.value}, nObjs: {model.nObjs.value}")
            logger.error(f"Number of output cases: {len(output_cases)}")
            raise
        
        # Export VSME to file
        logger.info("Exporting VSME model")
        gmoo_file = model.development.export_vsme()
        assert os.path.exists(gmoo_file), f"GMOO file was not created: {gmoo_file}"
        
        # Cleanup
        model.development.unload_vsme()
        
        # Try to load the model for application to verify it works
        model.application.load_model()
        model.application.unload_vsme()
        
        logger.info("Test completed successfully")
    
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        pytest.fail(f"Case results import test failed: {e}")
    
    finally:
        # Cleanup
        try:
            if hasattr(model, 'development'):
                model.development.unload_vsme()
            if hasattr(model, 'application'):
                model.application.unload_vsme()
        except:
            pass
        
        # Remove test files
        backup_file = f"{filename}.VPRJ"
        if os.path.exists(backup_file):
            try:
                os.remove(backup_file)
                logger.info(f"Removed backup file: {backup_file}")
            except:
                logger.warning(f"Could not remove backup file: {backup_file}")
        
        gmoo_file = f"{filename}.gmoo"
        if os.path.exists(gmoo_file):
            try:
                os.remove(gmoo_file)
                logger.info(f"Removed GMOO file: {gmoo_file}")
            except:
                logger.warning(f"Could not remove GMOO file: {gmoo_file}")

if __name__ == "__main__":
    test_case_results_import()
