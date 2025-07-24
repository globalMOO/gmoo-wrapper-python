"""
Test specifically focused on diagnosing memory issues with resource management.
"""

import os
import sys
import ctypes
import time
import gc
import numpy as np
import pytest
import logging
import platform

# Configure logging
logging.basicConfig(level=logging.WARNING,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("memory_diagnosis_test")
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

@pytest.fixture(scope="function")
def loaded_dll():
    """Load the GMOO DLL with explicit cleanup."""
    dll_path = os.environ.get('MOOLIB')
    if not dll_path:
        pytest.skip("MOOLIB environment variable is not set")
    
    if not os.path.exists(dll_path):
        pytest.skip(f"DLL file {dll_path} does not exist")
    
    logger.info("Loading DLL")
    
    try:
        # Handle Intel MPI paths if needed
        intel_redist = os.environ.get('I_MPI_ROOT')
        if intel_redist and os.path.exists(intel_redist):
            os.add_dll_directory(intel_redist)
        else:
            default_path = r'C:\Program Files (x86)\Common Files\Intel\Shared Libraries\intel64'
            if os.path.exists(default_path):
                os.add_dll_directory(default_path)
        
        # Load the DLL
        dll = ctypes.CDLL(dll_path)
        logger.info("DLL loaded successfully")
        yield dll
    
    except Exception as e:
        logger.error(f"Failed to load DLL: {e}")
        pytest.skip(f"Failed to load DLL: {e}")
    
    finally:
        # Try to free the DLL explicitly
        logger.info("Cleanup after test - forcing garbage collection")
        gc.collect()
        logger.info("Memory cleanup completed")

@pytest.fixture(scope="function", autouse=True)
def cleanup_before_and_after():
    """Run cleanup before and after each test"""
    # Pre-test memory cleanup
    logger.info("Running pre-test cleanup")
    
    if platform.system() == "Windows":
        # On Windows, try to manually unload any DLLs
        try:
            # This is a hacky way to force unload
            dll_path = os.environ.get('MOOLIB')
            if dll_path and os.path.exists(dll_path):
                # Force garbage collection to clean up any dangling references
                gc.collect()
        except Exception as e:
            logger.warning(f"Pre-test cleanup error: {e}")
    
    yield  # Test runs here
    
    # Post-test memory cleanup
    logger.info("Running post-test cleanup")
    gc.collect()
    logger.info("Post-test cleanup completed")

def test_case_results_import_with_cleanup(loaded_dll):
    """Test focused on case results import with explicit resource cleanup."""
    # Use simple relative path for save directory
    save_dir = "."
    
    # Create a unique filename with timestamp
    timestamp = int(time.time())
    filename = f"memory_test_{timestamp}"
    
    logger.info(f"Using filename: {filename}")
    
    # Create model
    model = GMOOAPI(
        vsme_windll=loaded_dll,
        vsme_input_filename=filename,
        var_mins=[0.0, 0.0, 0.0],
        var_maxs=[10.0, 10.0, 10.0],
        num_output_vars=3,
        model_function=simple_function,
        save_file_dir=save_dir
    )
    
    logger.info("Model created successfully")
    
    try:
        # Development setup with memory tracking
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
        
        # Generate and evaluate cases
        logger.info("Generating and evaluating cases")
        input_cases = []
        output_cases = []
        
        for kk in range(1, case_count + 1):
            case_vars = model.development.poke_case_variables(kk)
            input_cases.append(case_vars)
            evaluation = model.model_function(case_vars)
            output_cases.append(evaluation)
        
        logger.info("Cases generated successfully")
        
        # Direct initialization of outcomes
        logger.info("Initializing outcomes directly")
        model.development.initialize_outcomes()
        
        # Load each case result
        logger.info(f"Loading {len(output_cases)} case results")
        errors = []
        for kk in range(1, len(output_cases) + 1):
            output = output_cases[kk-1]
            try:
                model.development.load_case_results(kk, output)
            except Exception as e:
                errors.append((kk, str(e)))
                logger.error(f"Error loading case {kk}: {e}")
                raise
        
        if not errors:
            logger.info("All case results loaded successfully")
        
        # Develop the VSME model
        logger.info("Developing VSME model")
        model.development.develop_vsme()
        
        # Export VSME to file
        logger.info("Exporting VSME model")
        gmoo_file = model.development.export_vsme()
        
        # Explicit unload at each stage
        logger.info("Explicitly unloading development VSME")
        model.development.unload_vsme()
        
        # Try to load the model for application to verify it works
        logger.info("Loading model for application")
        model.application.load_model()
        
        # Explicitly unload application VSME
        logger.info("Explicitly unloading application VSME")
        model.application.unload_vsme()
        
        # Add force memory cleanup
        gc.collect()
        
        logger.info("Test completed successfully")
    
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        try:
            logger.info("Running cleanup in finally block")
            
            # Explicit unload for all subsystems
            if hasattr(model, 'development'):
                try:
                    model.development.unload_vsme()
                    logger.info("Development VSME unloaded in cleanup")
                except Exception as e:
                    logger.warning(f"Error unloading development VSME: {e}")
            
            if hasattr(model, 'application'):
                try:
                    model.application.unload_vsme()
                    logger.info("Application VSME unloaded in cleanup")
                except Exception as e:
                    logger.warning(f"Error unloading application VSME: {e}")
                    
            # Force garbage collection
            gc.collect()
            
            # Remove test files
            backup_file = f"{filename}.VPRJ"
            if os.path.exists(backup_file):
                try:
                    os.remove(backup_file)
                    logger.info(f"Removed backup file: {backup_file}")
                except:
                    logger.warning(f"Could not remove backup file: {backup_file}")
                    
            # Cleanup any dev_test files that might have been created
            for file in os.listdir('.'):
                if file.startswith('dev_test_') and file.endswith('.VPRJ'):
                    try:
                        os.remove(file)
                        logger.info(f"Removed found dev_test file: {file}")
                    except:
                        logger.warning(f"Could not remove found dev_test file: {file}")
            
            gmoo_file = f"{filename}.gmoo"
            if os.path.exists(gmoo_file):
                try:
                    os.remove(gmoo_file)
                    logger.info(f"Removed GMOO file: {gmoo_file}")
                except:
                    logger.warning(f"Could not remove GMOO file: {gmoo_file}")
                    
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")

def test_simple_unload_check():
    """
    This test just checks that unloading works properly without doing anything else.
    """
    logger.info("Starting simple unload check test")
    
    # Get DLL path from environment variable
    dll_path = os.environ.get('MOOLIB')
    if not dll_path:
        pytest.skip("MOOLIB environment variable is not set")
    
    if not os.path.exists(dll_path):
        pytest.skip(f"DLL file {dll_path} does not exist")
    
    try:
        # Handle Intel MPI paths if needed
        intel_redist = os.environ.get('I_MPI_ROOT')
        if intel_redist and os.path.exists(intel_redist):
            os.add_dll_directory(intel_redist)
        else:
            default_path = r'C:\Program Files (x86)\Common Files\Intel\Shared Libraries\intel64'
            if os.path.exists(default_path):
                os.add_dll_directory(default_path)
        
        # Load the DLL
        vsme_windll = ctypes.CDLL(dll_path)
        logger.info("DLL loaded successfully")
        
        # Just try calling some unload function if available
        if hasattr(vsme_windll, 'vsme_unload'):
            logger.info("Calling vsme_unload directly")
            try:
                result = vsme_windll.vsme_unload()
                logger.info(f"vsme_unload result: {result}")
            except Exception as e:
                logger.error(f"Error calling vsme_unload: {e}")
        
        # Try to find any unload functions available in the DLL
        unload_funcs = [name for name in dir(vsme_windll) if 'unload' in name.lower()]
        logger.info(f"Available unload functions: {unload_funcs}")
        
        # Force garbage collection
        gc.collect()
        logger.info("Test completed successfully")
    
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        pytest.fail(f"Simple unload check failed: {e}")

if __name__ == "__main__":
    test_case_results_import_with_cleanup(load_dll())
