"""
GMOO Test Suite Configuration

This file contains common fixtures and configurations for GMOO wrapper tests.
"""

import os
import sys
import ctypes
import logging
import numpy as np
import pytest
from numpy.linalg import norm
import glob

# Import the GMOOAPI class from the gmoo_sdk package
from gmoo_sdk.dll_interface import GMOOAPI, GMOOException

# Import test configuration
try:
    from test_config import get_dll_path, setup_test_environment
    # Set up test environment when conftest is loaded
    setup_test_environment()
except ImportError:
    # Fallback if test_config doesn't exist
    get_dll_path = None

# Configure logging for all tests
logging.basicConfig(level=logging.WARNING, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add a custom marker for tests that expect errors
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "expected_errors: mark test as expecting ERROR logs (suppresses them)"
    )

# Fixture to suppress expected ERROR logs
@pytest.fixture
def suppress_expected_errors(request):
    """Suppress ERROR logs for tests marked with @pytest.mark.expected_errors."""
    if request.node.get_closest_marker('expected_errors'):
        # Get all loggers that might emit expected errors
        loggers_to_suppress = [
            'gmoo_sdk',
            'gmoo_sdk.dll_interface', 
            'gmoo_sdk.workflows',
            'gmoo_sdk.development',
            'gmoo_sdk.application'
        ]
        
        # Store original levels
        original_levels = {}
        for logger_name in loggers_to_suppress:
            logger = logging.getLogger(logger_name)
            original_levels[logger_name] = logger.level
            logger.setLevel(logging.CRITICAL)
        
        yield
        
        # Restore original levels
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)
    else:
        yield
logger = logging.getLogger("gmoo_test")
# Set INFO level only for specific important messages
logger.setLevel(logging.INFO)

# Counter for unique test names
class TestCounter:
    count = 0
    
    @classmethod
    def next(cls):
        cls.count += 1
        return cls.count

# Test function definitions - reused across multiple tests
def simple_function(input_arr):
    """
    Implementation of a simple test problem.
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

def rosenbrock_function(input_arr):
    """
    Implementation of the Rosenbrock function - a more complex test problem.
    This is a common optimization test function with a global minimum at (1,1,...,1).
    """
    # Ensure input_arr is a numpy array
    input_arr = np.array(input_arr, ndmin=1)
    
    # Rosenbrock components
    n = len(input_arr)
    sum1 = 0
    sum2 = 0
    sum3 = 0
    
    for i in range(n-1):
        sum1 += 100 * (input_arr[i+1] - input_arr[i]**2)**2
        sum2 += (input_arr[i] - 1)**2
        sum3 += input_arr[i] * input_arr[i+1]
    
    # Create three different outputs based on Rosenbrock to make it multi-objective
    o1 = sum1  # Standard Rosenbrock
    o2 = sum2  # Deviation from the target point (1,1,...,1)
    o3 = sum3  # Cross-product sum (additional nonlinearity)
    
    return np.array([o1, o2, o3])

# Fixtures for reuse across tests
@pytest.fixture
def dll_path():
    """
    Get the DLL path using load_dll's logic.
    This will check in order: parameter (None here) -> env vars -> .env file
    """
    from gmoo_sdk.load_dll import load_dll
    
    try:
        # Try to load the DLL to verify it exists and get its path
        # load_dll will handle all the precedence logic
        dll = load_dll()
        
        # Get the DLL path from the loaded library
        # On Windows, _handle attribute contains the path
        if hasattr(dll, '_name'):
            return dll._name
        
        # If we can't get the path but DLL loaded, we know it exists somewhere
        # Return None to let load_dll handle it again in loaded_dll fixture
        return None
        
    except Exception as e:
        pytest.skip(f"Could not load DLL: {e}")

@pytest.fixture
def loaded_dll(dll_path):
    """
    Load the GMOO DLL.
    """
    try:
        # Don't add dll directories here - let load_dll handle it
        # This avoids duplicate calls to add_dll_directory
        
        # Set license path if needed
        if not os.environ.get('MOOLIC'):
            logger.warning("MOOLIC environment variable is not set. Some functionality may be limited.")
        
        # Use load_dll function which handles paths properly
        from gmoo_sdk.load_dll import load_dll
        # Pass dll_path if we have it, otherwise let load_dll use its default logic
        return load_dll(dll_path if dll_path else None)

    except Exception as e:
        # Check for bad image error specifically
        if any(phrase in str(e).lower() for phrase in ["bad image", "0xc000012f", "is either not designed"]):
            pytest.skip(f"DLL compatibility issue (Bad Image): {e}")
        pytest.skip(f"Failed to load the DLL: {e}")

@pytest.fixture
def simple_model(loaded_dll, request):
    """
    Create a simple GMOOAPI model for testing.
    
    Parameters can be customized through indirect parametrization.
    """
    # Default parameters
    params = {
        'filename': f"test_simple_{TestCounter.next()}",
        'num_input_vars': 3,
        'num_output_vars': 3,
        'var_mins': [0.0, 0.0, 0.0],
        'var_maxs': [10.0, 10.0, 10.0],
        'model_function': simple_function,
        'save_file_dir': '.',
        'var_types': None,
        'categories_list': None
    }
    
    # Override with any parameters passed through indirect
    if hasattr(request, 'param'):
        params.update(request.param)
    
    # Log the model creation parameters
    logger.info(f"Creating complex_model with filename: {params['filename']}")
    
    # Create the model
    model = GMOOAPI(
        vsme_windll=loaded_dll,
        vsme_input_filename=params['filename'],
        var_mins=params['var_mins'],
        var_maxs=params['var_maxs'],
        num_output_vars=params['num_output_vars'],
        model_function=params['model_function'],
        save_file_dir=params['save_file_dir'],
        var_types=params['var_types'],
        categories_list=params['categories_list']
    )
    
    # Log the actual filename used
    logger.info(f"Created complex_model with vsme_input_filename: {model.vsme_input_filename}")
    expected_gmoo_file = f"{model.vsme_input_filename}.gmoo"
    logger.info(f"Expected .gmoo file will be: {expected_gmoo_file}")
    
    yield model
    
    # Cleanup after test
    gmoo_file = f"{params['filename']}.gmoo"
    if os.path.exists(gmoo_file):
        try:
            os.remove(gmoo_file)
            logger.info(f"Cleaned up {gmoo_file}")
        except:
            logger.warning(f"Could not remove {gmoo_file}")

@pytest.fixture
def complex_model(loaded_dll, request):
    """
    Create a more complex GMOOAPI model with Rosenbrock function.
    """
    # Default parameters
    params = {
        'filename': f"test_complex_{TestCounter.next()}",
        'num_input_vars': 4,
        'num_output_vars': 3,
        'var_mins': [0.0, 0.0, 0.0, 0.0],
        'var_maxs': [2.0, 2.0, 2.0, 2.0],
        'model_function': rosenbrock_function,
        'save_file_dir': '.',
        'var_types': None,
        'categories_list': None
    }
    
    # Override with any parameters passed through indirect
    if hasattr(request, 'param'):
        params.update(request.param)
    
    # Log the model creation parameters
    logger.info(f"Creating complex_model with filename: {params['filename']}")
    
    # Create the model
    model = GMOOAPI(
        vsme_windll=loaded_dll,
        vsme_input_filename=params['filename'],
        var_mins=params['var_mins'],
        var_maxs=params['var_maxs'],
        num_output_vars=params['num_output_vars'],
        model_function=params['model_function'],
        save_file_dir=params['save_file_dir'],
        var_types=params['var_types'],
        categories_list=params['categories_list']
    )
    
    # Log the actual filename used
    logger.info(f"Created complex_model with vsme_input_filename: {model.vsme_input_filename}")
    expected_gmoo_file = f"{model.vsme_input_filename}.gmoo"
    logger.info(f"Expected .gmoo file will be: {expected_gmoo_file}")
    
    yield model
    
    # Cleanup after test with enhanced file detection
    logger.info("Cleaning up after complex_model fixture")
    
    # Try multiple possible file paths
    possible_files = [
        f"{params['filename']}.gmoo",                      # Filename from params
        f"{model.vsme_input_filename}.gmoo",                # Filename from model
        os.path.join('.', f"{params['filename']}.gmoo"),   # Explicitly in current directory
        "test_complex_5.gmoo"                               # Specific problem file
    ]
    
    # First unload the model to release any file handles
    try:
        model.development.unload_vsme()
        model.application.unload_vsme()
    except Exception as e:
        logger.warning(f"Error unloading model in fixture cleanup: {e}")
        
    # Force garbage collection
    import gc
    gc.collect()
    
    # Try to remove each possible file
    for gmoo_file in possible_files:
        if os.path.exists(gmoo_file):
            try:
                os.remove(gmoo_file)
                logger.info(f"Successfully cleaned up {gmoo_file} in fixture")
            except Exception as e:
                logger.warning(f"Could not remove {gmoo_file} in fixture: {e}")

# Helper functions for use in tests
def develop_model(model):
    """
    Common function to develop a model through the full workflow with robust error handling.
    Returns the path to the generated .gmoo file.
    """
    logger.info("Starting robust model development")
    
    # Force the save directory to be the simple current directory
    model.save_file_dir = "."
    
    # Use a simple filename without path manipulation
    if os.path.sep in model.vsme_input_filename:
        model.vsme_input_filename = os.path.basename(model.vsme_input_filename)
    
    logger.info(f"Using input filename: {model.vsme_input_filename}")
    
    # Ensure the save directory exists
    os.makedirs(model.save_file_dir, exist_ok=True)
    
    # Clearing the current VSME memory
    model.development.unload_vsme()
    model.application.unload_vsme()

    # Initialize development process
    logger.info("Setting up development environment")
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
    
    # Skip backup file entirely to avoid issues
    # Generate input cases
    logger.info("Generating and evaluating cases")
    input_dev = []
    for kk in range(1, case_count + 1):
        case_vars = model.development.poke_case_variables(kk)
        input_dev.append(case_vars)
    
    # Run the model function on all development cases
    output_dev = []
    for case in input_dev:
        evaluation = model.model_function(case)
        output_dev.append(evaluation)
    
    # Initialize outcomes directly - easier to debug
    logger.info("Initializing outcomes")
    model.development.initialize_outcomes()
    
    # Load each case result with validation
    logger.info(f"Loading {len(output_dev)} case results")
    errors = []
    for kk in range(1, len(output_dev) + 1):
        # Validate output
        output = output_dev[kk-1]
        assert len(output) == model.nObjs.value, f"Output dimension mismatch for case {kk}"
        
        try:
            model.development.load_case_results(kk, output)
        except Exception as e:
            errors.append((kk, str(e)))
            logger.error(f"Error loading case {kk}: {e}")
            raise
    
    if not errors:
        logger.info(f"All {len(output_dev)} cases loaded successfully")
    
    # Develop the VSME model with error handling
    logger.info("Developing VSME model")
    try:
        model.development.develop_vsme()
    except Exception as e:
        logger.error(f"Failed to develop VSME model: {e}")
        logger.error(f"Number of cases: {case_count}, Number of outputs loaded: {len(output_dev)}")
        raise
    
    # Export VSME to file
    logger.info("Exporting VSME model")
    gmoo_file = model.development.export_vsme()
    logger.info(f"Model exported to: {gmoo_file}")
    
    # Verify the file exists
    if not os.path.exists(gmoo_file):
        raise FileNotFoundError(f"GMOO file not created: {gmoo_file}")
    
    # Unload the VSME model from development mode
    logger.info("Unloading VSME model")
    model.development.unload_vsme()
    
    # We'll move the file cleanup to the end of the test to ensure it doesn't
    # interfere with the test functionality
    
    return gmoo_file

def perform_inverse_optimization(model, target_outputs, objective_types=None, max_iterations=30):
    """
    Common function to perform inverse optimization with a developed model.
    Returns the best solution found.
    """
    # Load the model in application mode
    model.application.load_model()
    
    # Set default objective types if not provided
    if objective_types is None:
        objective_types = [1] * len(target_outputs)  # Default to percentage error
    
    # Assign target objectives
    model.application.assign_objectives_target(target_outputs, objective_types)
    
    # Set uncertainty bounds (±3%)
    uncertainty = [3.0] * len(target_outputs)
    model.application.load_objective_uncertainty(uncertainty, uncertainty)
    
    # Initial guess - start from center of the variable range
    initial_guess = np.mean([model.aVarLimMin, model.aVarLimMax], axis=0)
    next_input_vars = initial_guess
    next_output_vars = model.model_function(next_input_vars)
    
    # Storage for tracking
    best_l1 = float("inf")
    best_l1_case = None
    best_l1_output = None
    
    # Run the inverse optimization loop
    for iteration in range(1, max_iterations + 1):
        # Perform a single inverse iteration
        next_vars, l1norm, l2norm = model.application.perform_inverse_iteration(
            target_outputs=target_outputs,
            current_inputs=next_input_vars,
            current_outputs=next_output_vars,
            objective_types=objective_types,
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
    
    # Unload the VSME model
    model.application.unload_vsme()
    
    return {
        'inputs': best_l1_case,
        'outputs': best_l1_output,
        'l1_error': best_l1,
        'iterations': max_iterations
    }

@pytest.fixture(scope="session", autouse=True)
def cleanup_all_test_files():
    """Clean up any leftover test files after all tests have run."""
    # This will run before all tests
    yield
    # This will run after all tests
    logger.info("Performing final cleanup of test files")
    
    # Add a short pause to ensure no files are still in use
    import time
    time.sleep(0.1)
    
    # Force release any resources
    import gc
    gc.collect()
    
    # Clean up any dev_test files
    for file in glob.glob('dev_test_*.VPRJ'):
        try:
            os.remove(file)
            logger.info(f"Cleaned up {file}")
        except Exception as e:
            logger.warning(f"Could not remove {file}: {e}")
    
    # Clean up test_rescoped.gmoo file
    if os.path.exists('test_rescoped.gmoo'):
        try:
            os.remove('test_rescoped.gmoo')
            logger.info("Cleaned up test_rescoped.gmoo")
        except Exception as e:
            logger.warning(f"Could not remove test_rescoped.gmoo: {e}")
    
    # Clean up all test_*.gmoo files
    for pattern in ['test_simple_*.gmoo', 'test_complex_*.gmoo']:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                logger.info(f"Cleaned up {file}")
            except Exception as e:
                logger.warning(f"Could not remove {file}: {e}")
    
    # Special handling for test_complex_5.gmoo
    if os.path.exists('test_complex_5.gmoo'):
        try:
            os.remove('test_complex_5.gmoo')
            logger.info("Specifically cleaned up test_complex_5.gmoo")
        except Exception as e:
            logger.warning(f"Could not remove test_complex_5.gmoo: {e}")
    
    # Also look in the tests directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    for pattern in ['test_simple_*.gmoo', 'test_complex_*.gmoo']:
        for file in glob.glob(os.path.join(test_dir, pattern)):
            try:
                os.remove(file)
                logger.info(f"Cleaned up {file} from tests directory")
            except Exception as e:
                logger.warning(f"Could not remove {file} from tests directory: {e}")


