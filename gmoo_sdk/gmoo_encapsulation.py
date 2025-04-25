"""
GMOO SDK Encapsulation Module

This module provides higher-level, stateless encapsulation functions around the GMOO DLL
interface. It simplifies common optimization workflows by abstracting away the details
of DLL management and memory handling.

The functions in this module follow a consistent pattern:
1. Load the DLL
2. Perform the requested operation
3. Unload the DLL
4. Return the results

This approach allows for simpler integration with external programs and reduces the
need for detailed knowledge of the DLL interface.

Authors: Matt Freeman, Jason Hopkins
Version: 2.0.0 (Updated with PEP8 compliance)
"""

import os
import ctypes
import numpy as np
import platform
from typing import Generator, List, Dict, Tuple, Union, Optional, Callable, Any, Set
import time

from gmoo_sdk.helpers import fortran_hollerith_string, CtypesHelper
from gmoo_sdk.dll_interface import GMOOAPI
from contextlib import contextmanager

# Path to DLL to be loaded
# Leave empty unless overriding the environment variable's DLL path
# Recommended usage is to set the 'MOOLIB' environment variable to
# the full path of the .dll or .so, including filename and extension
dll_path = ""

# Track directories that have been added to the DLL search path
added_paths: Set[str] = set()


def load_dll(dll_path: str = "") -> ctypes.CDLL:
    """
    Load the GMOO DLL with fallback license path handling.
    
    This function attempts to load the GMOO DLL from the specified path or
    from the MOOLIB environment variable. It also handles setting up necessary
    Intel MPI paths and a temporary license path if needed.
    
    Args:
        dll_path: Optional explicit path to the DLL file. If not provided,
                 the function will try to use the MOOLIB environment variable.
                
    Returns:
        ctypes.CDLL: Loaded DLL object.
        
    Raises:
        FileNotFoundError: If neither dll_path nor the MOOLIB environment variable is set.
        OSError: If the DLL cannot be loaded.
    """
    try:
        # Handle Intel MPI paths first
        intel_redist = os.environ.get('I_MPI_ROOT')
        
        if intel_redist:
            add_to_dll_path(intel_redist)
        else:
            default_path = r'C:\Program Files (x86)\Common Files\Intel\Shared Libraries\intel64'
            add_to_dll_path(default_path)
        
        # Handle MOOLIB path for the DLL location
        if not dll_path:
            dll_path = os.environ.get('MOOLIB')
        
        if not dll_path:
            raise FileNotFoundError(
                "Environment variable MOOLIB and hard coded 'dll_path' are both not set. "
                "Please use either to provide the full path to the VSME.dll file "
                "(including filename and extension)."
            )
        
        # Check if MOOLIC is already set
        if os.environ.get('MOOLIC'):
            # If MOOLIC exists, use it as is
            return ctypes.CDLL(dll_path)
        else:
            # If MOOLIC doesn't exist, use temporary path
            os.environ['MOOLIC'] = r"XXXXX"  # Replace XXXXX with actual license path
            
            return ctypes.CDLL(dll_path)

    except OSError as e:
        print(f"Failed to load the DLL: {e}")
        raise


def add_to_dll_path(path: str) -> None:
    """
    Add a directory to the DLL search path in a platform-specific way.
    
    This function adds the specified directory to the search path for
    dynamically loaded libraries. It handles the differences between
    Windows and Linux.
    
    Args:
        path: Directory path to add to the search path.
    """
    global added_paths
    
    if path and path not in added_paths:
        # Check if the operating system is Windows
        if platform.system() == "Windows":
            os.add_dll_directory(path)
        # Check if the operating system is Linux
        elif platform.system() == "Linux":
            # Modify LD_LIBRARY_PATH for Linux
            current_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
            os.environ['LD_LIBRARY_PATH'] = path + ":" + current_ld_library_path
        
        added_paths.add(path)


class DLLEnvironmentWrapper:
    """
    Context manager for safely loading and using the GMOO DLL with a specific license.
    
    This class provides a way to temporarily set environment variables needed
    for the DLL, load the DLL, use it, and then restore the original environment.
    
    Attributes:
        dll_path: Path to the DLL file.
        license_path: Path to the license file.
        _original_env: Original MOOLIC environment variable value.
        _dll: Loaded DLL object.
    """
    
    def __init__(self, dll_path: str, license_path: str):
        """
        Initialize the DLL environment wrapper.
        
        Args:
            dll_path: Path to the DLL file.
            license_path: Path to the license file.
        """
        self.dll_path = dll_path
        self.license_path = license_path
        self._original_env = None
        self._dll = None
    
    def __enter__(self) -> ctypes.CDLL:
        """
        Set up the environment and load the DLL.
        
        Returns:
            ctypes.CDLL: The loaded DLL object.
        """
        # Method 1: Temporarily set environment variable
        self._original_env = os.environ.get('MOOLIC')
        os.environ['MOOLIC'] = self.license_path
        
        # Load the DLL
        self._dll = ctypes.CDLL(self.dll_path)
        return self._dll
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Restore the original environment when exiting the context.
        
        Args:
            exc_type: Exception type, if any.
            exc_val: Exception value, if any.
            exc_tb: Exception traceback, if any.
        """
        # Restore original environment
        if self._original_env is None:
            del os.environ['MOOLIC']
        else:
            os.environ['MOOLIC'] = self._original_env


@contextmanager
def temporary_env_dll(dll_path: str, license_path: str) -> Generator[ctypes.CDLL, None, None]:
    """
    Alternative context manager function for loading the DLL with a temporary license path.
    
    This function provides the same functionality as DLLEnvironmentWrapper but in a
    function-based context manager style.
    
    Args:
        dll_path: Path to the DLL file.
        license_path: Path to the license file.
        
    Yields:
        ctypes.CDLL: The loaded DLL object.
    """
    original_env = os.environ.get('MOOLIC')
    os.environ['MOOLIC'] = license_path
    dll = ctypes.CDLL(dll_path)
    try:
        yield dll
    finally:
        if original_env is None:
            del os.environ['MOOLIC']
        else:
            os.environ['MOOLIC'] = original_env


def get_development_cases_encapsulation(
    var_mins: List[float],
    var_maxs: List[float],
    filename: str,
    var_types: Optional[List[int]] = None,
    categories_list: Optional[List[List[str]]] = None,
    save_file_dir: str = ".",
    problem_difficulty: str = "low"
) -> List[List[float]]:
    """
    Generate development cases for GMOO model training.
    
    This function creates a set of input cases that will be used to explore the
    design space and train the surrogate model. It handles DLL loading and unloading.
    
    Args:
        var_mins: List of minimum values for each variable.
        var_maxs: List of maximum values for each variable.
        filename: Base name for GMOO files.
        var_types: Optional list of variable types (1=float, 2=int, 3=logical, 4=categorical).
        categories_list: Optional list of category lists for categorical variables.
        save_file_dir: Directory where files will be saved.
        problem_difficulty: Complexity level of the problem, affects sampling density:
                          - "low": Default, standard sampling
                          - "medium": More dense sampling with parameter adjustments
                          - "high": Highest density sampling for complex problems
                          
    Returns:
        List[List[float]]: List of input variable arrays (development cases).
    """
    # Load VSME DLL
    vsme_windll = load_dll(dll_path)
    num_output_vars = 0  # For dev case generation, no outputs are needed yet
    
    # Create model interface
    model = GMOOAPI(
        vsme_windll,
        filename,
        var_mins,
        var_maxs,
        num_output_vars,
        None,  # No model function needed for case generation
        save_file_dir=save_file_dir,
        var_types=var_types,
        categories_list=categories_list
    )
    
    # Set parameters based on problem difficulty
    if problem_difficulty == "low":
        params = None
    elif problem_difficulty == "medium":
        params = {
            "e1": 100,
            "e4": 16,
            "e5": 2
        }
    elif problem_difficulty == "high":
        params = {
            "e1": 150,
            "e4": 16,
            "e5": 4,
            "e6": 4
        }
    else:
        params = None
        print("Invalid problem difficulty assignment.")

    # Setup development process and get input cases
    input_dev = model.development.setup_and_design_cases(params=params)

    # Clean up
    model.development.unload_vsme()
    model.application.unload_vsme()
    
    return input_dev


def load_user_cases_encapsulation(
    additional_case_vars: List[List[float]],
    additional_case_outcomes: List[List[float]],
    var_mins: List[float],
    var_maxs: List[float],
    num_output_vars: int,
    filename: str,
    var_types: Optional[List[int]] = None,
    categories_list: Optional[List[List[str]]] = None,
    save_file_dir: str = "."
) -> None:
    """
    Load additional user-provided cases into the VSME model.
    
    This function allows augmenting the automatically generated development cases
    with user-specified cases, which can be helpful for focusing the model on
    specific regions of interest.
    
    Args:
        additional_case_vars: Variable values for each case.
        additional_case_outcomes: Outcome values for each case.
        var_mins: Minimum values for each variable.
        var_maxs: Maximum values for each variable.
        num_output_vars: Number of output variables.
        filename: Base name for GMOO files.
        var_types: Optional list of variable types.
        categories_list: Optional list of category lists for categorical variables.
        save_file_dir: Directory to save files.
        
    Raises:
        ValueError: If the number of cases doesn't match between variables and outcomes.
    """
    # Validate inputs
    if len(additional_case_vars) != len(additional_case_outcomes):
        raise ValueError("Number of cases must match between variables and outcomes")
        
    # Load VSME DLL
    vsme_windll = load_dll(dll_path)
    
    # Create model interface
    model = GMOOAPI(
        vsme_windll,
        filename,
        var_mins,
        var_maxs,
        num_output_vars,
        None,  # No model function needed for loading cases
        save_file_dir=save_file_dir,
        var_types=var_types,
        categories_list=categories_list
    )
    
    # Read the backup file to continue from existing state
    model.development.read_backup_file()
    model.development.initialize_outcomes()
    model.application.load_model()

    # Load the additional user cases
    model.development.load_user_cases(additional_case_vars, additional_case_outcomes)

    # Export VSME to file
    model.development.export_vsme()

    # Wait for export to complete
    while not model.development.check_export_status():
        print("VSME file write ongoing ... ")
        time.sleep(1.0)

    # Clean up
    model.development.unload_vsme()
    model.application.unload_vsme()


def load_development_cases_encapsulation(
    output_dev: List[Optional[List[float]]],
    var_mins: List[float],
    var_maxs: List[float],
    num_output_vars: int,
    filename: str,
    var_types: Optional[List[int]] = None,
    categories_list: Optional[List[List[str]]] = None,
    save_file_dir: str = ".",
    extra_inputs: Optional[List[List[float]]] = None,
    extra_outputs: Optional[List[List[float]]] = None
) -> None:
    """
    Load development case results and train the VSME model.
    
    This function takes the results from evaluating development cases and uses
    them to train the surrogate model. It can also incorporate additional cases
    specified via extra_inputs and extra_outputs.
    
    Args:
        output_dev: Outcome values for development cases. Can contain None values
                   for cases that should be skipped.
        var_mins: Minimum values for each variable.
        var_maxs: Maximum values for each variable.
        num_output_vars: Number of output variables.
        filename: Base name for GMOO files.
        var_types: Optional list of variable types.
        categories_list: Optional list of category lists for categorical variables.
        save_file_dir: Directory to save files.
        extra_inputs: Optional additional input cases to include in training.
        extra_outputs: Optional additional outcome values to include in training.
    """
    # Load VSME DLL
    vsme_windll = load_dll(dll_path)

    # Create model interface
    model = GMOOAPI(
        vsme_windll,
        filename,
        var_mins,
        var_maxs,
        num_output_vars,
        None,  # No model function needed for loading cases
        save_file_dir=save_file_dir,
        var_types=var_types,
        categories_list=categories_list
    )
    
    # Check for invalid entries in output_dev
    invalid_entries = []
    for ii, entry in enumerate(output_dev):
        if entry is not None:
            if len(entry) != num_output_vars:
                print("INVALID OUTPUT ARRAY LENGTH IN DEVELOPMENT GENERATION. Resetting to `None` value.")
                invalid_entries.append(ii)
                
    for ii in invalid_entries:
        output_dev[ii] = None

    # Load the development cases and train the model
    model.development.load_results_and_develop(output_dev, extra_inputs, extra_outputs)

    # Get nonlinearity measure (optional diagnostic step)
    nonlinearity_value = model.development.poke_nonlinearity()
    print(f"Model nonlinearity: {nonlinearity_value}")

    # Clean up
    model.development.unload_vsme()
    model.application.unload_vsme()


def retrieve_nonlinearity_encapsulation(
    var_mins: List[float],
    var_maxs: List[float],
    num_output_vars: int,
    filename: str,
    save_file_dir: str = "."
) -> float:
    """
    Retrieve the nonlinearity measure from a trained VSME model.
    
    This function loads a trained model and returns its nonlinearity measure,
    which indicates how complex the surrogate model is.
    
    Args:
        var_mins: Minimum values for each variable.
        var_maxs: Maximum values for each variable.
        num_output_vars: Number of output variables.
        filename: Base name for GMOO files.
        save_file_dir: Directory where files are saved.
        
    Returns:
        float: Nonlinearity value, where:
              - Negative value: VSME has not been developed
              - 0.0: Model is linear
              - 0.1 to 10.0: Increasing severity of nonlinearity
    """
    # Load VSME DLL
    vsme_windll = load_dll(dll_path)

    # Create model interface
    model = GMOOAPI(
        vsme_windll,
        filename,
        var_mins,
        var_maxs,
        num_output_vars,
        None,  # No model function needed
        save_file_dir=save_file_dir,
    )
    
    # Load the model and check nonlinearity
    model.development.read_backup_file()
    model.development.initialize_outcomes()
    model.application.load_model()

    nonlinearity_value = model.development.poke_nonlinearity()

    # Clean up
    model.development.unload_vsme()
    model.application.unload_vsme()

    return nonlinearity_value


def get_dimensions_encapsulation(
    var_mins: List[float],
    var_maxs: List[float],
    num_output_vars: int,
    filename: str,
    var_types: Optional[List[int]] = None,
    categories_list: Optional[List[List[str]]] = None,
    save_file_dir: str = "."
) -> Tuple[ctypes.c_int, ...]:
    """
    Retrieve dimension information from a trained VSME model.
    
    This function loads a trained model and returns its dimension information,
    including counts of variables, objectives, bias solutions, and more.
    
    Args:
        var_mins: Minimum values for each variable.
        var_maxs: Maximum values for each variable.
        num_output_vars: Number of output variables.
        filename: Base name for GMOO files.
        var_types: Optional list of variable types.
        categories_list: Optional list of category lists for categorical variables.
        save_file_dir: Directory where files are saved.
        
    Returns:
        Tuple containing dimension information:
            nVars: Number of input variables
            nOuts: Number of output values/objectives
            nBias: Number of bias solutions
            nGens: Number of genetic ensembles
            nCats: Number of categorical variables
            nLrns: Number of learned cases
            n7-n10: Reserved for future use
    """
    # Load VSME DLL
    vsme_windll = load_dll(dll_path)

    # Create model interface
    model = GMOOAPI(
        vsme_windll,
        filename,
        var_mins,
        var_maxs,
        num_output_vars,
        None,  # No model function needed
        save_file_dir=save_file_dir,
        var_types=var_types,
        categories_list=categories_list
    )
    
    # Load the model and get dimensions
    model.application.load_model()

    dimensions = model.application.poke_dimensions10()

    # Clean up
    model.development.unload_vsme()
    model.application.unload_vsme()
    return dimensions


def get_observed_min_maxes_encapsulation(
    var_mins: List[float],
    var_maxs: List[float],
    num_output_vars: int,
    filename: str,
    var_types: Optional[List[int]] = None,
    categories_list: Optional[List[List[str]]] = None,
    save_file_dir: str = "."
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the observed minimum and maximum values for each output from the model.
    
    This function loads a trained model and returns the minimum and maximum values
    observed for each output variable during the development process.
    
    Args:
        var_mins: Minimum values for each variable.
        var_maxs: Maximum values for each variable.
        num_output_vars: Number of output variables.
        filename: Base name for GMOO files.
        var_types: Optional list of variable types.
        categories_list: Optional list of category lists for categorical variables.
        save_file_dir: Directory where files are saved.
        
    Returns:
        Tuple containing:
            np.ndarray: Minimum values observed for each output.
            np.ndarray: Maximum values observed for each output.
    """
    # Load VSME DLL
    vsme_windll = load_dll(dll_path)

    # Create model interface
    model = GMOOAPI(
        vsme_windll,
        filename,
        var_mins,
        var_maxs,
        num_output_vars,
        None,  # No model function needed
        save_file_dir=save_file_dir,
    )
    
    # Load the model and get min/max values
    model.application.load_model()

    observed_mins = model.application.poke_outcome_dev_limit_min(num_output_vars)
    observed_maxs = model.application.poke_outcome_dev_limit_max(num_output_vars)

    # Clean up
    model.development.unload_vsme()
    model.application.unload_vsme()
    return observed_mins, observed_maxs


def inverse_encapsulation(
    testcase_outcomes: List[float],
    var_mins: List[float],
    var_maxs: List[float],
    num_output_vars: int,
    current_input_vars: List[float],
    current_output_vars: List[float],
    filename: str,
    var_types: Optional[List[int]] = None,
    objective_types: Optional[List[int]] = None,
    objectives_uncertainty_minus: Optional[List[float]] = None,
    objectives_uncertainty_plus: Optional[List[float]] = None,
    learned_case_inputs: List[List[float]] = [],
    learned_case_outputs: List[List[float]] = [],
    save_file_dir: str = ".",
    inverse_iteration: int = 1,
    objective_status: Optional[List[int]] = None,
    categories_list: Optional[List[List[str]]] = None
) -> Tuple[np.ndarray, float, List[List[float]], List[List[float]]]:
    """
    Perform a single iteration of inverse optimization.
    
    This function uses a trained surrogate model to perform one step of inverse
    optimization, generating a new set of input variables that are expected to 
    produce outputs closer to the target outcomes.
    
    Args:
        testcase_outcomes: Target outcome values to match.
        var_mins: Minimum values for each variable.
        var_maxs: Maximum values for each variable.
        num_output_vars: Number of output variables.
        current_input_vars: Current input variable values.
        current_output_vars: Current output variable values.
        filename: Base name for GMOO files.
        var_types: Optional list of variable types.
        objective_types: Optional list of objective types (0=exact, 1=percentage, etc.).
        objectives_uncertainty_minus: Optional list of negative uncertainties.
        objectives_uncertainty_plus: Optional list of positive uncertainties.
        learned_case_inputs: List of previously learned case inputs.
        learned_case_outputs: List of previously learned case outputs.
        save_file_dir: Directory where files are saved.
        inverse_iteration: Current iteration number.
        objective_status: Optional list of objective status flags (1=active, 0=inactive).
        categories_list: Optional list of category lists for categorical variables.
        
    Returns:
        Tuple containing:
            np.ndarray: New input variable values suggested for the next iteration.
            float: L1 norm of the error between current outputs and targets.
            List[List[float]]: Updated list of learned case inputs.
            List[List[float]]: Updated list of learned case outputs.
    """
    # Load VSME DLL
    vsme_windll = load_dll(dll_path)

    # Create model interface
    model = GMOOAPI(
        vsme_windll,
        filename,
        var_mins,
        var_maxs,
        num_output_vars,
        None,  # No model function needed for this function
        save_file_dir=save_file_dir,
        var_types=var_types,
        categories_list=categories_list
    )
    
    # Clean up any previous instances
    model.development.unload_vsme()
    model.application.unload_vsme()

    # Load the model
    model.application.load_model()
    
    # Get current variable limits from the model
    min_vars_loaded = np.array(model.application.poke_variable_dev_limit_min())
    max_vars_loaded = np.array(model.application.poke_variable_dev_limit_max())

    # Reset variable limits to the intersection of specified limits and model limits
    var_mins_reset = np.maximum(min_vars_loaded, np.array(var_mins))
    var_maxs_reset = np.minimum(max_vars_loaded, np.array(var_maxs))

    model.application.load_variable_limit_min(var_mins_reset)
    model.application.load_variable_limit_max(var_maxs_reset)

    # Handle special case for minimize/maximize objectives
    if objective_types is not None:
        if 21 in objective_types or 22 in objective_types:
            outcome_mins = model.application.poke_outcome_dev_limit_min(num_output_vars)
            outcome_maxs = model.application.poke_outcome_dev_limit_max(num_output_vars)
    
            for ii, v in enumerate(objective_types):
                # For minimize objectives with no lower limit
                if v == 21 and testcase_outcomes[ii] == -999.25:
                    testcase_outcomes[ii] = outcome_mins[ii] * 0.8
                # For maximize objectives with no upper limit
                elif v == 22 and testcase_outcomes[ii] == -999.25:
                    testcase_outcomes[ii] = outcome_maxs[ii] * 1.2
    
    # Load the dynamically learned inverse cases
    for ii, learned_in, learned_out in zip(range(len(learned_case_inputs)), 
                                        learned_case_inputs, 
                                        learned_case_outputs):
        model.application.load_learned_case(len(var_mins), num_output_vars, learned_in, learned_out)

    # Perform inverse iteration
    next_input_vars, l1norm, _ = model.application.perform_inverse_iteration(
        target_outputs=testcase_outcomes,
        current_inputs=current_input_vars,
        current_outputs=current_output_vars,
        objective_types=objective_types,
        objective_status=objective_status,
        objective_uncertainty_minus=objectives_uncertainty_minus,
        objective_uncertainty_plus=objectives_uncertainty_plus
    )

    # Get the number of learned cases after this iteration
    learned_cases_after = model.application.poke_dimensions10()[5]

    # Reconstruct learned case inputs and outputs
    reconstructed_learned_case_inputs = []
    reconstructed_learned_case_outputs = []
    
    for ii in range(1, learned_cases_after.value + 1):
        learned_input, learned_output = model.application.poke_learned_case(len(var_mins), num_output_vars, ii)
        reconstructed_learned_case_inputs.append(learned_input)
        reconstructed_learned_case_outputs.append(learned_output)

    # Clean up
    model.development.unload_vsme()
    model.application.unload_vsme()

    return next_input_vars, l1norm, reconstructed_learned_case_inputs, reconstructed_learned_case_outputs


def minmax_encapsulation(
    var_mins: List[float],
    var_maxs: List[float],
    num_output_vars: int,
    filename: str,
    var_types: Optional[List[int]] = None,
    categories_list: Optional[List[List[str]]] = None,
    save_file_dir: str = "."
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the variable limits from the model.
    
    This function loads a trained model and returns the actual minimum and maximum
    values for each variable used during development.
    
    Args:
        var_mins: Minimum values for each variable.
        var_maxs: Maximum values for each variable.
        num_output_vars: Number of output variables.
        filename: Base name for GMOO files.
        var_types: Optional list of variable types.
        categories_list: Optional list of category lists for categorical variables.
        save_file_dir: Directory where files are saved.
        
    Returns:
        Tuple containing:
            np.ndarray: Minimum variable values from the model.
            np.ndarray: Maximum variable values from the model.
    """
    # Load VSME DLL
    vsme_windll = load_dll(dll_path)

    # Create model interface
    model = GMOOAPI(
        vsme_windll,
        filename,
        var_mins,
        var_maxs,
        num_output_vars,
        None,  # No model function needed
        save_file_dir=save_file_dir,
        var_types=var_types,
        categories_list=categories_list
    )
    
    # Load the model and get variable limits
    model.application.load_model()

    min_vars_loaded = np.array(model.application.poke_variable_dev_limit_min())
    max_vars_loaded = np.array(model.application.poke_variable_dev_limit_max())

    # Clean up
    model.development.unload_vsme()
    model.application.unload_vsme()

    return min_vars_loaded, max_vars_loaded


def rescope_search_space(
    all_inputs: Optional[List[List[float]]],
    all_outcomes: List[List[float]],
    objectives_array: List[float],
    var_mins: List[float],
    var_maxs: List[float],
    num_output_vars: int,
    filename: str,
    save_file_dir: str = "."
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptively rescope the search space based on optimization results.
    
    This function analyzes previous optimization results to identify a more
    promising region of the input space, potentially accelerating convergence
    by focusing subsequent optimizations on that region.
    
    Args:
        all_inputs: Previous input variable arrays. If None, development cases will be generated.
        all_outcomes: Previous outcome arrays.
        objectives_array: Target outcome values.
        var_mins: Current minimum values for each variable.
        var_maxs: Current maximum values for each variable.
        num_output_vars: Number of output variables.
        filename: Base name for GMOO files.
        save_file_dir: Directory where files are saved.
        
    Returns:
        Tuple containing:
            np.ndarray: New minimum variable values for the rescoped search space.
            np.ndarray: New maximum variable values for the rescoped search space.
    """
    # If no previous inputs are provided, generate development cases
    if all_inputs is None:
        input_dev = get_development_cases_encapsulation(
            var_mins,
            var_maxs,
            filename
        )
        all_inputs = input_dev
        
    # Load VSME DLL
    vsme_windll = load_dll(dll_path)

    # Create model interface
    model = GMOOAPI(
        vsme_windll,
        filename,
        var_mins,
        var_maxs,
        num_output_vars,
        None,  # No model function needed
        save_file_dir=save_file_dir
    )
    
    # Load the model
    model.application.load_model()
    
    # Use the last few results to generate suggestions for new input variables
    inputs_suggested = []
    for i, o in zip(all_inputs[-10:], all_outcomes[-10:]):
        next_inputs, _, _ = model.application.perform_inverse_iteration(
            target_outputs=objectives_array,
            current_inputs=i,
            current_outputs=o
        )
        inputs_suggested.append(next_inputs)
    
    # Clean up
    model.development.unload_vsme()
    model.application.unload_vsme()
    
    # Stack the arrays to find the min and max along each dimension
    stacked_arrays = np.stack(inputs_suggested)

    # Find the min and max along the first axis (across arrays)
    min_values = np.amin(stacked_arrays, axis=0)
    max_values = np.amax(stacked_arrays, axis=0)

    return min_values, max_values


def reset_mins_maxs_encapsulation(
    new_mins: List[float],
    new_maxs: List[float],
    var_mins: List[float],
    var_maxs: List[float],
    num_output_vars: int,
    filename: str,
    save_file_dir: str = "."
) -> None:
    """
    Reset the search space limits in the model.
    
    This function updates the minimum and maximum variable limits in the model,
    which can be used after rescoping the search space.
    
    Args:
        new_mins: New minimum values for each variable.
        new_maxs: New maximum values for each variable.
        var_mins: Original minimum values for each variable.
        var_maxs: Original maximum values for each variable.
        num_output_vars: Number of output variables.
        filename: Base name for GMOO files.
        save_file_dir: Directory where files are saved.
    """
    # Load VSME DLL
    vsme_windll = load_dll(dll_path)

    # Create model interface
    model = GMOOAPI(
        vsme_windll,
        filename,
        var_mins,
        var_maxs,
        num_output_vars,
        None,  # No model function needed
        save_file_dir=save_file_dir,
    )
    
    # Load the model
    model.application.load_model()
    
    # Reset search space limits
    model.application.load_variable_limit_min(new_mins)
    model.application.load_variable_limit_max(new_maxs)
    
    # Clean up
    model.development.unload_vsme()
    model.application.unload_vsme()


def random_search_encapsulation(
    var_mins: List[float],
    var_maxs: List[float],
    cases: int
) -> List[List[float]]:
    """
    Generate random search cases for initial space exploration.
    
    This function creates a set of random input variable combinations within
    the specified limits, which can be used for initial search space exploration
    or for global optimization approaches.
    
    Args:
        var_mins: Minimum values for each variable.
        var_maxs: Maximum values for each variable.
        cases: Number of random cases to generate.
        
    Returns:
        List[List[float]]: List of randomly generated input variable arrays.
    """
    # Generate random search cases
    search_cases = []
    for _ in range(0, cases):
        searchcase = np.random.uniform(var_mins, var_maxs)
        search_cases.append(searchcase.copy())
        
    return search_cases