import numpy as np
from numpy.linalg import norm
import logging
import os
import ctypes
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dll():
    """
    Load the GMOO DLL with fallback license path handling.
    """
    try:
        # Try to get the DLL path from environment variable
        dll_path = os.environ.get('MOOLIB')
        
        if not dll_path:
            raise FileNotFoundError(
                "Environment variable MOOLIB is not set. "
                "Please set it to the full path to the VSME.dll file "
                "(including filename and extension)."
            )
        
        # Handle Intel MPI paths if needed
        intel_redist = os.environ.get('I_MPI_ROOT')
        if intel_redist:
            os.add_dll_directory(intel_redist)
        else:
            default_path = r'C:\Program Files (x86)\Common Files\Intel\Shared Libraries\intel64'
            try:
                os.add_dll_directory(default_path)
            except:
                pass  # Ignore if it doesn't exist
        
        # Set temporary license path if needed
        if not os.environ.get('MOOLIC'):
            os.environ['MOOLIC'] = r"XXXXX"  # Replace with actual license path
            
        return ctypes.CDLL(dll_path)

    except OSError as e:
        print(f"Failed to load the DLL: {e}")
        raise

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
    o02 = (v01 - 2.0) * (v01 - 2.0)  * v03 * v03
    o03 = v01 * v02  * v03 * v03

    # Combine outputs into a 2D array
    output_arr = np.array([o01, o02, o03])

    # Remove the last dimension of size 1
    output_arr = np.squeeze(output_arr)

    return output_arr

if __name__ == "__main__":
    ######
    #
    #  Example of stateful framework (direct GMOOAPI calls).
    #
    ######

    ####
    # Set up basic model inputs.
    ####
    train_vsme = True
    filename = "simple"
    num_input_vars = 3
    num_output_vars = 3    
    var_mins = [0.0, 0.0, 0.0]
    var_maxs = [10.0, 10.0, 10.0]
    model_function = simple_test_function
    save_file_dir = "."

    # Load the VSME DLL
    vsme_windll = load_dll()

    # Create the model interface
    from gmoo_sdk.dll_interface import GMOOAPI

    model = GMOOAPI(
        vsme_windll=vsme_windll,
        vsme_input_filename=filename,
        var_mins=var_mins,
        var_maxs=var_maxs,
        num_output_vars=num_output_vars,
        model_function=model_function,
        save_file_dir=save_file_dir
    )

    # Print truthcase outcomes
    truthcase_inputs = [5.4321, 5.4321, 5.4321]
    print("Truthcase outcomes: ", model.model_function(truthcase_inputs))

    ####
    # Perform development case design and loading.
    ####
    if train_vsme:
        # Initialize development setup
        model.development.load_vsme_name()
        model.development.initialize_variables()
        model.development.load_variable_types()
        model.development.load_variable_limits()
        model.development.design_agents()
        model.development.design_cases()
        model.development.get_case_count()
        model.development.init_backup_file()

        # Generate input cases
        input_dev = []
        for kk in range(1, model.devCaseCount.value + 1):
            case_vars = model.development.poke_case_variables(kk)
            input_dev.append(case_vars)
        
        # User runs the development cases
        output_dev = []
        print("Training ... ")
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
        model.development.export_vsme()

    ####
    # Perform inverse solution workflow.
    ####
    # Outcomes corresponding to truthcase of [5.4321, 5.4321, 5.4321]
    objectives_array = np.array([870.70497364, 347.58048041, 870.70497364]) 

    # Load the VSME model
    model.application.load_model()
    
    # Assign target objectives
    objective_types = [1] * len(objectives_array)  # Type 1 = percentage error
    model.application.assign_objectives_target(objectives_array, objective_types)

    # Starting search from center point
    initial_guess = np.mean([var_mins, var_maxs], axis=0)
    next_input_vars = initial_guess
    next_output_vars = model.model_function(next_input_vars)

    # Initializations and parameters for inverse loop
    best_l1 = float("inf")
    best_l1_case = None
    iterations = 1000
    convergence = 0.1
    learned_case_inputs, learned_case_outputs = [], []

    # Converge to within +/- 2% of the true value
    objectives_uncertainty_minus = [2.0] * len(objectives_array)
    objectives_uncertainty_plus = [2.0] * len(objectives_array)
    
    # Set uncertainty for objectives
    model.application.load_objective_uncertainty(
        objectives_uncertainty_plus, 
        objectives_uncertainty_minus
    )

    print("Inverse ... ")
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
        
        # Check for convergence based on percentage errors
        percentage_errors = np.abs((next_output_vars - objectives_array) / objectives_array) * 100
        if np.all(percentage_errors <= objectives_uncertainty_plus):
            best_l1_case = next_input_vars.copy()  # Store values, not pointer
            best_l1_output = next_output_vars.copy()
            print(f"Converged in {ii} iterations based on percentage errors.")
            break
        else:
            l1current = norm(next_output_vars - objectives_array, ord=1)
            if l1current < best_l1:
                best_l1 = l1current
                best_l1_case = next_input_vars.copy()
                best_l1_output = next_output_vars.copy()
        if ii == iterations:
            print("Convergence failed.")
    
    print(f"Best case inputs (variables) solution: {best_l1_case} vs. {truthcase_inputs}")
    print(f"Best case outputs (outcomes): {best_l1_output} vs. {objectives_array}")

    ###
    #  Rescoping of variables using the existing learned cases
    ###
    print("Rescoping search space...")
    # Use the last few learned cases to generate suggestions for new input variables
    inputs_suggested = []
    for i, o in zip(learned_case_inputs[-10:], learned_case_outputs[-10:]):
        next_inputs, _, _ = model.application.perform_inverse_iteration(
            target_outputs=objectives_array,
            current_inputs=i,
            current_outputs=o
        )
        inputs_suggested.append(next_inputs)
    
    # Stack the arrays to find the min and max along each dimension
    stacked_arrays = np.stack(inputs_suggested)
    min_values_2 = np.amin(stacked_arrays, axis=0)
    max_values_2 = np.amax(stacked_arrays, axis=0)
    
    print(f"New (rescoped) minimum values: {min_values_2}")
    print(f"New (rescoped) maximum values: {max_values_2}")
    
    ###
    #  Retraining and repeating inverse phase with new rescoped search space. 
    #  This pass will converge much faster.
    ###
    # Clean up current VSME
    model.development.unload_vsme()
    model.application.unload_vsme()
    
    # Create a new model with the rescoped search space
    filename2 = "rescope"
    model2 = GMOOAPI(
        vsme_windll=vsme_windll,
        vsme_input_filename=filename2,
        var_mins=min_values_2.tolist(),
        var_maxs=max_values_2.tolist(),
        num_output_vars=num_output_vars,
        model_function=model_function,
        save_file_dir=save_file_dir
    )
    
    # Initialize development setup for rescoped model
    model2.development.load_vsme_name()
    model2.development.initialize_variables()
    model2.development.load_variable_types()
    model2.development.load_variable_limits()
    model2.development.design_agents()
    model2.development.design_cases()
    model2.development.get_case_count()
    model2.development.init_backup_file()

    # Generate input cases for rescoped model
    input_dev = []
    for kk in range(1, model2.devCaseCount.value + 1):
        case_vars = model2.development.poke_case_variables(kk)
        input_dev.append(case_vars)
    
    # User runs the development cases for rescoped model
    output_dev = []
    print("Training (rescoped, narrowed search space) ... ")
    for case in input_dev:
        evaluation = model2.model_function(case)
        output_dev.append(evaluation)

    # Load development cases into the endpoint and train rescoped model
    model2.development.read_backup_file()
    model2.development.initialize_outcomes()
            
    # Load each case result
    for kk in range(1, len(output_dev) + 1):
        model2.development.load_case_results(kk, output_dev[kk-1])

    # Develop the rescoped VSME model
    model2.development.develop_vsme()
    
    # Export VSME to file
    model2.development.export_vsme()
    
    # Load the rescoped VSME model for inverse optimization
    model2.application.load_model()
    model2.application.assign_objectives_target(objectives_array)
    
    print("Inverse 2 (rescoped, narrowed search space) ... ")
    next_input_vars = np.mean([min_values_2, max_values_2], axis=0)
    next_output_vars = model2.model_function(next_input_vars)
    best_l1 = float("inf")
    best_l1_case = None
    
    for ii in range(1, iterations + 1):
        # Perform a single inverse iteration
        next_vars, l1norm, l2norm = model2.application.perform_inverse_iteration(
            target_outputs=objectives_array,
            current_inputs=next_input_vars,
            current_outputs=next_output_vars
        )
        
        # Get the new input and output variables
        next_input_vars = next_vars
        next_output_vars = model2.model_function(next_input_vars)
        
        l1current = norm(next_output_vars - objectives_array, ord=1)
        
        if l1current < best_l1:
            best_l1 = l1current
            best_l1_case = next_input_vars.copy()
            best_l1_output = next_output_vars.copy()
            
            if best_l1 < convergence:
                print(f"Converged (after rescoping to narrower search) in {ii} iterations.")
                break
                
        if ii == iterations: 
            print("Convergence failed.")
            
    print(f"Best case inputs (variables) solution: {best_l1_case} vs. {truthcase_inputs}")
    print(f"Best case outputs (outcomes): {best_l1_output} vs. {objectives_array}")
    
    # Clean up
    model2.development.unload_vsme()
    model2.application.unload_vsme()