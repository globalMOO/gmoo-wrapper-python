import numpy as np
from numpy.linalg import norm
import gmoo_sdk.gmoo_encapsulation as gmapi

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
    #  Example of stateless encapsulation framework.
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

    truthcase_inputs = [5.4321, 5.4321, 5.4321]
    print("Truthcase outcomes: ", simple_test_function(truthcase_inputs))

    ####
    # Perform development case design and loading.
    ####
    if train_vsme:
        # Set up development, get inputs for development cases, create VPRJ file. 
        # VPRJ file should be tracked externally.      
        input_dev = gmapi.get_development_cases_encapsulation(
            var_mins,
            var_maxs,
            filename
        )
        
        
        # User runs the development cases.
        output_dev = []
        print("Training ... ")
        for case in input_dev:
            evaluation = model_function(case)
            output_dev.append(evaluation)

        # Load development cases into the endpoint, train, create .gmoo file. 
        # .gmoo file should be tracked externally.
        gmapi.load_development_cases_encapsulation(
            output_dev,
            var_mins,
            var_maxs,
            num_output_vars,
            filename
        )

    ####
    # Perform inverse solution workflow.
    ####
    # Outcomes corresponding to truthcase of [5.4321, 5.4321, 5.4321]
    objectives_array = np.array([870.70497364, 347.58048041, 870.70497364]) 

    # Starting search from center point
    initial_guess = np.mean([var_mins, var_maxs], axis=0)
    next_input_vars = initial_guess
    next_output_vars = model_function(next_input_vars)

    # Initializations and parameters for inverse loop
    best_l1 = float("inf")
    best_l1_case = None
    iterations = 1000
    convergence = 0.1
    learned_case_inputs, learned_case_outputs = [], []

    objective_types = [1] * len(objectives_array)

    # Converge to within +/- 2% of the true value.
    objectives_uncertainty_minus = [2.0] * len(objectives_array)
    objectives_uncertainty_plus = [2.0] * len(objectives_array)

    print("Inverse ... ")
    for ii in range(1, iterations + 1):
        # gMOO API takes whatever the current next_input_vars, next_output_vars are and 
        # suggests a next input variable point for the user to try.
        next_input_vars, l1norm_full, learned_case_inputs, learned_case_outputs = gmapi.inverse_encapsulation(
            objectives_array,
            var_mins,
            var_maxs,
            num_output_vars,
            next_input_vars,
            next_output_vars,
            filename,
            objective_types=objective_types,
            objectives_uncertainty_minus=objectives_uncertainty_minus,
            objectives_uncertainty_plus=objectives_uncertainty_plus,
            learned_case_inputs=learned_case_inputs, 
            learned_case_outputs=learned_case_outputs,
            inverse_iteration=ii
        )

        next_output_vars = model_function(next_input_vars)
        
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
    #  Rescoping of variables using the existing (saved) development input-output cases.
    ###
    input_dev = [arr.tolist() for arr in input_dev]
    min_values_2, max_values_2 = gmapi.rescope_search_space(
        input_dev,
        output_dev,
        objectives_array,
        var_mins,
        var_maxs,
        num_output_vars,
        filename
    )
    print(f"New (rescoped) minimum values: {min_values_2}")
    print(f"New (rescoped) maximum values: {max_values_2}")
    
    ###
    #  Retraining and repeating inverse phase with new rescoped search space. 
    #  This pass will converge much faster.
    ###
    filename2 = "rescope"
    input_dev = gmapi.get_development_cases_encapsulation(
        min_values_2,
        max_values_2,
        filename2
    )
    
    # User runs development cases
    output_dev = []
    print("Training (rescoped, narrowed search space) ... ")
    for case in input_dev:
        evaluation = model_function(case)
        output_dev.append(evaluation)
        
    # Load development cases into the endpoint, train, create .gmoo file.
    gmapi.load_development_cases_encapsulation(
        output_dev,
        min_values_2,
        max_values_2,
        num_output_vars,
        filename2
    )
    
    print("Inverse 2 (rescoped, narrowed search space) ... ")
    next_input_vars = initial_guess
    next_output_vars = model_function(next_input_vars)
    best_l1 = float("inf")
    best_l1_case = None
    learned_case_inputs, learned_case_outputs = [], []

    for ii in range(1, iterations + 1):
        # gMOO API takes whatever the current next_input_vars, next_output_vars are and 
        # suggests a next input variable point for the user to try.
        next_input_vars, l1norm_full, learned_case_inputs, learned_case_outputs = gmapi.inverse_encapsulation(
            objectives_array,
            min_values_2,
            max_values_2,
            num_output_vars,
            next_input_vars,
            next_output_vars,
            filename2,
            learned_case_inputs=learned_case_inputs, 
            learned_case_outputs=learned_case_outputs,
            inverse_iteration=ii
        )
        
        next_output_vars = model_function(next_input_vars)
        l1current = norm(next_output_vars - objectives_array, ord=1)
        
        if l1current < best_l1:
            best_l1 = l1current
            best_l1_case = next_input_vars.copy()  # Store values, not pointer
            best_l1_output = next_output_vars.copy()
            
            if best_l1 < convergence:
                print(f"Converged (after rescoping to narrower search) in {ii} iterations.")
                break
                
        if ii == iterations: 
            print("Convergence failed.")
            
    print(f"Best case inputs (variables) solution: {best_l1_case} vs. {truthcase_inputs}")
    print(f"Best case outputs (outcomes): {best_l1_output} vs. {objectives_array}")