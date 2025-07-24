"""
Stateful Example Suite for GMOO SDK

This module provides a stateful test runner that uses the GMOOAPI directly
for comparison with the stateless wrapper approach.

Authors: Matt Freeman, Jason Hopkins
Version: 2.0.0
"""

import numpy as np
import logging
import os
from typing import List, Optional, Tuple
from dotenv import load_dotenv

# Add parent directory to path to access src
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

# Now we can import from the reorganized structure
from gmoo_sdk.satisfaction import check_satisfaction
from gmoo_sdk.load_dll import load_dll

# Import from examples using relative path since we're in examples directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.test_configs import TEST_CONFIGS
from test_utilities import (
    prepare_model_function,
    generate_random_starting_points,
    print_test_summary,
    print_satisfaction_summary,
    print_overall_summary,
    determine_target_outputs,
    safe_evaluate_model
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_test(vsme_version: str, 
             test_name: str, 
             iterations: int = 100, 
             pipes: int = 1, 
             train_vsme: bool = True, 
             manual_starting_point: bool = True) -> Tuple[List[float], List[float], bool, float, Optional[int]]:
    """
    Run a specific test case using the stateful GMOOAPI approach.
    
    Args:
        vsme_version: Path to the VSME DLL
        test_name: Name of the test configuration to run
        iterations: Maximum number of iterations for optimization
        pipes: Number of parallel optimization pipes
        train_vsme: Whether to train the inverse model before optimization
        manual_starting_point: Whether to use manual starting points
        
    Returns:
        tuple: (best_inputs, best_outputs, converged, best_error, iterations_taken)
    """
    if test_name not in TEST_CONFIGS:
        raise ValueError(f"Unknown test: {test_name}. Available tests: {list(TEST_CONFIGS.keys())}")
    
    config = TEST_CONFIGS[test_name]
    logger.info(f"Running {test_name} test...")
    
    # Extract configuration and prepare model function
    modelFunction = prepare_model_function(config['function'], config)
    
    varMins = config['var_mins']
    varMaxs = config['var_maxs']
    varTypes = config['var_types']
    numInputVars = config['num_inputs']
    numOutputVars = config['num_outputs']
    categories_list = config['categories_list']
    truthcase_inputs = config['truth_case']
    initialGuess = config['initial_guess']
    objective_types = config['objective_types']
    uncertainty_minus = config['uncertainty_minus']
    uncertainty_plus = config['uncertainty_plus']
    
    # Load DLL and create GMOOAPI
    vsme_windll = load_dll(vsme_version)
    
    from gmoo_sdk.dll_interface import GMOOAPI
    
    logger.info(f"[STATEFUL] Starting {test_name}")
    
    # Create GMOOAPI instance
    gmoo_api_client = GMOOAPI(
        vsme_windll=vsme_windll,
        vsme_input_filename=test_name,
        var_mins=varMins,
        var_maxs=varMaxs,
        num_output_vars=numOutputVars,
        model_function=modelFunction,
        save_file_dir=".",
        var_types=varTypes,
        categories_list=categories_list
    )
    
    if train_vsme:
        # Development phase
        gmoo_api_client.development.load_vsme_name()
        gmoo_api_client.development.initialize_variables()
        gmoo_api_client.development.load_variable_types()
        gmoo_api_client.development.load_variable_limits()
        gmoo_api_client.development.design_agents()
        gmoo_api_client.development.design_cases()
        gmoo_api_client.development.get_case_count()
        gmoo_api_client.development.init_backup_file()

        # Generate development cases
        input_dev = []
        for kk in range(1, gmoo_api_client.devCaseCount.value + 1):
            case_vars = gmoo_api_client.development.poke_case_variables(kk)
            input_dev.append(list(case_vars))

        # Evaluate development cases
        development_outcome_arrs = []
        for i, input_arr in enumerate(input_dev):
            output = safe_evaluate_model(modelFunction, input_arr, f"development case {i}")
            development_outcome_arrs.append(output)

        # Load development case results
        gmoo_api_client.development.read_backup_file()
        gmoo_api_client.development.initialize_outcomes()
        
        for kk in range(1, len(development_outcome_arrs) + 1):
            gmoo_api_client.development.load_case_results(kk, development_outcome_arrs[kk-1])

        # Develop VSME
        gmoo_api_client.development.develop_vsme()
        gmoo_api_client.development.export_vsme()

    logger.info(f"[STATEFUL] Development cases complete.")

    # Inverse solution phase
    
    # Initialize current input/output variables
    if pipes > 0:
        # Generate starting points for all pipes
        starting_points = generate_random_starting_points(
            pipes, varMins, varMaxs, varTypes, initialGuess
        )
        currentInputVars = [np.array(sp) for sp in starting_points]
        
        # Evaluate the model function for each input
        currentOutputVars = []
        for pipe in range(pipes):
            output = safe_evaluate_model(modelFunction, currentInputVars[pipe], f"initial point {pipe}")
            if output is None:
                currentOutputVars.append([float('inf')] * numOutputVars)
            else:
                currentOutputVars.append(output)
        
        bestcase = currentInputVars[0].copy()
        bestoutput = currentOutputVars[0].copy()
        
    else:
        currentInputVars = [initialGuess]
        currentOutputVars = [list(modelFunction(currentInputVars[0]))]
        bestcase = currentInputVars[0]
        bestoutput = currentOutputVars[0]

    # Target objectives
    if 'target_outputs' in config:
        objectivesArray = config['target_outputs']
        logger.info(f"[STATEFUL] Using target outputs from config: {objectivesArray}")
    else:
        objectivesArray = list(modelFunction(truthcase_inputs))
        logger.info(f"[STATEFUL] Computing objectives from truth case: {objectivesArray}")

    logger.info("[STATEFUL] Objective types and targets:")
    for i, (obj_type, target) in enumerate(zip(objective_types, objectivesArray)):
        logger.info(f"  Output {i+1}: Type {obj_type}, Target {target:.4f}")
    
    logger.info("[STATEFUL] Solving inverse...")
    converged = False
    best_error_sum = float("inf")
    
    # Load model for this iteration
    gmoo_api_client.application.load_model()
    
    # Initialize variables for multiple pipes
    gmoo_api_client.application.init_variables(nPipes=pipes)

    for ii in range(1, iterations + 1):
        if converged:
            break
            
        try:
            # Perform inverse iteration for all pipes at once
            if pipes > 1:
                # Multi-pipe case: pass lists of lists
                next_inputs_list, l1norms, l2norms = gmoo_api_client.application.perform_inverse_iteration(
                    target_outputs=objectivesArray,
                    current_inputs=currentInputVars,  # Pass all pipes at once
                    current_outputs=currentOutputVars,  # Pass all pipes at once
                    objective_types=objective_types,
                    objective_uncertainty_minus=uncertainty_minus,
                    objective_uncertainty_plus=uncertainty_plus
                )
                # Update all pipe inputs
                for pipe in range(pipes):
                    currentInputVars[pipe] = next_inputs_list[pipe].copy()
            else:
                # Single pipe case
                next_inputs, l1norm, l2norm = gmoo_api_client.application.perform_inverse_iteration(
                    target_outputs=objectivesArray,
                    current_inputs=currentInputVars[0],
                    current_outputs=currentOutputVars[0],
                    objective_types=objective_types,
                    objective_uncertainty_minus=uncertainty_minus,
                    objective_uncertainty_plus=uncertainty_plus
                )
                currentInputVars[0] = next_inputs.copy()

            # Evaluate new outputs
            currentOutputVars = []
            for pipe in range(pipes):
                output = safe_evaluate_model(modelFunction, currentInputVars[pipe], f"pipe {pipe} in iteration {ii}")
                if output is None:
                    currentOutputVars.append([float('inf')] * numOutputVars)
                else:
                    currentOutputVars.append(output)

            # Check for satisfaction
            for pipe in range(pipes):
                satisfied, error_metrics = check_satisfaction(
                    currentOutputVars[pipe],
                    objectivesArray,
                    objective_types,
                    uncertainty_minus,
                    uncertainty_plus
                )
                
                current_error_sum = sum(error_metrics)
                
                if satisfied:
                    bestcase = currentInputVars[pipe].copy()
                    bestoutput = currentOutputVars[pipe].copy()
                    best_error_sum = current_error_sum
                    logger.info(f"Converged in {ii} iterations!")
                    logger.info("Final errors:")
                    for i, error in enumerate(error_metrics):
                        logger.info(f"Output {i+1} (Type {objective_types[i]}): {error:.6f}")
                    
                    logger.info(f"[STATEFUL] Final result: {currentInputVars}")
                    converged = True
                    best_error_sum = 0.0
                    break
                elif current_error_sum < best_error_sum:
                    best_error_sum = current_error_sum
                    bestcase = currentInputVars[pipe].copy()
                    bestoutput = currentOutputVars[pipe].copy()

        except Exception as e:
            logger.error(f"Error in iteration {ii}: {e}")
            break
        
        if ii % 10 == 0:
            logger.info(f"Iteration {ii}: Best error = {best_error_sum:.6f}")

    # Clean up
    try:
        gmoo_api_client.development.unload_vsme()
        gmoo_api_client.application.unload_vsme()
    except:
        pass

    iterations_taken = ii if converged else None
    return list(bestcase), list(bestoutput), converged, best_error_sum, iterations_taken

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Default parameters
    iterations = 200
    pipes = 5
    manual_starting_point = True
    
    print("Running all test cases (STATEFUL)...")
    print("=" * 60)
    print(f"Parameters: iterations={iterations}, pipes={pipes}")
    print("=" * 60)
    
    # Store results for comparison
    all_results = {}
    
    # Get DLL path from environment or use fallback list
    env_dll_path = os.environ.get('MOOLIB')
    if env_dll_path and os.path.exists(env_dll_path):
        test_vsme_list = [env_dll_path]
        print(f"Using DLL from .env: {env_dll_path}")
    else:
        # Fallback to hardcoded list for benchmarking multiple versions
        test_vsme_list = [
            # Add DLL paths here for benchmarking different versions
            # Example: r"C:\path\to\VSME_v1.dll",
            # Example: r"C:\path\to\VSME_v2.dll",
        ]
        # If no paths are specified, exit
        if not test_vsme_list:
            print("ERROR: No DLL path found. Please either:")
            print("1. Set MOOLIB in your .env file")
            print("2. Set MOOLIB environment variable")
            print("3. Add DLL paths to test_vsme_list in this script")
            exit(1)
        print(f"Using hardcoded DLL list (set MOOLIB in .env to override)")
    
    print("=" * 60)
    
    # Uncomment to test specific configurations
    # TEST_CONFIGS = {'linear_small': TEST_CONFIGS['linear_small']}
    
    for vsme_version in test_vsme_list:
        all_results[vsme_version] = {}
        
        # Run each test in sequence
        for test_name in TEST_CONFIGS.keys():
            print("\n" + "=" * 60)
            print(f"Starting {test_name.upper()} test")
            print("=" * 60)
            
            try:
                best_inputs, best_outputs, satisfied, final_error, iters = run_test(
                    vsme_version,
                    test_name,
                    iterations=iterations,
                    pipes=pipes,
                    manual_starting_point=manual_starting_point
                )
                
                target_outputs = determine_target_outputs(TEST_CONFIGS[test_name])
                
                all_results[vsme_version][test_name] = {
                    'best_inputs': best_inputs,
                    'best_outputs': best_outputs,
                    'target_inputs': TEST_CONFIGS[test_name]['truth_case'],
                    'target_outputs': target_outputs,
                    'final_error': final_error,
                    'success': satisfied,
                    'iterations': iters
                }
                
            except Exception as e:
                logger.error(f"Error in {test_name} test: {str(e)}")
                all_results[vsme_version][test_name] = {'error': str(e)}
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY OF ALL TESTS")
        print("=" * 60)
        
        for test_name, results in all_results[vsme_version].items():
            print_test_summary(test_name, results, TEST_CONFIGS[test_name])
        
        # Print overall success rate
        print_overall_summary({vsme_version: all_results[vsme_version]})

    # Print satisfaction summary
    print_satisfaction_summary(all_results, TEST_CONFIGS, iterations)
