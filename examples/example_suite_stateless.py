"""
# examples_suite_stateless.py
Statless Test Runner for GMOO SDK

This module provides a comprehensive test suite for the GMOO optimization library
using a stateless wrapper approach. It runs various optimization problems to validate
functionality across different problem types.

Authors: Matt Freeman, Jason Hopkins
Version: 2.0.0
"""

import numpy as np
import os
import logging
import traceback
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv

# Add parent directory to path to access src
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from gmoo_sdk.stateless_wrapper import GmooStatelessWrapper
from gmoo_sdk.load_dll import load_dll
from gmoo_sdk.satisfaction import check_satisfaction

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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_test(vsme_version: str, 
             test_name: str, 
             iterations: int = 100, 
             pipes: int = 1, 
             train_vsme: bool = True, 
             manual_starting_point: bool = True,
             meta_parameters: Optional[Dict] = None) -> Tuple[List[float], List[float], bool, float, Optional[int]]:
    """
    Run a specific test case with the given parameters.
    
    Args:
        vsme_version: Path to the VSME DLL
        test_name: Name of the test configuration to run
        iterations: Maximum number of iterations for optimization
        pipes: Number of parallel optimization pipes
        train_vsme: Whether to train the inverse model before optimization
        manual_starting_point: Whether to use manual starting points
        meta_parameters: Optional meta-parameters for optimization
        
    Returns:
        tuple: (best_inputs, best_outputs, satisfied, best_error, iterations_taken)
    """
    
    if test_name not in TEST_CONFIGS:
        raise ValueError(f"Unknown test: {test_name}. Available tests: {list(TEST_CONFIGS.keys())}")
    
    config = TEST_CONFIGS[test_name]
    logger.info(f"Running {test_name} test...")
    
    # Extract configuration
    minimum_list = config['var_mins']
    maximum_list = config['var_maxs']
    input_type_list = config['var_types']
    numInputVars = config['num_inputs']
    num_outcomes = config['num_outputs']
    category_list = config['categories_list']
    
    if 'truth_case' in config:
        truthcase_inputs = config['truth_case']
    else:
        truthcase_inputs = [None] * len(minimum_list)
        
    initialGuess = config['initial_guess']
    objective_types = config['objective_types']
    uncertainty_minus = config['uncertainty_minus']
    uncertainty_plus = config['uncertainty_plus']
    base_func = config['function']
    
    if 'target_outputs' in config:
        objectivesArray = config['target_outputs']
        logger.info(f"Using target outputs for {test_name}: {objectivesArray}")
    else:
        objectivesArray = None
    
    # Prepare model function
    modelFunction = prepare_model_function(base_func, config)
    
    # Training phase
    if train_vsme:
        # Create stateless wrapper for development
        logger.info("Creating development wrapper")
        gmoo_api_client_stateless = GmooStatelessWrapper(
            minimum_list, 
            maximum_list, 
            input_type_list, 
            category_list, 
            filename_prefix=test_name, 
            output_directory=r".",
            dll_path=vsme_version,
            num_outcomes=0  # Set to zero initially
        )

        logger.info("Calling develop_cases (stateless)")
        input_dev = gmoo_api_client_stateless.develop_cases(params=meta_parameters)
        logger.info(f"Number of development cases: {len(input_dev)}")
        
        # Evaluate development cases
        development_outputs_list = []
        for i, case in enumerate(input_dev):
            output = safe_evaluate_model(modelFunction, case, f"development case {i}")
            development_outputs_list.append(output)
            
        logger.info("Creating training wrapper")
        gmoo_api_client_stateless = GmooStatelessWrapper(
            minimum_list, 
            maximum_list, 
            input_type_list, 
            category_list, 
            filename_prefix=test_name, 
            output_directory=r".",
            dll_path=vsme_version,
            num_outcomes=num_outcomes
        )
        
        logger.info("Calling load_development_cases (stateless)")
        gmoo_api_client_stateless.load_development_cases(
            num_outcomes,
            development_outputs_list
        )

    # Initialize optimization variables
    if objectivesArray is None:
        objectivesArray = list(modelFunction(truthcase_inputs))
        logger.info(f"Computing objectives for {test_name}: {objectivesArray}")

    # Initialize starting points for multiple pipes
    if pipes > 0:
        # Use the same starting point generation as stateful
        starting_points = generate_random_starting_points(
            pipes, minimum_list, maximum_list, input_type_list, initialGuess
        )
        nextInputVars = starting_points
        nextOutputVars = []
        
        # Evaluate the model function for each starting point
        for pipe in range(pipes):
            output = safe_evaluate_model(modelFunction, nextInputVars[pipe], f"initial point {pipe}")
            if output is None:
                nextOutputVars.append([float('inf')] * num_outcomes)
            else:
                nextOutputVars.append(output)
    else:
        nextInputVars = [initialGuess]
        nextOutputVars = [list(modelFunction(nextInputVars[0]))]

    logger.info("Objective types and targets:")
    for i, (obj_type, target) in enumerate(zip(objective_types, objectivesArray)):
        logger.info(f"Output {i+1}: Type {obj_type}, Target {target:.4f}")
    
    # Add more detailed logging for the first iteration
    logger.info("About to start optimization loop")
    logger.info(f"Initial input variables: {nextInputVars}")
    logger.info(f"Initial output variables: {nextOutputVars}")
    logger.info(f"Number of pipes: {pipes}")
    
    # Optimization loop
    best_error_sum = float("inf")
    bestcase = None
    bestoutput = None
    satisfied_flag = False
    
    # Initialize learned cases for each pipe
    if pipes > 0:
        learned_case_inputs = [[] for _ in range(pipes)]
        learned_case_outputs = [[] for _ in range(pipes)]
    
    # Create wrapper for application/optimization phase
    logger.info("Creating application wrapper for optimization")
    gmoo_api_client_stateless = GmooStatelessWrapper(
        minimum_list, 
        maximum_list, 
        input_type_list, 
        category_list, 
        filename_prefix=test_name, 
        output_directory=r".",
        dll_path=vsme_version,
        num_outcomes=num_outcomes
    )
    
    for ii in range(1, iterations + 1):
        logger.info(f"Starting iteration {ii}")
        if satisfied_flag:
            logger.info(f"Satisfied flag is True, breaking at iteration {ii}")
            break
        
        try:
            logger.info(f"About to call inverse() for iteration {ii}")
            logger.debug(f"Iteration {ii}: Parameters - objectives: {objectivesArray}")
            logger.debug(f"Iteration {ii}: Parameters - pipes: {pipes}")
            logger.debug(f"Iteration {ii}: Current inputs length: {len(nextInputVars)}")
            logger.debug(f"Iteration {ii}: Learned cases input length: {len(learned_case_inputs)}")
            
            # Call inverse directly using the existing wrapper
            logger.info(f"Calling inverse() on existing wrapper for iteration {ii}")
            nextInputVars, l1norm_full, learned_case_inputs, learned_case_outputs = gmoo_api_client_stateless.inverse(
                iteration_count=ii,
                current_iteration_inputs_list=nextInputVars,
                current_iteration_outputs_list=nextOutputVars,
                objectives_list=objectivesArray,
                learned_case_input_list=learned_case_inputs,
                learned_case_output_list=learned_case_outputs,
                objective_types_list=objective_types,
                objective_status_list=[1] * num_outcomes,
                minimum_objective_bound_list=uncertainty_minus,
                maximum_objective_bound_list=uncertainty_plus,
                pipe_num=pipes
            )
            
            logger.info(f"Inverse() completed for iteration {ii}")
            
        except Exception as e:
            logger.error(f"Error in iteration {ii}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            break

        # Evaluate new inputs
        nextOutputVars = []
        for pipe in range(pipes):
            output = safe_evaluate_model(modelFunction, nextInputVars[pipe], f"pipe {pipe} in iteration {ii}")
            if output is None:
                nextOutputVars.append([float('inf')] * num_outcomes)
            else:
                nextOutputVars.append(output)
        
        logger.info(f"Iteration {ii}: New outputs: {[out[:2] if len(out) > 2 else out for out in nextOutputVars]}...")  # Show first 2 elements

        # Check for satisfaction in all pipes
        for pipe in range(pipes):
            nextOutputVarsSingle = nextOutputVars[pipe]
            nextInputVarsSingle = nextInputVars[pipe]

            satisfied, error_metrics = check_satisfaction(
                nextOutputVarsSingle,
                objectivesArray,
                objective_types,
                uncertainty_minus,
                uncertainty_plus
            )
            
            # Use sum of error metrics for tracking best case
            current_error_sum = sum(error_metrics)

            if satisfied:
                bestcase = nextInputVarsSingle.copy()
                bestoutput = nextOutputVarsSingle.copy()
                best_error_sum = current_error_sum
                logger.info(f"Converged in {ii} iterations!")
                logger.info("Final errors:")
                for i, error in enumerate(error_metrics):
                    logger.info(f"Output {i+1} (Type {objective_types[i]}): {error:.6f}")
                satisfied_flag = True
                break
            
            elif current_error_sum < best_error_sum:
                best_error_sum = current_error_sum
                bestcase = nextInputVarsSingle.copy()
                bestoutput = nextOutputVarsSingle.copy()
                logger.info(f"Iteration {ii}, Pipe {pipe}: New best error = {best_error_sum:.6f}")

            else:
                logger.info(f"Iteration {ii}, Pipe {pipe}: Current error = {current_error_sum:.6f}")
            
        if ii % 10 == 0:  # Print progress every 10 iterations
            logger.info(f"Iteration {ii}: Best total error = {best_error_sum:.6f}")
        
        if ii == iterations:
            logger.info(f"Maximum iterations reached. Best total error: {best_error_sum:.6f}")
            if bestcase is not None:
                satisfied, error_metrics = check_satisfaction(
                    bestoutput,
                    objectivesArray,
                    objective_types,
                    uncertainty_minus,
                    uncertainty_plus
                )
                logger.info("Final errors:")
                for i, error in enumerate(error_metrics):
                    logger.info(f"Output {i+1} (Type {objective_types[i]}): {error:.6f}")
    
    iterations_taken = ii if satisfied_flag else None
    return bestcase, bestoutput, satisfied_flag, best_error_sum, iterations_taken

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Default parameters
    iterations = 200
    pipes = 5
    
    print("Running GMOO test cases...")
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
    
    # You can uncomment specific tests or run all
    # TEST_CONFIGS = {'linear_small': TEST_CONFIGS['linear_small']}
    
    # Meta parameters for optimization
    meta_parameters = {
        "e1": 1,
        "e4": 16,
        "e5": 4,
        "e6": 4,
        "r": 0.5
    }
    
    for vsme_version in test_vsme_list:
        all_results[vsme_version] = {}
        
        # Run each test in sequence
        for test_name in TEST_CONFIGS.keys():
            print("\n" + "=" * 60)
            print(f"Starting {test_name.upper()} test")
            print("=" * 60)
            
            try:
                # Determine target outputs
                target_outputs = determine_target_outputs(TEST_CONFIGS[test_name])
                print(f"Target outputs: {target_outputs}")

                logger.info(f"Starting test execution for {test_name}")
                best_inputs, best_outputs, satisfied, final_error, iters = run_test(
                    vsme_version,
                    test_name,
                    iterations=iterations,
                    pipes=pipes,
                    meta_parameters=meta_parameters
                )
                logger.info(f"Completed test execution for {test_name}")
                
                if 'truth_case' in TEST_CONFIGS[test_name]:
                    truthcase_target = TEST_CONFIGS[test_name]['truth_case']
                else:
                    truthcase_target = [None] * len(best_inputs) if best_inputs else []
                    
                all_results[vsme_version][test_name] = {
                    'best_inputs': best_inputs,
                    'best_outputs': best_outputs,
                    'target_inputs': truthcase_target,
                    'target_outputs': target_outputs,
                    'final_error': final_error,
                    'success': satisfied,
                    'iterations': iters
                }
                
            except Exception as e:
                logger.error(f"Error in {test_name} test: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                all_results[vsme_version][test_name] = {'error': str(e)}
                
                # Try to clean up any remaining DLL state
                try:
                    logger.info("Attempting cleanup after test failure")
                    # This is a best-effort cleanup - we don't have access to the wrapper here
                    # but we log the attempt
                except Exception as cleanup_error:
                    logger.warning(f"Cleanup after test failure also failed: {cleanup_error}")
        
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
