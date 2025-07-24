# -*- coding: utf-8 -*-
"""
GMOO Workflows Module

This module provides high-level workflow functions for common GMOO optimization tasks.
These functions orchestrate the process of training inverse models and performing
optimization, managing the interaction between Python and the underlying GMOO engine.

The module supports both sequential and parallel execution of computationally intensive
tasks, as well as both direct and CSV-based interaction modes for external programs.

Authors: Matt Freeman, Jason Hopkins
Version: 2.0.0 (Refactored)
"""

import ctypes
import csv
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.linalg import norm

from .helpers import fortran_hollerith_string, validate_nan
from .dll_interface import GMOOAPI

# Configure logging
logger = logging.getLogger(__name__)


def process_batch(
    start: int, 
    end: int, 
    process_num: int, 
    input_arrs: List[np.ndarray], 
    mod_func: Callable
) -> Tuple[int, List[np.ndarray]]:
    """
    Process a batch of input arrays using a specified model function.
    
    This function is designed for parallel execution, where each process handles
    a subset of the input arrays. Each input array is processed by the model
    function, and the results are collected.
    
    Args:
        start: Starting index in the input_arrs list for this batch.
        end: Ending index (exclusive) in the input_arrs list for this batch.
        process_num: Identifier for the parallel process running this batch.
        input_arrs: List of input arrays to be processed.
        mod_func: Model function that processes each input array.
                Must accept an input array and a process number.
                
    Returns:
        Tuple containing:
            int: The process number that performed this batch.
            List[np.ndarray]: List of output arrays from the model function.
    """
    output_arrs = []
    for kk in range(start, end):
        input_arr = input_arrs[kk]
        # Pass the process number to model_function
        output_arr = mod_func(input_arr, process=process_num)
        output_arrs.append(output_arr)
    return process_num, output_arrs


def pyVSMEDevelopment(
    model: GMOOAPI,
    unbias_vector: Optional[List[float]] = None,
    params: Optional[Dict[str, Any]] = None,
    parallel_processes: int = 1,
    csv_mode: bool = False
) -> Tuple[GMOOAPI, int, Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    """
    Calculate learning cases, run them, load results, and develop the VSME model.
    
    This function manages the complete workflow for training an inverse model:
    1. Design the development cases (inputs to evaluate)
    2. Run the model function on these cases
    3. Load the results back into the VSME
    4. Build the inverse model
    5. Save the model as a .gmoo file
    
    Args:
        model: The GMOOAPI object for this problem.
        unbias_vector: Optional multiplier for de-biasing the input variables.
                     If None, all values are set to 1.0.
        params: Optional dictionary of custom parameters for VSME development.
        parallel_processes: Number of parallel processes to use for model evaluation.
                         A value of 1 runs evaluations sequentially.
        csv_mode: If True, cases are exported to CSV and results are read from CSV.
                This is useful when the model function is implemented externally.
                
    Returns:
        Tuple containing:
            GMOOAPI: The updated model with development data loaded.
            int: Number of development cases that were executed.
            List[np.ndarray] or None: List of input arrays used in training.
            List[np.ndarray] or None: List of output arrays from training.
    """
    # Initialize unbias vector if not provided
    if unbias_vector is None:
        unbias_vector = [1.0] * model.nVars.value

    # Setup and design cases using development module
    input_vectors = model.development.setup_and_design_cases(params=params)

    # Handle different modes of operation
    if csv_mode:
        # CSV Mode: Wait for external program to process cases
        input_vectors = None
        outcome_vectors = None
        output_vectors = None
        
        csv_filename = model.development.export_case_csv()
        
        logger.info(f"Development cases output to {csv_filename}. "
                    f"Waiting for outcomes to be processed by user's external model.")
              
        # Read outcomes from the CSV file produced by the external program
        outcome_vectors = model.development.read_outcomes_csv()
        logger.info(f"Read {len(outcome_vectors)} outcome vectors from CSV")

        # Load results and develop the model
        model.development.load_results_and_develop(outcome_vectors)
            
    elif not csv_mode:
        if parallel_processes > 1:
            # Parallel execution mode
            output_vectors = []
            output_dict = {}
            
            # Divide work among processes
            batch_size = model.devCaseCount.value // parallel_processes
            
            # Execute in parallel
            with ProcessPoolExecutor(max_workers=parallel_processes) as executor:
                futures = []
                for i in range(parallel_processes):
                    start = i * batch_size
                    end = (i + 1) * batch_size if i != parallel_processes - 1 else model.devCaseCount.value
                    futures.append(executor.submit(
                        process_batch, start, end, i, input_vectors, model.modelFunction)
                    )

                # Collect results as they complete
                for future in as_completed(futures):
                    process_number, batch_output = future.result()
                    output_dict[process_number] = batch_output

                # Combine results in the original order
                for i in sorted(output_dict.keys()):
                    output_vectors.extend(output_dict[i])

            # Load results and develop the model
            model.development.load_results_and_develop(output_vectors)
        else:
            # Sequential execution mode
            output_vectors = []
            
            # Process each case one by one
            for i, input_vector in enumerate(input_vectors, 1):
                logger.info(f"Running development case {i}/{len(input_vectors)}")
                output_vector = model.modelFunction(input_vector)
                
                # Validate output vector - check for NaN values
                try:
                    validate_nan(output_vector)
                except ValueError:
                    logger.error("Encountered NaN during model development process. Exiting.")
                    sys.exit(1)
                    
                output_vectors.append(output_vector)

            # Load results and develop the model
            model.development.load_results_and_develop(output_vectors)
    else:
        logger.error(f"Invalid value {csv_mode} for `csv_mode` in pyVSMEDevelopment, exiting.")
        sys.exit(1)

    return model, model.devCaseCount.value, input_vectors, output_vectors


def pyVSMEDevelopmentSetup(
    model: GMOOAPI,
    unbias_vector: Optional[List[float]] = None,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[GMOOAPI, List[np.ndarray]]:
    """
    Calculate the learning set of cases for later execution.
    
    This function performs the initial setup for VSME development, designing
    the cases that need to be evaluated but not actually running them. This
    is useful when the evaluation will be performed by an external process.
    
    Args:
        model: The GMOOAPI object for this problem.
        unbias_vector: Optional multiplier for de-biasing the input variables.
                     If None, all values are set to 1.0.
        params: Optional dictionary of custom parameters for VSME development.
        
    Returns:
        Tuple containing:
            GMOOAPI: The model with development initialization performed.
            List[np.ndarray]: List of input arrays to be evaluated by an external process.
    """
    # Initialize unbias vector if not provided
    if unbias_vector is None:
        unbias_vector = [1.0] * model.nVars.value

    # Setup and design cases
    input_vectors = model.development.setup_and_design_cases(params=params)

    return model, input_vectors


def pyVSMEDevelopmentLoad(
    model: GMOOAPI,
    outcome_vectors: List[Optional[np.ndarray]],
    extra_inputs: Optional[List[List[float]]] = None,
    extra_outputs: Optional[List[List[float]]] = None
) -> GMOOAPI:
    """
    Load results and develop the VSME model after external evaluation.
    
    This function is called after pyVSMEDevelopmentSetup, once the external
    process has evaluated the development cases. It loads the results and
    builds the inverse model.
    
    Args:
        model: The GMOOAPI object for this problem.
        outcome_vectors: List of output arrays from evaluating the development cases.
                       Can contain None entries for cases that failed.
        extra_inputs: Optional additional input cases to include in training.
        extra_outputs: Optional additional output values to include in training.
        
    Returns:
        GMOOAPI: The model with development data loaded and training completed.
    """
    # Load results and develop the model
    model.development.load_results_and_develop(
        outcome_vectors, extra_inputs=extra_inputs, extra_outputs=extra_outputs
    )

    return model


def pyVSMEInverse(
    model: GMOOAPI,           
    objectives_target: List[float],
    satisfaction_threshold: float,
    unbias_vector: Optional[List[float]] = None,
    override_init: Union[bool, List[float]] = False,
    max_inverse_iterations: int = 1000,
    outer_loops: int = 1,
    no_improvement_limit: Optional[int] = None,
    global_evaluations: int = 0,
    best_of_random: Optional[int] = None,
    objective_types: Optional[List[int]] = None,
    boundary_check: bool = False,
    fix_vars: Optional[List[bool]] = None,
    objectives_status: Optional[List[int]] = None
) -> Tuple[GMOOAPI, bool, str, List[float], int, float, float, Optional[np.ndarray]]:
    """
    Perform inverse optimization to find inputs that produce target outputs.
    
    This function runs an iterative optimization process to find input variables
    that will produce outputs matching the specified target values. It supports
    multiple satisfaction criteria, constraints, and restart strategies.
    
    Args:
        model: The GMOOAPI object for this problem.
        objectives_target: Target output values to be matched.
        satisfaction_threshold: L2 norm satisfaction criterion for successful optimization.
        unbias_vector: Optional multiplier for de-biasing the input variables.
        override_init: Initial guess for optimization. If False, starts from random point.
        max_inverse_iterations: Maximum number of iterations per optimization attempt.
        outer_loops: Number of optimization attempts with different starting points.
        no_improvement_limit: Maximum iterations without improvement before restarting.
        global_evaluations: Counter for total model evaluations across attempts.
        best_of_random: Number of random cases to evaluate when restarting.
        objective_types: Types of objectives (exact match, minimize, etc.).
        boundary_check: Whether to prevent solutions from getting too close to variable limits.
        fix_vars: Boolean flags indicating which variables should be fixed.
        objectives_status: Flags indicating which objectives are active (1) or inactive (0).
        
    Returns:
        Tuple containing:
            GMOOAPI: The model with optimization results loaded.
            bool: Success flag (True if converged, False otherwise).
            str: Message explaining the result.
            List[float]: Final input variable values.
            int: Total number of model evaluations performed.
            float: Final L1 norm of the error.
            float: Final L2 norm of the error.
            np.ndarray or None: Best case found based on L1 norm.
    """
    # Load the VSME model
    model.application.load_model()
        
    # Assign the target objectives
    model.application.assign_objectives_target(objectives_target, objective_types=objective_types)

    # Set which objectives are active, if specified
    if objectives_status is not None:
        model.application.load_objective_status(objectives_status)
    
    # Initialize unbias vector if not provided
    if unbias_vector is None:
        unbias_vector = [1.0] * model.nVars.value

    # Initialize optimization tracking
    cases = []
    l1norm, l2norm = float("inf"), float("inf")
    best_l1_case = None
    best_l1_norm = float("inf")
    
    # Outer loop - try multiple optimization attempts
    for oo in range(0, outer_loops):
        # Initialize variables for this attempt
        current_vars = model.application.calculate_initial_solution(override_init)

        # Set up improvement tracking if needed
        if no_improvement_limit is not None:
            from collections import deque
            deque_l1s = deque(maxlen=no_improvement_limit)
        
        mem_idx = 1
        
        # Inner loop - optimization iterations
        for kk in range(0, max_inverse_iterations):
            global_evaluations += 1
            
            # Periodically print progress
            print_freq = 100
            if global_evaluations % print_freq == 0 and kk > 0:
                logger.info(f"Inverse iteration: {global_evaluations} "
                            f"L1: {l1norm} "
                            f"L2: {l2norm} "
                            f"Input: {input_arr} "
                            f"Output: {output_arr} "
                            f"Mismatch: {output_arr - objectives_target}")
            
            # Apply unbias and evaluate model
            input_arr = np.array(current_vars) * unbias_vector
            output_arr = model.modelFunction(input_arr)

            cases.append(input_arr)
            
            # Check for NaN values
            try:
                validate_nan(input_arr)
                validate_nan(output_arr)
            except ValueError:
                logger.error("ENCOUNTERED NaN INSIDE INVERSE LOOP!")
                return (model, False, 
                        f"Solution aborted in {max_inverse_iterations} iterations for {outer_loops} "
                        f"outer loops due to NaN value.",
                        current_vars, global_evaluations, l1norm, l2norm, best_l1_case)

            # Calculate L2 norm for satisfaction check
            l2norm = norm(output_arr - objectives_target, ord=2)
          
            # Check for satisfaction based on L2 norm
            if l2norm < satisfaction_threshold:
                logger.info(f"Satisfaction achieved in inverse loop, l2={l2norm} on iteration "
                            f"{kk}/{max_inverse_iterations} with solution {input_arr}")
                return (model, True, "pyVSMEInverse succeeded.", current_vars, 
                        global_evaluations, l1norm, l2norm, best_l1_case)
            
            # Calculate full L1 norm for tracking
            l1norm_full = norm(output_arr - objectives_target, ord=1)
            
            # Calculate active L1 norm (only for active objectives)
            if objectives_status is not None:
                l1norm = norm(output_arr * objectives_status - objectives_target * objectives_status, ord=1)
            else:
                l1norm = norm(output_arr - objectives_target, ord=1)
            
            # Track no-improvement iterations if enabled
            if no_improvement_limit is not None:
                deque_l1s.append(l1norm)

            # Update best case if this is better
            if l1norm < best_l1_norm:
                best_l1_case = input_arr.copy()
                best_l1_norm = l1norm
            
            # Check for improvement stagnation
            if no_improvement_limit is not None and len(deque_l1s) == no_improvement_limit:
                if l1norm < deque_l1s[0]:
                    # Still improving, continue
                    pass
                else:
                    # No improvement, try random restart
                    best_l1_value, min_err_case_array, global_evaluations = model.application.perform_min_error_search(
                        objectives_target, 
                        global_evaluations, 
                        min_err_cases=best_of_random
                    )
                    
                    logger.info(f"Best new random case selected {min_err_case_array} with l1norm = {best_l1_value}")
                    override_init = model.application.calculate_initial_solution(min_err_case_array)
                    
                    # Reset improvement tracking
                    deque_l1s = deque(maxlen=no_improvement_limit)
                    
                    # Break inner loop to start fresh with new initial point
                    break
                
            # Perform a single inverse iteration
            next_vars, l1norm, l2norm = model.application.perform_inverse_iteration(
                target_outputs=objectives_target,
                current_inputs=current_vars,
                current_outputs=output_arr,
                objective_types=objective_types,
                objective_status=objectives_status
            )

            mem_idx += 1
            
            # Handle fixed variables if specified
            if fix_vars is not None:
                for ff, fix in enumerate(fix_vars):
                    if fix == False:  # Only update variables that are not fixed
                        current_vars[ff] = next_vars[ff]
            else:
                current_vars = next_vars

            # Apply boundary check if enabled
            if boundary_check:
                boundary_distance = 0.01
                lowEnd = current_vars - model.aVarLimMin
                highEnd = model.aVarLimMax - current_vars
                range_val = model.aVarLimMax - model.aVarLimMin
                
                for ev in range(0, len(range_val)):
                    if lowEnd[ev] / range_val[ev] < boundary_distance:
                        logger.info(f"Low end tripped for {ev}")
                        current_vars[ev] = current_vars[ev] + boundary_distance * range_val[ev]
                    if highEnd[ev] / range_val[ev] < boundary_distance:
                        logger.info(f"High end tripped for {ev}")
                        current_vars[ev] = current_vars[ev] - boundary_distance * range_val[ev]
                              
    # If we get here, we didn't converge
    return (model, False, 
            f"Solution failed to converge in {max_inverse_iterations} iterations for {outer_loops} outer loops.",
            current_vars, global_evaluations, l1norm, l2norm, best_l1_case)


def pyVSMEInverseSingleIter(
    model: GMOOAPI,           
    objectivesTarget: List[float],
    currentInputVars: List[float],
    currentOutcomeVars: List[float],
    iteration: int,
    objectiveTypes: Optional[List[int]] = None,
    objectives_status: Optional[List[int]] = None,
    objectives_uncertainty_minus: Optional[List[float]] = None,
    objectives_uncertainty_plus: Optional[List[float]] = None,
    reinitializeModel: bool = True
) -> Tuple[GMOOAPI, str, np.ndarray, float, float]:
    """
    Perform a single iteration of inverse optimization.
    
    This function executes one step of the inverse optimization process,
    starting from the current input and output values and generating
    a new suggestion for the next iteration.
    
    Args:
        model: The GMOOAPI object for this problem.
        objectivesTarget: Target output values to be matched.
        currentInputVars: Current input variable values.
        currentOutcomeVars: Current output values.
        iteration: Current iteration number.
        objectiveTypes: Types of objectives (exact match, minimize, etc.).
        objectives_status: Flags indicating which objectives are active (1) or inactive (0).
        objectives_uncertainty_minus: Lower uncertainty bounds for objectives.
        objectives_uncertainty_plus: Upper uncertainty bounds for objectives.
        reinitializeModel: Whether to reload the model (True) or reuse existing (False).
        
    Returns:
        Tuple containing:
            GMOOAPI: The model with optimization results loaded.
            str: Message indicating the result status.
            np.ndarray: Suggested input variable values for the next iteration.
            float: L1 norm of the error.
            float: L2 norm of the error.
    """
    # Initialize model if requested
    if reinitializeModel:
        model.application.load_model()

    # Load categorical variable information if needed
    if model.categories_list is not None:
        model.development.load_category_labels()
 
    # Perform a single inverse iteration
    next_vars, l1norm, l2norm = model.application.perform_inverse_iteration(
        target_outputs=objectivesTarget,
        current_inputs=currentInputVars,
        current_outputs=currentOutcomeVars,
        objective_types=objectiveTypes,
        objective_status=objectives_status
    )

    return model, "Success.", next_vars, l1norm, l2norm


def pyVSMEInverseCSV(
    model: GMOOAPI,           
    objectives_target: List[float],
    satisfaction_threshold: float,
    unbias_vector: Optional[List[float]] = None,
    override_init: Union[bool, List[float]] = False,
    max_inverse_iterations: int = 1000,
    outer_loops: int = 1,
    no_improvement_limit: int = 50,
    global_evaluations: int = 0,
    objective_types: Optional[List[int]] = None,
    boundary_check: bool = False
) -> Tuple[GMOOAPI, bool, str, List[float], int, float, float, Optional[np.ndarray]]:
    """
    Perform inverse optimization using CSV files for external model evaluation.
    
    This function is similar to pyVSMEInverse, but it writes variable values to
    CSV files and reads outcome values from CSV files, allowing for optimization
    with external model evaluations.
    
    Args:
        model: The GMOOAPI object for this problem.
        objectives_target: Target output values to be matched.
        satisfaction_threshold: L2 norm satisfaction criterion for successful optimization.
        unbias_vector: Optional multiplier for de-biasing the input variables.
        override_init: Initial guess for optimization. If False, starts from random point.
        max_inverse_iterations: Maximum number of iterations per optimization attempt.
        outer_loops: Number of optimization attempts with different starting points.
        no_improvement_limit: Maximum iterations without improvement before restarting.
        global_evaluations: Counter for total model evaluations across attempts.
        objective_types: Types of objectives (exact match, minimize, etc.).
        boundary_check: Whether to prevent solutions from getting too close to variable limits.
        
    Returns:
        Tuple containing:
            GMOOAPI: The model with optimization results loaded.
            bool: Success flag (True if converged, False otherwise).
            str: Message explaining the result.
            List[float]: Final input variable values.
            int: Total number of model evaluations performed.
            float: Final L1 norm of the error.
            float: Final L2 norm of the error.
            np.ndarray or None: Best case found based on L1 norm.
    """
    # Initialize unbias vector if not provided
    if unbias_vector is None:
        unbias_vector = [1.0] * model.nVars.value
    
    # Load the VSME model
    model.application.load_model()
        
    # Assign the target objectives
    model.application.assign_objectives_target(objectives_target, objective_types=objective_types)

    # Initialize optimization tracking
    l1norm, l2norm = float("inf"), float("inf")
    best_l1_case = None
    best_l1_norm = float("inf")
    
    # Outer loop - try multiple optimization attempts
    for oo in range(0, outer_loops):
        # Initialize variables for this attempt
        current_vars = model.application.calculate_initial_solution(override_init)

        # Set up improvement tracking
        from collections import deque
        deque_l1s = deque(maxlen=no_improvement_limit)
        
        mem_idx = 1
        
        # Inner loop - optimization iterations
        for kk in range(0, max_inverse_iterations):
            global_evaluations += 1
            
            # Periodically print progress
            if global_evaluations % 100 == 0:
                logger.info(f"Inverse iteration: {global_evaluations} "
                            f"L1: {l1norm} "
                            f"L2: {l2norm} "
                            f"Input: {input_arr} "
                            f"Output: {output_arr} "
                            f"Mismatch: {output_arr - objectives_target}")
            
            # Apply unbias
            input_arr = np.array(current_vars) * unbias_vector

            # Write input variables to CSV file for external program
            this_iter_inv_filename = f"{model.vsme_input_filename}_INVVAR{str(mem_idx).zfill(5)}"
            fname = f"{this_iter_inv_filename}.csv"
            with open(fname, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(input_arr.tolist())
                
            # Write 'done' file to signal availability of input data
            fname_done = f"{this_iter_inv_filename}.done"
            with open(fname_done, "w") as fd:
                pass

            logger.info(f"Inverse iteration data written to {this_iter_inv_filename}.csv for values {input_arr}. "
                        f"Waiting for outcomes to be processed by user's external model.")
        
            # Wait for the external program to process the inputs and write the outputs
            while not os.path.exists(f"{model.vsme_input_filename}_INVOUT{str(mem_idx).zfill(5)}.done"):
                time.sleep(0.1)  # Short pause to avoid CPU overuse

            # Read the output data from the CSV file
            with open(f"{model.vsme_input_filename}_INVOUT{str(mem_idx).zfill(5)}.csv", 'r') as file:
                reader = csv.reader(file)
                output_arr = [list(map(float, row[0:])) for row in reader][0]
                logger.info(f"Obtained output array from file: {output_arr}")

            # Convert to numpy arrays for calculation
            output_arr = np.array(output_arr)
            objectives_target_arr = np.array(objectives_target)

            # Calculate error metrics
            l1norm = norm(output_arr - objectives_target_arr, ord=1)
            l2norm = norm(output_arr - objectives_target_arr, ord=2)
            logger.info(f"L2: {l2norm} on iteration {kk} of {max_inverse_iterations}")

            # Update best case if this is better
            if l1norm < best_l1_norm:
                best_l1_case = input_arr
                best_l1_norm = l1norm
            
            # Check for satisfaction based on L2 norm
            if l2norm < satisfaction_threshold:
                logger.info(f"Satisfaction achieved in inverse loop, l2={l2norm} on iteration "
                            f"{kk}/{max_inverse_iterations} with solution {input_arr}")
                return (model, True, "pyVSMEInverse succeeded.", current_vars, 
                        global_evaluations, l1norm, l2norm, best_l1_case)
            
            # Track improvement for potential restart
            deque_l1s.append(l1norm)

            # Perform a single inverse iteration - manual approach for CSV mode
            model.application.load_variable_values(current_vars)
            model.application.load_outcome_values(output_arr)
            model.application.run_vsme_app(global_evaluations)
            
            mem_idx += 1
            
            # Get suggested variables for next iteration
            current_vars = model.application.fetch_variables_for_next_iteration()

            # Apply boundary check if enabled
            if boundary_check:
                boundary_distance = 0.01
                lowEnd = current_vars - model.aVarLimMin
                highEnd = model.aVarLimMax - current_vars
                range_val = model.aVarLimMax - model.aVarLimMin
                
                for ev in range(0, len(range_val)):
                    if lowEnd[ev] / range_val[ev] < boundary_distance:
                        logger.info(f"Low end tripped for {ev}")
                        current_vars[ev] = current_vars[ev] + boundary_distance * range_val[ev]
                    if highEnd[ev] / range_val[ev] < boundary_distance:
                        logger.info(f"High end tripped for {ev}")
                        current_vars[ev] = current_vars[ev] - boundary_distance * range_val[ev]
                                    
    # If we get here, we didn't converge
    return (model, False, 
            f"Solution failed to converge in {max_inverse_iterations} iterations for {outer_loops} outer loops.",
            current_vars, global_evaluations, l1norm, l2norm, best_l1_case)


def pyVSMEBias(
    model: GMOOAPI,
    model_function: Callable,
    objectives_target: List[float],
    unbias_vector: List[float],
    bias_iterations: int,
    show_debug_prints: bool
) -> Tuple[GMOOAPI, bool, str, List[float]]:
    """
    Calculate the bias solution for target objectives.
    
    This function implements the bias optimization algorithm, which can help
    in complex multi-objective scenarios by analyzing the bias in the solution space.
    
    Args:
        model: The GMOOAPI object for this problem.
        model_function: Function that evaluates the model for a given input.
        objectives_target: Target output values to be matched.
        unbias_vector: Multiplier for de-biasing the input variables.
        bias_iterations: Number of bias iterations to perform.
        show_debug_prints: Whether to show detailed debug output.
        
    Returns:
        Tuple containing:
            GMOOAPI: The model with bias results loaded.
            bool: Success flag (False if satisfaction failed).
            str: Message explaining the result.
            List[float]: Bias factors solution.
    """
    # Configure bias and genetic algorithm options
    gen_i_option = -1  # Genetic algorithm deactivated
    bias_i_option = 0  # Bias algorithm activated, started (or restarted)
    bias_option = 2    # Report normalized bias values
    n_bias = bias_iterations

    # Load the VSME model
    model.application.load_model()

    # Assign the target objectives
    model.application.assign_objectives_target(objectives_target)
    
    # Initialize genetic algorithm (disabled in this case)
    model.application.initialize_genetic_algorithm(
        iOption=gen_i_option,
        rLog10Gen=1,    # Log10 of number of genetic ensembles
        nParents=0,
        iError=2        # L2 norm
    )

    # Initialize bias functionality
    model.application.initialize_bias(iOption=bias_i_option, nBias=n_bias)

    # Configure debug output
    inner_loop_print_debug = False
    
    # Bias iteration loop
    for ll in range(1, n_bias + 1):
        logger.info(f"Starting outer loop iteration {ll}")
        # If iOption = 0 : bias procedure is started or restarted from scratch
        # If iOption == 0 and nBias > 0, then this value is adopted for nBias
        
        # Initialize variables for this bias iteration
        model.application.init_variables(iBias=ll)
        
        # Get initial values from bias initialization
        iBiasToUse = 0
        dVarValueInits = model.application.fetch_variables_for_next_iteration(iBias=iBiasToUse)
        
        # Apply unbias to get actual input values
        dVarjValues = dVarValueInits * unbias_vector
            
        # Evaluate the model
        input_arr = np.array(dVarjValues)
        output_arr = model_function(input_arr)
        
        # Load values into the model
        # Note: Use the internal values (without unbias) when loading into model
        model.application.load_variable_values(dVarValueInits, iBias=ll)     
        model.application.load_outcome_values(output_arr, iBias=ll)
        
        # Inner loop - inverse solution iterations
        max_inverse_iterations = 500
        for kk in range(1, max_inverse_iterations + 1):
            # Run optimization iteration
            model.application.run_vsme_app(kk)

            # Check for errors or satisfaction
            if model.last_status > 0: 
                return model, False, f"VSMEAPP failed in pyVSMEBias with code {model.last_status}", False
            elif model.last_status < 0:
                logger.info(f"Satisfaction (code {model.last_status}) on outer loop iteration {ll}, "
                           f"inner loop iteration ({kk})!")
                break
            else:
                if show_debug_prints and inner_loop_print_debug:
                    logger.info(f"VSMEAPP succeeded (but not converged) in pyVSMEBias with code "
                              f"{model.last_status}")

            # Get suggested variables for next iteration
            # Note: The "optimal/suggested" next value is associated with iBias=0
            dVarValueLL = model.application.fetch_variables_for_next_iteration(iBias=iBiasToUse)
            
            # Apply unbias to get actual input values
            dVarjValues = dVarValueLL * unbias_vector

            # Evaluate the model
            input_arr = np.array(dVarjValues)
            output_arr = model_function(input_arr)
            
            # Load outcome values into the model
            model.application.load_outcome_values(output_arr, iBias=ll)

            # Check for maximum iterations
            if kk == max_inverse_iterations:
                logger.warning("Inverse solution failed! Increase iteration count.")
            
        # Get bias values after this iteration
        bias_values = model.application.poke_bias(iBiasOption=bias_option)

        # Display bias values
        logger.info(f"Latest bias values ({ll};{kk}):")
        for ii, jj in enumerate(bias_values):
            logger.info(f"{ii}: {jj}")
        
    # Return the final bias values
    return model, False, "pyVSMEBias failed to converge in max outer loops.", bias_values


def random_search(
    model: GMOOAPI, 
    parallel_processes: int = 1, 
    csv_mode: bool = False, 
    unbias_vector: Optional[List[float]] = None
) -> Tuple[GMOOAPI, int, List[np.ndarray], List[np.ndarray]]:
    """
    Perform a random search of the input space to find optimal solutions.
    
    This function generates and evaluates a set of random input combinations,
    which can be used for global optimization or initial exploration.
    
    Args:
        model: The GMOOAPI object for this problem.
        parallel_processes: Number of parallel processes to use (default 1).
        csv_mode: Whether to use CSV files for external evaluation.
        unbias_vector: Optional multiplier for de-biasing the input variables.
        
    Returns:
        Tuple containing:
            GMOOAPI: The model with random search results loaded.
            int: Number of random cases executed.
            List[np.ndarray]: List of input arrays used.
            List[np.ndarray]: List of output arrays obtained.
    """
    # Initialize unbias vector if not provided
    if unbias_vector is None:
        unbias_vector = [1.0] * model.nVars.value

    # Generate random input vectors
    input_vectors = []
    for i in range(model.devCaseCount.value):
        input_vector = model.application.generate_random_exploration_case(
            [log.value == 1 for log in model.logspace]
        )
        input_vectors.append(input_vector)

    # Handle different modes of operation
    if csv_mode:
        # CSV Mode: Wait for external program to process cases
        # Export input cases to CSV
        with open(f"{model.vsme_input_filename}_RAND_VARS.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            for i, input_vector in enumerate(input_vectors):
                writer.writerow([f"Case{i}"] + input_vector.tolist())
        
        # Write 'done' file to signal availability of input data
        with open(f"{model.vsme_input_filename}_RAND_VARS.done", "w") as fd:
            pass
            
        logger.info(f"Random cases output to {model.vsme_input_filename}_RAND_VARS.csv. "
                    f"Waiting for outcomes to be processed by user's external model.")
                    
        # Read outcomes from the CSV file produced by the external program
        outcome_vectors = model.development.read_outcomes_csv(runs_type="RAND")
        
        # Store the results in the model
        for i, outcome_vector in enumerate(outcome_vectors, 1):
            model.development.load_case_results(i, outcome_vector)

        # Develop the VSME model
        model.development.develop_vsme()

        # Export VSME to file
        model.development.export_vsme()
        
        # Return None for output_vectors in CSV mode
        return model, len(input_vectors), input_vectors, None
            
    elif not csv_mode:
        if parallel_processes > 1:
            # Parallel execution mode
            output_vectors = []
            output_dict = {}
            
            # Divide work among processes
            batch_size = len(input_vectors) // parallel_processes
            
            # Execute in parallel
            with ProcessPoolExecutor(max_workers=parallel_processes) as executor:
                futures = []
                for i in range(parallel_processes):
                    start = i * batch_size
                    end = (i + 1) * batch_size if i != parallel_processes - 1 else len(input_vectors)
                    futures.append(executor.submit(
                        process_batch, start, end, i, input_vectors, model.modelFunction)
                    )

                # Collect results as they complete
                for future in as_completed(futures):
                    process_number, batch_output = future.result()
                    output_dict[process_number] = batch_output

                # Combine results in the original order
                for i in sorted(output_dict.keys()):
                    output_vectors.extend(output_dict[i])
        else:
            # Sequential execution mode
            output_vectors = []
            
            # Process each case one by one
            for i, input_vector in enumerate(input_vectors):
                logger.info(f"Running random case {i+1}/{len(input_vectors)}")
                output_vector = model.modelFunction(input_vector)
                
                # Validate output vector - check for NaN values
                try:
                    validate_nan(output_vector)
                except ValueError:
                    logger.error("Encountered NaN during random search. Exiting.")
                    sys.exit(1)
                    
                output_vectors.append(output_vector)
    else:
        logger.error(f"Invalid value {csv_mode} for `csv_mode` in random_search, exiting.")
        sys.exit(1)

    # Store the results in the model
    for i, (input_vector, output_vector) in enumerate(zip(input_vectors, output_vectors), 1):
        model.development.load_case_results(i, output_vector)

    # Develop the VSME model
    model.development.develop_vsme()

    # Export VSME to file
    model.development.export_vsme()

    return model, len(input_vectors), input_vectors, output_vectors