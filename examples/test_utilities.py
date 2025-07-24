"""
Test Utilities for GMOO SDK

This module provides common utilities for running and reporting GMOO optimization tests,
reducing redundancy between stateful and stateless test runners.

Authors: Matt Freeman, Jason Hopkins
Version: 2.0.0
"""

import numpy as np
import inspect
import logging
from functools import partial
from typing import List, Dict, Tuple, Optional, Callable, Any
from functions.example_functions import generate_coefficients

logger = logging.getLogger(__name__)


# Constant mappings
OBJECTIVE_TYPE_DESCRIPTIONS = {
    0: "Exact Match",
    1: "Percentage Error", 
    2: "Absolute Error",
    11: "Less Than",
    12: "Less Than or Equal",
    13: "Greater Than",
    14: "Greater Than or Equal",
    21: "Minimize",
    22: "Maximize"
}

# Variable type mapping (1-based as used by the DLL)
VARIABLE_TYPE_NAMES = {
    1: 'Real',       # Continuous
    2: 'Integer',
    3: 'Logical',
    4: 'Categorical'
}


def prepare_model_function(base_function: Callable, config: Dict) -> Callable:
    """
    Prepare a model function by handling coefficient generation and partial application.
    
    Args:
        base_function: The base model function from the test configuration
        config: The test configuration dictionary containing num_inputs, num_outputs, etc.
        
    Returns:
        Callable: The prepared model function ready for evaluation
    """
    # Check function parameters
    params = inspect.signature(base_function).parameters
    partial_kwargs = {}
    
    # Handle seed parameter (for coefficient generation)
    if 'seed' in params:
        C = generate_coefficients(
            n=config['num_inputs'],
            m=config['num_outputs'],
            seed=43
        )
        partial_kwargs['C'] = C
        partial_kwargs['n'] = config['num_inputs']
        partial_kwargs['m'] = config['num_outputs']
    
    # Handle other common parameters
    if 'm' in params and 'm' not in partial_kwargs:
        partial_kwargs['m'] = config['num_outputs']
    if 'n' in params and 'n' not in partial_kwargs:
        partial_kwargs['n'] = config['num_inputs']
    
    # Create partial function if needed
    return partial(base_function, **partial_kwargs) if partial_kwargs else base_function


def generate_random_starting_points(num_pipes: int, 
                                  var_mins: List[float],
                                  var_maxs: List[float], 
                                  var_types: List[int],
                                  initial_guess: Optional[List[float]] = None) -> List[List[float]]:
    """
    Generate random starting points for multiple optimization pipes.
    
    Args:
        num_pipes: Number of parallel optimization pipes
        var_mins: Minimum values for each variable
        var_maxs: Maximum values for each variable
        var_types: Variable types (1=real, 2=integer, 3=logical, 4=categorical)
        initial_guess: Optional initial guess for the first pipe
        
    Returns:
        List of starting points for each pipe
    """
    starting_points = []
    num_vars = len(var_mins)
    
    for pipe in range(num_pipes):
        if pipe == 0 and initial_guess is not None:
            # Use provided initial guess for first pipe
            starting_points.append(list(initial_guess))
        else:
            # Generate random starting point
            random_inputs = []
            for i in range(num_vars):
                min_val = var_mins[i]
                max_val = var_maxs[i]
                
                # Handle different variable types (1-based as used by DLL)
                if var_types[i] == 1:  # Real/continuous type
                    random_val = np.random.uniform(min_val, max_val)
                elif var_types[i] == 2:  # Integer type
                    random_val = int(np.random.uniform(min_val, max_val + 1))
                elif var_types[i] == 3:  # Logical type
                    random_val = int(np.random.randint(min_val, max_val + 1))
                elif var_types[i] == 4:  # Categorical type
                    random_val = int(np.random.randint(min_val, max_val + 1))
                else:  # Unknown type, treat as real
                    random_val = np.random.uniform(min_val, max_val)
                
                random_inputs.append(random_val)
            
            starting_points.append(random_inputs)
            
        logger.info(f"Starting point for pipe {pipe}: {starting_points[pipe]}")
    
    return starting_points


def print_test_summary(test_name: str, results: Dict[str, Any], test_config: Dict) -> None:
    """
    Print a formatted summary of test results.
    
    Args:
        test_name: Name of the test
        results: Dictionary containing test results
        test_config: Test configuration from TEST_CONFIGS
    """
    print(f"\n{test_name.upper()}:")
    
    if 'error' in results:
        print(f"  Failed with error: {results['error']}")
        return
    
    print(f"  Success: {results['success']}")
    print(f"  Final Total Error: {results['final_error']:.6f}")
    
    # Print objective-specific results
    if results.get('best_outputs') and results.get('target_outputs'):
        print("  Objectives:")
        objective_types = test_config['objective_types']
        
        for i, (best, target, obj_type) in enumerate(zip(
            results['best_outputs'],
            results['target_outputs'],
            objective_types
        )):
            type_desc = OBJECTIVE_TYPE_DESCRIPTIONS.get(obj_type, f"Type {obj_type}")
            
            print(f"    Output {i+1}: {type_desc}")
            print(f"      Target: {target:.4f}")
            print(f"      Achieved: {best:.4f}")
            
            # Print uncertainty bounds for types 1 and 2
            if obj_type in [1, 2]:
                minus = test_config['uncertainty_minus'][i]
                plus = test_config['uncertainty_plus'][i]
                if obj_type == 1:  # Percentage error
                    print(f"      Uncertainty: ±{plus}%")
                else:  # Absolute error
                    print(f"      Bounds: [{target-minus:.4f}, {target+plus:.4f}]")
    
    # Print input variables
    if results.get('best_inputs') and results.get('target_inputs'):
        print("  Input Variables:")
        for i, (best, target) in enumerate(zip(results['best_inputs'], 
                                              results['target_inputs'])):
            if target is not None:
                var_type = test_config['var_types'][i]
                type_name = VARIABLE_TYPE_NAMES.get(var_type, f'Type {var_type}')
                print(f"    {type_name}: {best:.4f} (target: {target:.4f})")


def print_satisfaction_summary(all_results: Dict[str, Dict], test_configs: Dict, 
                             iterations: int) -> None:
    """
    Print a satisfaction summary table for all test results.
    
    Args:
        all_results: Dictionary of results organized by VSME version and test name
        test_configs: TEST_CONFIGS dictionary
        iterations: Maximum iterations used
    """
    print("\nSatisfaction Summary:")
    print("-" * 93)
    print(f"{'VSME Version':<20} {'Test Name':<20} {'Inputs':<12} {'Outputs':<12} {'Satisfied?':<12} {'Iterations':<10}")
    print("-" * 93)
    
    for vsme_version, version_results in all_results.items():
        # Extract just the filename from the full path
        vsme_short = vsme_version.split('\\')[-1] if '\\' in vsme_version else vsme_version
        vsme_short = vsme_short[-17:] if len(vsme_short) > 17 else vsme_short
        
        for test_name, results in version_results.items():
            if 'error' in results:
                status = "ERROR"
                iters = "N/A"
            else:
                status = "Yes" if results['success'] else "No"
                iters = str(results['iterations']) if results['iterations'] else f">{iterations}"
            
            print(f"{vsme_short:<20} {test_name:<20} "
                  f"{test_configs[test_name]['num_inputs']:<12} "
                  f"{test_configs[test_name]['num_outputs']:<12} "
                  f"{status:<12} {iters:<10}")


def print_overall_summary(all_results: Dict[str, Dict]) -> None:
    """
    Print overall success rate summary.
    
    Args:
        all_results: Dictionary of results organized by VSME version
    """
    for vsme_version, version_results in all_results.items():
        successful_tests = sum(1 for r in version_results.values() 
                             if 'success' in r and r['success'])
        total_tests = len(version_results)
        
        print("\n" + "=" * 60)
        print(f"Overall Success Rate: {successful_tests}/{total_tests} tests passed")
        print("=" * 60)


def determine_target_outputs(config: Dict) -> List[float]:
    """
    Determine the target outputs for a test configuration.
    
    Args:
        config: Test configuration dictionary
        
    Returns:
        List of target output values
    """
    if 'target_outputs' in config:
        return config['target_outputs']
    elif 'truth_case' in config and config['truth_case'] is not None:
        # Prepare the function and evaluate it
        model_func = prepare_model_function(config['function'], config)
        return list(model_func(config['truth_case']))
    else:
        # No target outputs defined
        return []


def safe_evaluate_model(model_function: Callable, inputs: List[float], 
                       case_id: Optional[Any] = None) -> Optional[List[float]]:
    """
    Safely evaluate a model function with error handling.
    
    Args:
        model_function: The model function to evaluate
        inputs: Input values
        case_id: Optional identifier for logging
        
    Returns:
        List of output values or None if evaluation failed
    """
    try:
        outputs = model_function(inputs)
        return list(outputs)
    except Exception as e:
        case_str = f" {case_id}" if case_id is not None else ""
        logger.warning(f"Failed to evaluate{case_str}: {e}")
        return None