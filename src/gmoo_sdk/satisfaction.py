"""
Objective Satisfaction Checking for GMOO Optimization

This module provides functions to check objective satisfaction for different objective types
in multi-objective optimization problems. This includes checking constraints, bounds,
and target achievement rather than just convergence.

Authors: Matt Freeman, Jason Hopkins
Version: 2.0.0
"""

def check_satisfaction(outputs, targets, objective_types, uncertainty_minus, uncertainty_plus, default_l1_threshold=0.00001):
    """
    Check objective satisfaction for multiple objective types.
    
    This function evaluates whether current outputs satisfy the specified objectives,
    which may include exact targets, percentage/absolute error bounds, inequality
    constraints, or minimization/maximization goals.
    
    Args:
        outputs: Current output values
        targets: Target output values
        objective_types: List of objective types for each output
            0: Exact value match
            1: Percentage error tolerance
            2: Absolute error bounds
            11: Less than constraint
            12: Less than or equal constraint
            13: Greater than constraint
            14: Greater than or equal constraint
            21: Minimize (always returns satisfied=True)
            22: Maximize (always returns satisfied=True)
        uncertainty_minus: Lower uncertainty bounds (used for types 1,2)
        uncertainty_plus: Upper uncertainty bounds (used for types 1,2)
        default_l1_threshold: Default L1 norm threshold for type 0 objectives
        
    Returns:
        tuple: (satisfied, error_metrics)
            satisfied: Boolean indicating if all objectives are satisfied
            error_metrics: List of error/violation values for each objective
    """
    num_objectives = len(outputs)
    satisfied = True
    error_metrics = []
    l1_sum = 0.0  # Track total L1 error for type 0 objectives
    num_l1_objectives = 0  # Count how many type 0 objectives we have
    
    for i in range(num_objectives):
        obj_type = objective_types[i]
        current = outputs[i]
        target = targets[i]
        
        if obj_type == 0:  # Exact value
            error = abs(current - target)
            error_metrics.append(error)
            l1_sum += error
            num_l1_objectives += 1
            satisfied &= error < default_l1_threshold
            
        elif obj_type == 1:  # Percentage error
            error = abs((current - target) / target) * 100
            error_metrics.append(error)
            satisfied &= error <= uncertainty_plus[i]
            
        elif obj_type == 2:  # Absolute error bounds
            error = abs(current - target)
            error_metrics.append(error)
            satisfied &= (current >= target + uncertainty_minus[i] and 
                        current <= target + uncertainty_plus[i])
            
        elif obj_type == 11:  # Less than
            error = max(0, current - target)
            error_metrics.append(error)
            satisfied &= current < target
            
        elif obj_type == 12:  # Less than or equal
            error = max(0, current - target)
            error_metrics.append(error)
            satisfied &= current <= target
            
        elif obj_type == 13:  # Greater than
            error = max(0, target - current)
            error_metrics.append(error)
            satisfied &= current > target
            
        elif obj_type == 14:  # Greater than or equal
            error = max(0, target - current)
            error_metrics.append(error)
            satisfied &= current >= target
            
        elif obj_type in (21, 22):  # Minimize/Maximize
            # Minimize (21) and Maximize (22) objectives cannot 
            # be satisfied in the traditional sense, but we track 
            # the difference between the current and target.
            satisfied = True
            error = abs(current - target)
            error_metrics.append(error)
        
    # Check L1 norm satisfaction for type 0 objectives
    if num_l1_objectives > 0:
        l1_satisfied = l1_sum < default_l1_threshold
        satisfied &= l1_satisfied
    
    return satisfied, error_metrics