import numpy as np
from numpy.linalg import norm
import time
import gmoo_sdk.gmoo_encapsulation as gmapi
from functools import partial
import inspect
from typing import List, Optional

def generate_coefficients(n: int, m: int, seed: Optional[int] = None) -> List[List[float]]:
    """Generate coefficient matrix C[i,j] for the equations."""
    if seed is not None:
        np.random.seed(seed)
    return [np.random.uniform(-1, 1, n).tolist() for _ in range(m)]

def linear_model(x: np.ndarray, m: int = 3, n: int = None, C: Optional[List[List[float]]] = None, seed: Optional[int] = None) -> np.ndarray:
    """Linear model implementation."""
    x = np.array(x, ndmin=1)
    if n is None:
        n = len(x)
    
    if C is None:
        C = generate_coefficients(n, m, seed)
    
    y = np.zeros(m)
    for j in range(m):
        y[j] = sum((1 + C[j][i] * x[i]) for i in range(n))
    return y

def quadratic_model(x: np.ndarray, m: int = 3, n: int = 3, C: Optional[List[List[float]]] = None, seed: Optional[int] = None) -> np.ndarray:
    """Quadratic model implementation."""
    x = np.array(x, ndmin=1)
    if C is None:
        C = generate_coefficients(n, m, seed)
    
    y = np.zeros(m)
    for j in range(m):
        y[j] = sum((1 + C[j][i] * x[i])**2 for i in range(n))
    return y

def cubic_model(x: np.ndarray, m: int = 3, n: int = 3, C: Optional[List[List[float]]] = None, seed: Optional[int] = None) -> np.ndarray:
    """Cubic model implementation."""
    x = np.array(x, ndmin=1)
    if C is None:
        C = generate_coefficients(n, m, seed)
    
    y = np.zeros(m)
    for j in range(m):
        y[j] = sum((1 + C[j][i] * x[i])**3 for i in range(n))
    return y

def nonlinear_model1(x: np.ndarray, m: int = 3, n: int = 3, C: Optional[List[List[float]]] = None, seed: Optional[int] = None) -> np.ndarray:
    """First nonlinear model implementation."""
    x = np.array(x, ndmin=1)
    if C is None:
        C = generate_coefficients(n, m, seed)
    
    y = np.zeros(m)
    for j in range(m):
        base_term = (1 + C[j][0] * x[0])**2
        sum_term = sum((1 + C[j][i-1] * x[i-1]) * (1 + C[j][i] * x[i]) for i in range(1, n))
        y[j] = base_term + sum_term
    return y

def nonlinear_model2(x: np.ndarray, m: int = 3, n: int = 3, C: Optional[List[List[float]]] = None, seed: Optional[int] = None) -> np.ndarray:
    """Second nonlinear model implementation."""
    x = np.array(x, ndmin=1)
    if C is None:
        C = generate_coefficients(n, m, seed)
    
    y = np.zeros(m)
    for j in range(m):
        first_term = (1 + C[j][n-1] * x[n-1]) * (1 + C[j][0] * x[0])**2
        sum_term = sum((1 + C[j][i-1] * x[i-1]) * (1 + C[j][i] * x[i])**2 for i in range(1, n))
        y[j] = first_term + sum_term
    return y

def java_test_function(input_arr):
    """Nonlinear test function with nine outputs."""
    input_arr = np.array(input_arr, ndmin=1)
    v1, v2, v3 = input_arr[0], input_arr[1], input_arr[2]
    
    output1 = v1 * v2 + 6.54321
    output2 = v2 * v3
    output3 = v1 + v2 + v3
    output4 = 2.0 * v1 + v2 + 5.0
    output5 = 3.0 * v1 + v2
    output6 = 4.0 * v1 + v2 + 70.0
    output7 = 2.0 * v1 + v3
    output8 = 3.0 * v1 + v3 - 5.0
    output9 = 4.0 * v1 + v3
    
    return np.array([output1, output2, output3, output4, output5, output6, output7, output8, output9])

# Test Function Definitions
def simple_test_function(input_arr):
    """Nonlinear test function with continuous variables only."""
    input_arr = np.array(input_arr, ndmin=1)
    v01, v02, v03 = input_arr[0], input_arr[1], input_arr[2]
    
    o01 = v01 * v01 * v03 * v03
    o02 = (v01 - 2.0) * (v01 - 2.0) * v03
    o03 = (v02 - 2.0) * (v02 - 2.0) * v03
    o04 = v01 * v02 * v03 * v03
    
    return np.array([o01, o02, o03, o04])

def nonlinear_test_function(input_arr):
    """Nonlinear test function with continuous variables only."""
    input_arr = np.array(input_arr, ndmin=1)
    v01, v02, v03 = input_arr[0], input_arr[1], input_arr[2]
    
    o01 = v01 * v01 * v03 * v03 / v02
    o02 = (v01 - 2.0) * (v01 - 2.0) * v03 + v01 * v02
    o03 = (v02 - 2.0) * (v02 - 2.0) * v03 + v02 * v03
    o04 = v01 * v02 * v03 * v03 + v01 * v01
    
    return np.array([o01, o02, o03, o04])

def integer_test_function(input_arr):
    """Test function with mixed integer and continuous variables."""
    input_arr = np.array(input_arr, ndmin=1)
    v01, v02 = input_arr[0], input_arr[1]  # Real, Integer
    
    o01 = 1.0 * v01 + v02
    o02 = 2.0 * v01 + v02
    o03 = 3.0 * v01 + v02
    o04 = v02 + 7.0 * v01
    o05 = 1.5 * v01 + v02
    o06 = 2.5 * v01 + v02
    o07 = 3.5 * v01 + v02
    o08 = v01 * v02
    o09 = 1.5 * v01 * v02
    
    return np.array([o01, o02, o03, o04, o05, o06, o07, o08, o09])

def logical_test_function(input_arr):
    """Test function with logical/boolean variables."""
    input_arr = np.array(input_arr, ndmin=1)
    v01, v02 = input_arr[0], input_arr[1]  # Real, Logical
    
    logical_term = 100.0 if (v02 == 0 or v02 == 0.0) else -5.4321
    
    o01 = 1.0 * v01 * logical_term
    o02 = 2.0 * v01 * logical_term
    o03 = 3.0 * v01 + logical_term
    o04 = v01 / logical_term
    o05 = 1.5 * v01 * logical_term
    o06 = 2.5 * v01 * logical_term
    o07 = 3.5 * v01 + logical_term
    o08 = v01 * logical_term
    o09 = 1.5 * v01 + logical_term / 1000.0
    
    return np.array([o01, o02, o03, o04, o05, o06, o07, o08, o09])

def complex_mixed_type_function(input_arr):
    """Test function with mixed variable types: real, integer, and logical."""
    input_arr = np.array(input_arr, ndmin=1)
    
    v01 = input_arr[0]  # Real type variable
    v02 = input_arr[1]  # Integer type variable
    v03 = input_arr[2]  # Real type variable
    v04 = input_arr[3]  # Logical type variable
    v05 = input_arr[4]  # Logical type variable
    
    # coeff is defined by the logical variable
    if(v04 == 0 or v04 == 0.0):
        coeff = -7.3
    else:
        coeff = 0.0

    if(v05 == 0 or v05 == 0.0):
        categorical_coeff = 1.0
    else:
        categorical_coeff = 0.0

    o01 = 0.5 * v01 + v02 * categorical_coeff
    o02 = 0.5 * v02 + v03 * categorical_coeff
    o03 = 0.5 * v03 + coeff * categorical_coeff
    o04 = 0.5 * coeff + v05
    o05 = 0.5 * v05 + v01
    o06 = 0.5 * v01 + v03
    o07 = 0.5 * v02 + coeff * categorical_coeff
    o08 = 0.5 * v03 + v05 * categorical_coeff
    o09 = v05 * coeff * categorical_coeff
    o10 = v01 * v03 + 1.0 * coeff
    o11 = v01 * v03 + 2.0 * coeff
    o12 = v01 * v02 + 1.0 * coeff
    o13 = v01 * v02 + 2.0 * coeff * categorical_coeff
    o14 = v02 * v05 * categorical_coeff
    o15 = v02 * v05 * categorical_coeff

    return np.array([o01, o02, o03, o04, o05, o06, o07, o08, o09, o10, o11, o12, o13, o14, o15])

def categorical_test_function(input_arr):
    """Test function with categorical variables."""
    input_arr = np.array(input_arr, ndmin=1)
    v01, v02, v03 = input_arr[0:3]
    additive_terms = {
        1: 5.0, #1.0,
        2: 1.0, #5.0,
        3: 10.0,
        4: 100.0,
        5: 1000.0,  
        6: -5.0027
    }
    
    a = additive_terms.get(int(v03))
    if a is None:
        raise ValueError(f"Invalid categorical value: {v03}")
    
    # Linear combinations: c1*v01 + c2*v02 + c3*a
    outputs = [
        1.0*v01 + 0*v02 + 1.0*a,  # o01
        2.0*v01 + 0*v02 + 1.0*a,  # o02
        3.0*v01 + 0*v02 + 1.0*a,  # o03
        1.0*v01 + 0*v02 + 1.0*a,  # o04
        1.5*v01 + 0*v02 + 1.0*a,  # o05 
        3.0*v01 + 0*v02 + 1.0*a,  # o06
        4.5*v01 + 0*v02 + 1.0*a,  # o07
        1.0*v01 + 0*v02 + 1.0*a,  # o08
        1.5*v01 + 0*v02 + 1.0*a,  # o09
        0*v01 + 1.0*v02 + 1.0*a,  # o10
        0*v01 + 2.0*v02 + 1.0*a,  # o11
        0*v01 + 3.0*v02 + 1.0*a,  # o12
        0*v01 + 1.0*v02 + 1.0*a,  # o13
        0*v01 + 1.5*v02 + 1.0*a,  # o14
        0*v01 + 3.0*v02 + 1.0*a,  # o15
        0*v01 + 4.5*v02 + 1.0*a,  # o16
        0*v01 + 1.0*v02 + 1.0*a,  # o17
        0*v01 + 1.5*v02 + 1.0*a,  # o18
        1.0*v01 + 1.0*v02 + 1.0*a,  # o19
        1.0*v01 + 1.0*v02 + 2.0*a,   # o20
    ]
    
    return np.array(outputs)

TEST_CONFIGS = {
    'nonlinear2_huge': {
        'function': nonlinear_model2,
        'num_inputs': 35,
        'num_outputs': 150,
        'var_mins': [-1.0]*35,
        'var_maxs': [1.0]*35,
        'var_types': [1]*35,
        'categories_list': [[],]*35,
        'truth_case': [0.15]*35,
        'initial_guess': [0.0]*35,
        'objective_types': [0]*150,
        'uncertainty_minus': [0.0]*150,
        'uncertainty_plus': [0.0]*150
    },
    'linear_small': {
        'function': linear_model,
        'num_inputs': 2,
        'num_outputs': 2,
        'var_mins': [-1.0, -1.0],
        'var_maxs': [1.0, 1.0],
        'var_types': [1, 1],
        'categories_list': [[], []],
        'truth_case': [0.5, 0.5],
        'initial_guess': [0.0, 0.0],
        'objective_types': [0, 0],
        'uncertainty_minus': [0.0, 0.0],
        'uncertainty_plus': [0.0, 0.0]
    },
    'quadratic_medium': {
        'function': quadratic_model,
        'num_inputs': 3,
        'num_outputs': 3,
        'var_mins': [-1.0, -1.0, -1.0],
        'var_maxs': [1.0, 1.0, 1.0],
        'var_types': [1, 1, 1],
        'categories_list': [[], [], []],
        'truth_case': [0.3, 0.3, 0.3],
        'initial_guess': [0.0, 0.0, 0.0],
        'objective_types': [0, 0, 0],
        'uncertainty_minus': [0.0, 0.0, 0.0],
        'uncertainty_plus': [0.0, 0.0, 0.0]
    },
    'cubic_large': {
        'function': cubic_model,
        'num_inputs': 4,
        'num_outputs': 4,
        'var_mins': [-1.0, -1.0, -1.0, -1.0],
        'var_maxs': [1.0, 1.0, 1.0, 1.0],
        'var_types': [1, 1, 1, 1],
        'categories_list': [[], [], [], []],
        'truth_case': [0.2, 0.2, 0.2, 0.2],
        'initial_guess': [0.0, 0.0, 0.0, 0.0],
        'objective_types': [0, 0, 0, 0],
        'uncertainty_minus': [0.0, 0.0, 0.0, 0.0],
        'uncertainty_plus': [0.0, 0.0, 0.0, 0.0]
    },
    'nonlinear1_medium': {
        'function': nonlinear_model1,
        'num_inputs': 4,
        'num_outputs': 3,
        'var_mins': [-1.0, -1.0, -1.0, -1.0],
        'var_maxs': [1.0, 1.0, 1.0, 1.0],
        'var_types': [1, 1, 1, 1],
        'categories_list': [[], [], [], []],
        'truth_case': [0.25, 0.25, 0.25, 0.25],
        'initial_guess': [0.0, 0.0, 0.0, 0.0],
        'objective_types': [0, 0, 0],
        'uncertainty_minus': [0.0, 0.0, 0.0],
        'uncertainty_plus': [0.0, 0.0, 0.0]
    },
    'nonlinear2_large': {
        'function': nonlinear_model2,
        'num_inputs': 5,
        'num_outputs': 4,
        'var_mins': [-1.0, -1.0, -1.0, -1.0, -1.0],
        'var_maxs': [1.0, 1.0, 1.0, 1.0, 1.0],
        'var_types': [1, 1, 1, 1, 1],
        'categories_list': [[], [], [], [], []],
        'truth_case': [0.15, 0.15, 0.15, 0.15, 0.15],
        'initial_guess': [0.0, 0.0, 0.0, 0.0, 0.0],
        'objective_types': [0, 0, 0, 0],
        'uncertainty_minus': [0.0, 0.0, 0.0, 0.0],
        'uncertainty_plus': [0.0, 0.0, 0.0, 0.0]
    },
    'java': {
        'function': java_test_function,
        'num_inputs': 3,
        'num_outputs': 9,
        'var_mins': [0.3, 0.3, 0.3],
        'var_maxs': [0.5, 0.5, 0.5],
        'var_types': [1, 1, 1],
        'categories_list': [[], [], []],
        'truth_case': [0.4321, 0.321, 0.415],
        'initial_guess': [0.4, 0.4, 0.4],
        'objective_types': [0] * 9,
        'uncertainty_minus': [0] * 9,
        'uncertainty_plus': [0] * 9
    },
    'simple': {
        'function': simple_test_function,
        'num_inputs': 3,
        'num_outputs': 4,
        'var_mins': [0.0, 0.0, 0.0],
        'var_maxs': [10.0, 10.0, 10.0],
        'var_types': [1, 1, 1],
        'categories_list': [[], [], []],
        'truth_case': [5.4321, 5.4321, 5.4321],
        'initial_guess': [5.0, 5.0, 5.0],
        'objective_types': [1, 1, 1, 1],
        'uncertainty_minus': [-2.0, -2.0, -2.0, -2.0],
        'uncertainty_plus': [2.0, 2.0, 2.0, 2.0]
    },
    'nonlinear': {
        'function': nonlinear_test_function,
        'num_inputs': 3,
        'num_outputs': 4,
        'var_mins': [0.0, 0.0, 0.0],
        'var_maxs': [10.0, 10.0, 10.0],
        'var_types': [1, 1, 1],
        'categories_list': [[], [], []],
        'truth_case': [5.4321, 5.4321, 5.4321],
        'initial_guess': [5.0, 5.0, 5.0],
        'objective_types': [0, 0, 0, 0],
        'uncertainty_minus': [0.0, 0.0, 0.0, 0.0],
        'uncertainty_plus': [0.0, 0.0, 0.0, 0.0]
    },
    'integer': {
        'function': integer_test_function,
        'num_inputs': 2,
        'num_outputs': 9,
        'var_mins': [0.0, 0],
        'var_maxs': [10.0, 7],
        'var_types': [1, 2],
        'categories_list': [[], []],
        'truth_case': [5.4321, 4],
        'initial_guess': [6.21, 2],
        'objective_types': [0] * 9,
        'uncertainty_minus': [0] * 9,
        'uncertainty_plus': [0] * 9
    },
    'logical': {
        'function': logical_test_function,
        'num_inputs': 2,
        'num_outputs': 9,
        'var_mins': [0.0, 0],
        'var_maxs': [10.0, 1],
        'var_types': [1, 3],
        'categories_list': [[], []],
        'truth_case': [5.4321, 1],
        'initial_guess': [6.21, 0],
        'objective_types': [1] * 9,
        'uncertainty_minus': [2.0] * 9,
        'uncertainty_plus': [2.0] * 9
    },
    'mixed_types': {
        'function': complex_mixed_type_function,
        'num_inputs': 5,
        'num_outputs': 15,
        'var_mins': [-15.0, 199.00, 30.0, 0.0, 0.0],
        'var_maxs': [-0.01, 300.0, 50.0, 1.0, 1.0],
        'var_types': [1, 2, 1, 3, 3],  # 1=real, 2=integer, 3=logical
        'categories_list': [[], [], [], [], []],
        'truth_case': [-1.2345, 250.0, 32.0, 0.0, 1.0],
        'initial_guess': [-5.0, 225.0, 40.0, 0.0, 1.0],
        'objective_types': [0] * 15, # Exact match, L1 convergence
        'uncertainty_minus': [0] * 15,
        'uncertainty_plus': [0] * 15
    },
    'categorical': {
        'function': categorical_test_function,
        'num_inputs': 3,
        'num_outputs': 20,
        'var_mins': [0.0, 0.0, 1.0],  # Min value for categorical is 1
        'var_maxs': [10.0, 10.0, 4.0],  # Max value represents number of categories
        'var_types': [1, 1, 4],  # 1=real, 4=categorical
        'categories_list': [[], [], ["cat", "dog", "fish", "mouse"]],
        'truth_case': [5.4321, 1.2345, 3.0],
        'initial_guess': [5.0, 5.0, 3.0],
        'objective_types': [0] * 20,  # Exact match for all outputs
        'uncertainty_minus': [0.0] * 20,
        'uncertainty_plus': [0.0] * 20
    },
    'multi_obj': {
        'function': simple_test_function,
        'num_inputs': 3,
        'num_outputs': 4,
        'var_mins': [5.0, 5.0, 5.0],
        'var_maxs': [6.0, 6.0, 6.0],
        'var_types': [1, 1, 1],
        'categories_list': [[], [], []],
        'truth_case': [5.4321, 5.4321, 5.4321],
        'initial_guess': [5.5, 5.5, 5.5],
        'objective_types': [
            0,    # Exact match for first output
            1,    # Percentage-based error for second output
            12,   # Less than or equal for third output
            21    # Minimize to target for fourth output
        ],
        'uncertainty_minus': [0.0, -2.0, 0.0, 0.0],  # Only used for type 1 and 2
        'uncertainty_plus': [0.0, 2.0, 0.0, 0.0]    # Only used for type 1 and 2
    }
}

def check_convergence(outputs, targets, objective_types, uncertainty_minus, uncertainty_plus, default_l1_convergence=0.00001):
    """
    Check convergence using multiple objective types.
    
    Args:
        outputs: Current output values
        targets: Target output values
        objective_types: List of objective types for each output
        uncertainty_minus: Lower uncertainty bounds (used for types 1,2)
        uncertainty_plus: Upper uncertainty bounds (used for types 1,2)
        default_l1_convergence: Default convergence threshold for L1 norm
        
    Returns:
        tuple: (satisfied, error_metrics)
    """
    num_objectives = len(outputs)
    satisfied = True
    error_metrics = []
    
    for i in range(num_objectives):
        obj_type = objective_types[i]
        current = outputs[i]
        target = targets[i]
        
        if obj_type == 0:  # Exact value
            error = abs(current - target)
            error_metrics.append(error)
            satisfied &= error < default_l1_convergence
            
        elif obj_type == 1:  # Percentage error
            error = abs((current - target) / target) * 100
            error_metrics.append(error)
            satisfied &= error <= uncertainty_plus[i]
            
        elif obj_type == 2:  # Absolute error bounds
            error = abs(current - target)
            error_metrics.append(error)
            satisfied &= (current >= target - uncertainty_minus[i] and 
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
            error = abs(current - target)
            error_metrics.append(error)
            if obj_type == 21:
                satisfied &= current <= target
            else:
                satisfied &= current >= target
    
    return satisfied, error_metrics

def run_test(test_name, iterations=100, train_vsme=True):
    """
    Run a specific test case with the given parameters.
    
    Args:
        test_name: Name of the test configuration to run
        iterations: Maximum number of iterations for optimization
        train_vsme: Whether to train the surrogate model before optimization
    
    Returns:
        tuple: (best_inputs, best_outputs, satisfied, best_error, iterations_taken)
    """
    if test_name not in TEST_CONFIGS:
        raise ValueError(f"Unknown test: {test_name}. Available tests: {list(TEST_CONFIGS.keys())}")
    
    config = TEST_CONFIGS[test_name]
    print(f"\nRunning {test_name} test...")
    
    # Extract configuration
    var_mins = config['var_mins']
    var_maxs = config['var_maxs']
    var_types = config['var_types']
    num_input_vars = config['num_inputs']
    num_output_vars = config['num_outputs']
    categories_list = config['categories_list']
    truthcase_inputs = config['truth_case']
    initial_guess = config['initial_guess']
    objective_types = config['objective_types']
    uncertainty_minus = config['uncertainty_minus']
    uncertainty_plus = config['uncertainty_plus']
    base_func = config['function']
    
    # Check function parameters
    params = inspect.signature(base_func).parameters
    partial_kwargs = {}
    if 'm' in params:
        partial_kwargs['m'] = config['num_outputs']
    if 'n' in params:
        partial_kwargs['n'] = config['num_inputs']
    if 'seed' in params:
        partial_kwargs['seed'] = 43
    model_function = partial(base_func, **partial_kwargs) if partial_kwargs else base_func
    
    # Training phase
    if train_vsme:
        input_dev = gmapi.get_development_cases_encapsulation(
            var_mins, var_maxs, test_name,
            varTypes=var_types,
            categories_list=categories_list
        )
        
        output_dev = []
        for case in input_dev:
            evaluation = model_function(case)
            output_dev.append(evaluation)
            
        gmapi.load_development_cases_encapsulation(
            output_dev, var_mins, var_maxs, num_output_vars, test_name,
            varTypes=var_types,
            categories_list=categories_list
        )
    
    # Initialize optimization variables
    objectives_array = model_function(truthcase_inputs)
    next_input_vars = initial_guess
    next_output_vars = model_function(next_input_vars)
    
    print(f"Initial guess inputs: {next_input_vars}")
    print(f"Initial guess outputs: {next_output_vars}")
    print("\nObjective types and targets:")
    for i, (obj_type, target) in enumerate(zip(objective_types, objectives_array)):
        print(f"Output {i+1}: Type {obj_type}, Target {target:.4f}")
    
    # Optimization loop
    best_error_sum = float("inf")
    bestcase = None
    bestoutput = None
    satisfied = False
    learned_case_inputs, learned_case_outputs = [], []
    
    print("\nStarting inverse optimization...")
    for ii in range(1, iterations + 1):
        next_input_vars, l1norm_full, learned_case_inputs, learned_case_outputs = gmapi.inverse_encapsulation(
            objectives_array,
            var_mins,
            var_maxs,
            num_output_vars,
            next_input_vars,
            next_output_vars,
            test_name,
            learned_case_inputs=learned_case_inputs,
            learned_case_outputs=learned_case_outputs,
            inverse_iteration=ii,
            varTypes=var_types,
            categories_list=categories_list,
            objectiveTypes=objective_types,
            objectives_uncertainty_minus=uncertainty_minus,
            objectives_uncertainty_plus=uncertainty_plus
        )
        
        next_output_vars = model_function(next_input_vars)
        satisfied, error_metrics = check_convergence(
            next_output_vars,
            objectives_array,
            objective_types,
            uncertainty_minus,
            uncertainty_plus
        )
        
        # Use sum of error metrics for tracking best case
        current_error_sum = sum(error_metrics)
        
        if satisfied:
            bestcase = next_input_vars.copy()
            bestoutput = next_output_vars.copy()
            best_error_sum = current_error_sum
            print(f"\nConverged in {ii} iterations!")
            print("Final errors:")
            for i, error in enumerate(error_metrics):
                print(f"Output {i+1} (Type {objective_types[i]}): {error:.6f}")
            break
        
        elif current_error_sum < best_error_sum:
            best_error_sum = current_error_sum
            bestcase = next_input_vars.copy()
            bestoutput = next_output_vars.copy()
        
        if ii % 10 == 0:  # Print progress every 10 iterations
            print(f"Iteration {ii}: Best total error = {best_error_sum:.6f}")
        
        if ii == iterations:
            print(f"\nMaximum iterations reached. Best total error: {best_error_sum:.6f}")
            print("Final errors:")
            for i, error in enumerate(error_metrics):
                print(f"Output {i+1} (Type {objective_types[i]}): {error:.6f}")
    
    iterations_taken = ii if satisfied else None
    return bestcase, bestoutput, satisfied, best_error_sum, iterations_taken

if __name__ == "__main__":
    # Default parameters
    iterations = 200
    
    print("Running all test cases...")
    print("=" * 60)
    print(f"Parameters: iterations={iterations}")
    print("=" * 60)
    
    # Store results for comparison
    all_results = {}
    
    # Run each test in sequence
    for test_name in TEST_CONFIGS.keys(): #["categorical"]: #
        print("\n" + "=" * 60)
        print(f"Starting {test_name.upper()} test")
        print("=" * 60)
        
        try:
            best_inputs, best_outputs, satisfied, final_error, iters = run_test(
                test_name,
                iterations=iterations
            )
            
            all_results[test_name] = {
                'best_inputs': best_inputs,
                'best_outputs': best_outputs,
                'target_inputs': TEST_CONFIGS[test_name]['truth_case'],
                'target_outputs': TEST_CONFIGS[test_name]['function'](TEST_CONFIGS[test_name]['truth_case']),
                'final_error': final_error,
                'success': satisfied,
                'iterations': iters
            }
            
        except Exception as e:
            print(f"Error in {test_name} test: {str(e)}")
            all_results[test_name] = {'error': str(e)}
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY OF ALL TESTS")
    print("=" * 60)
    
    for test_name, results in all_results.items():
        print(f"\n{test_name.upper()}:")
        if 'error' in results:
            print(f"  Failed with error: {results['error']}")
        else:
            print(f"  Success: {results['success']}")
            print(f"  Final Total Error: {results['final_error']:.6f}")
            
            # Print objective-specific results
            print("  Objectives:")
            objective_types = TEST_CONFIGS[test_name]['objective_types']
            for i, (best, target, obj_type) in enumerate(zip(
                results['best_outputs'],
                results['target_outputs'],
                objective_types
            )):
                type_desc = {
                    0: "Exact Match",
                    1: "Percentage Error",
                    2: "Absolute Error",
                    11: "Less Than",
                    12: "Less Than or Equal",
                    13: "Greater Than",
                    14: "Greater Than or Equal",
                    21: "Minimize",
                    22: "Maximize"
                }.get(obj_type, f"Type {obj_type}")
                
                print(f"    Output {i+1}: {type_desc}")
                print(f"      Target: {target:.4f}")
                print(f"      Achieved: {best:.4f}")
                
                # Print uncertainty bounds for types 1 and 2
                if obj_type in [1, 2]:
                    minus = TEST_CONFIGS[test_name]['uncertainty_minus'][i]
                    plus = TEST_CONFIGS[test_name]['uncertainty_plus'][i]
                    if obj_type == 1:
                        print(f"      Uncertainty: ±{plus}%")
                    else:
                        print(f"      Bounds: [{target-minus:.4f}, {target+plus:.4f}]")
            
            print("  Input Variables:")
            for i, (best, target) in enumerate(zip(results['best_inputs'], 
                                                 results['target_inputs'])):
                var_type = TEST_CONFIGS[test_name]['var_types'][i]
                type_name = {1: 'Real', 2: 'Integer', 
                            3: 'Logical', 4: 'Categorical'}[var_type]
                print(f"    {type_name}: {best:.4f} (target: {target:.4f})")
    
    # Print overall success rate
    successful_tests = sum(1 for r in all_results.values() 
                         if 'success' in r and r['success'])
    total_tests = len(all_results)
    print("\n" + "=" * 60)
    print(f"Overall Success Rate: {successful_tests}/{total_tests} tests passed")
    print("=" * 60)

    # Add this after the success rate print:
    print("\nConvergence Summary:")
    print("-" * 60)
    print(f"{'Test Name':<20} {'Converged?':<12} {'Iterations':<10}")
    print("-" * 60)
    for test_name, results in all_results.items():
        if 'error' in results:
            status = "ERROR"
            iters = "N/A"
        else:
            status = "Yes" if results['success'] else "No"
            iters = str(results['iterations']) if results['iterations'] else ">"+ str(iterations)
        print(f"{test_name:<20} {status:<12} {iters:<10}")