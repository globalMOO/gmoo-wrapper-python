"""
Example Functions for GMOO Test Suite

This module contains various test functions used to validate the GMOO optimization
functionality across different problem types including linear, nonlinear, mixed
variable types, and constrained optimization problems.

Authors: Matt Freeman, Jason Hopkins  
Version: 2.0.0
"""

from typing import List, Optional
import numpy as np

def generate_coefficients(n: int, m: int, seed: Optional[int] = None) -> List[List[float]]:
    """Generate coefficient matrix C[i,j] for the equations."""
    if seed is not None:
        np.random.seed(seed)
    return [np.random.uniform(-1, 1, n).tolist() for _ in range(m)]

# =====================================
# Original Test Functions from example_suite.py
# =====================================

def constrained_maximization_function(inputs):
    """
    Two-humped function with one local maximum and one global maximum.
    
    The function has three outputs:
    1. The objective to maximize - a two-humped function with:
       - Local maximum around (3.12, 3.12) with value ~10.02
       - Global maximum around (8.06, 8.06) with value ~15.86
       - A gently sloping surface across the domain  
    2. First constraint: x1 + x2, will be constrained to < 11.5
    3. Second constraint: x1 * x2, will be constrained to < 22
    
    When both constraints are applied, the global maximum at (8, 8) becomes
    unreachable, forcing the optimizer to find the local maximum at (3.12, 3.12).
    
    Args:
        inputs: List of two input values [x1, x2]
        
    Returns:
        List of three outputs [objective_to_maximize, constraint1, constraint2]
    """
    x1, x2 = inputs
    
    # Primary objective to maximize - a two-humped function using rational polynomials
    # First hump (local maximum) around (3.12, 3.12)
    # Second hump (global maximum) around (8, 8)
    local_hump = 7.0 / (1 + 0.3 * ((x1 - 3)**2 + (x2 - 3)**2))
    global_hump = 9.0 / (1 + 0.3 * ((x1 - 8)**2 + (x2 - 8)**2))
    
    # Add a gentle slope across the domain
    gentle_slope = 0.4 * (x1 + x2)
    
    objective_to_maximize = local_hump + global_hump + gentle_slope
    
    # Constraint 1: Should be less than a threshold
    # This will exclude the global maximum region
    constraint1 = x1 + x2  # Must be less than 11.5, which excludes the point (8, 8)
    
    # Constraint 2: Another constraint to further shape the feasible region
    constraint2 = x1 * x2  # Must be less than 22, which also excludes points near (8, 8)
    
    return np.array([objective_to_maximize, constraint1, constraint2])

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

def JavaTestFunction(inputArr):
    """Nonlinear test function with nine outputs."""
    inputArr = np.array(inputArr, ndmin=1)
    v1, v2, v3 = inputArr[0], inputArr[1], inputArr[2]
    
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
def SimpleTestFunction(inputArr):
    """Nonlinear test function with continuous variables only."""
    inputArr = np.array(inputArr, ndmin=1)
    v01, v02, v03 = inputArr[0], inputArr[1], inputArr[2]
    
    o01 = v01 * v01 * v03 * v03
    o02 = (v01 - 2.0) * (v01 - 2.0) * v03
    o03 = (v02 - 2.0) * (v02 - 2.0) * v03
    o04 = v01 * v02 * v03 * v03
    
    return np.array([o01, o02, o03, o04])

def NonlinearTestFunction(inputArr):
    """Nonlinear test function with continuous variables only."""
    inputArr = np.array(inputArr, ndmin=1)
    v01, v02, v03 = inputArr[0], inputArr[1], inputArr[2]
    
    o01 = v01 * v01 * v03 * v03 / (v02+2.20123)
    o02 = (v01 - 2.0) * (v01 - 2.0) * v03 + v01 * v02
    o03 = (v02 - 2.0) * (v02 - 2.0) * v03 + v02 * v03
    o04 = v01 * v02 * v03 * v03 + v01 * v01
    
    return np.array([o01, o02, o03, o04])

def IntegerTestFunction(inputArr):
    """Test function with mixed integer and continuous variables."""
    inputArr = np.array(inputArr, ndmin=1)
    v01, v02 = inputArr[0], inputArr[1]  # Real, Integer
    
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

def LogicalTestFunction(inputArr):
    """Test function with logical/boolean variables."""
    inputArr = np.array(inputArr, ndmin=1)
    v01, v02 = inputArr[0], inputArr[1]  # Real, Logical
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

def ComplexMixedTypeFunction(inputArr):
    """Test function with mixed variable types: real, integer, and logical."""
    inputArr = np.array(inputArr, ndmin=1)
    
    v01 = inputArr[0]  # Real type variable
    v02 = inputArr[1]  # Integer type variable
    v03 = inputArr[2]  # Real type variable
    v04 = inputArr[3]  # Logical type variable
    v05 = inputArr[4]  # Logical type variable
    
    # coeff is defined by the logical variable
    if(v04 == 0 or v04 == 0.0):
        logical_coeff_1 = 1.3
    else:
        logical_coeff_1 = 2.5

    if(v05 == 0 or v05 == 0.0):
        logical_coeff_2 = 0.9
    else:
        logical_coeff_2 = 0.7

    o01 = 0.5 * v01 + v02 * logical_coeff_2
    o02 = 0.5 * v02 + v03 * logical_coeff_2
    o03 = 0.5 * v03 + logical_coeff_1 * logical_coeff_2
    o04 = 0.5 * logical_coeff_1 + v05
    o05 = 0.5 * v05 + v01
    o06 = 0.5 * v01 + v03
    o07 = 0.5 * v02 + logical_coeff_1 * logical_coeff_2
    o08 = 0.5 * v03 + v05 * logical_coeff_2
    o09 = v05 * logical_coeff_1 * logical_coeff_2
    o10 = v01 * v03 + 1.0 * logical_coeff_1
    o11 = v01 * v03 + 2.0 * logical_coeff_1
    o12 = v01 * v02 + 1.0 * logical_coeff_1
    o13 = v01 * v02 + 2.0 * logical_coeff_1 * logical_coeff_2
    o14 = v02 * v05 * logical_coeff_2
    o15 = v02 * v05 * logical_coeff_2

    return np.array([o01, o02, o03, o04, o05, o06, o07, o08, o09, o10, o11, o12, o13, o14, o15])

def CategoricalTestFunction(inputArr):
   inputArr = np.array(inputArr, ndmin=1)
   v01, v02, v03 = inputArr[0:3]
   additive_terms = {
       1: 5.883322,
       2: 7.343434, #0.51515,
       3: 6.121212, #17.654321,
       4: 3.010101,
       5: -1700.703,  
       6: -5.0027
   }
   
   a = additive_terms.get(int(v03))
   if a is None:
       raise ValueError(f"Invalid categorical value: {v03}")
   
   # Linear combinations: c1*v01 + c2*v02 + c3*a
   outputs = [
       1.0*v01 + 0*v02 + 1.0*a,  # o01
       2.0*v01 + 0*v02 + 2.0*a,  # o02
       3.0*v01 + 0*v02 + 1.0*a,  # o03
       1.0*v01 + 0*v02 + 2.0*a,  # o04
       1.5*v01 + 0*v02 + 1.0*a,  # o05 
       3.0*v01 + 0*v02 + 2.0*a,  # o06
       4.5*v01 + 0*v02 + 1.0*a,  # o07
       1.0*v01 + 0*v02 + 2.0*a,  # o08
       1.5*v01 + 0*v02 + 1.0*a,  # o09
       0*v01 + 1.0*v02 + 2.0*a,  # o10
       0*v01 + 2.0*v02 + 1.0*a,  # o11
       0*v01 + 3.0*v02 + 2.0*a,  # o12
       0*v01 + 1.0*v02 + 1.0*a,  # o13
       0*v01 + 1.5*v02 + 2.0*a,  # o14
       0*v01 + 3.0*v02 + 1.0*a,  # o15
       0*v01 + 4.5*v02 + 2.0*a,  # o16
       0*v01 + 1.0*v02 + 1.0*a,  # o17
       0*v01 + 1.5*v02 + 2.0*a,  # o18
       1.0*v01 + 1.0*v02 + 0.0*a,  # o19
       1.0*v01 + 1.0*v02 + 1.0*a,  # o20
       1.0*v01 + 1.0*v02 + 2.0*a,   # o21
       1.0*v01 + 1.0*v02 + 3.0*a,
       1.0*v01 + 1.0*v02 + 4.0*a,
       1.0*v01 + 1.0*v02 + 5.0*a,
       1.0*v01 + 1.0*v02 * 0.0*a,  # o1
       1.0*v01 + 1.0*v02 * 1.0*a,  # o1
       1.0*v01 + 1.0*v02 * 2.0*a,   # o20
       1.0*v01 + 1.0*v02 * 3.0*a,
       1.0*v01 + 1.0*v02 * 4.0*a,
       1.0*v01 + 1.0*v02 * 5.0*a,
   ]
   
   return np.array(outputs)

def ComplexMixedFunction(inputArr):
    """Test function combining exponential, polynomial, and logarithmic components."""
    inputArr = np.array(inputArr, ndmin=1)
    v1, v2, v3, v4, v5 = inputArr[0], inputArr[1], inputArr[2], inputArr[3], inputArr[4]
    
    # Exponential components
    o1 = np.exp(0.5 * v1) + v2**2
    o2 = np.exp(-0.3 * v3) + v4 * v5
    
    # Polynomial components
    o3 = v1**3 + v2**2 + v1
    o4 = (v1 - 2)**2 * v2
    
    # Logarithmic components
    o5 = np.log(v1 + 2) * v2 + v3
    o6 = np.log10(v4 + 3) * v5**2
    
    # Mixed components
    o7 = np.exp(0.1 * v1) * np.log(v2 + 2) + v3**2
    o8 = v4**3 * np.log10(v5 + 1.5)
    
    # Complex combinations
    o9 = np.exp(0.2 * v1) * v2**2 * np.log(v3 + 1.5) + v4 * v5
    o10 = np.log10(v1 + 2) * v2**2 + np.exp(0.15 * v3) + v4**2 * v5
    
    return np.array([o1, o2, o3, o4, o5, o6, o7, o8, o9, o10])

# =====================================
# Additional Test Functions from examples
# =====================================

def ExactLinearTestFunction(inputs):
    """
    Simple linear problem with 3 inputs and 5 outputs from exact_objective_example.py.
    
    This specific implementation is used in examples that demonstrate exact objective types
    and requires very precise matching to target values.
    
    Args:
        inputs: List of three input values [v01, v02, v03]
        
    Returns:
        Array of five outputs
    """
    v01, v02, v03 = inputs
    
    # Each output is a distinct linear combination of inputs
    o01 = 1.25 * v01 + 2.0 * v02 + 2.75 * v03
    o02 = 1.19 * (v01 - 2.1) + 0.71 * (v02 - 2.2) + 0.51 * v03
    o03 = 1.11 * (v01 - 2.3) + 0.65 * v02 + 0.56 * (v03 - 2.4)
    o04 = 1.02 * v01 + 0.49 * (v02 - 2.65) + 0.62 * (v03 - 2.5)
    o05 = 3.0 * v01 + 2.1 * v02 + 1.0 * v03
    
    return np.array([o01, o02, o03, o04, o05])

def MultiObjectiveLinearTestFunction(inputs):
    """
    Linear function with different coefficients from multiple_objective_types_example.py.
    
    This implementation is used to demonstrate different objective types
    (percentage, value, greater than, less than) all in the same optimization.
    
    Args:
        inputs: List of three input values [v01, v02, v03]
        
    Returns:
        Array of five outputs
    """
    v01, v02, v03 = inputs
    
    o01 = v01 + v02 + v03
    o02 = 1.001*(v01 - 2.0) + 1.02*(v02 - 2.0) + v03
    o03 = 0.997*(v01 - 2.0) + v02 + 0.995*(v03 - 2.0)
    o04 = 1.002*v01 + (v02 - 2.02) + (v03 - 2.0)
    o05 = 3.002 * v01 + 2.0 * v02 + 1.0 * v03
    
    return np.array([o01, o02, o03, o04, o05])

def SingleHumpFunction(inputs):
    """
    Single-humped function with constraints.
    
    Args:
        inputs: List of two input values [x1, x2]
        
    Returns:
        List of four outputs [objective, constraint1, constraint2, constraint3]
    """
    x1, x2 = inputs
    
    # Primary objective to maximize - a single-hump function 
    # Global maximum around (5, 5)
    objective = 10.0 - 0.2 * ((x1 - 5)**2 + 1.5*(x2 - 5)**2)
    
    # Constraint 1: Should be less than or equal to a threshold (e.g., ≤ 10.3)
    constraint1 = x1 + x2  
    
    # Constraint 2: Upper bound constraint (e.g., ≤ 22)
    constraint2 = x1 * x2  
    
    # Constraint 3: Lower bound constraint (e.g., ≥ 9)
    constraint3 = x1 * x2  
    
    return np.array([objective, constraint1, constraint2, constraint3])

def ConvolutionalFunction(inputs):
    """
    Convolutional function with 3 inputs and 1 output.
    
    This function demonstrates a system with more inputs than outputs,
    resulting in an underdetermined system with multiple possible solutions.
    
    Args:
        inputs: List of 3 input values [x1, x2, x3]
        
    Returns:
        Array with a single output calculated from the inputs
    """
    x1, x2, x3 = inputs
    
    # Calculate a single output using a function with multiple valid solutions
    # This is a weighted sum of the inputs with some nonlinear terms
    output = 0.5 * x1 + 0.3 * x2 + 0.7 * x3
    
    return np.array([output])

def MultiOutputFunction(inputs):
    """
    Function that produces 10 different outcomes from 3 inputs.
    
    This function demonstrates a system with multiple inputs and outputs.
    
    Args:
        inputs: List of 3 input values [x1, x2, x3]
        
    Returns:
        Array of 10 different outcomes calculated from the inputs
    """
    x1, x2, x3 = inputs
    
    # Calculate 10 different outputs with various relationships
    o1 = 1.0*x1 + 0.5*x2 + 0.3*x3
    o2 = 0.6*x1 + 1.0*x2 + 0.2*x3
    o3 = 0.3*x1 + 0.7*x2 + 1.0*x3
    o4 = 0.9*x1 + 0.2*x2 + 0.6*x3
    o5 = 0.5*x1 + 1.2*x2 + 0.8*x3
    o6 = 1.2*x1 + 0.7*x2 + 0.4*x3
    o7 = x1**2 + x2 + np.sin(x3) + np.log(x3 + 1)
    o8 = x1*x2 + x3*x3 + x1
    o9 = x1**2 + x2 + np.sin(x3) + np.log(x1 + 1) + np.sqrt(abs(x3) + 0.1)
    o10 = np.exp(x1/10) + x2**2 + np.cos(x3)
    
    return np.array([o1, o2, o3, o4, o5, o6, o7, o8, o9, o10])

def ReadmeFunction(inputs):
    """
    Simple 2-input, 3-output linear function from the README example.
    
    Args:
        inputs: List of two input values [x, y]
        
    Returns:
        Array of three output values
    """
    x, y = inputs
    return np.array([
        x + y,          # Output 1: simple sum
        2 * x + y,      # Output 2: weighted sum
        x + 2 * y       # Output 3: different weighted sum
    ])

def CubicSpecificFunction(inputs):
    """
    Cubic polynomial function with 4 inputs that define polynomial coefficients.
    
    Args:
        inputs: List of four input values [A, B, C, D] representing polynomial coefficients
        
    Returns:
        Array of 20 output values representing the polynomial evaluated at 20 points
    """
    A, B, C, D = inputs

    X = np.array([float(i)/10 for i in range(0,20)])

    Y = A*X**3 + B*X**2 + C*X + D

    return Y


