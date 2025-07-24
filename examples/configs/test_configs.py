"""
Test Configurations for GMOO Example Suite

This module defines test configurations for various optimization problems including
linear, nonlinear, mixed variable types, and constrained optimization.

Authors: Matt Freeman, Jason Hopkins
Version: 2.0.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.example_functions import *

TEST_CONFIGS = {
    'exact_linear': {
        'function': ExactLinearTestFunction,
        'num_inputs': 3,
        'num_outputs': 5,
        'var_mins': [1.0, 1.0, 1.0],
        'var_maxs': [10.0, 10.0, 10.0],
        'var_types': [1, 1, 1],
        'categories_list': [[], [], []],
        'truth_case': [5.4321, 4.321, 3.21],
        'initial_guess': [5.0, 5.0, 5.0],
        'objective_types': [0, 0, 0, 0, 0],  # All exact match
        'uncertainty_minus': [0.0] * 5,
        'uncertainty_plus': [0.0] * 5
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
        'initial_guess': [0.1, 0.1],
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
        'initial_guess': [0.1, 0.1, 0.1],
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
        'initial_guess': [0.1, 0.1, 0.2, 0.2],
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
        'initial_guess': [0.1, 0.1, 0.1, 0.1],
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
        'initial_guess': [0.1, 0.1, 0.1, 0.1, 0.1],
        'objective_types': [0, 0, 0, 0],
        'uncertainty_minus': [0.0, 0.0, 0.0, 0.0],
        'uncertainty_plus': [0.0, 0.0, 0.0, 0.0]
    },
    'java': {
        'function': JavaTestFunction,
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
        'function': SimpleTestFunction,
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
        'function': NonlinearTestFunction,
        'num_inputs': 3,
        'num_outputs': 4,
        'var_mins': [8.0,   8.0,  8.0],
        'var_maxs': [10.0, 10.0, 10.0],
        'var_types': [1, 1, 1],
        'categories_list': [[], [], []],
        'truth_case': [8.4321, 8.4321, 8.4321],
        'initial_guess': [9.0, 9.0, 9.0],
        'objective_types': [0, 0, 0, 0],
        'uncertainty_minus': [0.0, 0.0, 0.0, 0.0],
        'uncertainty_plus': [0.0, 0.0, 0.0, 0.0]
    },
    'integer': {
        'function': IntegerTestFunction,
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
        'function': LogicalTestFunction,
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
        'function': ComplexMixedTypeFunction,
        'num_inputs': 5,
        'num_outputs': 15,
        'var_mins': [-15.0, 199.00, 30.0, 0.0, 0.0],
        'var_maxs': [-0.01, 300.0, 50.0, 1.0, 1.0],
        'var_types': [1, 2, 1, 3, 3],  # 1=real, 2=integer, 3=logical
        'categories_list': [[], [], [], [], []],
        'truth_case': [-1.2345, 250.0, 32.0, 0.0, 0.0],
        'initial_guess': [-5.0, 225.0, 40.0, 0.0, 1.0],
        'objective_types': [0] * 15, # Exact match, L1 satisfaction
        'uncertainty_minus': [0] * 15,
        'uncertainty_plus': [0] * 15
    },
    'categorical': {
        'function': CategoricalTestFunction,
        'num_inputs': 3,
        'num_outputs': 30,
        'var_mins': [0.0, 0.0, 1.0],  # Min value for categorical is 1
        'var_maxs': [10.0, 10.0, 4.0],  # Max value represents number of categories
        'var_types': [1, 1, 4],  # 1=real, 4=categorical
        'categories_list': [[], [], ["cat", "dog", "fish", "bear"]], #, "house"]],
        'truth_case': [2.3456, 1.2345, 1.0],
        'initial_guess': [5.0, 5.0, 4.0],
        'objective_types': [1] * 30,  # Percent match for all outputs
        'uncertainty_minus': [-1.0] * 30,
        'uncertainty_plus': [1.0] * 30
    },
    'multi_obj': {
        'function': SimpleTestFunction,
        'num_inputs': 3,
        'num_outputs': 4,
        'var_mins': [5.0, 5.0, 5.0],
        'var_maxs': [6.0, 6.0, 6.0],
        'var_types': [1, 1, 1],
        'categories_list': [[], [], []],
        'truth_case': [5.4321, 5.4321, 5.4321],
        'initial_guess': [5.5, 5.5, 5.5],
        'objective_types': [
            1,    # Exact match for first output
            1,    # Percentage-based error for second output
            12,   # Less than or equal for third output
            21    # Minimize to target for fourth output
        ],
        'uncertainty_minus': [-5.0e-1, -2.0, 0.0, 0.0],  # Only used for type 1 and 2
        'uncertainty_plus': [5.0e-1, 2.0, 0.0, 0.0]    # Only used for type 1 and 2
    },
    'complex_mixed': {
        'function': ComplexMixedFunction,
        'num_inputs': 5,
        'num_outputs': 10,
        'var_mins': [0.1, 0.1, 0.1, 0.1, 0.1],  # Keeping positive and nonzero to avoid log(negative)
        'var_maxs': [5.0, 5.0, 5.0, 5.0, 5.0],
        'var_types': [1, 1, 1, 1, 1],  # All continuous float variables
        'categories_list': [[], [], [], [], []],  # No categoricals
        'truth_case': [2.5, 1.5, 3.0, 2.0, 1.0],
        'initial_guess': [1.0, 1.0, 1.0, 1.0, 1.0],
        'objective_types': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # All percentage-based
        'uncertainty_minus': [-0.1]*10, # Only a small percent deviation permitted
        'uncertainty_plus': [0.1]*10 # Only a small percent deviation permitted
    },
    'constrained_max': {
        'function': constrained_maximization_function,
        'num_inputs': 2,
        'num_outputs': 3,
        'var_mins': [0.0, 0.0], 
        'var_maxs': [10.0, 10.0],
        'var_types': [1, 1],  # All continuous float variables
        'categories_list': [[], []],  # No categoricals
        'truth_case': [2.5, 1.5], # Ignore
        'target_outputs': [12.0, 11.5, 22.0],
        'initial_guess': [5.0, 5.0],
        'objective_types': [22, 12, 12],  # Maximize, <=, <=
        'uncertainty_minus': [0.0]*3, # 
        'uncertainty_plus': [0.0]*3 # 
    },
    
    # New test configurations from additional examples
    'single_hump': {
        'function': SingleHumpFunction,
        'num_inputs': 2,
        'num_outputs': 4,
        'var_mins': [0.0, 0.0],
        'var_maxs': [10.0, 10.0],
        'var_types': [1, 1],
        'categories_list': [[], []],
        'truth_case': [5.0, 5.0],
        'target_outputs': [10.0, 10.3, 22.0, 9.0],
        'initial_guess': [7.0, 6.0],
        'objective_types': [22, 12, 12, 13],  # Maximize, <=, <=, >=
        'uncertainty_minus': [0.0, 0.0, 0.0, 0.0],
        'uncertainty_plus': [0.0, 0.0, 0.0, 0.0]
    },
    'convolutional': {
        'function': ConvolutionalFunction,
        'num_inputs': 3,
        'num_outputs': 1,
        'var_mins': [0.0, 0.0, 0.0],
        'var_maxs': [10.0, 10.0, 10.0],
        'var_types': [1, 1, 1],
        'categories_list': [[], [], []],
        'truth_case': [2.0, 3.0, 4.0],
        'initial_guess': [5.0, 5.0, 5.0],
        'objective_types': [1],  # Percentage-based
        'uncertainty_minus': [-0.1],
        'uncertainty_plus': [0.1]
    },
    'multi_output': {
        'function': MultiOutputFunction,
        'num_inputs': 3,
        'num_outputs': 10,
        'var_mins': [0.0, 0.0, 0.0],
        'var_maxs': [10.0, 10.0, 10.0],
        'var_types': [1, 1, 1],
        'categories_list': [[], [], []],
        'truth_case': [4.0, 4.0, 4.0],
        'initial_guess': [5.0, 5.0, 5.0],
        'objective_types': [1] * 10,  # All percentage-based
        'uncertainty_minus': [-0.1] * 10,
        'uncertainty_plus': [0.1] * 10
    },
    'readme': {
        'function': ReadmeFunction,
        'num_inputs': 2,
        'num_outputs': 3,
        'var_mins': [0.0, 0.0],
        'var_maxs': [10.0, 10.0],
        'var_types': [1, 1],
        'categories_list': [[], []],
        'truth_case': [1.0, 1.0],
        'target_outputs': [2.0, 3.0, 3.0],
        'initial_guess': [5.0, 5.0],
        'objective_types': [1, 1, 1],  # All percentage-based
        'uncertainty_minus': [-1.0] * 3,
        'uncertainty_plus': [1.0] * 3
    },
}
