"""
Extended tests for stateless_wrapper module to improve coverage.

This module contains additional tests for the stateless wrapper functionality
to improve code coverage and test real-world usage scenarios.
"""

import pytest
import numpy as np
import os
from unittest.mock import Mock, patch
import tempfile

from gmoo_sdk.stateless_wrapper import GmooStatelessWrapper
from gmoo_sdk.dll_interface import GMOOException


# Test functions for use in tests
def simple_test_function(inputs):
    """Simple function for testing: f(x,y) = [x+y, x-y, x*y]"""
    x, y = inputs
    return np.array([x + y, x - y, x * y])


def quadratic_test_function(inputs):
    """Quadratic function for testing: f(x,y,z) = [x^2, y^2, z^2, x*y*z]"""
    x, y, z = inputs
    return np.array([x**2, y**2, z**2, x*y*z])


class TestStatelessWrapperBasicOperations:
    """Test core stateless wrapper operations."""
    
    @pytest.fixture
    def dll_path(self):
        """Get DLL path from environment or skip test."""
        path = os.environ.get('MOOLIB')
        if not path or not os.path.exists(path):
            pytest.skip("MOOLIB environment variable not set or DLL not found")
        return path
    
    def test_basic_workflow(self, dll_path):
        """Test the main workflow: develop_cases → load_development_cases → inverse."""
        # Initialize wrapper for development
        wrapper = GmooStatelessWrapper(
            minimum_list=[0.0, 0.0],
            maximum_list=[10.0, 10.0],
            input_type_list=[1, 1],  # Both continuous (1=real)
            category_list=[[], []],
            filename_prefix="test_basic_workflow",
            output_directory=".",
            dll_path=dll_path,
            num_outcomes=0  # Unknown initially
        )
        
        # Step 1: Develop cases
        dev_cases = wrapper.develop_cases()
        assert isinstance(dev_cases, list)
        assert len(dev_cases) > 0
        assert all(len(case) == 2 for case in dev_cases)
        
        # Step 2: Evaluate cases with our test function
        dev_outputs = []
        for case in dev_cases:
            output = simple_test_function(case)
            dev_outputs.append(list(output))
        
        # Step 3: Load development cases
        wrapper2 = GmooStatelessWrapper(
            minimum_list=[0.0, 0.0],
            maximum_list=[10.0, 10.0],
            input_type_list=[1, 1],  # 1=real
            category_list=[[], []],
            filename_prefix="test_basic_workflow",
            output_directory=".",
            dll_path=dll_path,
            num_outcomes=3  # Now we know
        )
        
        wrapper2.load_development_cases(
            num_outcomes=3,
            development_outputs_list=dev_outputs
        )
        
        # Step 4: Perform inverse optimization
        target_outputs = [5.0, 1.0, 6.0]  # x+y=5, x-y=1, x*y=6 → x=3, y=2
        current_inputs = [[5.0, 5.0]]  # Starting guess
        current_outputs = [list(simple_test_function(current_inputs[0]))]
        
        wrapper3 = GmooStatelessWrapper(
            minimum_list=[0.0, 0.0],
            maximum_list=[10.0, 10.0],
            input_type_list=[1, 1],  # 1=real
            category_list=[[], []],
            filename_prefix="test_basic_workflow",
            output_directory=".",
            dll_path=dll_path,
            num_outcomes=3
        )
        
        next_inputs, l1_norm, learned_inputs, learned_outputs = wrapper3.inverse(
            iteration_count=1,
            current_iteration_inputs_list=current_inputs,
            current_iteration_outputs_list=current_outputs,
            objectives_list=target_outputs,
            learned_case_input_list=[[]],
            learned_case_output_list=[[]],
            objective_types_list=[0, 0, 0],  # Exact match
            objective_status_list=[1, 1, 1],  # All active
            minimum_objective_bound_list=[0.0, 0.0, 0.0],
            maximum_objective_bound_list=[0.0, 0.0, 0.0],
            pipe_num=1
        )
        
        assert isinstance(next_inputs, list)
        assert len(next_inputs) == 1
        assert len(next_inputs[0]) == 2
        
        # Clean up
        gmoo_file = f"test_basic_workflow.gmoo"
        if os.path.exists(gmoo_file):
            os.remove(gmoo_file)
    
    
    def test_mixed_variable_types(self, dll_path):
        """Test with mixed variable types (continuous, integer, categorical)."""
        wrapper = GmooStatelessWrapper(
            minimum_list=[0.0, 0, 0],
            maximum_list=[10.0, 5, 2],
            input_type_list=[1, 2, 4],  # 1=real, 2=integer, 4=categorical
            category_list=[[], [], ["small", "medium", "large"]],
            filename_prefix="test_mixed_types",
            output_directory=".",
            dll_path=dll_path,
            num_outcomes=0
        )
        
        # Develop cases
        dev_cases = wrapper.develop_cases()
        
        # Verify development cases respect bounds but not necessarily integer constraints
        for case in dev_cases:
            # Continuous variable should be within bounds
            assert 0.0 <= case[0] <= 10.0, f"Continuous variable {case[0]} out of bounds"
            
            # Integer variable bounds (but may be continuous during development)
            # The DLL generates continuous values even for integer types during development
            assert 0 <= case[1] <= 5, f"Integer variable {case[1]} out of bounds"
            
            # Categorical variable should be within index bounds
            assert 0 <= case[2] <= 2, f"Categorical variable {case[2]} out of bounds"
        
        # Clean up
        gmoo_file = f"test_mixed_types.gmoo"
        if os.path.exists(gmoo_file):
            os.remove(gmoo_file)
    
    def test_error_handling(self, dll_path):
        """Test error handling in stateless wrapper."""
        import pytest
        
        # Test mismatched array lengths
        with pytest.raises(ValueError, match="must match"):
            GmooStatelessWrapper(
                minimum_list=[0.0, 0.0],
                maximum_list=[10.0],  # Wrong length
                input_type_list=[1, 1],  # 1=real
                category_list=[[], []],
                filename_prefix="test_error",
                output_directory=".",
                dll_path=dll_path,
                num_outcomes=1
            )
        
        # Test NaN values
        with pytest.raises(ValueError, match="contains NaN"):
            GmooStatelessWrapper(
                minimum_list=[0.0, np.nan],
                maximum_list=[10.0, 10.0],
                input_type_list=[1, 1],  # 1=real
                category_list=[[], []],
                filename_prefix="test_error",
                output_directory=".",
                dll_path=dll_path,
                num_outcomes=1
            )
        
        # Test invalid bounds
        with pytest.raises(ValueError, match="cannot be greater than maximum"):
            GmooStatelessWrapper(
                minimum_list=[10.0, 0.0],
                maximum_list=[5.0, 10.0],  # min > max for first variable
                input_type_list=[1, 1],  # 1=real
                category_list=[[], []],
                filename_prefix="test_error",
                output_directory=".",
                dll_path=dll_path,
                num_outcomes=1
            )
        
        # Test invalid variable type
        with pytest.raises(ValueError, match="invalid type"):
            GmooStatelessWrapper(
                minimum_list=[0.0, 0.0],
                maximum_list=[10.0, 10.0],
                input_type_list=[1, 5],  # Invalid type 5
                category_list=[[], []],
                filename_prefix="test_error",
                output_directory=".",
                dll_path=dll_path,
                num_outcomes=1
            )
        
        # Test categorical without categories
        with pytest.raises(ValueError, match="categorical variable must have"):
            GmooStatelessWrapper(
                minimum_list=[0.0, 0.0],
                maximum_list=[10.0, 3.0],
                input_type_list=[1, 4],  # Second is categorical (4=categorical)
                category_list=[[], []],  # But no categories provided
                filename_prefix="test_error",
                output_directory=".",
                dll_path=dll_path,
                num_outcomes=1
            )
    
    def test_parameter_validation(self, dll_path):
        """Test parameter validation in wrapper methods."""
        wrapper = GmooStatelessWrapper(
            minimum_list=[0.0, 0.0],
            maximum_list=[10.0, 10.0],
            input_type_list=[1, 1],  # 1=real
            category_list=[[], []],
            filename_prefix="test_validation",
            output_directory=".",
            dll_path=dll_path,
            num_outcomes=2
        )
        
        # Test develop_cases with parameters
        params = {
            "e1": 1,
            "e4": 8,
            "e5": 2,
            "e6": 2,
            "r": 0.5
        }
        dev_cases = wrapper.develop_cases(params=params)
        assert len(dev_cases) > 0
        
        # Clean up
        gmoo_file = f"test_validation.gmoo"
        if os.path.exists(gmoo_file):
            os.remove(gmoo_file)