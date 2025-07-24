"""
Tests for user case loading functionality in development.py.
"""

import pytest
import numpy as np
import os
from gmoo_sdk.dll_interface import GMOOAPI


class TestDevelopmentUserCases:
    """Test user case loading in the development module."""
    
    def test_load_single_user_case(self, loaded_dll):
        """Test loading a single user-defined case."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_single_user_case",
            var_mins=[0.0, 0.0],
            var_maxs=[10.0, 10.0],
            num_output_vars=2,
            model_function=lambda x: np.array([x[0] + x[1], x[0] * x[1]]),
            save_file_dir='.'
        )
        
        # Set up development
        model.development.load_vsme_name()
        model.development.initialize_variables()
        model.development.load_variable_types()
        model.development.load_variable_limits()
        model.development.design_agents()
        model.development.design_cases()
        case_count = model.development.get_case_count()
        
        # Initialize outcomes first
        model.development.initialize_outcomes()
        
        # Evaluate ALL development cases (required by DLL)
        for i in range(1, case_count + 1):
            case_vars = model.development.poke_case_variables(i)
            outputs = model.model_function(case_vars)
            model.development.load_case_results(i, outputs)
        
        # Load a single user case
        user_inputs = [5.0, 3.0]
        user_outputs = model.model_function(user_inputs)
        model.development.load_user_case(user_inputs, list(user_outputs))
        
        # Develop VSME with the added user case
        model.development.develop_vsme()
        
        # Clean up
        model.development.export_vsme()
        model.development.unload_vsme()
        gmoo_file = f"{model.vsme_input_filename}.gmoo"
        if os.path.exists(gmoo_file):
            os.remove(gmoo_file)
        vprj_file = f"{model.vsme_input_filename}.VPRJ"
        if os.path.exists(vprj_file):
            os.remove(vprj_file)
    
    def test_load_multiple_user_cases(self, loaded_dll):
        """Test loading multiple user-defined cases."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_multi_user_cases",
            var_mins=[0.0, 0.0, 0.0],
            var_maxs=[5.0, 5.0, 5.0],
            num_output_vars=3,
            model_function=lambda x: np.array([sum(x), np.prod(x), x[0]**2 + x[1]**2 + x[2]**2]),
            save_file_dir='.'
        )
        
        # Set up development
        model.development.load_vsme_name()
        model.development.initialize_variables()
        model.development.load_variable_types()
        model.development.load_variable_limits()
        model.development.design_agents()
        model.development.design_cases()
        case_count = model.development.get_case_count()
        
        # Initialize outcomes first
        model.development.initialize_outcomes()
        
        # Evaluate ALL development cases (required by DLL)
        for i in range(1, case_count + 1):
            case_vars = model.development.poke_case_variables(i)
            outputs = model.model_function(case_vars)
            model.development.load_case_results(i, outputs)
        
        # Prepare multiple user cases
        user_cases_inputs = [
            [1.0, 2.0, 3.0],
            [2.5, 2.5, 2.5],
            [4.0, 1.0, 0.5],
            [0.0, 0.0, 5.0]
        ]
        
        user_cases_outputs = []
        for inputs in user_cases_inputs:
            outputs = model.model_function(inputs)
            user_cases_outputs.append(list(outputs))
        
        # Load all user cases at once
        model.development.load_user_cases(user_cases_inputs, user_cases_outputs)
        
        # Develop VSME with the added user cases
        model.development.develop_vsme()
        
        # Export and verify
        gmoo_file = model.development.export_vsme()
        assert os.path.exists(gmoo_file)
        
        # Clean up
        model.development.unload_vsme()
        if os.path.exists(gmoo_file):
            os.remove(gmoo_file)
        vprj_file = f"{model.vsme_input_filename}.VPRJ"
        if os.path.exists(vprj_file):
            os.remove(vprj_file)
    
    def test_load_user_case_wrong_dimensions(self, loaded_dll):
        """Test error handling when loading user cases with wrong dimensions."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_wrong_dims",
            var_mins=[0.0, 0.0],
            var_maxs=[10.0, 10.0],
            num_output_vars=2,
            model_function=lambda x: np.array([x[0] + x[1], x[0] * x[1]]),
            save_file_dir='.'
        )
        
        # Set up development
        model.development.load_vsme_name()
        model.development.initialize_variables()
        model.development.initialize_outcomes()
        
        # Try to load user case with wrong number of inputs
        with pytest.raises(ValueError, match="Expected 2 variables"):
            model.development.load_user_case([1.0, 2.0, 3.0], [3.0, 2.0])
        
        # Try to load user case with wrong number of outputs
        with pytest.raises(ValueError, match="Expected 2 outcomes"):
            model.development.load_user_case([1.0, 2.0], [3.0])
        
        # Clean up
        model.development.unload_vsme()
    
    def test_load_user_cases_mismatched_counts(self, loaded_dll):
        """Test error handling when input and output counts don't match."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_mismatched_counts",
            var_mins=[0.0],
            var_maxs=[1.0],
            num_output_vars=1,
            model_function=lambda x: np.array([x[0]**2]),
            save_file_dir='.'
        )
        
        # Set up minimal development
        model.development.load_vsme_name()
        model.development.initialize_variables()
        model.development.initialize_outcomes()
        
        # Try to load mismatched number of cases
        inputs = [[0.5], [0.7]]
        outputs = [[0.25]]  # Only one output case
        
        with pytest.raises(ValueError, match="Number of cases must match"):
            model.development.load_user_cases(inputs, outputs)
        
        # Clean up
        model.development.unload_vsme()
    
    def test_load_user_case_after_development(self, loaded_dll):
        """Test adding user cases to improve an already developed model."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_improve_model",
            var_mins=[-5.0, -5.0],
            var_maxs=[5.0, 5.0],
            num_output_vars=1,
            model_function=lambda x: np.array([x[0]**2 + x[1]**2]),  # Simple paraboloid
            save_file_dir='.'
        )
        
        # Initial development with limited cases
        model.development.load_vsme_name()
        model.development.initialize_variables()
        model.development.load_variable_types()
        model.development.load_variable_limits()
        model.development.design_agents()
        model.development.design_cases()
        case_count = model.development.get_case_count()
        
        # Initialize outcomes first
        model.development.initialize_outcomes()
        
        # Evaluate ALL development cases (required by DLL)
        for i in range(1, case_count + 1):
            case_vars = model.development.poke_case_variables(i)
            outputs = model.model_function(case_vars)
            model.development.load_case_results(i, outputs)
        model.development.develop_vsme()
        
        # Now add strategic user cases around the optimum
        additional_cases_inputs = [
            [0.0, 0.0],    # Global minimum
            [0.1, 0.1],    # Near minimum
            [-0.1, -0.1],  # Near minimum
            [0.0, 0.5],    # Along axes
            [0.5, 0.0]
        ]
        
        additional_cases_outputs = []
        for inputs in additional_cases_inputs:
            outputs = model.model_function(inputs)
            additional_cases_outputs.append(list(outputs))
        
        # Re-develop with additional cases
        model.development.load_user_cases(additional_cases_inputs, additional_cases_outputs)
        model.development.develop_vsme()
        
        # Export final model
        gmoo_file = model.development.export_vsme()
        
        # Clean up
        model.development.unload_vsme()
        if os.path.exists(gmoo_file):
            os.remove(gmoo_file)
        vprj_file = f"{model.vsme_input_filename}.VPRJ"
        if os.path.exists(vprj_file):
            os.remove(vprj_file)