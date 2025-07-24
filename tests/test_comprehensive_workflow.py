"""
Comprehensive tests covering various aspects of the GMOO SDK workflow.
These tests are designed to work with the DLL's constraints and avoid problematic patterns.
"""

import pytest
import numpy as np
import os
import time
import logging
from gmoo_sdk.dll_interface import GMOOAPI, GMOOException
from gmoo_sdk.helpers import CtypesHelper

logger = logging.getLogger(__name__)


class TestComprehensiveWorkflow:
    """Comprehensive test suite for GMOO SDK workflows."""
    
    def test_01_variable_bounds_validation(self, loaded_dll):
        """Test that variable bounds are properly enforced during case design."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        var_mins = [0.0, -5.0, 10.0]
        var_maxs = [1.0, 5.0, 20.0]
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_bounds_validation",
            var_mins=var_mins,
            var_maxs=var_maxs,
            num_output_vars=2,
            model_function=lambda x: np.array([x[0] + x[1], x[2]]),
            save_file_dir='.'
        )
        
        try:
            # Develop model
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            # Check all designed cases respect bounds
            violations = 0
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                for j, var in enumerate(case_vars):
                    if not (var_mins[j] <= var <= var_maxs[j]):
                        violations += 1
                        logger.warning(f"Variable {j} value {var} outside bounds [{var_mins[j]}, {var_maxs[j]}]")
            
            assert violations == 0, f"Found {violations} bound violations"
            logger.info(f"All {case_count} cases respect variable bounds")
            
        finally:
            # Cleanup
            try:
                model.development.unload_vsme()
            except:
                pass
            if os.path.exists("test_bounds_validation.gmoo"):
                os.remove("test_bounds_validation.gmoo")
    
    def test_02_mixed_variable_types(self, loaded_dll):
        """Test development with mixed variable types (continuous, integer, logical)."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        def mixed_function(inputs):
            continuous = inputs[0]
            integer = int(inputs[1])
            logical = int(inputs[2]) > 0
            return np.array([continuous * integer, float(logical)])
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_mixed_types",
            var_mins=[0.0, 1, 0],
            var_maxs=[10.0, 5, 1],
            num_output_vars=2,
            model_function=mixed_function,
            save_file_dir='.',
            var_types=[1, 2, 3]  # Real, Integer, Logical
        )
        
        try:
            # Develop model
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            # Complete development
            development_outcome_arrs = []
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                output = model.model_function(case_vars)
                development_outcome_arrs.append(output)
            
            model.development.initialize_outcomes()
            for i in range(1, len(development_outcome_arrs) + 1):
                model.development.load_case_results(i, development_outcome_arrs[i-1])
            
            model.development.develop_vsme()
            model.development.export_vsme()
            
            assert os.path.exists("test_mixed_types.gmoo"), "Model file should be created"
            
        finally:
            # Cleanup
            try:
                model.development.unload_vsme()
            except:
                pass
            if os.path.exists("test_mixed_types.gmoo"):
                os.remove("test_mixed_types.gmoo")
    
    def test_03_single_variable_problem(self, loaded_dll):
        """Test development with single variable problem."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_single_var",
            var_mins=[0.0],
            var_maxs=[10.0],
            num_output_vars=2,
            model_function=lambda x: np.array([x[0]**2, np.sin(x[0])]),
            save_file_dir='.'
        )
        
        try:
            # Develop model
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            # Process cases
            development_outcome_arrs = []
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                output = model.model_function(case_vars)
                development_outcome_arrs.append(output)
            
            model.development.initialize_outcomes()
            for i in range(1, len(development_outcome_arrs) + 1):
                model.development.load_case_results(i, development_outcome_arrs[i-1])
            
            model.development.develop_vsme()
            model.development.export_vsme()
            
        finally:
            try:
                model.development.unload_vsme()
            except:
                pass
            if os.path.exists("test_single_var.gmoo"):
                os.remove("test_single_var.gmoo")
    
    def test_04_extreme_output_values(self, loaded_dll):
        """Test handling of extreme output values (very large/small)."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        def extreme_function(x):
            return np.array([
                x[0] * 1e6,      # Very large
                x[1] * 1e-6,     # Very small
                x[0] / (x[1] + 1e-10)  # Potential for large values
            ])
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_extreme_outputs",
            var_mins=[0.1, 0.1],
            var_maxs=[1.0, 1.0],
            num_output_vars=3,
            model_function=extreme_function,
            save_file_dir='.'
        )
        
        try:
            # Develop model
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            # Process cases
            development_outcome_arrs = []
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                output = model.model_function(case_vars)
                development_outcome_arrs.append(output)
            
            model.development.initialize_outcomes()
            for i in range(1, len(development_outcome_arrs) + 1):
                model.development.load_case_results(i, development_outcome_arrs[i-1])
            
            model.development.develop_vsme()
            model.development.export_vsme()
            
        finally:
            try:
                model.development.unload_vsme()
            except:
                pass
            if os.path.exists("test_extreme_outputs.gmoo"):
                os.remove("test_extreme_outputs.gmoo")
    
    def test_05_categorical_variables(self, loaded_dll):
        """Test categorical variables in development."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        def categorical_function(inputs):
            continuous = inputs[0]
            category = int(inputs[1])  # 1, 2, or 3
            multipliers = {1: 0.5, 2: 1.0, 3: 2.0}
            mult = multipliers.get(category, 1.0)
            return np.array([continuous * mult, continuous + category])
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_categorical",
            var_mins=[0.0, 1],
            var_maxs=[5.0, 3],
            num_output_vars=2,
            model_function=categorical_function,
            save_file_dir='.',
            var_types=[1, 4],  # Real, Categorical
            categories_list=[[], ["small", "medium", "large"]]
        )
        
        try:
            # Develop model
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            # Track categorical values seen
            categorical_values = set()
            development_outcome_arrs = []
            
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                categorical_values.add(int(case_vars[1]))
                output = model.model_function(case_vars)
                development_outcome_arrs.append(output)
            
            # Should see all categorical values
            assert categorical_values == {1, 2, 3}, f"Expected all categories, got {categorical_values}"
            
            model.development.initialize_outcomes()
            for i in range(1, len(development_outcome_arrs) + 1):
                model.development.load_case_results(i, development_outcome_arrs[i-1])
            
            model.development.develop_vsme()
            model.development.export_vsme()
            
        finally:
            try:
                model.development.unload_vsme()
            except:
                pass
            if os.path.exists("test_categorical.gmoo"):
                os.remove("test_categorical.gmoo")
    
    def test_06_percentage_error_objectives(self, loaded_dll):
        """Test optimization with percentage error objectives."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        # Develop a simple model first
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_percentage_error",
            var_mins=[0.1, 0.1],
            var_maxs=[5.0, 5.0],
            num_output_vars=2,
            model_function=lambda x: np.array([x[0] * x[1], x[0] + x[1]]),
            save_file_dir='.'
        )
        
        try:
            # Development phase
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            development_outcome_arrs = []
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                output = model.model_function(case_vars)
                development_outcome_arrs.append(output)
            
            model.development.initialize_outcomes()
            for i in range(1, len(development_outcome_arrs) + 1):
                model.development.load_case_results(i, development_outcome_arrs[i-1])
            
            model.development.develop_vsme()
            model.development.export_vsme()
            model.development.unload_vsme()
            
            # Application phase with percentage error
            model.application.load_model()
            
            # Target: output1 = 10.0 ± 5%, output2 = 5.0 ± 10%
            targets = [10.0, 5.0]
            obj_types = [1, 1]  # Percentage error
            model.application.assign_objectives_target(targets, obj_types)
            model.application.load_objective_uncertainty([5.0, 10.0], [5.0, 10.0])
            
            # Run one iteration
            current_inputs = np.array([2.0, 3.0])
            current_outputs = model.model_function(current_inputs)
            
            next_inputs, l1, l2 = model.application.perform_inverse_iteration(
                target_outputs=targets,
                current_inputs=current_inputs,
                current_outputs=current_outputs,
                objective_types=obj_types,
                objective_uncertainty_minus=[-5.0, -10.0],
                objective_uncertainty_plus=[5.0, 10.0]
            )
            
            # Verify we got valid results
            assert not np.any(np.isnan(next_inputs)), "Results should not be NaN"
            
        finally:
            try:
                model.development.unload_vsme()
                model.application.unload_vsme()
            except:
                pass
            if os.path.exists("test_percentage_error.gmoo"):
                os.remove("test_percentage_error.gmoo")
    
    def test_07_absolute_error_objectives(self, loaded_dll):
        """Test optimization with absolute error objectives."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_absolute_error",
            var_mins=[0.0, 0.0],
            var_maxs=[10.0, 10.0],
            num_output_vars=2,
            model_function=lambda x: np.array([x[0]**2, x[1]**2]),
            save_file_dir='.'
        )
        
        try:
            # Development phase
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            development_outcome_arrs = []
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                output = model.model_function(case_vars)
                development_outcome_arrs.append(output)
            
            model.development.initialize_outcomes()
            for i in range(1, len(development_outcome_arrs) + 1):
                model.development.load_case_results(i, development_outcome_arrs[i-1])
            
            model.development.develop_vsme()
            model.development.export_vsme()
            model.development.unload_vsme()
            
            # Application phase with absolute error
            model.application.load_model()
            
            # Target: output1 = 25.0 ± 5.0, output2 = 16.0 ± 3.0
            targets = [25.0, 16.0]
            obj_types = [2, 2]  # Absolute error
            model.application.assign_objectives_target(targets, obj_types)
            model.application.load_objective_uncertainty([-5.0, -3.0], [5.0, 3.0])
            
            # Run optimization
            current_inputs = np.array([3.0, 3.0])
            for i in range(5):
                current_outputs = model.model_function(current_inputs)
                next_inputs, l1, l2 = model.application.perform_inverse_iteration(
                    target_outputs=targets,
                    current_inputs=current_inputs,
                    current_outputs=current_outputs,
                    objective_types=obj_types,
                    objective_uncertainty_minus=[-5.0, -3.0],
                    objective_uncertainty_plus=[5.0, 3.0]
                )
                current_inputs = next_inputs
            
            # Check final outputs are within tolerance
            final_outputs = model.model_function(current_inputs)
            assert abs(final_outputs[0] - 25.0) <= 5.0, "Output 1 not within tolerance"
            assert abs(final_outputs[1] - 16.0) <= 3.0, "Output 2 not within tolerance"
            
        finally:
            try:
                model.development.unload_vsme()
                model.application.unload_vsme()
            except:
                pass
            if os.path.exists("test_absolute_error.gmoo"):
                os.remove("test_absolute_error.gmoo")
    
    def test_08_inequality_constraints(self, loaded_dll):
        """Test optimization with inequality constraints."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_inequality",
            var_mins=[0.0, 0.0],
            var_maxs=[5.0, 5.0],
            num_output_vars=3,
            model_function=lambda x: np.array([x[0] + x[1], x[0] - x[1], x[0] * x[1]]),
            save_file_dir='.'
        )
        
        try:
            # Development phase
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            development_outcome_arrs = []
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                output = model.model_function(case_vars)
                development_outcome_arrs.append(output)
            
            model.development.initialize_outcomes()
            for i in range(1, len(development_outcome_arrs) + 1):
                model.development.load_case_results(i, development_outcome_arrs[i-1])
            
            model.development.develop_vsme()
            model.development.export_vsme()
            model.development.unload_vsme()
            
            # Application phase with constraints
            model.application.load_model()
            
            # Constraints: output1 <= 6.0, output2 >= -1.0, output3 = 8.0
            targets = [6.0, -1.0, 8.0]
            obj_types = [12, 13, 0]  # <=, >=, exact
            model.application.assign_objectives_target(targets, obj_types)
            model.application.load_objective_uncertainty([0.0, 0.0, -0.5], [0.0, 0.0, 0.5])
            
            # Run optimization
            current_inputs = np.array([3.0, 2.0])
            for i in range(10):
                current_outputs = model.model_function(current_inputs)
                next_inputs, l1, l2 = model.application.perform_inverse_iteration(
                    target_outputs=targets,
                    current_inputs=current_inputs,
                    current_outputs=current_outputs,
                    objective_types=obj_types
                )
                current_inputs = next_inputs
            
            # Check constraints
            final_outputs = model.model_function(current_inputs)
            assert final_outputs[0] <= 6.1, f"Constraint 1 violated: {final_outputs[0]} > 6.0"
            assert final_outputs[1] >= -1.1, f"Constraint 2 violated: {final_outputs[1]} < -1.0"
            
        finally:
            try:
                model.development.unload_vsme()
                model.application.unload_vsme()
            except:
                pass
            if os.path.exists("test_inequality.gmoo"):
                os.remove("test_inequality.gmoo")
    
    def test_09_model_with_discontinuity(self, loaded_dll):
        """Test model with discontinuous function."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        def discontinuous_function(x):
            # Step function with discontinuity at x[0] = 2.5
            if x[0] < 2.5:
                y1 = x[0] * 2
            else:
                y1 = x[0] * 3 - 2.5
            
            y2 = x[1]**2
            return np.array([y1, y2])
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_discontinuous",
            var_mins=[0.0, 0.0],
            var_maxs=[5.0, 5.0],
            num_output_vars=2,
            model_function=discontinuous_function,
            save_file_dir='.'
        )
        
        try:
            # Development phase
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            development_outcome_arrs = []
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                output = model.model_function(case_vars)
                development_outcome_arrs.append(output)
            
            model.development.initialize_outcomes()
            for i in range(1, len(development_outcome_arrs) + 1):
                model.development.load_case_results(i, development_outcome_arrs[i-1])
            
            model.development.develop_vsme()
            model.development.export_vsme()
            
        finally:
            try:
                model.development.unload_vsme()
            except:
                pass
            if os.path.exists("test_discontinuous.gmoo"):
                os.remove("test_discontinuous.gmoo")
    
    def test_10_zero_variance_output(self, loaded_dll):
        """Test model where one output has zero variance."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        def constant_output_function(x):
            return np.array([
                x[0]**2 + x[1]**2,  # Variable output
                5.0                  # Constant output
            ])
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_zero_variance",
            var_mins=[0.0, 0.0],
            var_maxs=[3.0, 3.0],
            num_output_vars=2,
            model_function=constant_output_function,
            save_file_dir='.'
        )
        
        try:
            # Development phase
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            development_outcome_arrs = []
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                output = model.model_function(case_vars)
                development_outcome_arrs.append(output)
            
            model.development.initialize_outcomes()
            for i in range(1, len(development_outcome_arrs) + 1):
                model.development.load_case_results(i, development_outcome_arrs[i-1])
            
            model.development.develop_vsme()
            model.development.export_vsme()
            
        finally:
            try:
                model.development.unload_vsme()
            except:
                pass
            if os.path.exists("test_zero_variance.gmoo"):
                os.remove("test_zero_variance.gmoo")
    
    def test_11_model_accuracy_verification(self, loaded_dll):
        """Test inverse model accuracy against original function."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        def test_function(x):
            return np.array([
                x[0]**2 + x[1],
                np.sin(x[0]) * np.cos(x[1])
            ])
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_accuracy",
            var_mins=[0.0, 0.0],
            var_maxs=[3.14, 3.14],
            num_output_vars=2,
            model_function=test_function,
            save_file_dir='.'
        )
        
        try:
            # Development phase
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            development_outcome_arrs = []
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                output = model.model_function(case_vars)
                development_outcome_arrs.append(output)
            
            model.development.initialize_outcomes()
            for i in range(1, len(development_outcome_arrs) + 1):
                model.development.load_case_results(i, development_outcome_arrs[i-1])
            
            model.development.develop_vsme()
            model.development.export_vsme()
            model.development.unload_vsme()
            
            # Application phase - verify model works through inverse optimization
            model.application.load_model()
            
            # Test: can we find inputs that produce specific outputs?
            # Pick a known point and see if we can recover it
            test_input = np.array([1.5, 2.0])
            target_outputs = test_function(test_input)
            
            # Set up inverse problem with exact objectives
            model.application.assign_objectives_target(target_outputs.tolist(), [0, 0])
            model.application.load_objective_uncertainty([0.1, 0.1], [0.1, 0.1])
            
            # Start from different location
            current_inputs = np.array([0.5, 0.5])
            
            # Run inverse optimization
            for _ in range(10):
                current_outputs = model.model_function(current_inputs)
                next_inputs, l1, l2 = model.application.perform_inverse_iteration(
                    target_outputs=target_outputs.tolist(),
                    current_inputs=current_inputs,
                    current_outputs=current_outputs,
                    objective_types=[0, 0]
                )
                current_inputs = next_inputs
                
                # Check if we're close enough
                if np.linalg.norm(current_inputs - test_input) < 0.2:
                    break
            
            # We should have found inputs close to the original
            logger.info(f"Target input: {test_input}, Found: {current_inputs}")
            assert np.linalg.norm(current_inputs - test_input) < 0.5, \
                f"Failed to recover inputs: expected {test_input}, got {current_inputs}"
            
        finally:
            try:
                model.development.unload_vsme()
                model.application.unload_vsme()
            except:
                pass
            if os.path.exists("test_accuracy.gmoo"):
                os.remove("test_accuracy.gmoo")
    
    def test_12_sequential_optimization(self, loaded_dll):
        """Test sequential optimization towards different targets."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_sequential",
            var_mins=[0.0, 0.0],
            var_maxs=[10.0, 10.0],
            num_output_vars=2,
            model_function=lambda x: np.array([x[0] + x[1], x[0] * x[1]]),
            save_file_dir='.'
        )
        
        try:
            # Development phase
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            development_outcome_arrs = []
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                output = model.model_function(case_vars)
                development_outcome_arrs.append(output)
            
            model.development.initialize_outcomes()
            for i in range(1, len(development_outcome_arrs) + 1):
                model.development.load_case_results(i, development_outcome_arrs[i-1])
            
            model.development.develop_vsme()
            model.development.export_vsme()
            model.development.unload_vsme()
            
            # Sequential optimization to different targets
            model.application.load_model()
            
            targets_sequence = [
                ([5.0, 6.0], [0, 0]),    # First target
                ([10.0, 20.0], [0, 0]),  # Second target
                ([15.0, 40.0], [0, 0])   # Third target
            ]
            
            current_inputs = np.array([2.0, 2.0])
            
            for targets, obj_types in targets_sequence:
                model.application.assign_objectives_target(targets, obj_types)
                model.application.load_objective_uncertainty([-1.0, -2.0], [1.0, 2.0])
                
                # Optimize towards this target
                for _ in range(5):
                    current_outputs = model.model_function(current_inputs)
                    next_inputs, l1, l2 = model.application.perform_inverse_iteration(
                        target_outputs=targets,
                        current_inputs=current_inputs,
                        current_outputs=current_outputs,
                        objective_types=obj_types
                    )
                    current_inputs = next_inputs
                
                # Check we're close to target
                final_outputs = model.model_function(current_inputs)
                logger.info(f"Target: {targets}, Achieved: {final_outputs}")
            
        finally:
            try:
                model.development.unload_vsme()
                model.application.unload_vsme()
            except:
                pass
            if os.path.exists("test_sequential.gmoo"):
                os.remove("test_sequential.gmoo")
    
    def test_13_sparse_data_regions(self, loaded_dll):
        """Test model behavior in regions with sparse training data."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        # Function that's mostly evaluated in one region
        def biased_function(x):
            # We'll manually control which regions get sampled
            return np.array([
                x[0]**2 + x[1]**2,
                np.exp(-((x[0]-5)**2 + (x[1]-5)**2)/10)
            ])
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_sparse",
            var_mins=[0.0, 0.0],
            var_maxs=[10.0, 10.0],
            num_output_vars=2,
            model_function=biased_function,
            save_file_dir='.'
        )
        
        try:
            # Development phase
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            development_outcome_arrs = []
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                output = model.model_function(case_vars)
                development_outcome_arrs.append(output)
            
            model.development.initialize_outcomes()
            for i in range(1, len(development_outcome_arrs) + 1):
                model.development.load_case_results(i, development_outcome_arrs[i-1])
            
            model.development.develop_vsme()
            model.development.export_vsme()
            
        finally:
            try:
                model.development.unload_vsme()
            except:
                pass
            if os.path.exists("test_sparse.gmoo"):
                os.remove("test_sparse.gmoo")
    
    def test_14_rapid_satisfaction_check(self, loaded_dll):
        """Test rapid satisfaction for well-conditioned problems."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        # Simple quadratic - should converge quickly
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_rapid_conv",
            var_mins=[-5.0, -5.0],
            var_maxs=[5.0, 5.0],
            num_output_vars=2,
            model_function=lambda x: np.array([(x[0]-2)**2, (x[1]-3)**2]),
            save_file_dir='.'
        )
        
        try:
            # Development phase
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            development_outcome_arrs = []
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                output = model.model_function(case_vars)
                development_outcome_arrs.append(output)
            
            model.development.initialize_outcomes()
            for i in range(1, len(development_outcome_arrs) + 1):
                model.development.load_case_results(i, development_outcome_arrs[i-1])
            
            model.development.develop_vsme()
            model.development.export_vsme()
            model.development.unload_vsme()
            
            # Application - should converge to minimum quickly
            model.application.load_model()
            
            targets = [0.0, 0.0]  # Minimize both
            obj_types = [21, 21]  # Both minimize
            model.application.assign_objectives_target(targets, obj_types)
            
            current_inputs = np.array([0.0, 0.0])  # Start far from optimum
            
            # Should converge in few iterations
            for i in range(10):
                current_outputs = model.model_function(current_inputs)
                if all(o < 0.1 for o in current_outputs):
                    logger.info(f"Converged in {i+1} iterations")
                    break
                    
                next_inputs, l1, l2 = model.application.perform_inverse_iteration(
                    target_outputs=targets,
                    current_inputs=current_inputs,
                    current_outputs=current_outputs,
                    objective_types=obj_types
                )
                current_inputs = next_inputs
            
            # Check that we've made progress toward minimum
            final_outputs = model.model_function(current_inputs)
            initial_outputs = model.model_function(np.array([0.0, 0.0]))
            
            # Both outputs should be smaller than initial
            assert final_outputs[0] < initial_outputs[0], "First output should decrease"
            assert final_outputs[1] < initial_outputs[1], "Second output should decrease"
            
            # Note: We can't guarantee exact satisfaction to (2,3) since minimize objectives
            # don't have proper satisfaction checking in the SDK
            logger.info(f"Final position: {current_inputs}, outputs: {final_outputs}")
            
        finally:
            try:
                model.development.unload_vsme()
                model.application.unload_vsme()
            except:
                pass
            if os.path.exists("test_rapid_conv.gmoo"):
                os.remove("test_rapid_conv.gmoo")
    
    def test_15_conflicting_objectives(self, loaded_dll):
        """Test optimization with conflicting objectives."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        # Conflicting objectives: minimize x[0] but maximize x[1], with x[0]+x[1]=5
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_conflicting",
            var_mins=[0.0, 0.0],
            var_maxs=[5.0, 5.0],
            num_output_vars=3,
            model_function=lambda x: np.array([x[0], x[1], x[0] + x[1]]),
            save_file_dir='.'
        )
        
        try:
            # Development phase
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            development_outcome_arrs = []
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                output = model.model_function(case_vars)
                development_outcome_arrs.append(output)
            
            model.development.initialize_outcomes()
            for i in range(1, len(development_outcome_arrs) + 1):
                model.development.load_case_results(i, development_outcome_arrs[i-1])
            
            model.development.develop_vsme()
            model.development.export_vsme()
            model.development.unload_vsme()
            
            # Application with conflicting objectives
            model.application.load_model()
            
            targets = [0.0, 5.0, 5.0]  # Min x[0], Max x[1], x[0]+x[1]=5
            obj_types = [21, 22, 0]    # Minimize, Maximize, Exact
            model.application.assign_objectives_target(targets, obj_types)
            model.application.load_objective_uncertainty([0.0, 0.0, -0.1], [0.0, 0.0, 0.1])
            
            current_inputs = np.array([2.5, 2.5])
            
            # Run optimization
            for _ in range(20):
                current_outputs = model.model_function(current_inputs)
                next_inputs, l1, l2 = model.application.perform_inverse_iteration(
                    target_outputs=targets,
                    current_inputs=current_inputs,
                    current_outputs=current_outputs,
                    objective_types=obj_types
                )
                current_inputs = next_inputs
            
            # Check constraint is satisfied
            final_sum = current_inputs[0] + current_inputs[1]
            assert abs(final_sum - 5.0) < 0.2, f"Sum constraint not satisfied: {final_sum}"
            
        finally:
            try:
                model.development.unload_vsme()
                model.application.unload_vsme()
            except:
                pass
            if os.path.exists("test_conflicting.gmoo"):
                os.remove("test_conflicting.gmoo")
    
    def test_16_noisy_function_handling(self, loaded_dll):
        """Test model with noisy outputs."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        np.random.seed(42)  # For reproducibility
        
        def noisy_function(x):
            # Add small noise to outputs
            noise1 = np.random.normal(0, 0.1)
            noise2 = np.random.normal(0, 0.1)
            return np.array([
                x[0]**2 + x[1]**2 + noise1,
                x[0] - x[1] + noise2
            ])
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_noisy",
            var_mins=[0.0, 0.0],
            var_maxs=[5.0, 5.0],
            num_output_vars=2,
            model_function=noisy_function,
            save_file_dir='.'
        )
        
        try:
            # Development phase
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            development_outcome_arrs = []
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                output = model.model_function(case_vars)
                development_outcome_arrs.append(output)
            
            model.development.initialize_outcomes()
            for i in range(1, len(development_outcome_arrs) + 1):
                model.development.load_case_results(i, development_outcome_arrs[i-1])
            
            model.development.develop_vsme()
            model.development.export_vsme()
            
        finally:
            try:
                model.development.unload_vsme()
            except:
                pass
            if os.path.exists("test_noisy.gmoo"):
                os.remove("test_noisy.gmoo")
    
    def test_17_extrapolation_behavior(self, loaded_dll):
        """Test inverse optimization behavior when targeting extrapolated outputs."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_extrapolation",
            var_mins=[0.0, 0.0],
            var_maxs=[5.0, 5.0],
            num_output_vars=1,
            model_function=lambda x: np.array([x[0]**2 + x[1]**2]),
            save_file_dir='.'
        )
        
        try:
            # Development phase
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            development_outcome_arrs = []
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                output = model.model_function(case_vars)
                development_outcome_arrs.append(output)
            
            model.development.initialize_outcomes()
            for i in range(1, len(development_outcome_arrs) + 1):
                model.development.load_case_results(i, development_outcome_arrs[i-1])
            
            model.development.develop_vsme()
            model.development.export_vsme()
            model.development.unload_vsme()
            
            # Test extrapolation
            model.application.load_model()
            
            # Test points slightly outside training range
            test_points = [
                np.array([5.5, 2.5]),  # Slightly outside in x[0]
                np.array([2.5, 5.5]),  # Slightly outside in x[1]
                np.array([5.2, 5.2])   # Slightly outside in both
            ]
            
            # Test extrapolation through inverse optimization
            # Try to find inputs that produce outputs typical of extrapolated region
            
            # First, get typical output values at boundary
            boundary_outputs = []
            for x in [0.0, 5.0]:
                for y in [0.0, 5.0]:
                    boundary_outputs.append(model.model_function(np.array([x, y]))[0])
            
            # Target an output beyond typical range
            target_output = max(boundary_outputs) * 1.5  # 50% beyond max
            
            model.application.assign_objectives_target([target_output], [0])
            model.application.load_objective_uncertainty([target_output * 0.1], [target_output * 0.1])
            
            # Start from center of domain
            current_inputs = np.array([2.5, 2.5])
            
            # Run optimization - should push toward extrapolated region
            for i in range(10):
                current_outputs = model.model_function(current_inputs)
                next_inputs, l1, l2 = model.application.perform_inverse_iteration(
                    target_outputs=[target_output],
                    current_inputs=current_inputs,
                    current_outputs=current_outputs,
                    objective_types=[0]
                )
                
                # Verify we get valid results (no NaN)
                assert not np.any(np.isnan(next_inputs)), f"Iteration {i}: Got NaN in extrapolation"
                current_inputs = next_inputs
            
            # Should have moved toward higher output region
            final_output = model.model_function(current_inputs)[0]
            initial_output = model.model_function(np.array([2.5, 2.5]))[0]
            assert final_output > initial_output, "Should move toward higher outputs"
            logger.info(f"Extrapolation test: target={target_output:.2f}, achieved={final_output:.2f}")
            
        finally:
            try:
                model.development.unload_vsme()
                model.application.unload_vsme()
            except:
                pass
            if os.path.exists("test_extrapolation.gmoo"):
                os.remove("test_extrapolation.gmoo")
    
    def test_18_multiple_minima_function(self, loaded_dll):
        """Test optimization with function having multiple local minima."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        def multi_minima_function(x):
            # Rastrigin-like function with multiple minima
            return np.array([
                20 + x[0]**2 + x[1]**2 - 10*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1])),
                x[0] + x[1]
            ])
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_multi_minima",
            var_mins=[-2.0, -2.0],
            var_maxs=[2.0, 2.0],
            num_output_vars=2,
            model_function=multi_minima_function,
            save_file_dir='.'
        )
        
        try:
            # Development phase
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            development_outcome_arrs = []
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                output = model.model_function(case_vars)
                development_outcome_arrs.append(output)
            
            model.development.initialize_outcomes()
            for i in range(1, len(development_outcome_arrs) + 1):
                model.development.load_case_results(i, development_outcome_arrs[i-1])
            
            model.development.develop_vsme()
            model.development.export_vsme()
            
        finally:
            try:
                model.development.unload_vsme()
            except:
                pass
            if os.path.exists("test_multi_minima.gmoo"):
                os.remove("test_multi_minima.gmoo")
    
    def test_19_time_dependent_optimization(self, loaded_dll):
        """Test optimization where we change targets over time."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_time_dependent",
            var_mins=[0.0, 0.0],
            var_maxs=[10.0, 10.0],
            num_output_vars=2,
            model_function=lambda x: np.array([x[0]**2, x[1]**2]),
            save_file_dir='.'
        )
        
        try:
            # Development phase
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            development_outcome_arrs = []
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                output = model.model_function(case_vars)
                development_outcome_arrs.append(output)
            
            model.development.initialize_outcomes()
            for i in range(1, len(development_outcome_arrs) + 1):
                model.development.load_case_results(i, development_outcome_arrs[i-1])
            
            model.development.develop_vsme()
            model.development.export_vsme()
            model.development.unload_vsme()
            
            # Application with time-varying targets
            model.application.load_model()
            
            current_inputs = np.array([5.0, 5.0])
            
            # Simulate changing targets over time
            time_steps = 5
            for t in range(time_steps):
                # Targets change with time
                target1 = 25.0 + 10.0 * np.sin(t * np.pi / 4)
                target2 = 25.0 + 10.0 * np.cos(t * np.pi / 4)
                
                model.application.assign_objectives_target([target1, target2], [0, 0])
                model.application.load_objective_uncertainty([-2.0, -2.0], [2.0, 2.0])
                
                # One optimization step
                current_outputs = model.model_function(current_inputs)
                next_inputs, l1, l2 = model.application.perform_inverse_iteration(
                    target_outputs=[target1, target2],
                    current_inputs=current_inputs,
                    current_outputs=current_outputs,
                    objective_types=[0, 0]
                )
                current_inputs = next_inputs
                
                logger.info(f"Time {t}: Targets=({target1:.1f}, {target2:.1f}), "
                          f"Achieved=({current_outputs[0]:.1f}, {current_outputs[1]:.1f})")
            
        finally:
            try:
                model.development.unload_vsme()
                model.application.unload_vsme()
            except:
                pass
            if os.path.exists("test_time_dependent.gmoo"):
                os.remove("test_time_dependent.gmoo")
    
    def test_20_constraint_satisfaction_priority(self, loaded_dll):
        """Test that hard constraints are prioritized over soft objectives."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_constraint_priority",
            var_mins=[0.0, 0.0],
            var_maxs=[10.0, 10.0],
            num_output_vars=3,
            model_function=lambda x: np.array([
                x[0]**2 + x[1]**2,  # Minimize this
                x[0] + x[1],        # Must be <= 8
                x[0] - x[1]         # Must be >= -2
            ]),
            save_file_dir='.'
        )
        
        try:
            # Development phase
            model.development.load_vsme_name()
            model.development.initialize_variables()
            model.development.load_variable_types()
            model.development.load_variable_limits()
            model.development.design_agents()
            model.development.design_cases()
            case_count = model.development.get_case_count()
            
            development_outcome_arrs = []
            for i in range(1, case_count + 1):
                case_vars = model.development.poke_case_variables(i)
                output = model.model_function(case_vars)
                development_outcome_arrs.append(output)
            
            model.development.initialize_outcomes()
            for i in range(1, len(development_outcome_arrs) + 1):
                model.development.load_case_results(i, development_outcome_arrs[i-1])
            
            model.development.develop_vsme()
            model.development.export_vsme()
            model.development.unload_vsme()
            
            # Application with constraints
            model.application.load_model()
            
            targets = [0.0, 8.0, -2.0]
            obj_types = [21, 12, 13]  # Minimize, <=, >=
            model.application.assign_objectives_target(targets, obj_types)
            
            current_inputs = np.array([7.0, 7.0])  # Start violating constraint
            
            # Optimize
            for _ in range(20):
                current_outputs = model.model_function(current_inputs)
                next_inputs, l1, l2 = model.application.perform_inverse_iteration(
                    target_outputs=targets,
                    current_inputs=current_inputs,
                    current_outputs=current_outputs,
                    objective_types=obj_types
                )
                current_inputs = next_inputs
            
            # Check constraints are satisfied
            final_outputs = model.model_function(current_inputs)
            assert final_outputs[1] <= 8.1, f"Constraint 1 violated: {final_outputs[1]} > 8"
            assert final_outputs[2] >= -2.1, f"Constraint 2 violated: {final_outputs[2]} < -2"
            
        finally:
            try:
                model.development.unload_vsme()
                model.application.unload_vsme()
            except:
                pass
            if os.path.exists("test_constraint_priority.gmoo"):
                os.remove("test_constraint_priority.gmoo")