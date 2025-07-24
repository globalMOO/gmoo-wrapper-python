"""
Tests for error handling paths in application.py.
"""

import os
import pytest
import numpy as np
from gmoo_sdk.dll_interface import GMOOAPI, GMOOException


class TestApplicationErrorPaths:
    """Test error handling in the application module."""
    
    def test_invalid_objective_assignment(self, loaded_dll):
        """Test assigning objectives with invalid dimensions."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_invalid_obj",
            var_mins=[0.0],
            var_maxs=[1.0],
            num_output_vars=2,
            model_function=lambda x: np.array([x[0]**2, x[0]**3]),
            save_file_dir='.'
        )
        
        # Develop model first
        model.development.load_vsme_name()
        model.development.initialize_variables()
        model.development.load_variable_types()
        model.development.load_variable_limits()
        model.development.design_agents()
        model.development.design_cases()
        case_count = model.development.get_case_count()
        
        # Generate development cases
        development_outcome_arrs = []
        for i in range(1, case_count + 1):
            case_vars = model.development.poke_case_variables(i)
            outputs = model.model_function(case_vars)
            development_outcome_arrs.append(outputs)
        
        # Initialize outcomes and load results (skip backup file)
        model.development.initialize_outcomes()
        
        for i in range(1, len(development_outcome_arrs) + 1):
            model.development.load_case_results(i, development_outcome_arrs[i-1])
        
        model.development.develop_vsme()
        model.development.export_vsme()
        
        # Now test invalid objective assignment
        model.application.load_model()
        
        # Mismatched array lengths should raise ValueError
        with pytest.raises(ValueError, match="Array length mismatch"):
            model.application.assign_objectives_target([1.0, 2.0, 3.0], [0, 0])  # 3 targets, 2 types
        
        # Clean up
        model.development.unload_vsme()
        model.application.unload_vsme()
        gmoo_file = f"{model.vsme_input_filename}.gmoo"
        if os.path.exists(gmoo_file):
            os.remove(gmoo_file)
    
    def test_perform_inverse_with_nan(self, loaded_dll):
        """Test perform_inverse_iteration with NaN inputs."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_nan_inverse",
            var_mins=[0.0, 0.0],
            var_maxs=[10.0, 10.0],
            num_output_vars=2,
            model_function=lambda x: np.array([x[0] + x[1], x[0] - x[1]]),
            save_file_dir='.'
        )
        
        # Quick development
        model.development.load_vsme_name()
        model.development.initialize_variables()
        model.development.load_variable_types()
        model.development.load_variable_limits()
        model.development.design_agents()
        model.development.design_cases()
        case_count = model.development.get_case_count()
        
        # Generate development cases
        development_outcome_arrs = []
        for i in range(1, case_count + 1):
            case_vars = model.development.poke_case_variables(i)
            outputs = model.model_function(case_vars)
            development_outcome_arrs.append(outputs)
        
        # Initialize outcomes and load results (skip backup file)
        model.development.initialize_outcomes()
        
        for i in range(1, len(development_outcome_arrs) + 1):
            model.development.load_case_results(i, development_outcome_arrs[i-1])
        
        model.development.develop_vsme()
        model.development.export_vsme()
        
        # Load model and set objectives
        model.application.load_model()
        model.application.assign_objectives_target([5.0, 1.0], [0, 0])
        
        # Test with NaN in current inputs
        with pytest.raises(ValueError, match="contains NaN"):
            model.application.perform_inverse_iteration(
                target_outputs=[5.0, 1.0],
                current_inputs=[np.nan, 2.0],
                current_outputs=[4.0, 0.0],
                objective_types=[0, 0]
            )
        
        # Test with NaN in current outputs
        with pytest.raises(ValueError, match="contains NaN"):
            model.application.perform_inverse_iteration(
                target_outputs=[5.0, 1.0],
                current_inputs=[3.0, 2.0],
                current_outputs=[5.0, np.nan],
                objective_types=[0, 0]
            )
        
        # Test with NaN in target outputs
        with pytest.raises(ValueError, match="contains NaN"):
            model.application.perform_inverse_iteration(
                target_outputs=[np.nan, 1.0],
                current_inputs=[3.0, 2.0],
                current_outputs=[5.0, 1.0],
                objective_types=[0, 0]
            )
        
        # Clean up
        model.development.unload_vsme()
        model.application.unload_vsme()
        gmoo_file = f"{model.vsme_input_filename}.gmoo"
        if os.path.exists(gmoo_file):
            os.remove(gmoo_file)
    
