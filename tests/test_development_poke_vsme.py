"""
Tests for poke_vsme_name and other development utility functions.
"""

import pytest
import os
import numpy as np
from gmoo_sdk.dll_interface import GMOOAPI


class TestDevelopmentUtilities:
    """Test utility functions in the development module."""
    
    def test_poke_vsme_name(self, loaded_dll):
        """Test poke_vsme_name functionality."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        # Create a simple model
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_poke_name",
            var_mins=[0.0, 0.0],
            var_maxs=[10.0, 10.0],
            num_output_vars=2,
            model_function=lambda x: np.array([x[0] + x[1], x[0] * x[1]]),
            save_file_dir='.'
        )
        
        # Load the VSME name
        model.development.load_vsme_name()
        
        # Poke (retrieve) the VSME name
        retrieved_name = model.development.poke_vsme_name()
        
        # The name should match what we set (without path)
        assert retrieved_name.strip() == "test_poke_name"
        
        # Clean up
        model.development.unload_vsme()
    
    
    def test_export_case_csv(self, loaded_dll):
        """Test export_case_csv functionality."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_csv_export",
            var_mins=[0.0, 0.0, 0.0],
            var_maxs=[10.0, 10.0, 10.0],
            num_output_vars=2,
            model_function=lambda x: np.array([sum(x), np.prod(x)]),
            save_file_dir='.'
        )
        
        # Set up development
        model.development.load_vsme_name()
        model.development.initialize_variables()
        model.development.load_variable_types()
        model.development.load_variable_limits()
        model.development.design_agents()
        model.development.design_cases()
        model.development.get_case_count()
        
        # Export cases to CSV
        csv_filename = model.development.export_case_csv()
        
        # Check that files were created
        assert os.path.exists(csv_filename)
        assert os.path.exists(f"{model.vsme_input_filename}_DEVVARS.done")
        
        # Read and verify CSV has content
        with open(csv_filename, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 1  # Should have at least header and data
        
        # Clean up
        os.remove(csv_filename)
        os.remove(f"{model.vsme_input_filename}_DEVVARS.done")
        model.development.unload_vsme()
    
    def test_poke_nonlinearity(self, loaded_dll):
        """Test poke_nonlinearity functionality."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        # Create a nonlinear model
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_nonlinearity",
            var_mins=[0.0, 0.0],
            var_maxs=[5.0, 5.0],
            num_output_vars=2,
            model_function=lambda x: np.array([x[0]**2 + x[1]**2, np.sin(x[0]) * np.cos(x[1])]),
            save_file_dir='.'
        )
        
        # Before developing VSME, nonlinearity should be negative
        model.development.load_vsme_name()
        nonlinearity = model.development.poke_nonlinearity()
        assert nonlinearity < 0, "Nonlinearity should be negative before VSME development"
        
        # Develop the model
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
        
        # After development, check nonlinearity
        nonlinearity = model.development.poke_nonlinearity()
        assert nonlinearity >= 0, "Nonlinearity should be non-negative after VSME development"
        # For a nonlinear function, we expect nonlinearity > 0
        assert nonlinearity > 0.1, f"Expected nonlinearity > 0.1 for nonlinear function, got {nonlinearity}"
        
        # Clean up
        model.development.export_vsme()
        model.development.unload_vsme()
        gmoo_file = f"{model.vsme_input_filename}.gmoo"
        if os.path.exists(gmoo_file):
            os.remove(gmoo_file)
        vprj_file = f"{model.vsme_input_filename}.VPRJ"
        if os.path.exists(vprj_file):
            os.remove(vprj_file)