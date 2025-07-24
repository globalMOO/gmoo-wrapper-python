"""
Development mode tests for the GMOO wrapper.

These tests verify the model development functionality including
case generation, model training, and model export.
"""

import pytest
import numpy as np
import os
from numpy.linalg import norm

from conftest import develop_model

def test_model_development(simple_model):
    """Test the full model development process."""
    try:
        # Develop the model
        gmoo_file = develop_model(simple_model)
        
        # Verify the file exists
        assert os.path.exists(gmoo_file), f"GMOO file {gmoo_file} should exist"
        
        # Clean up - delete the file
        os.remove(gmoo_file)
    except Exception as e:
        pytest.fail(f"Model development failed: {e}")

@pytest.mark.parametrize('simple_model', [{
    'filename': 'test_categorical',
    'var_mins': [0.0, 0.0, 1.0],
    'var_maxs': [10.0, 10.0, 3.0],
    'num_input_vars': 3,
    'num_output_vars': 3,
    'model_function': lambda x: np.array([x[0]*x[2], x[1]*x[2], x[0]+x[1]+x[2]]),
    'var_types': [1, 1, 4],  # Float, Float, Categorical
    'categories_list': [None, None, ["Low", "Medium", "High"]]
}], indirect=True)
def test_categorical_variables(simple_model):
    """Test that categorical variables are handled correctly."""
    try:
        # Clearing the current VSME memory
        simple_model.development.unload_vsme()
        simple_model.application.unload_vsme()

        # Initialize development setup
        simple_model.development.load_vsme_name()
        simple_model.development.initialize_variables()
        simple_model.development.load_variable_types()
        simple_model.development.load_variable_limits()
        simple_model.development.load_category_labels()
        
        # Design agents and cases
        simple_model.development.design_agents()
        simple_model.development.design_cases()
        
        # Get case count
        case_count = simple_model.development.get_case_count()
        assert case_count > 0, "Expected at least one case to be designed"
        
        # Check a few cases to ensure categorical variables are within range
        for i in range(1, min(10, case_count + 1)):
            case_vars = simple_model.development.poke_case_variables(i)
            # The third variable should be categorical (1, 2, or 3)
            categorical_value = case_vars[2]
            assert 1 <= categorical_value <= 3, f"Categorical value {categorical_value} out of range [1, 3]"
        
        # Check category labels
        # Note: Not testing VSMEdevPokeCategoryLabel since it doesn't exist in the DLL
        # Just verify that the categorical variable is in the correct range
        assert case_vars[2] <= 3 and case_vars[2] >= 1, "Categorical variable should be in range [1,3]"
    
    except Exception as e:
        pytest.fail(f"Categorical variables test failed: {e}")
