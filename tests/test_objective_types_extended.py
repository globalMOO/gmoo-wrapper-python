"""
Extended tests for all objective types in GMOO.

This module tests all 23 objective types (0-22) to ensure proper functionality
and improve code coverage in the application module.
"""

import pytest
import numpy as np
import os
import logging
from gmoo_sdk.dll_interface import GMOOAPI

from conftest import develop_model

logger = logging.getLogger("gmoo_test")


def constrained_test_function(inputs):
    """Test function with constraints for testing inequality objectives."""
    x, y = inputs
    # Outputs:
    # 1. Main objective: x^2 + y^2 (to minimize or maximize)
    # 2. Constraint 1: x + y (for inequality constraints)
    # 3. Constraint 2: x * y (for inequality constraints)
    # 4. Extra output: x - y
    return np.array([x**2 + y**2, x + y, x * y, x - y])


class TestAllObjectiveTypes:
    """Test all objective types (0-22) supported by GMOO."""
    
    @pytest.fixture
    def constrained_model(self, loaded_dll):
        """Create a model for testing constrained optimization."""
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_constrained",
            var_mins=[0.0, 0.0],
            var_maxs=[10.0, 10.0],
            num_output_vars=4,
            model_function=constrained_test_function,
            save_file_dir='.'
        )
        
        # Develop the model
        gmoo_file = develop_model(model)
        
        yield model
        
        # Cleanup
        if os.path.exists(gmoo_file):
            try:
                model.development.unload_vsme()
                model.application.unload_vsme()
                os.remove(gmoo_file)
            except:
                pass
    
    def test_type_0_exact_match(self, simple_model):
        """Test Type 0: Exact match objective."""
        # Develop model
        develop_model(simple_model)
        
        # Load in application mode
        simple_model.application.load_model()
        
        # Set exact match targets
        target = [10.0, 5.0, 15.0]
        simple_model.application.assign_objectives_target(target, [0, 0, 0])
        
        # No uncertainty for exact match
        simple_model.application.load_objective_uncertainty([0.0]*3, [0.0]*3)
        
        # Test that objective is set correctly
        # We can't easily test the full optimization here without running many iterations
        # but we verify the setup works
        assert True  # Setup successful
        
        simple_model.application.unload_vsme()
    
    def test_type_1_percentage_error(self, simple_model):
        """Test Type 1: Percentage error objective."""
        develop_model(simple_model)
        simple_model.application.load_model()
        
        # Set percentage error targets with 5% tolerance
        target = [10.0, 5.0, 15.0]
        simple_model.application.assign_objectives_target(target, [1, 1, 1])
        
        # 5% uncertainty
        simple_model.application.load_objective_uncertainty([-5.0]*3, [5.0]*3)
        
        simple_model.application.unload_vsme()
    
    def test_type_2_absolute_error(self, simple_model):
        """Test Type 2: Absolute error objective."""
        develop_model(simple_model)
        simple_model.application.load_model()
        
        # Set absolute error targets with ±1.0 tolerance
        target = [10.0, 5.0, 15.0]
        simple_model.application.assign_objectives_target(target, [2, 2, 2])
        
        # ±1.0 absolute uncertainty
        simple_model.application.load_objective_uncertainty([-1.0]*3, [1.0]*3)
        
        simple_model.application.unload_vsme()
    
    def test_type_11_less_than(self, constrained_model):
        """Test Type 11: Less than constraint."""
        constrained_model.application.load_model()
        
        # Output 1: minimize
        # Output 2: x + y < 5.0
        # Output 3: x * y < 6.0
        # Output 4: don't care
        targets = [0.0, 5.0, 6.0, 0.0]
        obj_types = [21, 11, 11, 0]  # minimize, <, <, exact
        
        constrained_model.application.assign_objectives_target(targets, obj_types)
        constrained_model.application.load_objective_uncertainty([0.0]*4, [0.0]*4)
        
        constrained_model.application.unload_vsme()
    
    def test_type_12_less_than_equal(self, constrained_model):
        """Test Type 12: Less than or equal constraint."""
        constrained_model.application.load_model()
        
        targets = [0.0, 5.0, 6.0, 0.0]
        obj_types = [21, 12, 12, 0]  # minimize, <=, <=, exact
        
        constrained_model.application.assign_objectives_target(targets, obj_types)
        constrained_model.application.load_objective_uncertainty([0.0]*4, [0.0]*4)
        
        constrained_model.application.unload_vsme()
    
    def test_type_13_greater_than(self, constrained_model):
        """Test Type 13: Greater than constraint."""
        constrained_model.application.load_model()
        
        # Output 2: x + y > 3.0
        # Output 3: x * y > 2.0
        targets = [0.0, 3.0, 2.0, 0.0]
        obj_types = [21, 13, 13, 0]  # minimize, >, >, exact
        
        constrained_model.application.assign_objectives_target(targets, obj_types)
        constrained_model.application.load_objective_uncertainty([0.0]*4, [0.0]*4)
        
        constrained_model.application.unload_vsme()
    
    def test_type_14_greater_than_equal(self, constrained_model):
        """Test Type 14: Greater than or equal constraint."""
        constrained_model.application.load_model()
        
        targets = [0.0, 3.0, 2.0, 0.0]
        obj_types = [21, 14, 14, 0]  # minimize, >=, >=, exact
        
        constrained_model.application.assign_objectives_target(targets, obj_types)
        constrained_model.application.load_objective_uncertainty([0.0]*4, [0.0]*4)
        
        constrained_model.application.unload_vsme()
    
    def test_type_21_minimize(self, constrained_model):
        """Test Type 21: Minimize objective."""
        constrained_model.application.load_model()
        
        # Minimize x^2 + y^2
        targets = [0.0, 5.0, 0.0, 0.0]  # Target value doesn't matter for minimize
        obj_types = [21, 12, 0, 0]  # minimize, <=, exact, exact
        
        constrained_model.application.assign_objectives_target(targets, obj_types)
        constrained_model.application.load_objective_uncertainty([0.0]*4, [0.0]*4)
        
        constrained_model.application.unload_vsme()
    
    def test_type_22_maximize(self, constrained_model):
        """Test Type 22: Maximize objective."""
        constrained_model.application.load_model()
        
        # Maximize x^2 + y^2 (subject to constraints)
        targets = [100.0, 5.0, 0.0, 0.0]  # Target value doesn't matter for maximize
        obj_types = [22, 12, 0, 0]  # maximize, <=, exact, exact
        
        constrained_model.application.assign_objectives_target(targets, obj_types)
        constrained_model.application.load_objective_uncertainty([0.0]*4, [0.0]*4)
        
        constrained_model.application.unload_vsme()
    
    def test_mixed_objective_types(self, constrained_model):
        """Test combination of different objective types."""
        constrained_model.application.load_model()
        
        # Complex multi-objective problem:
        # - Minimize main objective
        # - Constraint: x + y <= 8.0
        # - Constraint: x * y >= 4.0
        # - Target: x - y = 2.0 ± 0.5
        targets = [0.0, 8.0, 4.0, 2.0]
        obj_types = [21, 12, 14, 2]  # minimize, <=, >=, absolute_error
        
        constrained_model.application.assign_objectives_target(targets, obj_types)
        constrained_model.application.load_objective_uncertainty([0.0, 0.0, 0.0, -0.5], 
                                                                [0.0, 0.0, 0.0, 0.5])
        
        constrained_model.application.unload_vsme()
    
    @pytest.mark.parametrize('obj_type', range(23))
    def test_all_objective_types_individually(self, simple_model, obj_type):
        """Parametrized test to ensure all objective types 0-22 are tested."""
        develop_model(simple_model)
        simple_model.application.load_model()
        
        # Set up appropriate targets for each type
        if obj_type in [0, 1, 2]:  # Match-based objectives
            targets = [10.0, 5.0, 15.0]
        elif obj_type in [11, 12]:  # Less than constraints
            targets = [20.0, 20.0, 20.0]  # Upper bounds
        elif obj_type in [13, 14]:  # Greater than constraints
            targets = [1.0, 1.0, 1.0]  # Lower bounds
        elif obj_type in [21, 22]:  # Min/Max objectives
            targets = [0.0, 0.0, 0.0]  # Dummy targets
        else:
            # For any future objective types
            targets = [10.0, 5.0, 15.0]
        
        try:
            simple_model.application.assign_objectives_target(targets, [obj_type]*3)
            
            # Set appropriate uncertainty
            if obj_type == 1:  # Percentage
                simple_model.application.load_objective_uncertainty([-5.0]*3, [5.0]*3)
            elif obj_type == 2:  # Absolute
                simple_model.application.load_objective_uncertainty([-1.0]*3, [1.0]*3)
            else:
                simple_model.application.load_objective_uncertainty([0.0]*3, [0.0]*3)
            
            # If we get here, the objective type is supported
            assert True
        except Exception as e:
            if obj_type > 22:
                # Expected to fail for undefined objective types
                pytest.skip(f"Objective type {obj_type} not yet implemented")
            else:
                raise e
        finally:
            simple_model.application.unload_vsme()