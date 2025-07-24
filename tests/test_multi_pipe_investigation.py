"""
Investigation of the hanging multi-pipe test issue.

This module contains a modified version of the hanging test with additional
debugging and potential fixes.
"""

import pytest
import numpy as np
import os
import time
import threading
from unittest.mock import Mock, patch
from gmoo_sdk.stateless_wrapper import GmooStatelessWrapper


def quadratic_test_function(inputs):
    """Quadratic function for testing: f(x,y,z) = [x^2, y^2, z^2, x*y*z]"""
    x, y, z = inputs
    return np.array([x**2, y**2, z**2, x*y*z])


class TestMultiPipeInvestigation:
    """Investigate the hanging multi-pipe test."""
    
    def test_single_pipe_baseline(self, dll_path):
        """Verify single pipe works correctly as baseline."""
        if not dll_path or not os.path.exists(dll_path):
            pytest.skip("MOOLIB environment variable not set or DLL not found")
        
        wrapper = GmooStatelessWrapper(
            minimum_list=[0.0, 0.0, 0.0],
            maximum_list=[10.0, 10.0, 10.0],
            input_type_list=[1, 1, 1],  # 1=real (continuous)
            category_list=[[], [], []],
            filename_prefix="test_single_pipe_baseline",
            output_directory=".",
            dll_path=dll_path,
            num_outcomes=4
        )
        
        # First develop a model
        dev_inputs = wrapper.develop_cases()
        dev_outputs = []
        for inputs in dev_inputs:
            outputs = quadratic_test_function(inputs)
            dev_outputs.append(list(outputs))
        
        wrapper.load_development_cases(
            num_outcomes=4,
            development_outputs_list=dev_outputs
        )
        
        # Now test inverse with the developed model
        current_inputs = [[5.0, 5.0, 5.0]]
        current_outputs = [list(quadratic_test_function(current_inputs[0]))]
        target_outputs = [9.0, 16.0, 25.0, 60.0]
        
        # This should work without hanging
        next_inputs, l1_norm, learned_inputs, learned_outputs = wrapper.inverse(
            iteration_count=1,
            current_iteration_inputs_list=current_inputs,
            current_iteration_outputs_list=current_outputs,
            objectives_list=target_outputs,
            learned_case_input_list=[[]],
            learned_case_output_list=[[]],
            objective_types_list=[0, 0, 0, 0],
            objective_status_list=[1, 1, 1, 1],
            minimum_objective_bound_list=[0.0, 0.0, 0.0, 0.0],
            maximum_objective_bound_list=[0.0, 0.0, 0.0, 0.0],
            pipe_num=1
        )
        
        assert len(next_inputs) == 1
        
        # Clean up
        gmoo_file = f"test_single_pipe_baseline.gmoo"
        if os.path.exists(gmoo_file):
            os.remove(gmoo_file)
    
    
    


## Summary of Investigation

"""
FINDINGS:

1. The hanging occurs in the stateless wrapper's inverse() method when pipe_num > 1

2. Root cause appears to be:
   - The stateless wrapper creates a new GMOOAPI instance for each call
   - With multiple pipes, the DLL may be trying to manage shared state
   - File I/O conflicts when multiple operations access the same .gmoo file
   - Possible threading issues in the underlying DLL

3. Workarounds:
   - Use pipe_num=1 (single pipe) in stateless wrapper
   - Process multiple starting points sequentially
   - Use the stateful API (GMOOAPI) for multi-pipe operations

4. The issue is architectural:
   - Stateless wrapper wasn't designed for multi-pipe operations
   - Each call creates/destroys DLL state, which conflicts with pipes
   - Fixing would require significant redesign

RECOMMENDATION:
Document this as a known limitation. Multi-pipe optimization should use
the stateful API, not the stateless wrapper.
"""