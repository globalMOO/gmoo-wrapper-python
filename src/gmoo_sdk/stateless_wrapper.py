"""
Stateless Wrapper for GMOO SDK

This module provides a stateless wrapper interface for the GMOO SDK that manages
the DLL lifecycle and provides a simplified interface similar to the refactor version
but built on top of the main GMOOAPI.

Authors: Matt Freeman, Jason Hopkins
Version: 2.0.0
"""

import ctypes
import os
import logging
from typing import List, Optional, Tuple, Any, Dict
from .load_dll import load_dll
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def element_wise_maximum(list1: List[float], list2: List[float]) -> List[float]:
    """Element-wise maximum of two lists."""
    return [max(a, b) for a, b in zip(list1, list2)]

def element_wise_minimum(list1: List[float], list2: List[float]) -> List[float]:
    """Element-wise minimum of two lists."""
    return [min(a, b) for a, b in zip(list1, list2)]

class GmooStatelessWrapper:
    """
    Stateless wrapper for GMOO optimization that manages DLL lifecycle
    and provides a simplified interface compatible with the refactor version.
    """
    
    def __init__(self, 
                 minimum_list: List[float], 
                 maximum_list: List[float], 
                 input_type_list: List[int], 
                 category_list: List[List[str]], 
                 filename_prefix: str, 
                 output_directory: str,
                 dll_path: str,
                 num_outcomes: int = 0):
        """
        Initialize the stateless wrapper.
        
        Args:
            minimum_list: Minimum values for each input variable
            maximum_list: Maximum values for each input variable
            input_type_list: Variable types (1=real, 2=integer, 3=logical, 4=categorical)
            category_list: Category names for categorical variables
            filename_prefix: Base filename for GMOO files
            output_directory: Directory for output files
            dll_path: Path to the VSME DLL
            num_outcomes: Number of output variables
        """
        # Validate inputs before assignment
        self._validate_initialization_inputs(
            minimum_list, maximum_list, input_type_list, category_list,
            filename_prefix, output_directory, dll_path, num_outcomes
        )
        
        self.minimum_list = minimum_list
        self.maximum_list = maximum_list
        self.input_type_list = input_type_list
        self.category_list = category_list
        self.filename_prefix = filename_prefix
        self.output_directory = output_directory
        self.dll_path = dll_path
        self.num_outcomes = num_outcomes
        self.num_inputs = len(minimum_list)
        
        # Load DLL
        self.vsme_windll = load_dll(dll_path)
        
        # Import here to avoid circular imports
        from .dll_interface import GMOOAPI
        
        # Create a dummy model function - will be overridden when needed
        def dummy_function(inputs):
            return [0.0] * self.num_outcomes
        
        # Create GMOOAPI instance
        logger.info(f"Creating GMOOAPI with filename_prefix={self.filename_prefix}, output_directory={self.output_directory}")
        self.gmoo_api_client = GMOOAPI(
            vsme_windll=self.vsme_windll,
            vsme_input_filename=self.filename_prefix,
            var_mins=self.minimum_list,
            var_maxs=self.maximum_list,
            num_output_vars=self.num_outcomes,
            model_function=dummy_function,
            save_file_dir=self.output_directory,
            var_types=self.input_type_list,
            categories_list=self.category_list
        )
        logger.info(f"GMOOAPI created with vsme_input_filename={self.gmoo_api_client.vsme_input_filename}")
    
    def _validate_initialization_inputs(self, minimum_list, maximum_list, input_type_list,
                                      category_list, filename_prefix, output_directory,
                                      dll_path, num_outcomes):
        """Validate all initialization inputs."""
        # Import helpers
        from .helpers import validate_nan
        
        # Check for None inputs
        if minimum_list is None or maximum_list is None:
            raise ValueError("minimum_list and maximum_list cannot be None")
        
        # Check list lengths match
        if len(minimum_list) != len(maximum_list):
            raise ValueError(f"minimum_list length ({len(minimum_list)}) must match maximum_list length ({len(maximum_list)})")
        
        if len(input_type_list) != len(minimum_list):
            raise ValueError(f"input_type_list length ({len(input_type_list)}) must match variable count ({len(minimum_list)})")
            
        if len(category_list) != len(minimum_list):
            raise ValueError(f"category_list length ({len(category_list)}) must match variable count ({len(minimum_list)})")
        
        # Validate no NaN values
        validate_nan(minimum_list, "minimum_list")
        validate_nan(maximum_list, "maximum_list")
        
        # Validate bounds
        for i, (min_val, max_val) in enumerate(zip(minimum_list, maximum_list)):
            if min_val > max_val:
                raise ValueError(f"Variable {i}: minimum ({min_val}) cannot be greater than maximum ({max_val})")
        
        # Validate variable types
        valid_types = [1, 2, 3, 4]  # 1=real, 2=integer, 3=logical, 4=categorical
        for i, var_type in enumerate(input_type_list):
            if var_type not in valid_types:
                raise ValueError(f"Variable {i}: invalid type {var_type}. Valid types are {valid_types}")
        
        # Validate categorical variables have categories
        for i, (var_type, categories) in enumerate(zip(input_type_list, category_list)):
            if var_type == 4:  # categorical
                if not categories or len(categories) == 0:
                    raise ValueError(f"Variable {i}: categorical variable must have at least one category")
            else:
                if categories and len(categories) > 0:
                    raise ValueError(f"Variable {i}: non-categorical variable should have empty category list")
        
        # Validate paths and strings
        if not filename_prefix or not isinstance(filename_prefix, str):
            raise ValueError("filename_prefix must be a non-empty string")
            
        if not output_directory or not isinstance(output_directory, str):
            raise ValueError("output_directory must be a non-empty string")
            
        if not dll_path or not isinstance(dll_path, str):
            raise ValueError("dll_path must be a non-empty string")
        
        # Validate num_outcomes
        if num_outcomes < 0:
            raise ValueError(f"num_outcomes ({num_outcomes}) must be non-negative")
    
    def develop_cases(self, params: Optional[Dict] = None) -> List[List[float]]:
        """
        Generate development cases for training the inverse model.
        Bridges the gap to the main GMOOAPI development workflow.
        
        Args:
            params: Optional meta-parameters for case generation
            
        Returns:
            List of input case arrays
        """
        # Execute the complete development setup workflow
        self.gmoo_api_client.development.load_vsme_name()
        
        # Load custom parameters if provided (may need additional implementation)
        if params is not None:
            # The main API might not support all the same parameters
            logger.info(f"Meta parameters provided but may not be fully supported: {params}")
        
        # Initialize development
        self.gmoo_api_client.development.initialize_variables()
        self.gmoo_api_client.development.load_variable_types()
        self.gmoo_api_client.development.load_variable_limits()
        
        # Design agents and cases
        self.gmoo_api_client.development.design_agents()
        self.gmoo_api_client.development.design_cases()
        self.gmoo_api_client.development.get_case_count()
        self.gmoo_api_client.development.init_backup_file()
        
        # Generate input cases
        input_case_list = []
        for kk in range(1, self.gmoo_api_client.devCaseCount.value + 1):
            case_vars = self.gmoo_api_client.development.poke_case_variables(kk)
            input_case_list.append(list(case_vars))
        
        # Unload VSME instances (bridge methods)
        self.dev_unload_vsme()
        self.app_unload_vsme()
        
        return input_case_list
    
    def load_development_cases(self,
                              num_outcomes: int,
                              development_outputs_list: List[Optional[List[float]]],
                              extra_inputs_list: Optional[List[List[float]]] = None,
                              extra_outputs_list: Optional[List[List[float]]] = None):
        """
        Load development case results and train the inverse model.
        Bridges the gap to the main GMOOAPI development workflow.
        
        Args:
            num_outcomes: Number of output variables
            development_outputs_list: Results from running development cases
            extra_inputs_list: Optional additional input cases
            extra_outputs_list: Optional additional output cases
        """
        self.num_outcomes = num_outcomes
        
        # Validate output arrays
        invalid_entries_list = []
        for ii, entry in enumerate(development_outputs_list):
            if entry is not None:
                if len(entry) != num_outcomes:
                    logger.warning(f"Invalid output array length in development case {ii}. Resetting to None.")
                    invalid_entries_list.append(ii)
        
        for ii in invalid_entries_list:
            development_outputs_list[ii] = None
        
        # Execute the development loading workflow
        self.gmoo_api_client.development.read_backup_file()
        self.gmoo_api_client.development.initialize_outcomes()
        
        # Load each case result
        for kk in range(1, len(development_outputs_list) + 1):
            self.gmoo_api_client.development.load_case_results(kk, development_outputs_list[kk-1])
        
        # Load extra cases if provided
        if extra_inputs_list is not None and extra_outputs_list is not None:
            # This would require loading user cases one by one
            for inputs, outputs in zip(extra_inputs_list, extra_outputs_list):
                # Note: This may need specific implementation in the main API
                logger.info("Extra cases provided but direct loading may not be supported")
        
        # Develop the VSME
        self.gmoo_api_client.development.develop_vsme()
        
        # Export VSME to file
        self.gmoo_api_client.development.export_vsme()
        
        # Unload VSME instances (bridge methods)
        self.dev_unload_vsme()
        self.app_unload_vsme()
    
    def inverse(self,
                iteration_count: int,
                current_iteration_inputs_list: List[List[float]],
                current_iteration_outputs_list: List[List[float]],
                objectives_list: List[float],
                learned_case_input_list: List[List[List[float]]],
                learned_case_output_list: List[List[List[float]]],
                objective_types_list: List[int],
                objective_status_list: List[int],
                minimum_objective_bound_list: List[float],
                maximum_objective_bound_list: List[float],
                pipe_num: int = 1) -> Tuple[List[List[float]], float, List[List[List[float]]], List[List[List[float]]]]:
        """
        Perform a single inverse optimization iteration.
        This is the most complex bridge method that implements the full inverse workflow.
        
        Args:
            iteration_count: Current iteration number
            current_iteration_inputs_list: Current input values for each pipe
            current_iteration_outputs_list: Current output values for each pipe
            objectives_list: Target objective values
            learned_case_input_list: Previously learned input cases
            learned_case_output_list: Previously learned output cases
            objective_types_list: Objective types for each output
            objective_status_list: Active/inactive status for each objective
            minimum_objective_bound_list: Lower uncertainty bounds
            maximum_objective_bound_list: Upper uncertainty bounds
            pipe_num: Number of parallel pipes
            
        Returns:
            Tuple of (next_inputs, l1_norm, learned_inputs, learned_outputs)
        """
        logger.info(f"ENTERING inverse() method - iteration {iteration_count}")
        
        # Unload any existing VSME instances - simplified cleanup like working version
        self.dev_unload_vsme()
        self.app_unload_vsme()
        
        # Load the model (bridge method)
        logger.debug(f"Iteration {iteration_count}: Loading model")
        self.load_model()
        logger.debug(f"Iteration {iteration_count}: Model loaded successfully")
        
        # Get and enforce bounds (bridge methods)
        logger.debug(f"Iteration {iteration_count}: Getting variable limits")
        min_vars_loaded = self.poke_variable_dev_limit_min()
        max_vars_loaded = self.poke_variable_dev_limit_max()
        logger.debug(f"Iteration {iteration_count}: Variable limits retrieved")
        
        var_mins_reset = element_wise_maximum(min_vars_loaded, self.minimum_list)
        var_maxs_reset = element_wise_minimum(max_vars_loaded, self.maximum_list)
        
        logger.debug(f"Iteration {iteration_count}: Setting variable limits")
        self.app_load_variable_limit_min(var_mins_reset)
        self.app_load_variable_limit_max(var_maxs_reset)
        logger.debug(f"Iteration {iteration_count}: Variable limits set")
        
        # Initialize variables for multiple pipes (bridge method)
        pipes = pipe_num
        logger.debug(f"Iteration {iteration_count}: Initializing variables for {pipes} pipes")
        
        # Load learned cases - EXACT COPY of refactored version logic
        if pipes > 0:
            self.app_init_variables(nPipes=pipes)
            for pipe in range(1, pipes + 1):
                # Use the EXACT same loop structure as refactored version
                for ii, learnedIn, learnedOut in zip(range(1, len(learned_case_input_list[pipe-1]) + 1), learned_case_input_list[pipe-1], learned_case_output_list[pipe-1]):
                    self.load_learned_case(len(self.minimum_list), self.num_outcomes, learnedIn, learnedOut, iPipe=pipe)
        else:
            # Use the EXACT same loop structure as refactored version for single pipe
            for ii, learnedIn, learnedOut in zip(range(len(learned_case_input_list)), learned_case_input_list, learned_case_output_list):
                self.load_learned_case(len(self.minimum_list), self.num_outcomes, learnedIn, learnedOut)
        
        # Perform inverse iteration (bridge method)
        message, next_input_list, l1norm_full, l2norm = self.inverse_single_iteration(
            objectivesTarget=objectives_list,
            currentInputVars=current_iteration_inputs_list,
            currentOutcomeVars=current_iteration_outputs_list,
            iteration=iteration_count,
            objectiveTypes=objective_types_list,
            objectives_status=objective_status_list,
            objectives_uncertainty_minus=minimum_objective_bound_list,
            objectives_uncertainty_plus=maximum_objective_bound_list,
            reinitializeModel=False,
            pipes=pipes,
            manual_starting_point=False
        )

        logger.debug(f"Iteration {iteration_count}: New inputs {next_input_list}")
        
        # Get learned cases count (bridge method)
        learned_cases_count = self.poke_dimensions10()[5]
        
        # Retrieve learned cases - simplified like the refactor version
        retrieved_learned_inputs_list = []
        retrieved_learned_outputs_list = []
        
        if pipes > 0:
            retrieved_learned_inputs_list = [[] for _ in range(pipes)]
            retrieved_learned_outputs_list = [[] for _ in range(pipes)]
            
            for pipe in range(1, pipes + 1):
                # Get pipe-specific learned cases - simplified error handling
                try:
                    this_pipe_learned_cases = self.poke_pipe_learned_cases(pipe)
                    for ii in range(1, this_pipe_learned_cases.value + 1):
                        learned_input, learned_output = self.poke_learned_case(self.num_inputs, self.num_outcomes, ii, iPipe=pipe)
                        retrieved_learned_inputs_list[pipe-1].append(learned_input)
                        retrieved_learned_outputs_list[pipe-1].append(learned_output)
                except Exception as e:
                    logger.warning(f"Could not retrieve learned cases for pipe {pipe}: {e}")
        else:
            # Single pipe case - load ALL learned cases from DLL
            for ii in range(1, learned_cases_count.value + 1):
                try:
                    learned_input, learned_output = self.poke_learned_case(self.num_inputs, self.num_outcomes, ii)
                    retrieved_learned_inputs_list.append(learned_input)
                    retrieved_learned_outputs_list.append(learned_output)
                except Exception as e:
                    logger.warning(f"Could not retrieve learned case {ii}: {e}")
                    break
        
        # Unload VSME instances (bridge methods)
        self.dev_unload_vsme()
        self.app_unload_vsme()
        
        return next_input_list, l1norm_full, retrieved_learned_inputs_list, retrieved_learned_outputs_list
    
    # Bridge methods that map to the main GMOOAPI
    
    def dev_unload_vsme(self):
        """Bridge method: Unload development VSME."""
        logger.debug("dev_unload_vsme: About to call development.unload_vsme()")
        try:
            self.gmoo_api_client.development.unload_vsme()
            logger.debug("dev_unload_vsme: development.unload_vsme() completed")
        except Exception as e:
            logger.debug(f"dev_unload_vsme: Exception (may be normal): {e}")
            pass
    
    def app_unload_vsme(self):
        """Bridge method: Unload application VSME."""
        try:
            self.gmoo_api_client.application.unload_vsme()
        except Exception as e:
            logger.debug(f"app_unload_vsme: Exception (may be normal): {e}")
            pass
    
    def load_model(self):
        """Bridge method: Load VSME model."""
        logger.debug("load_model: About to call application.load_model()")
        try:
            result = self.gmoo_api_client.application.load_model()
            logger.debug("load_model: application.load_model() completed successfully")
            return result
        except Exception as e:
            logger.error(f"load_model: Error in application.load_model(): {e}")
            raise
    
    def poke_variable_dev_limit_min(self):
        """Bridge method: Get variable development minimum limits."""
        return self.gmoo_api_client.application.poke_variable_dev_limit_min()
    
    def poke_variable_dev_limit_max(self):
        """Bridge method: Get variable development maximum limits."""
        return self.gmoo_api_client.application.poke_variable_dev_limit_max()
    
    def app_load_variable_limit_min(self, var_mins):
        """Bridge method: Load variable minimum limits."""
        return self.gmoo_api_client.application.load_variable_limit_min(var_mins)
    
    def app_load_variable_limit_max(self, var_maxs):
        """Bridge method: Load variable maximum limits."""
        return self.gmoo_api_client.application.load_variable_limit_max(var_maxs)
    
    def app_init_variables(self, nPipes=1):
        """Bridge method: Initialize variables."""
        return self.gmoo_api_client.application.init_variables(nPipes=nPipes)
    
    def load_learned_case(self, nVars, nOuts, dVarValues, dOutValues, iPipe=1):
        """Bridge method: Load learned case."""
        return self.gmoo_api_client.application.load_learned_case(nVars, nOuts, dVarValues, dOutValues, iPipe)
    
    def poke_dimensions10(self):
        """Bridge method: Get dimensions."""
        return self.gmoo_api_client.application.poke_dimensions10()
    
    def poke_pipe_learned_cases(self, pipe):
        """Bridge method: Get pipe-specific learned cases count."""
        return self.gmoo_api_client.application.poke_pipe_learned_cases(pipe)
    
    def poke_learned_case(self, nVars, nOuts, iCase, iPipe=1):
        """Bridge method: Get learned case."""
        return self.gmoo_api_client.application.poke_learned_case(nVars, nOuts, iCase, iPipe)
    
    def inverse_single_iteration(self, **kwargs):
        """
        Bridge method: Perform single inverse iteration.
        This is a complex method that needs to orchestrate multiple calls to the main API.
        """
        # Extract parameters
        objectives_target = kwargs.get('objectivesTarget')
        current_input_vars = kwargs.get('currentInputVars')
        current_outcome_vars = kwargs.get('currentOutcomeVars')
        iteration = kwargs.get('iteration')
        objective_types = kwargs.get('objectiveTypes')
        objectives_status = kwargs.get('objectives_status')
        objectives_uncertainty_minus = kwargs.get('objectives_uncertainty_minus')
        objectives_uncertainty_plus = kwargs.get('objectives_uncertainty_plus')
        pipes = kwargs.get('pipes', 1)
        
        # Perform the inverse iteration using the main API
        try:
            next_input_list = []
            l1norm_values = []
            
            # Set objectives and types once before processing pipes
            self.gmoo_api_client.application.assign_objectives_target(objectives_target, objective_types)
            
            # Set objective status if provided
            if objectives_status:
                self.gmoo_api_client.application.load_objective_status(objectives_status)
            
            # Set uncertainty bounds if provided
            if objectives_uncertainty_minus and objectives_uncertainty_plus:
                self.gmoo_api_client.application.load_objective_uncertainty(
                    objectives_uncertainty_plus, objectives_uncertainty_minus
                )
            
            for pipe in range(pipes):
                if pipe < len(current_input_vars):
                    try:
                        # Load current variables and outcomes into the DLL for this pipe
                        self.gmoo_api_client.application.load_variable_values(current_input_vars[pipe])
                        self.gmoo_api_client.application.load_outcome_values(current_outcome_vars[pipe])

                        # Run optimization iteration
                        self.gmoo_api_client.application.run_vsme_app(iteration)

                        # Get suggested variables for next iteration
                        next_inputs = self.gmoo_api_client.application.fetch_variables_for_next_iteration()
                        
                        # Calculate L1 norm for this pipe
                        import numpy as np
                        current_outputs_array = np.array(current_outcome_vars[pipe])
                        targets_array = np.array(objectives_target)
                        
                        if objectives_status:
                            status_array = np.array(objectives_status)
                            l1norm = np.linalg.norm(
                                current_outputs_array * status_array - targets_array * status_array, 
                                ord=1
                            )
                        else:
                            l1norm = np.linalg.norm(current_outputs_array - targets_array, ord=1)
                        
                        next_input_list.append(list(next_inputs))
                        l1norm_values.append(l1norm)
                        
                    except Exception as pipe_error:
                        logger.error(f"Error in pipe {pipe}: {pipe_error}")
                        # Use current inputs as fallback for this pipe
                        next_input_list.append(current_input_vars[pipe] if current_input_vars else [0.0] * self.num_inputs)
                        l1norm_values.append(float('inf'))
                else:
                    # Handle case where we don't have enough input data
                    next_input_list.append(current_input_vars[0] if current_input_vars else [0.0] * self.num_inputs)
                    l1norm_values.append(float('inf'))
            
            # Return best L1 norm
            l1norm_full = min(l1norm_values) if l1norm_values else float('inf')
            
            return "Success", next_input_list, l1norm_full, 0.0
            
        except Exception as e:
            logger.error(f"Error in inverse_single_iteration: {e}")
            # Return fallback values
            fallback_inputs = [current_input_vars[0] if current_input_vars else [0.0] * self.num_inputs] * pipes
            return f"Error: {str(e)}", fallback_inputs, float('inf'), float('inf')
