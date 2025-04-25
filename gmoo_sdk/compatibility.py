# -*- coding: utf-8 -*-
"""
GMOO SDK Compatibility Module

This module provides backward compatibility for code that was written
for earlier versions of the GMOO SDK. It translates deprecated function
calls and parameters to their modern equivalents in the refactored API.

The module consists of:
1. A GMOOAPILegacy class that wraps the modern GMOOAPI with legacy methods
2. Function wrappers that maintain the interface of deprecated global functions

IMPORTANT: This module is intended for transitional use only.
           New code should use the modern API directly.

Authors: Matt Freeman, Jason Hopkins
Version: 2.0.0
"""

import ctypes
import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.linalg import norm

from gmoo_sdk.dll_interface import GMOOAPI
from gmoo_sdk.helpers import (
    fortran_hollerith_string as fortranHollerithStringHelper,
    c_string_compatibility as cStringCompatibilityHelper,
    validate_nan as error_nan
)

# Configure logging
logger = logging.getLogger(__name__)


class GMOOAPILegacy:
    """
    Legacy compatibility wrapper for the GMOOAPI class.
    
    This class wraps the modern GMOOAPI to provide interfaces compatible with
    the earlier version of the API. It delegates to the appropriate methods
    on the underlying API, translating parameters as needed.
    
    Usage:
        # Create a modern API instance
        api = GMOOAPI(...)
        
        # Wrap it with the legacy interface
        legacy_api = GMOOAPILegacy(api)
        
        # Now use the legacy methods
        legacy_api.dev_load_vsme_name()
    """
    
    def __init__(self, api: GMOOAPI):
        """
        Initialize the legacy wrapper with a modern GMOOAPI instance.
        
        Args:
            api: An instance of the modern GMOOAPI class
        """
        self.api = api
        self._emit_deprecation_warning()
        
    def _emit_deprecation_warning(self):
        """Emit a deprecation warning when this class is instantiated."""
        warnings.warn(
            "The GMOOAPILegacy class is deprecated and will be removed in a future version. "
            "Please update your code to use the GMOOAPI class directly.",
            DeprecationWarning,
            stacklevel=3
        )
    
    # --- Development Methods ---
    
    def dev_load_vsme_name(self, compatibility_mode: bool = False) -> None:
        """Legacy method for loading VSME name."""
        self.api.development.load_vsme_name(compatibility_mode)
    
    def dev_poke_vsme_name(self, compatibility_mode: bool = False) -> str:
        """Legacy method for retrieving VSME name."""
        return self.api.development.poke_vsme_name(compatibility_mode)
    
    def dev_init_backup_file(self, override_name: bool = False) -> None:
        """Legacy method for initializing backup file."""
        self.api.development.init_backup_file(override_name if override_name is not False else None)
    
    def dev_read_backup_file(self) -> None:
        """Legacy method for reading backup file."""
        self.api.development.read_backup_file()
        
    def dev_load_parameters(self, e1: int = -1, e2: int = -1, e3: int = -1, 
                           e4: int = -1, e5: int = -1, e6: int = -1, r: float = 0.4) -> None:
        """Legacy method for loading parameters."""
        self.api.development.load_parameters(e1, e2, e3, e4, e5, e6, r)
        
    def dev_initialize_variables(self) -> None:
        """Legacy method for initializing variables."""
        self.api.development.initialize_variables()
        
    def dev_load_variable_limits(self) -> None:
        """Legacy method for loading variable limits."""
        self.api.development.load_variable_limits()
        
    def dev_load_variable_types(self) -> None:
        """Legacy method for loading variable types."""
        self.api.development.load_variable_types()
        
    def poke_variables(self) -> None:
        """Legacy method for displaying variable information."""
        warnings.warn(
            "poke_variables() is deprecated and will be removed. No direct replacement available.",
            DeprecationWarning
        )
        # Legacy implementation tried to print variables - no direct equivalent
        # We might log the variables instead
        logger.info(f"Variables: {[(i, v) for i, v in enumerate(self.api.dVarLimMin)]}")
    
    def check_loaded_bounds(self) -> None:
        """Legacy method for checking loaded bounds."""
        self.api.development.check_loaded_bounds()
        
    def poke_variable_count(self) -> None:
        """Legacy method for retrieving variable count."""
        # No direct replacement for this specific function, but we can access the information
        logger.info(f"Variable count: {self.api.nVars.value}")
        
    def dev_design_agents(self) -> None:
        """Legacy method for designing agents."""
        self.api.development.design_agents()
        
    def dev_design_cases(self) -> None:
        """Legacy method for designing cases."""
        self.api.development.design_cases()
        
    def dev_get_case_count(self) -> None:
        """Legacy method for retrieving case count."""
        self.api.development.get_case_count()
        
    def dev_initialize_outcomes(self) -> None:
        """Legacy method for initializing outcomes."""
        self.api.development.initialize_outcomes()
        
    def dev_poke_case_variables(self, i_case: int) -> None:
        """Legacy method for retrieving case variables."""
        case_variables = self.api.development.poke_case_variables(i_case)
        # In the legacy API, this populated the caseVariables attribute
        self.api.caseVariables = (ctypes.c_double * self.api.nVars.value)(*case_variables)
        
    def dev_load_case_results(self, i_case: int, dOutValues: Optional[List[float]]) -> None:
        """Legacy method for loading case results."""
        self.api.development.load_case_results(i_case, dOutValues)
        
    def dev_export_case_csv(self, csv_filename: Optional[str] = None) -> None:
        """Legacy method for exporting case variables to CSV."""
        self.api.development.export_case_csv(csv_filename)
        
    def export_vsme(self, vsme_output_filename: Optional[str] = None) -> None:
        """Legacy method for exporting VSME model."""
        self.api.development.export_vsme(vsme_output_filename)
        
    def dev_develop_vsme(self) -> None:
        """Legacy method for developing VSME model."""
        self.api.development.develop_vsme()
        
    def delete_vsme_file(self, vsme_output_filename: Optional[str] = None) -> None:
        """Legacy method for deleting VSME files."""
        self.api.development.delete_vsme_file(vsme_output_filename)
        
    def dev_unload_vsme(self) -> None:
        """Legacy method for unloading VSME in development mode."""
        self.api.development.unload_vsme()
        
    def app_unload_vsme(self) -> None:
        """Legacy method for unloading VSME in application mode."""
        self.api.application.unload_vsme()
        
    def dev_load_category_labels(self) -> None:
        """Legacy method for loading category labels."""
        self.api.development.load_category_labels()
        
    def dev_poke_category_label(self, var_id: int, cat_id: int) -> str:
        """Legacy method for retrieving a category label."""
        return self.api.development.poke_category_label(var_id, cat_id)
        
    def dev_load_user_cases(self, cases_var_values: List[List[float]], 
                           cases_out_values: List[List[float]]) -> None:
        """Legacy method for loading multiple user-defined cases."""
        self.api.development.load_user_cases(cases_var_values, cases_out_values)
        
    def dev_load_user_case(self, var_values: List[float], out_values: List[float]) -> None:
        """Legacy method for loading a single user-defined case."""
        self.api.development.load_user_case(var_values, out_values)
        
    def read_outcomes_csv(self, runs_type: str = "DEV") -> List[List[float]]:
        """Legacy method for reading outcomes from CSV file."""
        return self.api.development.read_outcomes_csv(runs_type)
        
    def vsme_dev_main(self, item: int) -> int:
        """Legacy method for checking VSME development status."""
        if item == 6:
            return 1 if self.api.development.check_development_status() else 0
        elif item == 7:
            return 1 if self.api.development.check_export_status() else 0
        else:
            return 0  # Default return for unsupported items
    
    # --- Application Methods ---
    
    def load_model(self, alternate_vsme_input_filename: Optional[str] = None, 
                  inspect_load: bool = False) -> None:
        """Legacy method for loading a VSME model."""
        self.api.application.load_model(alternate_vsme_input_filename, inspect_load)
        
    def assign_objectives_target(self, objectives_target: List[float], 
                                objective_types: Optional[List[int]] = None) -> None:
        """Legacy method for assigning target objectives."""
        self.api.application.assign_objectives_target(objectives_target, objective_types)
        
    def app_load_objective_status(self, objectives_status: List[int]) -> None:
        """Legacy method for loading objective status."""
        self.api.application.load_objective_status(objectives_status)
        
    def app_poke_objective_status(self) -> List[int]:
        """Legacy method for retrieving objective status."""
        return self.api.application.poke_objective_status()
        
    def app_load_objective_uncertainty(self, dUncPlus: List[float], dUncMinus: List[float]) -> None:
        """Legacy method for loading objective uncertainties."""
        self.api.application.load_objective_uncertainty(dUncPlus, dUncMinus)
        
    def initialize_variables(self, override_init: Union[bool, List[float]]) -> ctypes.Array:
        """Legacy method for initializing variables."""
        return self.api.application.initialize_variables(override_init)
        
    def calculate_initial_solution(self, override_init: Union[bool, List[float]], 
                                  iBias: int = 0, iGen: int = 0) -> ctypes.Array:
        """Legacy method for calculating initial solution."""
        return self.api.application.calculate_initial_solution(override_init, iBias, iGen)
        
    def load_variable_values(self, variable_arr: List[float], iBias: int = 0, iGen: int = 0) -> None:
        """Legacy method for loading variable values."""
        self.api.application.load_variable_values(variable_arr, iBias, iGen)
        
    def app_load_variable_limit_min(self, variable_min_arr: List[float]) -> None:
        """Legacy method for loading minimum variable limits."""
        self.api.application.load_variable_limit_min(variable_min_arr)
        
    def app_load_variable_limit_max(self, variable_max_arr: List[float]) -> None:
        """Legacy method for loading maximum variable limits."""
        self.api.application.load_variable_limit_max(variable_max_arr)
        
    def app_load_category_index(self, var_names: List[str], selected_indices: List[int]) -> None:
        """Legacy method for loading category indices."""
        self.api.application.load_category_index(var_names, selected_indices)
        
    def app_load_category_label(self, category_names: List[str], category_labels: List[str]) -> None:
        """Legacy method for loading category labels."""
        self.api.application.load_category_label(category_names, category_labels)
        
    def load_outcome_values(self, output_arr: List[float], iBias: int = 0, iGen: int = 0) -> None:
        """Legacy method for loading outcome values."""
        self.api.application.load_outcome_values(output_arr, iBias, iGen)
        
    def run_vsme_app(self, iteration_number: int) -> None:
        """Legacy method for running VSME app."""
        self.api.application.run_vsme_app(iteration_number)
        
    def fetch_variables_for_next_iteration(self, iBias: int = 0, iGen: int = 0) -> np.ndarray:
        """Legacy method for fetching variables for next iteration."""
        return self.api.application.fetch_variables_for_next_iteration(iBias, iGen)
        
    def app_init_variables(self, iBias: int = 0) -> None:
        """Legacy method for initializing variables in application mode."""
        self.api.application.init_variables(iBias)
        
    def initialize_genetic_algorithm(self, iOption: int = -1, rLog10Gen: int = 1,
                                    nParents: int = 0, iError: int = 2) -> None:
        """Legacy method for initializing genetic algorithm."""
        self.api.application.initialize_genetic_algorithm(iOption, rLog10Gen, nParents, iError)
        
    def initialize_bias(self, iOption: int = 0, nBias: int = 150) -> None:
        """Legacy method for initializing bias."""
        self.api.application.initialize_bias(iOption, nBias)
        
    def poke_bias(self, iBiasOption: int = 2, nBias: int = 150) -> np.ndarray:
        """Legacy method for retrieving bias values."""
        return self.api.application.poke_bias(iBiasOption, nBias)
        
    def poke_dimensions10(self) -> Tuple[ctypes.c_int, ...]:
        """Legacy method for retrieving model dimensions."""
        return self.api.application.poke_dimensions10()
        
    def poke_nonlinearity(self) -> float:
        """Legacy method for retrieving nonlinearity measure."""
        return self.api.development.poke_nonlinearity()
        
    def poke_learned_case(self, nVars: int, nOuts: int, iCase: int) -> Tuple[np.ndarray, np.ndarray]:
        """Legacy method for retrieving learned case values."""
        return self.api.application.poke_learned_case(nVars, nOuts, iCase)
        
    def load_learned_case(self, nVars: int, nOuts: int, dVarValues: List[float], 
                         dOutValues: List[float], iBias: int = 1) -> None:
        """Legacy method for loading learned case values."""
        self.api.application.load_learned_case(nVars, nOuts, dVarValues, dOutValues, iBias)
        
    def poke_outcome_dev_limit_min(self, nOuts: int) -> np.ndarray:
        """Legacy method for retrieving minimum outcome limits."""
        return self.api.application.poke_outcome_dev_limit_min(nOuts)
        
    def poke_outcome_dev_limit_max(self, nOuts: int) -> np.ndarray:
        """Legacy method for retrieving maximum outcome limits."""
        return self.api.application.poke_outcome_dev_limit_max(nOuts)
        
    def poke_variable_dev_limit_min(self) -> List[float]:
        """Legacy method for retrieving minimum variable limits."""
        return self.api.application.poke_variable_dev_limit_min()
        
    def poke_variable_dev_limit_max(self) -> List[float]:
        """Legacy method for retrieving maximum variable limits."""
        return self.api.application.poke_variable_dev_limit_max()

    # --- Utility Methods ---
    
    def perform_min_error_search(self, target_result: np.ndarray, 
                               total_VSME_model_evaluations: int,
                               min_err_cases: int = 10) -> Tuple[float, np.ndarray, int]:
        """Legacy method for performing minimum error search."""
        return self.api.application.perform_min_error_search(
            target_result, total_VSME_model_evaluations, min_err_cases
        )
        
    def generate_random_exploration_case(self, logspace: List[bool]) -> np.ndarray:
        """Legacy method for generating random exploration case."""
        return self.api.application.generate_random_exploration_case(logspace)
        
    def plot_behavior(self, highlight_point: np.ndarray, initial_points: List[np.ndarray],
                     improved_points: List[np.ndarray]) -> None:
        """Legacy method for plotting behavior."""
        self.api.visualization.plot_behavior(highlight_point, initial_points, improved_points)
        
    def print_an_array(self, arrayIn: np.ndarray) -> None:
        """Legacy method for printing an array."""
        self.api.visualization.print_an_array(arrayIn)


# Legacy global function wrappers
# These functions provide compatibility for code that uses the global functions from gmoo_encapsulation

def pyVSMEDevelopmentSetup(model: GMOOAPI, unbiasVector: Optional[List[float]] = None,
                           params: Optional[Dict[str, Any]] = None) -> Tuple[GMOOAPI, List[np.ndarray]]:
    """
    Legacy wrapper for setting up VSME development.
    
    Args:
        model: The GMOOAPI object for this problem.
        unbiasVector: Optional multiplier for de-biasing the input variables.
        params: Optional dictionary of custom parameters for VSME development.
        
    Returns:
        Tuple containing:
            GMOOAPI: The model with development initialization performed.
            List[np.ndarray]: List of input arrays to be evaluated by an external process.
    """
    warnings.warn(
        "pyVSMEDevelopmentSetup is deprecated. Use model.development.setup_and_design_cases() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Create a legacy wrapper if needed
    if not isinstance(model, GMOOAPILegacy):
        model_legacy = GMOOAPILegacy(model)
    else:
        model_legacy = model
    
    # Initialize unbias vector if not provided
    if unbiasVector is None:
        unbiasVector = [1.0] * model.nVars.value

    # Initialize variables to collect inputs
    input_vectors = []
    
    # Start the development setup process
    model_legacy.dev_load_vsme_name()
    
    # Load custom parameters
    if params is not None:
        model_legacy.dev_load_parameters(**params)

    # Initialize Variables
    model_legacy.dev_initialize_variables()
    
    # Load Variable Data Types or Logarithmic Status
    model_legacy.dev_load_variable_types()

    # Load Variable Limits
    model_legacy.dev_load_variable_limits()
    
    # Load Categorical Labels if needed
    if model.categories_list is not None:
        model_legacy.dev_load_category_labels()

    # Design Agents
    model_legacy.dev_design_agents()

    # Design Cases
    model_legacy.dev_design_cases()

    # Get Case Count
    model_legacy.dev_get_case_count()

    # Extract the case variables
    for kk in range(1, model.devCaseCount.value + 1):
        model_legacy.dev_poke_case_variables(kk)
        inputArr = np.array(model.caseVariables) * unbiasVector
        input_vectors.append(inputArr)

    # Initialize Backup File to save the current state
    model_legacy.dev_init_backup_file()

    return model, input_vectors


def pyVSMEDevelopmentLoad(model: GMOOAPI, outcome_vectors: List[Optional[np.ndarray]],
                         extra_inputs: Optional[List[List[float]]] = None,
                         extra_outputs: Optional[List[List[float]]] = None) -> GMOOAPI:
    """
    Legacy wrapper for loading results and developing the VSME model.
    
    Args:
        model: The GMOOAPI object for this problem.
        outcome_vectors: List of output arrays from evaluating the development cases.
        extra_inputs: Optional additional input cases to include in training.
        extra_outputs: Optional additional output values to include in training.
        
    Returns:
        GMOOAPI: The model with development data loaded and training completed.
    """
    warnings.warn(
        "pyVSMEDevelopmentLoad is deprecated. Use model.development.load_results_and_develop() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Create a legacy wrapper if needed
    if not isinstance(model, GMOOAPILegacy):
        model_legacy = GMOOAPILegacy(model)
    else:
        model_legacy = model

    # Restore created experimental design from backup file for consistency
    model_legacy.dev_read_backup_file()

    # Initialize outcomes in memory
    model_legacy.dev_initialize_outcomes()
            
    # Load each case result
    for kk in range(1, len(outcome_vectors) + 1):
        model_legacy.dev_load_case_results(kk, outcome_vectors[kk-1])
        
    # Add extra user-provided cases if available
    if extra_inputs is not None and extra_outputs is not None:
        model_legacy.dev_load_user_cases(extra_inputs, extra_outputs)

    # Develop the VSME model
    model_legacy.dev_develop_vsme()

    # Wait for development to complete
    while True:
        status = model_legacy.vsme_dev_main(6)  # Check if VSME is developed
        if status:
            break
        print("VSME development ongoing ... ")
        time.sleep(1.0)

    # Export VSME to file
    model_legacy.export_vsme()

    # Wait for export to complete
    while True:
        status = model_legacy.vsme_dev_main(7)  # Check if VSME is exported
        if status:
            break
        print("VSME file write ongoing ... ")
        time.sleep(1.0)

    return model


def pyVSMEInverseSingleIter(model: GMOOAPI, objectivesTarget: List[float],
                          currentInputVars: List[float], currentOutcomeVars: List[float],
                          iteration: int, objectiveTypes: Optional[List[int]] = None,
                          objectives_status: Optional[List[int]] = None,
                          objectives_uncertainty_minus: Optional[List[float]] = None,
                          objectives_uncertainty_plus: Optional[List[float]] = None,
                          reinitializeModel: bool = True) -> Tuple[GMOOAPI, str, np.ndarray, float, float]:
    """
    Legacy wrapper for performing a single iteration of inverse optimization.
    
    Args:
        model: The GMOOAPI object for this problem.
        objectivesTarget: Target output values to be matched.
        currentInputVars: Current input variable values.
        currentOutcomeVars: Current output values.
        iteration: Current iteration number.
        objectiveTypes: Types of objectives (exact match, minimize, etc.).
        objectives_status: Flags indicating which objectives are active (1) or inactive (0).
        objectives_uncertainty_minus: Lower uncertainty bounds for objectives.
        objectives_uncertainty_plus: Upper uncertainty bounds for objectives.
        reinitializeModel: Whether to reload the model (True) or reuse existing (False).
        
    Returns:
        Tuple containing:
            GMOOAPI: The model with optimization results loaded.
            str: Message indicating the result status.
            np.ndarray: Suggested input variable values for the next iteration.
            float: L1 norm of the error.
            float: L2 norm of the error.
    """
    warnings.warn(
        "pyVSMEInverseSingleIter is deprecated. Use model.application.perform_inverse_iteration() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Create a legacy wrapper if needed
    if not isinstance(model, GMOOAPILegacy):
        model_legacy = GMOOAPILegacy(model)
    else:
        model_legacy = model
    
    # Initialize model if requested
    if reinitializeModel:
        model_legacy.load_model()

    # Load categorical variable information if needed
    if model.categories_list is not None:
        model_legacy.dev_load_category_labels()
 
    # Assign the target objectives
    model_legacy.assign_objectives_target(objectivesTarget, objective_types=objectiveTypes)

    # Set which objectives are active, if specified
    if objectives_status is not None:
        model_legacy.app_load_objective_status(objectives_status)

    # Set objective uncertainties if provided
    if objectives_uncertainty_minus is not None and objectives_uncertainty_plus is not None:
        model_legacy.app_load_objective_uncertainty(
            dUncPlus=objectives_uncertainty_plus, 
            dUncMinus=objectives_uncertainty_minus
        )

    # Use the provided current values
    inputArr = currentInputVars
    outputArr = currentOutcomeVars

    # Check for NaN values
    if np.isnan(np.array(inputArr)).any() or np.isnan(np.array(outputArr)).any():
        print("ENCOUNTERED NaN INSIDE INVERSE LOOP!")
        return model, "NaN encountered in pyVSMEInverseSingleIter", None, None, None

    # Calculate error metrics
    l2norm = norm(np.array(outputArr) - np.array(objectivesTarget), ord=2)
    
    # Calculate active L1 norm (only for active objectives)
    if objectives_status is not None:
        l1norm = norm(
            np.array(outputArr) * np.array(objectives_status) - 
            np.array(objectivesTarget) * np.array(objectives_status), 
            ord=1
        )
    else:
        l1norm = norm(np.array(outputArr) - np.array(objectivesTarget), ord=1)

    # Load current variables and outcomes into the model
    model_legacy.load_variable_values(inputArr)
    model_legacy.load_outcome_values(outputArr)

    # Handle categorical variables if present
    if model.varTypes is not None and 4 in model.varTypes:
        model_legacy.app_load_category_index([''], [0])
            
    # Run optimization iteration
    model_legacy.run_vsme_app(iteration)

    # Get suggested variables for next iteration
    dVarjValuesNew = model_legacy.fetch_variables_for_next_iteration()

    return model, "Success.", dVarjValuesNew, l1norm, l2norm