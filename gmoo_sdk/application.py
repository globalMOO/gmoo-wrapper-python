# -*- coding: utf-8 -*-
"""
GMOO Application Operations Module

This module contains the ApplicationOperations class, which provides methods for
using trained GMOO models for inference and inverse optimization. This includes
loading models, setting objectives, and performing inverse design.

Classes:
    ApplicationOperations: Application mode functionality for the GMOO SDK

Authors: Matt Freeman, Jason Hopkins
Version: 2.0.0 (Refactored)
"""

import ctypes
import csv
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.linalg import norm

from gmoo_sdk.helpers import fortran_hollerith_string, CtypesHelper, validate_nan

# Configure logging
logger = logging.getLogger(__name__)


class ApplicationOperations:
    """
    Application mode functionality for the GMOO SDK.
    
    This class contains methods for using trained surrogate models,
    including loading models, performing inverse design, and handling
    bias and genetic algorithm components.
    """
    
    def __init__(self, api):
        """
        Initialize the ApplicationOperations object.
        
        Args:
            api: The parent GMOOAPI instance
        """
        self.api = api
        
    def unload_vsme(self) -> None:
        """
        Unload the VSME model from the DLL memory in application mode.
        """
        self.api.VSMEappUnloadVSME()
        
    def load_model(self, alternate_vsme_input_filename: Optional[str] = None, 
                  inspect_load: bool = False) -> None:
        """
        Load a VSME model from a file for application (inference) use.
        
        Args:
            alternate_vsme_input_filename: Optional alternative path to the .gmoo file.
            inspect_load: If True, prints debugging information about the loaded model.
        """
        if alternate_vsme_input_filename:
            vsme_input_filename = alternate_vsme_input_filename
        else:
            vsme_input_filename = f"{self.api.vsme_input_filename}.gmoo"
    
        self.api.c_path_filename = fortran_hollerith_string(vsme_input_filename, pad_len=256)
        
        # Load the model file
        self.api.VSMEappLoadVSME(self.api.c_path_filename)
                           
        # Load the VSME name (which isn't automatically loaded with the file)
        self.api.development.load_vsme_name()

        # Verify dimensions match the expected values
        nVars, nObjs, nBias, nGens, nCats, nLrns, n7, n8, n9, n10 = self.poke_dimensions10()

        # Check for inconsistencies
        if nVars.value != self.api.nVars.value:
            logger.warning(f"Implied nVars ({self.api.nVars.value}) is inconsistent with loaded nVars ({nVars.value}).")
        if nObjs.value != self.api.nObjs.value:
            logger.warning(f"Implied nObjs ({self.api.nObjs.value}) is inconsistent with loaded nObjs ({nObjs.value}).")
            
        # Print debugging information if requested
        if inspect_load:
            # Create a list of dimension names and values for display
            dim_info = list(zip(
                ["nVars", "nOuts", "nBias", "nGens", "nCats", "nLrns", "n7", "n8", "n9", "n10"],
                [a.value for a in [nVars, nObjs, nBias, nGens, nCats, nLrns, n7, n8, n9, n10]]
            ))
            logger.info(f"Model dimensions: {dim_info}")
            
            # Verify VSME name
            pyname = self.api.development.poke_vsme_name()
            logger.info(f"VSME name retrieved: {pyname}")

    def assign_objectives_target(self, objectives_target: List[float], 
                               objective_types: Optional[List[int]] = None) -> None:
        """
        Assign target objectives for the VSME model in application mode.
        
        Args:
            objectives_target: List of target objective values.
            objective_types: List of integer objective types. Available types:
                0: Exact match - the specified value must be achieved exactly.
                1: Percentage error - the value has % error allowed.
                2: Absolute error - the value has +/- error allowed.
                11: Less than - the objective must be less than the specified value.
                12: Less than or equal - the objective must be less than or equal to the specified value.
                13: Greater than - the objective must be greater than the specified value.
                14: Greater than or equal - the objective must be greater than or equal to the specified value.
                21: Minimize - the objective is to be minimized down to the specified value.
                22: Maximize - the objective is to be maximized up to the specified value.
        """
        # Set default objective types if not provided
        if objective_types is None:
            self.api.objective_types = CtypesHelper.create_int_array([0] * self.api.nObjs.value)
        else:
            self.api.objective_types = CtypesHelper.create_int_array(objective_types)

        # Convert target values to C-compatible format
        self.api.d_obj_values = CtypesHelper.create_double_array(objectives_target)
        
        # Load the objective values and types into the DLL
        self.api.VSMEappLoadObjectiveValues(
            ctypes.byref(self.api.nObjs), 
            ctypes.byref(self.api.objective_types), 
            ctypes.byref(self.api.d_obj_values)
        )

    def load_objective_status(self, objectives_status: List[int]) -> None:
        """
        Load the active/inactive status of the objectives into the DLL.
        
        Args:
            objectives_status: List of integer flags (1=active, 0=inactive) for each objective.
        """
        self.api.objStatus = CtypesHelper.create_int_array(objectives_status)
        self.api.VSMEappLoadObjectiveStatus(
            ctypes.byref(self.api.nObjs), 
            ctypes.byref(self.api.objStatus)
        )
    
    def poke_objective_status(self) -> List[int]:
        """
        Retrieve the active/inactive status of the objectives from the DLL.
        
        Returns:
            List[int]: Status flags for each objective (1=active, 0=inactive)
        """
        self.api.objStatus = (ctypes.c_int * self.api.nObjs.value)()
        self.api.VSMEappPokeObjectiveStatus(
            ctypes.byref(self.api.nObjs), 
            ctypes.byref(self.api.objStatus)
        )
                           
        return list(self.api.objStatus)

    def load_objective_uncertainty(self, dUncPlus: List[float], dUncMinus: List[float]) -> None:
        """
        Set the uncertainty associated with the objectives.
        
        Args:
            dUncPlus: Uncertainty on the positive side for each objective.
            dUncMinus: Uncertainty on the negative side for each objective.
                      For symmetric uncertainty, set equal to dUncPlus.
        """
        # Validate input lengths
        if len(dUncPlus) != self.api.nObjs.value or len(dUncMinus) != self.api.nObjs.value:
            raise ValueError("Length of dUncPlus and dUncMinus must match the number of objectives.")

        # Convert uncertainties to C-compatible arrays
        d_unc_plus = CtypesHelper.create_double_array(dUncPlus)
        d_unc_minus = CtypesHelper.create_double_array(dUncMinus)

        # Load the uncertainty values into the DLL
        self.api.VSMEappLoadObjectiveUncertainty(
            ctypes.byref(self.api.nObjs),
            ctypes.byref(d_unc_plus),
            ctypes.byref(d_unc_minus)
        )

    def initialize_variables(self, override_init: Union[bool, List[float]]) -> ctypes.Array:
        """
        Initialize variables for the inverse optimization process.
        
        Args:
            override_init: Specifies how to initialize the variables:
                         - If a list of float values, the variables will be initialized to these values.
                         - If False, the variables will be initialized randomly within the limits.
                         
        Returns:
            ctypes.Array: C-compatible array of initialized variable values.
        """
        # Case 1: Use provided initial values
        if not isinstance(override_init, bool):
            return CtypesHelper.create_double_array(override_init)
            
        # Case 2: Generate random values within limits
        elif override_init is False:
            random_values = np.random.uniform(
                np.array(self.api.dVarLimMin), 
                np.array(self.api.dVarLimMax)
            )
            return CtypesHelper.create_double_array(random_values)
            
        # Case 3: Invalid input
        else:
            raise ValueError("Invalid override_init. It should be a boolean value of False or a list of values.")

    def calculate_initial_solution(self, override_init: Union[bool, List[float]], 
                                  iBias: int = 0, iGen: int = 0) -> ctypes.Array:
        """
        Load an initial solution for the VSME model and return the values.
        
        Args:
            override_init: Specifies how to initialize the variables:
                         - If a list of float values, the variables will be initialized to these values.
                         - If False, the variables will be initialized randomly within the limits.
            iBias: The bias index to use (default 0, no bias).
            iGen: The genetic algorithm generation index (default 0).
            
        Returns:
            ctypes.Array: C-compatible array of the initial variable values.
        """
        self.api.iBias = ctypes.c_int(iBias)
        self.api.iGen = ctypes.c_int(iGen)
        
        # Initialize the variables
        self.api.d_varj_values = self.initialize_variables(override_init)
        
        # Load the initial values into the DLL
        self.api.VSMEappLoadVariableValues(
            ctypes.byref(self.api.nVars), 
            ctypes.byref(self.api.iBias), 
            ctypes.byref(self.api.iGen), 
            ctypes.byref(self.api.d_varj_values)
        )
                           
        return self.api.d_varj_values

    def load_variable_values(self, variable_arr: List[float], iBias: int = 0, iGen: int = 0) -> None:
        """
        Load specific variable values into the VSME model.
        
        Args:
            variable_arr: The array of variable values to load.
            iBias: The bias index (default 0, no bias).
            iGen: The genetic algorithm generation index (default 0).
        """
        self.api.iBias = ctypes.c_int(iBias)
        self.api.iGen = ctypes.c_int(iGen)
        
        # Convert input array to C-compatible format
        self.api.d_variable_values = CtypesHelper.create_double_array(variable_arr)
        
        # Load the values into the DLL
        self.api.VSMEappLoadVariableValues(
            ctypes.byref(self.api.nVars), 
            ctypes.byref(self.api.iBias), 
            ctypes.byref(self.api.iGen), 
            ctypes.byref(self.api.d_variable_values)
        )

    def load_variable_limit_min(self, variable_min_arr: List[float]) -> None:
        """
        Load new minimum variable limits into the VSME model during application.
        
        Args:
            variable_min_arr: List of new minimum values for each variable.
        """
        # Convert input array to C-compatible format
        self.api.d_variable_lim_min = CtypesHelper.create_double_array(variable_min_arr)
        
        # Load the new limits into the DLL
        self.api.VSMEappLoadVariableLimitMin(
            ctypes.byref(self.api.nVars), 
            ctypes.byref(self.api.d_variable_lim_min)
        )
        
    def load_variable_limit_max(self, variable_max_arr: List[float]) -> None:
        """
        Load new maximum variable limits into the VSME model during application.
        
        Args:
            variable_max_arr: List of new maximum values for each variable.
        """
        # Convert input array to C-compatible format
        self.api.d_variable_lim_max = CtypesHelper.create_double_array(variable_max_arr)
        
        # Load the new limits into the DLL
        self.api.VSMEappLoadVariableLimitMax(
            ctypes.byref(self.api.nVars), 
            ctypes.byref(self.api.d_variable_lim_max)
        )

    def load_category_index(self, var_names: List[str], selected_indices: List[int]) -> None:
        """
        Load category indices for categorical variables in application mode.
        
        Args:
            var_names: Names of the categorical variables (e.g., "VAR00003").
            selected_indices: 1-based indices of the selected category for each variable.
        """
        num_categorical_vars = len(var_names)
        if len(selected_indices) != num_categorical_vars:
            raise ValueError("Number of indices must match number of categorical variables")
            
        # Create the inputs for the DLL call
        nCats = ctypes.c_int(num_categorical_vars)
        
        # Create array for variable names (32 chars each)
        cCategory = (ctypes.c_char * (32 * num_categorical_vars))()
        
        # Fill the variable names array
        for i, name in enumerate(var_names):
            # Strip any existing padding to get base name
            base_name = name.strip()
            if len(base_name) > 31:
                raise ValueError(f"Base variable name '{base_name}' exceeds 31 characters")
                
            # Pad to exactly 32 characters
            name_bytes = base_name.encode('ascii').ljust(32, b'\0')
            for j, byte in enumerate(name_bytes):
                cCategory[i*32 + j] = byte
                
        # Create array for the selected indices
        iLabel = CtypesHelper.create_int_array(selected_indices)
        
        # Call the DLL function
        self.api.VSMEappLoadCategoryIndex(
            ctypes.byref(nCats),
            cCategory,
            ctypes.byref(iLabel)
        )

    def load_category_label(self, category_names: List[str], category_labels: List[str]) -> None:
        """
        Load category labels for categorical variables in application mode.
        
        Args:
            category_names: List of names for the categorical variables.
            category_labels: List of labels to select for each categorical variable.
        """
        if len(category_names) != len(category_labels):
            raise ValueError("Number of category names must match number of labels")
            
        nCats = ctypes.c_int(len(category_names))
        
        # Validate string lengths
        for name, label in zip(category_names, category_labels):
            if len(name) > 31:
                raise ValueError(f"Category name '{name}' exceeds 31 characters")
            if len(label) > 31:
                raise ValueError(f"Category label '{label}' exceeds 31 characters")
        
        # Create arrays for category names and labels
        cCategory = (ctypes.c_char * (32 * nCats.value))()
        cLabel = (ctypes.c_char * (32 * nCats.value))()
        
        # Fill the arrays with padded strings
        for i, (name, label) in enumerate(zip(category_names, category_labels)):
            # Convert and pad the name
            name_bytes = name.encode('ascii').ljust(32, b'\0')
            for j, byte in enumerate(name_bytes):
                cCategory[i*32 + j] = byte
                
            # Convert and pad the label
            label_bytes = label.encode('ascii').ljust(32, b'\0')
            for j, byte in enumerate(label_bytes):
                cLabel[i*32 + j] = byte
        
        # Call the DLL function
        self.api.VSMEappLoadCategoryLabel(
            ctypes.byref(nCats),
            cCategory,
            cLabel
        )

    def load_outcome_values(self, output_arr: List[float], iBias: int = 0, iGen: int = 0) -> None:
        """
        Load outcome values into the DLL for optimization.
        
        Args:
            output_arr: Array of outcome values to be loaded.
            iBias: The bias index (default 0, no bias).
            iGen: The genetic algorithm generation index (default 0).
        """
        self.api.iBias = ctypes.c_int(iBias)
        self.api.iGen = ctypes.c_int(iGen)
        
        # Convert the output array to C-compatible format
        self.api.d_out_values = CtypesHelper.create_double_array(output_arr)
        
        # Load the outcome values into the DLL
        self.api.VSMEappLoadOutcomeValues(
            ctypes.byref(self.api.nObjs), 
            ctypes.byref(self.api.iBias), 
            ctypes.byref(self.api.iGen), 
            ctypes.byref(self.api.d_out_values)
        )

    def run_vsme_app(self, iteration_number: int) -> None:
        """
        Run a single optimization iteration in application mode.
        
        Args:
            iteration_number: The current iteration count (used for tracking progress).
        """
        self.api.i_count = ctypes.c_int(iteration_number)
        self.api.VSMEapp(ctypes.byref(self.api.i_count))

    def fetch_variables_for_next_iteration(self, iBias: int = 0, iGen: int = 0) -> np.ndarray:
        """
        Retrieve the suggested variable values for the next optimization iteration.
        
        Args:
            iBias: The bias index (default 0, no bias).
            iGen: The genetic algorithm generation index (default 0).
            
        Returns:
            np.ndarray: Array of suggested variable values for the next iteration.
        """
        # Initialize a buffer to receive the variable values
        d_var_value_ll = (ctypes.c_double * self.api.nVars.value)(*np.zeros(self.api.nVars.value))
        
        self.api.iBias = ctypes.c_int(iBias)
        self.api.iGen = ctypes.c_int(iGen)
        
        # Retrieve the values from the DLL
        self.api.VSMEappPokeVariableValues(
            ctypes.byref(self.api.nVars), 
            ctypes.byref(self.api.iBias), 
            ctypes.byref(self.api.iGen), 
            ctypes.byref(d_var_value_ll)
        )
                           
        # Convert to numpy array for easier handling
        return np.array(d_var_value_ll)

    def init_variables(self, nPipes: int = 1) -> None:
        """
        Randomly initialize variables for the inverse application in the DLL.
        
        Args:
            nPipes: Number of simultaneuos parallel exectuting inverse 
                solutions (pipes). A value of 1 will retain normal single
                pass behavior.  Having more pipes is an excellent option 
                for models that take significant time to run but can be 
                run in parallel without much performance impact.
                This also provides a massive speedup for solutions that
                are highly sensitive to starting position.
        """
        self.api.nPipes = ctypes.c_int(nPipes)
        
        self.api.VSMEappInitVariables(ctypes.byref(self.api.nPipes))

    def initialize_genetic_algorithm(
        self, 
        iOption: int = -1,        # -1: No genetic algorithm
        rLog10Gen: int = 1,       # Log10 of number of genetic ensembles
        nParents: int = 0,        # Number of parents
        iError: int = 2           # Selection of Lp norm
    ) -> None:
        """
        Initialize the genetic algorithm parameters for optimization.
        
        Args:
            iOption: Genetic algorithm option:
                    -1: Disabled (no genetic algorithm)
                    Other values enable specific genetic algorithm modes.
            rLog10Gen: Log10 of the number of genetic ensembles.
                      For example, 1 means 10 ensembles, 2 means 100 ensembles.
            nParents: Number of parents to use in the genetic algorithm.
            iError: Error norm to use for fitness evaluation:
                   1: L1 norm (sum of absolute errors)
                   2: L2 norm (Euclidean distance)
        """
        self.api.iOption = ctypes.c_int(iOption)
        self.api.rLog10Gen = ctypes.c_int(rLog10Gen)
        self.api.nParents = ctypes.c_int(nParents)
        self.api.iError = ctypes.c_int(iError)
        
        self.api.VSMEGenInit(
            ctypes.byref(self.api.iOption),
            ctypes.byref(self.api.rLog10Gen),
            ctypes.byref(self.api.nParents),
            ctypes.byref(self.api.iError)
        )

    def initialize_bias(
        self, 
        iOption: int = 0,         # Bias algorithm option
        nBias: int = 150          # Number of bias ensembles
    ) -> None:
        """
        Initialize the bias parameters for optimization.
        
        Args:
            iOption: Bias algorithm option:
                    0: Start or restart the bias procedure from scratch
                    Other values enable specific bias algorithm modes.
            nBias: Number of bias ensembles to use.
        """
        self.api.iOption = ctypes.c_int(iOption)
        self.api.nBias = ctypes.c_int(nBias)
        
        self.api.VSMEBiasInit(
            ctypes.byref(self.api.iOption),
            ctypes.byref(self.api.nBias)
        )

    def poke_bias(
        self, 
        iBiasOption: int = 2,     # Option for bias value reporting
        nBias: int = 150          # Number of bias iterations
    ) -> np.ndarray:
        """
        Retrieve bias values from the DLL.
        
        Args:
            iBiasOption: Option to specify the type of bias values to retrieve:
                        2: Normalized bias values
                        Other values retrieve other forms of bias information.
            nBias: Number of bias iterations.
            
        Returns:
            np.ndarray: Array of bias values.
        """
        self.api.iBiasOption = ctypes.c_int(iBiasOption)
        self.api.nBias = ctypes.c_int(nBias)
        
        # Initialize buffer to receive bias values
        biasValues = (ctypes.c_double * self.api.nVars.value)(*np.zeros(self.api.nVars.value))
        
        # Retrieve bias values from the DLL
        self.api.VSMEBiasPokeBiasValues(
            ctypes.byref(self.api.nVars),
            ctypes.byref(self.api.iBiasOption),
            ctypes.byref(biasValues)
        )
                           
        return np.array(biasValues)

    def poke_dimensions10(self) -> Tuple[ctypes.c_int, ...]:
        """
        Retrieve various dimension values from the VSME model.
        
        Returns:
            Tuple of ctypes.c_int objects containing the following dimensions:
                nVars: Number of input variables
                nOuts: Number of output values/objectives
                nBias: Number of bias solutions
                nGens: Number of genetic ensembles
                nCats: Number of categorical variables
                nLrns: Number of learned cases
                n7-n10: Reserved for future use
        """
        # Initialize variables to receive the dimension values
        nVars = ctypes.c_int(0)
        nOuts = ctypes.c_int(0)
        nBias = ctypes.c_int(0)
        nGens = ctypes.c_int(0)
        nCats = ctypes.c_int(0)
        nLrns = ctypes.c_int(0)
        n7 = ctypes.c_int(0)
        n8 = ctypes.c_int(0)
        n9 = ctypes.c_int(0)
        n10 = ctypes.c_int(0)
        
        # Retrieve the dimension values from the DLL
        self.api.VSMEappPokeDimensions10(
            ctypes.byref(nVars),
            ctypes.byref(nOuts),
            ctypes.byref(nBias),
            ctypes.byref(nGens),
            ctypes.byref(nCats),
            ctypes.byref(nLrns),
            ctypes.byref(n7),
            ctypes.byref(n8),
            ctypes.byref(n9),
            ctypes.byref(n10)
        )
                           
        return nVars, nOuts, nBias, nGens, nCats, nLrns, n7, n8, n9, n10

    def poke_learned_case(self, nVars: int, nOuts: int, iCase: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the values of a specific learned case from the inverse process.
        
        Args:
            nVars: Number of variables in the model.
            nOuts: Number of outputs in the model.
            iCase: Index of the case to retrieve (1-based).

        Returns:
            Tuple containing:
                np.ndarray: Array of variable (input) values for the learned case.
                np.ndarray: Array of outcome (output) values for the learned case.
        """
        # Initialize buffers to receive the values
        dVarValue = (ctypes.c_double * nVars)()
        dOutValue = (ctypes.c_double * nOuts)()
        
        # Convert arguments to C-compatible types
        nVars_c = ctypes.c_int(nVars)
        nOuts_c = ctypes.c_int(nOuts)
        iCase_c = ctypes.c_int(iCase)
        iDummy = ctypes.c_int(1)  # Required by the DLL interface

        # Retrieve the learned case values from the DLL
        self.api.VSMEappPokeLearnedCase(
            ctypes.byref(nVars_c),
            ctypes.byref(nOuts_c),
            ctypes.byref(iDummy),
            ctypes.byref(iCase_c),
            ctypes.byref(dVarValue),
            ctypes.byref(dOutValue)
        )
                           
        return np.array(dVarValue), np.array(dOutValue)

    def load_learned_case(self, nVars: int, nOuts: int, dVarValues: List[float], 
                         dOutValues: List[float], iBias: int = 1) -> None:
        """
        Load the values of a previously learned case into the DLL.
        
        Args:
            nVars: Number of variables in the model.
            nOuts: Number of outputs in the model.
            dVarValues: Variable (input) values to load.
            dOutValues: Outcome (output) values to load.
            iBias: Bias solution identifier (default 1).
        """
        # Convert inputs to C-compatible types
        dVarValue = CtypesHelper.create_double_array(dVarValues)
        dOutValue = CtypesHelper.create_double_array(dOutValues)
        nVars_c = ctypes.c_int(nVars)
        nOuts_c = ctypes.c_int(nOuts)
        iDummy = ctypes.c_int(1)  # Required by the DLL interface

        # Load the learned case into the DLL
        self.api.VSMEappLoadLearnedCase(
            ctypes.byref(nVars_c),
            ctypes.byref(nOuts_c),
            ctypes.byref(iDummy),
            ctypes.byref(dVarValue),
            ctypes.byref(dOutValue)
        )

    def poke_outcome_dev_limit_min(self, nOuts: int) -> np.ndarray:
        """
        Retrieve the minimum values obtained for each outcome in the learning process.
        
        Args:
            nOuts: Number of output variables.
            
        Returns:
            np.ndarray: Array of minimum values for each outcome.
        """
        nOuts_c = ctypes.c_int(nOuts)
        OutLimMin = (ctypes.c_double * nOuts)()
        
        self.api.VSMEappPokeOutcomeDevLimitMin(
            ctypes.byref(nOuts_c),
            ctypes.byref(OutLimMin)
        )
                           
        return np.array(OutLimMin)

    def poke_outcome_dev_limit_max(self, nOuts: int) -> np.ndarray:
        """
        Retrieve the maximum values obtained for each outcome in the learning process.
        
        Args:
            nOuts: Number of output variables.
            
        Returns:
            np.ndarray: Array of maximum values for each outcome.
        """
        nOuts_c = ctypes.c_int(nOuts)
        OutLimMax = (ctypes.c_double * nOuts)()
        
        self.api.VSMEappPokeOutcomeDevLimitMax(
            ctypes.byref(nOuts_c),
            ctypes.byref(OutLimMax)
        )
                           
        return np.array(OutLimMax)
        
    def poke_variable_dev_limit_min(self) -> List[float]:
        """
        Retrieve the current minimum limits for development cases from the DLL.
        
        Returns:
            List[float]: Minimum limit values for each variable.
        """
        nVars = self.api.nVars
        dVarLimMin = (ctypes.c_double * self.api.nVars.value)()
        
        self.api.VSMEappPokeVariableDevLimitMin(
            ctypes.byref(nVars),
            ctypes.byref(dVarLimMin)
        )
                           
        return [v for v in dVarLimMin]

    def poke_variable_dev_limit_max(self) -> List[float]:
        """
        Retrieve the current maximum limits for development cases from the DLL.
        
        Returns:
            List[float]: Maximum limit values for each variable.
        """
        nVars = self.api.nVars
        dVarLimMax = (ctypes.c_double * self.api.nVars.value)()
        
        self.api.VSMEappPokeVariableDevLimitMax(
            ctypes.byref(nVars),
            ctypes.byref(dVarLimMax)
        )
                           
        return [v for v in dVarLimMax]
        
    # High-level convenience methods
    
    def perform_inverse_iteration(self, target_outputs: List[float],
                                 current_inputs: List[float],
                                 current_outputs: List[float],
                                 objective_types: Optional[List[int]] = None,
                                 objective_status: Optional[List[int]] = None,
                                 objective_uncertainty_minus: Optional[List[float]] = None,
                                 objective_uncertainty_plus: Optional[List[float]] = None) -> Tuple[np.ndarray, float, float]:
        """
        Perform a single iteration of inverse optimization.
        
        This method uses the current input/output values to generate a new suggestion
        for the next iteration.
        
        Args:
            target_outputs: Target output values to match
            current_inputs: Current input values
            current_outputs: Current output values
            objective_types: Types of objectives (0=exact, 1=percentage, etc.)
            objective_status: Active/inactive flags for each objective
            objective_uncertainty_minus: Lower uncertainty bounds
            objective_uncertainty_plus: Upper uncertainty bounds
            
        Returns:
            Tuple containing:
                np.ndarray: Suggested input values for the next iteration
                float: L1 norm (sum of absolute errors) measure of error
                float: L2 norm (Euclidean distance) measure of error
        """
        # Set objective targets and types
        self.assign_objectives_target(target_outputs, objective_types)
        
        # Set which objectives are active, if specified
        if objective_status is not None:
            self.load_objective_status(objective_status)

        # Set objective uncertainties if provided
        if objective_uncertainty_minus is not None and objective_uncertainty_plus is not None:
            self.load_objective_uncertainty(
                dUncPlus=objective_uncertainty_plus, 
                dUncMinus=objective_uncertainty_minus
            )

        # Calculate error metrics
        l2norm = norm(np.array(current_outputs) - np.array(target_outputs), ord=2)
        
        # Calculate L1 norm (only for active objectives if specified)
        if objective_status is not None:
            l1norm = norm(
                np.array(current_outputs) * np.array(objective_status) - 
                np.array(target_outputs) * np.array(objective_status), 
                ord=1
            )
        else:
            l1norm = norm(np.array(current_outputs) - np.array(target_outputs), ord=1)

        # Load current variables and outcomes into the DLL
        self.load_variable_values(current_inputs)
        self.load_outcome_values(current_outputs)

        # Run optimization iteration
        self.run_vsme_app(1)  # We're just running a single iteration

        # Get suggested variables for next iteration
        next_inputs = self.fetch_variables_for_next_iteration()

        return next_inputs, l1norm, l2norm
        
    def perform_min_error_search(self, target_result: np.ndarray, 
                               total_VSME_model_evaluations: int,
                               min_err_cases: int = 10) -> Tuple[float, np.ndarray, int]:
        """
        Search random cases for the one with minimum error compared to a target.
        
        Args:
            target_result: The target result to match.
            total_VSME_model_evaluations: Count of model evaluations so far.
            min_err_cases: Number of random cases to evaluate (default 10).
            
        Returns:
            Tuple containing:
                float: The best (lowest) L1 norm value found.
                np.ndarray: Variable values corresponding to the minimum error.
                int: Updated total count of model evaluations.
        """
        best_l1_value = float('inf')
        min_err_case_array = None
        
        # Generate and evaluate random cases
        for _ in range(min_err_cases):
            # Generate random values within the allowed ranges
            exploration_case = np.random.uniform(
                np.array(self.api.dVarLimMin), 
                np.array(self.api.dVarLimMax)
            )
            
            # Evaluate the model
            exploration_result = self.api.modelFunction(exploration_case)
            
            # Calculate the L1 norm (sum of absolute errors)
            l1norm = norm(exploration_result - target_result, ord=1)
            
            # Update best result if this is better
            if l1norm < best_l1_value:
                best_l1_value = l1norm
                min_err_case_array = exploration_case
                
            total_VSME_model_evaluations += 1
        
        return best_l1_value, min_err_case_array, total_VSME_model_evaluations
        
    def generate_random_exploration_case(self, logspace: List[bool]) -> np.ndarray:
        """
        Generate a random case for exploration, respecting variable limits and distribution.
        
        Args:
            logspace: Flags indicating whether each variable should be sampled logarithmically.
                     True for logarithmic sampling, False for linear sampling.
                     
        Returns:
            np.ndarray: Randomly generated variable values.
        """
        exploration_case = []
        
        # Generate a value for each variable
        for is_log, min_val, max_val in zip(logspace, self.api.dVarLimMin, self.api.dVarLimMax):
            if is_log:
                # For logarithmic variables, sample in log space and convert back
                min_val, max_val = np.log10(min_val), np.log10(max_val)
                sample = np.random.uniform(min_val, max_val)
                exploration_case.append(10 ** sample)
            else:
                # For linear variables, sample directly
                exploration_case.append(np.random.uniform(min_val, max_val))
                
        return np.array(exploration_case)
        
    def rescope_search_space(self, all_inputs: List[List[float]], 
                            all_outcomes: List[List[float]],
                            target_outcomes: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adaptively rescope the search space based on optimization results.
        
        This method analyzes previous results to identify a more promising region
        of the input space to focus subsequent optimizations on.
        
        Args:
            all_inputs: Previous input variable arrays
            all_outcomes: Previous outcome arrays
            target_outcomes: Target outcome values
            
        Returns:
            Tuple containing:
                np.ndarray: New minimum variable values for the rescoped search space
                np.ndarray: New maximum variable values for the rescoped search space
        """
        # Use the last few results to generate suggestions for new input variables
        inputs_suggested = []
        for i, o in zip(all_inputs[-10:], all_outcomes[-10:]):
            next_inputs, _, _ = self.perform_inverse_iteration(
                target_outputs=target_outcomes,
                current_inputs=i,
                current_outputs=o
            )
            inputs_suggested.append(next_inputs)
        
        # Stack the arrays to find the min and max along each dimension
        stacked_arrays = np.stack(inputs_suggested)

        # Find the min and max along the first axis (across arrays)
        min_values = np.amin(stacked_arrays, axis=0)
        max_values = np.amax(stacked_arrays, axis=0)

        return min_values, max_values