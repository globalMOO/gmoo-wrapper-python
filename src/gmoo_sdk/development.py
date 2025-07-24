# -*- coding: utf-8 -*-
"""
GMOO Development Operations Module

This module contains the DevelopmentOperations class, which provides methods for
creating and training inverse models using the GMOO DLL. This includes methods
for designing development cases, managing training data, and developing the VSME model.

Classes:
    DevelopmentOperations: Development mode functionality for the GMOO SDK

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

from .helpers import fortran_hollerith_string, CtypesHelper, validate_nan

# Configure logging
logger = logging.getLogger(__name__)


class DevelopmentOperations:
    """
    Development mode functionality for the GMOO SDK.
    
    This class contains methods for training inverse models, including
    case generation, training, and model development.
    """
    
    def __init__(self, api):
        """
        Initialize the DevelopmentOperations object.
        
        Args:
            api: The parent GMOOAPI instance
        """
        self.api = api
        
    def load_vsme_name(self, compatibility_mode: bool = False) -> None:
        """
        Load a VSME model by its filename into the DLL.

        Args:
            compatibility_mode: If True, uses compatibility mode for string handling.
        """
        self.api.cVSMEname = fortran_hollerith_string(
            self.api.vsme_input_filename.split("\\")[-1], 
            pad_len=32
        )
        
        if compatibility_mode:
            iVSME = ctypes.c_int(0)
            from .helpers import c_string_compatibility
            self.api.cVSMEname = c_string_compatibility(self.api.vsme_input_filename)
            self.api.vsme_windll.VSMEdevLoadVSMEName(
                ctypes.byref(iVSME), 
                ctypes.byref(self.api.cVSMEname)
            )
        else:
            self.api.VSMEdevLoadVSMEName(
                ctypes.byref(self.api.iVSME), 
                ctypes.byref(self.api.cVSMEname)
            )

    def poke_vsme_name(self, compatibility_mode: bool = False) -> str:
        """
        Retrieve the loaded VSME filename from the DLL.

        Args:
            compatibility_mode: If True, uses compatibility mode for string handling.

        Returns:
            str: The VSME filename stored in the DLL.
        """
        self.api.cVSMEname = fortran_hollerith_string(self.api.vsme_input_filename, pad_len=32)
        
        if compatibility_mode:
            from .helpers import c_string_compatibility
            self.api.cVSMEname = c_string_compatibility(self.api.vsme_input_filename)
            self.api.vsme_windll.VSMEdevPokeVSMEName(ctypes.byref(self.api.cVSMEname))
            # For c_wchar_p, just get the value directly
            pystring_value = self.api.cVSMEname.value.rstrip()
        else:
            self.api.VSMEdevPokeVSMEName(ctypes.byref(self.api.cVSMEname))
            # Convert C string back to Python string, removing padding
            pystring_value = ''.join(byte.decode('ascii') for byte in self.api.cVSMEname if byte != b' ')
        
        return pystring_value

    def init_backup_file(self, override_name: Optional[str] = None) -> None:
        """
        Initialize the VSME backup project file for incremental completion.
        
        Args:
            override_name: If provided, uses this name for the backup file instead of default.
        """
        import os
        if override_name:
            # For override, use it as full path or join with save_file_dir
            if os.path.isabs(override_name):
                pyPath = f"{override_name}.VPRJ"
            else:
                pyPath = os.path.join(self.api.save_file_dir, f"{override_name}.VPRJ")
        else:
            # Use the full vsme_input_filename which already includes the directory
            pyPath = f"{self.api.vsme_input_filename}.VPRJ"
        
        # Use just the filename without path, like the .gmoo export
        base_name = os.path.basename(pyPath)
        logger.info(f"Initializing backup file: {base_name} (in working directory)")
        
        self.api.cVSMEname = fortran_hollerith_string(base_name, pad_len=256)
        
        self.api.VSMEdevInitBackupFile(ctypes.byref(self.api.cVSMEname))
        
        # Check if file was created in current directory
        if os.path.exists(base_name):
            logger.info(f"Backup file created successfully: {base_name}")
        else:
            logger.warning(f"Backup file NOT created: {base_name}")
            # Check if it was created with a different name
            vprj_files = [f for f in os.listdir(".") if f.endswith('.VPRJ')]
            if vprj_files:
                logger.info(f"Found VPRJ files: {vprj_files}")
            else:
                logger.warning("No VPRJ files found in current directory")

    def read_backup_file(self, override_name: Optional[str] = None) -> None:
        """
        Read the VSME backup file to begin or continue incremental completion.
        
        Args:
            override_name: If provided, uses this name for the backup file instead of default.
        """
        import os
        if override_name:
            # For override, use it as full path or join with save_file_dir
            if os.path.isabs(override_name):
                path = f"{override_name}.VPRJ"
            else:
                path = os.path.join(self.api.save_file_dir, f"{override_name}.VPRJ")
        else:
            # Use the full vsme_input_filename which already includes the directory
            path = f"{self.api.vsme_input_filename}.VPRJ"
        
        # Use just the filename without path, like the .gmoo export
        base_name = os.path.basename(path)
        logger.info(f"Reading backup file: {base_name} (from working directory)")
        
        # Check if file exists in current directory
        if not os.path.exists(base_name):
            logger.error(f"Backup file does not exist in current directory: {base_name}")
            # List VPRJ files in current directory
            vprj_files = [f for f in os.listdir(".") if f.endswith('.VPRJ')]
            logger.info(f"VPRJ files in current directory: {vprj_files}")
            
        self.api.cVSMEname = fortran_hollerith_string(base_name, pad_len=256)
        
        self.api.VSMEdevReadBackupFile(ctypes.byref(self.api.cVSMEname))
                            
    def load_parameters(self, e1: int = -1, e2: int = -1, e3: int = -1, 
                        e4: int = -1, e5: int = -1, e6: int = -1, r: float = 0.4) -> None:
        """
        Load custom parameters for the VSME development process.
        
        Args:
            e1: Integer parameter 1, default -1 (use DLL defaults)
            e2: Integer parameter 2, default -1 (use DLL defaults)
            e3: Integer parameter 3, default -1 (use DLL defaults)
            e4: Integer parameter 4, default -1 (use DLL defaults)
            e5: Integer parameter 5, default -1 (use DLL defaults)
            e6: Integer parameter 6, default -1 (use DLL defaults)
            r: Floating-point parameter, default 0.4
        """
        self.api.nItems = ctypes.c_int(6)
        self.api.iParameter = CtypesHelper.create_int_array([e1, e2, e3, e4, e5, e6])
        self.api.rParameter = ctypes.c_double(r)
        
        self.api.VSMEdevLoadParameters(
            ctypes.byref(self.api.nItems),
            ctypes.byref(self.api.iParameter),
            ctypes.byref(self.api.rParameter)
        )

    def initialize_variables(self) -> None:
        """Initialize the development case variables in the DLL."""
        self.api.VSMEdevInitVariables(ctypes.byref(self.api.nVars))

    def load_variable_limits(self) -> None:
        """
        Load the variable ranges into the DLL and validate that they were loaded correctly.
        """
        self.api.VSMEdevLoadVariableLimits(
            ctypes.byref(self.api.nVars), 
            ctypes.byref(self.api.dVarLimMin), 
            ctypes.byref(self.api.dVarLimMax)
        )
        
        # Verify that limits were loaded correctly
        self.check_loaded_bounds()

    def load_variable_types(self) -> None:
        """
        Load the variable type information into the DLL.
        
        This method passes information about:
        1. Variable types (float, int, logical, categorical)
        2. Variable distribution (logarithmic or linear)
        """
        # Load variable types (float=1, int=2, logical=3, categorical=4)
        self.api.iVarType = CtypesHelper.create_int_array(self.api.var_types)
        
        # Load logarithmic distribution flags (0=linear, 1=logarithmic)
        self.api.lVarLog = CtypesHelper.create_int_array([log.value for log in self.api.logspace])
            
        self.api.VSMEdevLoadVariableTypes(
            ctypes.byref(self.api.nVars), 
            ctypes.byref(self.api.iVarType), 
            ctypes.byref(self.api.lVarLog)
        )

    def check_loaded_bounds(self) -> None:
        """
        Verify that variable bounds were correctly loaded into the DLL.
        
        This method compares the values loaded into the DLL with the values
        that were intended to be loaded, checking for potential discrepancies.
        """
        # Prepare containers for the DLL to fill
        nVars = self.api.nVars
        size = nVars.value * 32
        cVarNames = (ctypes.c_char * size)()
        dVarLimMin = (ctypes.c_double * self.api.nVars.value)()
        dVarLimMax = (ctypes.c_double * self.api.nVars.value)()
        iVarType = (ctypes.c_int * self.api.nVars.value)()
        lVarLog = (ctypes.c_int * self.api.nVars.value)()

        # Retrieve current variable information from the DLL
        self.api.vsme_windll.VSMEdevPokeVariables(
            ctypes.byref(nVars),
            ctypes.cast(cVarNames, ctypes.POINTER(ctypes.c_char*size)),
            ctypes.byref(dVarLimMin), 
            ctypes.byref(dVarLimMax), 
            ctypes.byref(iVarType), 
            ctypes.byref(lVarLog),
            ctypes.byref(ctypes.c_int())  # Status param
        )

        # Check for discrepancies
        if nVars.value != self.api.nVars.value:
            logger.error(f"Loaded nVars ({nVars.value}) is inconsistent with intended nVars ({self.api.nVars.value})")
       
        for i, val in enumerate(dVarLimMin):
            if self.api.dVarLimMin[i] != val:
                logger.error(f"Loaded dVarLimMin[{i}] ({val}) is inconsistent with intended dVarLimMin[{i}] ({self.api.dVarLimMin[i]})")
        
        for i, val in enumerate(dVarLimMax):
            if self.api.dVarLimMax[i] != val:
                logger.error(f"Loaded dVarLimMax[{i}] ({val}) is inconsistent with intended dVarLimMax[{i}] ({self.api.dVarLimMax[i]})")
        
        for i, val in enumerate(iVarType):
            if hasattr(self.api, 'iVarType') and val != self.api.iVarType[i]:
                logger.error(f"Loaded iVarType[{i}] ({val}) is inconsistent with intended iVarType[{i}] ({self.api.iVarType[i]})")
                
        for i, val in enumerate(lVarLog):
            if hasattr(self.api, 'lVarLog') and val != self.api.lVarLog[i]:
                logger.error(f"Loaded lVarLog[{i}] ({val}) is inconsistent with intended lVarLog[{i}] ({self.api.lVarLog[i]})")

    def design_agents(self) -> None:
        """
        Design the optimization agents during the development process.
        
        This method initializes the internal optimization agents that will be used
        to build the inverse model for the problem space.
        """
        self.api.VSMEdevDesignAgents()

    def design_cases(self) -> None:
        """
        Design the exploration cases for the development process.
        
        This method generates the set of input variable combinations that will be used 
        to explore the problem space and build the inverse model.
        """
        self.api.VSMEdevDesignCases(ctypes.byref(self.api.cRootName))

    def get_case_count(self) -> int:
        """
        Retrieve the number of development cases that were generated.
        
        Returns:
            int: The number of development cases
        """
        self.api.VSMEdevPokeCaseCount(ctypes.byref(self.api.devCaseCount))
        return self.api.devCaseCount.value

    def initialize_outcomes(self) -> None:
        """
        Initialize the outcome structures in the development process.
        
        This method sets up the data structures in the DLL that will hold the 
        outcome values corresponding to each development case.
        """
        self.api.VSMEdevInitOutcomes(ctypes.byref(self.api.nObjs))
                           
    def poke_case_variables(self, i_case: int) -> np.ndarray:
        """
        Retrieve the input variable values for a specific development case.
        
        Args:
            i_case: The 1-based index of the development case to retrieve.
            
        Returns:
            np.ndarray: Array of variable values for the specified case
        """
        self.api.iCase = ctypes.c_int(i_case)
        # Initialize an empty array to receive the variable values
        self.api.caseVariables = (ctypes.c_double * self.api.nVars.value)()
        
        self.api.VSMEdevPokeCaseVariables(
            ctypes.byref(self.api.iCase), 
            ctypes.byref(self.api.caseVariables)
        )
        
        return np.array(self.api.caseVariables)

    def load_case_results(self, i_case: int, dOutValues: Optional[List[float]]) -> None:
        """
        Load the evaluation results for a specific development case.
        
        Args:
            i_case: The 1-based index of the development case.
            dOutValues: The outcome values to load. If None, the case is skipped.
        """
        if dOutValues is None:
            logger.info(f"Not loading case {i_case} (outcome values are None)")
            return  # Skip this case
        
        self.api.iCase = ctypes.c_int(i_case)
        self.api.caseOutcomes = CtypesHelper.create_double_array(dOutValues)
        
        self.api.VSMEdevLoadCaseResults(
            ctypes.byref(self.api.iCase), 
            ctypes.byref(self.api.caseOutcomes)
        )
        
    def export_case_csv(self, csv_filename: Optional[str] = None) -> str:
        """
        Export the variables for all development cases to a CSV file.
        
        Args:
            csv_filename: Optional custom filename for the export CSV.
                         If None, uses <vsme_input_filename>_DEVVARS.csv.
                         
        Returns:
            str: Path to the created CSV file
        """
        if csv_filename is None:
            csv_filename = f"{self.api.vsme_input_filename}_DEVVARS.csv"
            
        self.api.c_path_csv_filename = fortran_hollerith_string(csv_filename, pad_len=256)
        
        self.api.VSMEdevExportCasesCSV(ctypes.byref(self.api.c_path_csv_filename))
                           
        # Create a .done file to signal completion
        done_filename = f"{self.api.vsme_input_filename}_DEVVARS.done"
        with open(done_filename, "w") as fh:
            pass  # Empty file, just marking completion
            
        return csv_filename

    def export_vsme(self, vsme_output_filename: Optional[str] = None) -> str:
        """
        Export the VSME model in the form of a .gmoo file.
        
        Args:
            vsme_output_filename: Optional custom filename for the .gmoo file.
                                 If None, uses <vsme_input_filename>.gmoo.
                                 
        Returns:
            str: Path to the created .gmoo file
        """

        # Create path with the .gmoo file extension
        gmoo_path = self.api.vsme_input_filename+".gmoo"

        # Create the directory if it doesn't exist
        output_dir = os.path.dirname(gmoo_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        if vsme_output_filename is None:
            # Use simple relative path without extension, let the DLL handle it
            base_dir = os.path.dirname(gmoo_path) or "."
            base_name = os.path.basename(gmoo_path)
            simple_path = base_name  # Simple filename without path or extension
            self.api.c_path_filename = fortran_hollerith_string(simple_path, pad_len=256)
        else:
            # For custom paths, still keep it simple
            self.api.c_path_filename = fortran_hollerith_string(vsme_output_filename, pad_len=256)
        
        # Export without adding extension
        self.api.VSMEdevExportVSME(self.api.c_path_filename)
        
        # Return the expected path with extension for reference
        return gmoo_path

    def develop_vsme(self) -> None:
        """
        Develop the VSME inverse model.
        
        This method builds the inverse model based on the loaded development cases.
        It should be called after all development cases have been evaluated and 
        their results loaded.
        """
        self.api.VSMEdevDevelopVSME()
        
        # Wait for development to complete if using async mode
        while True:
            status = self.check_development_status()
            if status:
                break
            logger.info("VSME development in progress...")
            time.sleep(1.0)
            
    def delete_vsme_file(self, vsme_output_filename: Optional[str] = None) -> None:
        """
        Delete a VSME (.gmoo) file.
        
        Args:
            vsme_output_filename: Path to the VSME file to delete.
                                 If None, uses the default name.
        """
        if vsme_output_filename is not None:
            try:
                os.remove(vsme_output_filename)
                logger.info(f"Deleted file: {vsme_output_filename}")
            except FileNotFoundError:
                logger.warning(f"Could not delete {vsme_output_filename} - file not found")
            except PermissionError:
                logger.warning(f"Could not delete {vsme_output_filename} - permission denied")

    def unload_vsme(self) -> None:
        """
        Unload the VSME model from the DLL memory in development mode.
        """
        self.api.VSMEdevUnloadVSME()
        
    def load_category_labels(self) -> None:
        """
        Define labels (categories) for all categorical variables during development.
        """
        if not self.api.categories_list:
            return
            
        for var_id, category_labels in enumerate(self.api.categories_list, start=1):
            # Skip non-categorical variables (empty category_labels)
            if not category_labels:
                continue
                
            # Prepare parameters for DLL call
            iVar = ctypes.c_int(var_id)
            nLabels = ctypes.c_int(len(category_labels))
            
            # Validate categorical variable configuration
            var_max_limit = int(self.api.aVarLimMax[var_id - 1])
            if nLabels.value != var_max_limit:
                raise ValueError(
                    f"Number of labels ({nLabels.value}) must match the Variable Maximum Limit "
                    f"({var_max_limit}) for variable {var_id}"
                )
            
            # Create a 1D array of characters to hold all labels
            # Each label gets 32 characters of space
            cCatLabels = (ctypes.c_char * (32 * nLabels.value))()
            
            # Fill the array with the category labels, properly encoded and padded
            for i, label in enumerate(category_labels):
                # Limit label length to 31 characters (leaving room for null terminator)
                label = label[:31]
                
                # Convert the label to bytes and pad with nulls
                label_bytes = label.encode('ascii').ljust(32, b'\0')
                
                # Copy the bytes into the array at the correct position
                for j, byte in enumerate(label_bytes):
                    cCatLabels[i*32 + j] = byte
            
            # Load the category labels into the DLL
            self.api.VSMEdevLoadCategoryLabels(
                ctypes.byref(iVar),
                ctypes.byref(nLabels),
                cCatLabels
            )
            

            
    def poke_nonlinearity(self) -> float:
        """
        Retrieve the internal measure of nonlinearity from the VSME model.
        
        Returns:
            float: A measure of the nonlinearity of the predictive model:
                - Negative value: VSME has not been developed
                - 0.0: Model is linear
                - 0.1 to 10.0: Increasing severity of nonlinearity
        """
        dVar = ctypes.c_double(0.0)
        
        self.api.VSMEdevPokeNonlinearity(ctypes.byref(dVar))
                           
        return dVar.value
            
    def load_user_cases(self, cases_var_values: List[List[float]], 
                        cases_out_values: List[List[float]]) -> None:
        """
        Load multiple user-defined cases with their variable and outcome values.
        
        Args:
            cases_var_values: Variable values for all cases - shape (n_cases, n_vars)
            cases_out_values: Outcome values for all cases - shape (n_cases, n_outs)
        """
        n_cases = len(cases_var_values)
                                         
        if n_cases != len(cases_out_values):
            raise ValueError("Number of cases must match between variables and outcomes")
            
        n_vars = len(cases_var_values[0])
        n_outs = len(cases_out_values[0])
            
        # Flatten the arrays for Fortran
        flat_var_values = [val for case in cases_var_values for val in case]
        flat_out_values = [val for case in cases_out_values for val in case]

        # Convert to C-compatible types
        nCases = ctypes.c_int(n_cases)
        nVars = ctypes.c_int(n_vars)
        nOuts = ctypes.c_int(n_outs)
        dVarValues = (ctypes.c_double * (self.api.nVars.value * n_cases))(*flat_var_values)
        dOutValues = (ctypes.c_double * (self.api.nObjs.value * n_cases))(*flat_out_values)
        
        # Load the user cases into the DLL
        self.api.VSMEdevLoadUserCases(
            ctypes.byref(nVars),
            ctypes.byref(nOuts),
            ctypes.byref(nCases),
            ctypes.byref(dVarValues),
            ctypes.byref(dOutValues)
        )

    def load_user_case(self, var_values: List[float], out_values: List[float]) -> None:
        """
        Load a single user-defined case with its variable and outcome values.
        
        Args:
            var_values: Variable values for the case
            out_values: Outcome values for the case
        """
        n_vars = len(var_values)
        n_outs = len(out_values)

        logger.info(f"Loading user case with variables: {var_values}")
        logger.info(f"and outcomes: {out_values}")

        # Validate dimensions
        if n_vars != self.api.nVars.value:
            raise ValueError(f"Expected {self.api.nVars.value} variables, got {n_vars}")
        if n_outs != self.api.nObjs.value:
            raise ValueError(f"Expected {self.api.nObjs.value} outcomes, got {n_outs}")

        # Convert to C-compatible types
        nVars = ctypes.c_int(n_vars)
        nOuts = ctypes.c_int(n_outs)
        dVarValues = CtypesHelper.create_double_array(var_values)
        dOutValues = CtypesHelper.create_double_array(out_values)

        # Load the user case into the DLL
        self.api.VSMEdevLoadUserCase(
            ctypes.byref(nVars),
            ctypes.byref(nOuts),
            ctypes.byref(dVarValues),
            ctypes.byref(dOutValues)
        )
        
    def read_outcomes_csv(self, runs_type: str = "DEV") -> List[List[float]]:
        """
        Read outcomes from a CSV file after it has been generated by an external process.
        
        Args:
            runs_type: Type of run, used in the filename (default "DEV" for development).
                     Other common values are "DEVOUT" or "INV" for inverse.
                     
        Returns:
            List[List[float]]: List of outcome arrays, one for each case.
        """
        # Construct the filename based on the run type
        outcomes_file_name = f"{self.api.vsme_input_filename}_{runs_type}OUT"
        
        # Wait for the .done file to appear
        while not os.path.exists(f"{outcomes_file_name}.done"):
            time.sleep(1)  # Pause for 1 second to avoid CPU overuse
            
        # Read the CSV file
        with open(f"{outcomes_file_name}.csv", 'r') as file:
            reader = csv.reader(file)
            # Skip metadata and headers
            next(reader)
            next(reader)

            # Extract data, converting strings to floats
            data = [list(map(float, row[1:])) for row in reader]
            
        return data
        
    def check_development_status(self) -> bool:
        """
        Check if VSME development is complete.
        
        Returns:
            bool: True if development is complete, False if still in progress
        """
        return self.api.vsme_dev_main(6)
        
    def check_export_status(self) -> bool:
        """
        Check if VSME file export is complete.
        
        Returns:
            bool: True if export is complete, False if still in progress
        """
        return self.api.vsme_dev_main(7)
        
    # High-level convenience methods
    
    def setup_and_design_cases(self, params: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        """
        Set up the development environment and design cases in one step.
        
        This is a convenience method that combines several common development steps.
        
        Args:
            params: Optional dictionary of custom parameters for VSME development.
            
        Returns:
            List[np.ndarray]: List of input arrays to be evaluated
        """
        # Load VSME name
        self.load_vsme_name()
        
        # Load custom parameters if provided
        if params is not None:
            self.load_parameters(**params)

        # Initialize Variables
        self.initialize_variables()
        
        # Load Variable Data Types
        self.load_variable_types()

        # Load Variable Limits
        self.load_variable_limits()
        
        # Load Categorical Labels if needed
        if self.api.categories_list is not None:
            self.load_category_labels()

        # Design Agents
        self.design_agents()

        # Design Cases
        self.design_cases()

        # Get Case Count
        case_count = self.get_case_count()
        
        # Initialize the backup file
        self.init_backup_file()

        # Extract the case variables
        input_vectors = []
        for kk in range(1, case_count + 1):
            input_vector = self.poke_case_variables(kk)
            input_vectors.append(input_vector)

        return input_vectors
        
    def load_results_and_develop(self, outcome_vectors: List[Optional[np.ndarray]],
                                extra_inputs: Optional[List[List[float]]] = None,
                                extra_outputs: Optional[List[List[float]]] = None) -> str:
        """
        Load results and develop the VSME model in one step.
        
        This is a convenience method that combines result loading and model development.
        
        Args:
            outcome_vectors: List of output arrays from evaluating the development cases.
                           Can contain None entries for cases that failed.
            extra_inputs: Optional additional input cases to include in training.
            extra_outputs: Optional additional output values to include in training.
            
        Returns:
            str: Path to the created .gmoo file
        """
        # Restore created experimental design from backup file for consistency
        self.read_backup_file()

        # Initialize outcomes in memory
        self.initialize_outcomes()
                
        # Load each case result
        for kk in range(1, len(outcome_vectors) + 1):
            self.load_case_results(kk, outcome_vectors[kk-1])
            
        # Add extra user-provided cases if available
        if extra_inputs is not None and extra_outputs is not None:
            self.load_user_cases(extra_inputs, extra_outputs)

        # Develop the VSME model
        self.develop_vsme()

        # Export VSME to file
        output_file = self.export_vsme()
        
        # Wait for export to complete
        while not self.check_export_status():
            logger.info("VSME file export in progress...")
            time.sleep(1.0)
            
        return output_file