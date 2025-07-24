# -*- coding: utf-8 -*-
"""
GMOO DLL Interface Module

This module provides a wrapper API for the VSME_DLL engine, facilitating interaction between
Python applications and the GMOO (Global Multi-Objective Optimization) compiled library.
The wrapper manages file I/O, memory allocation, and data type conversions required for
communication with the underlying optimization engine.

Classes:
    GMOOAPI: Main interface class for interacting with the GMOO optimization library

Authors: Matt Freeman, Jason Hopkins
Version: 1.0.0
"""

import ctypes
import time
import numpy as np
import logging
import os
import platform
import struct
from typing import List, Dict, Tuple, Union, Optional, Callable, Any

from .helpers import fortran_hollerith_string
from .helpers import CtypesHelper
from .application import ApplicationOperations
from .development import DevelopmentOperations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GMOOException(Exception):
    """Custom exception for GMOO-specific errors."""
    pass

class DLLFunction:
    """Descriptor for DLL function calls with error handling."""
    
    def __init__(self, func_name, return_status=True):
        self.func_name = func_name
        self.return_status = return_status
        
    def __get__(self, instance, owner):
        """Get method that returns a wrapper function for the DLL call."""
        if instance is None:
            return self
        
        dll_func = getattr(instance.vsme_windll, self.func_name)
        
        def wrapper(*args, **kwargs):
            try:
                if self.return_status:
                    # Initialize status code and add it to args
                    status = ctypes.c_int(-999)
                    args_with_status = args + (ctypes.byref(status),)
                    
                    # Call the function and retry if status is still -999
                    while True:
                        dll_func(*args_with_status)
                        if status.value != -999:
                            break
                        import time
                        time.sleep(0.1)  # Short pause before retry
                    
                    # Store status value in instance for later reference
                    instance.last_status = status.value
                    
                    # Check for error status
                    if status.value != 0 and self.func_name != "VSMEapp":
                        error_msg = f"Function {self.func_name} failed. Status: {status.value}: {instance.get_error_description(status.value)}"
                        logger.error(error_msg)
                        raise GMOOException(error_msg)
                else:
                    # Direct call without status handling
                    dll_func(*args)
                    
            except Exception as e:
                if not isinstance(e, GMOOException):
                    logger.error(f"An error occurred while calling function {self.func_name}", exc_info=True)
                raise e
                
        return wrapper


class GMOOAPI:
    """
    Wrapper API class for the VSME_DLL engine, providing optimization capabilities and file management.
    
    The class manages communication with the compiled GMOO DLL/shared library, handling
    data type conversions, memory allocation, and file operations necessary for running
    optimization workflows.
    """

    # Define DLL functions using the descriptor - grouped by type and alphabetized
    
    # VSMEdev* functions (development mode)
    VSMEdevCopyVSME = DLLFunction("VSMEdevCopyVSME")
    VSMEdevDesignAgents = DLLFunction("VSMEdevDesignAgents")
    VSMEdevDesignCases = DLLFunction("VSMEdevDesignCases")
    VSMEdevDevelopVSME = DLLFunction("VSMEdevDevelopVSME")
    VSMEdevExportCasesBIN = DLLFunction("VSMEdevExportCasesBIN")
    VSMEdevExportCasesCSV = DLLFunction("VSMEdevExportCasesCSV")
    VSMEdevExportVSME = DLLFunction("VSMEdevExportVSME")
    VSMEdevImportResultsBIN = DLLFunction("VSMEdevImportResultsBIN")
    VSMEdevImportResultsCSV = DLLFunction("VSMEdevImportResultsCSV")
    VSMEdevInitBackupFile = DLLFunction("VSMEdevInitBackupFile")
    VSMEdevInitOutcomes = DLLFunction("VSMEdevInitOutcomes")
    VSMEdevInitVariables = DLLFunction("VSMEdevInitVariables")
    VSMEdevLoadCaseResults = DLLFunction("VSMEdevLoadCaseResults")
    VSMEdevLoadCategoryLabels = DLLFunction("VSMEdevLoadCategoryLabels")
    VSMEdevLoadOutcomeNames = DLLFunction("VSMEdevLoadOutcomeNames")
    VSMEdevLoadOutcomeName = DLLFunction("VSMEdevLoadOutcomeName")
    VSMEdevLoadParameters = DLLFunction("VSMEdevLoadParameters")
    VSMEdevLoadUserCase = DLLFunction("VSMEdevLoadUserCase")
    VSMEdevLoadUserCases = DLLFunction("VSMEdevLoadUserCases")
    VSMEdevLoadVariableLimits = DLLFunction("VSMEdevLoadVariableLimits")
    VSMEdevLoadVariableNames = DLLFunction("VSMEdevLoadVariableNames")
    VSMEdevLoadVariableName = DLLFunction("VSMEdevLoadVariableName")
    VSMEdevLoadVariableTypes = DLLFunction("VSMEdevLoadVariableTypes")
    VSMEdevLoadVSMEName = DLLFunction("VSMEdevLoadVSMEName")
    VSMEdevMain = DLLFunction("VSMEdevMain", return_status=False)
    VSMEdevPokeCaseCount = DLLFunction("VSMEdevPokeCaseCount")
    VSMEdevPokeCaseVariables = DLLFunction("VSMEdevPokeCaseVariables")

    VSMEdevPokeNonlinearity = DLLFunction("VSMEdevPokeNonlinearity")
    VSMEdevPokeVariableCount = DLLFunction("VSMEdevPokeVariableCount")
    VSMEdevPokeVariables = DLLFunction("VSMEdevPokeVariables")
    VSMEdevPokeVSMEName = DLLFunction("VSMEdevPokeVSMEName")
    VSMEdevReadBackupFile = DLLFunction("VSMEdevReadBackupFile")
    VSMEdevUnloadVSME = DLLFunction("VSMEdevUnloadVSME")
    
    # VSMEapp* functions (application mode)
    VSMEapp = DLLFunction("VSMEapp")
    VSMEappExportCasesBIN = DLLFunction("VSMEappExportCasesBIN")
    VSMEappExportCasesCSV = DLLFunction("VSMEappExportCasesCSV")
    VSMEappImportResultsBIN = DLLFunction("VSMEappImportResultsBIN")
    VSMEappImportResultsCSV = DLLFunction("VSMEappImportResultsCSV")
    VSMEappInitVariables = DLLFunction("VSMEappInitVariables")
    VSMEappLoadCategoryIndex = DLLFunction("VSMEappLoadCategoryIndex")
    VSMEappLoadCategoryLabel = DLLFunction("VSMEappLoadCategoryLabel")
    VSMEappLoadLearnedCase = DLLFunction("VSMEappLoadLearnedCase")
    VSMEappLoadObjectiveStatus = DLLFunction("VSMEappLoadObjectiveStatus")
    VSMEappLoadObjectiveUncertainty = DLLFunction("VSMEappLoadObjectiveUncertainty")
    VSMEappLoadObjectiveValues = DLLFunction("VSMEappLoadObjectiveValues")
    VSMEappLoadOutcomeValues = DLLFunction("VSMEappLoadOutcomeValues")
    VSMEappLoadVariableLimitMax = DLLFunction("VSMEappLoadVariableLimitMax")
    VSMEappLoadVariableLimitMin = DLLFunction("VSMEappLoadVariableLimitMin")
    VSMEappLoadVariableValues = DLLFunction("VSMEappLoadVariableValues")
    VSMEappLoadVSME = DLLFunction("VSMEappLoadVSME")
    VSMEappPokeDimensions10 = DLLFunction("VSMEappPokeDimensions10")
    VSMEappPokeLearnedCase = DLLFunction("VSMEappPokeLearnedCase")
    VSMEappPokeNorm = DLLFunction("VSMEappPokeNorm")
    VSMEappPokeObjectiveStatus = DLLFunction("VSMEappPokeObjectiveStatus")
    VSMEappPokeOutcomeDevLimitMax = DLLFunction("VSMEappPokeOutcomeDevLimitMax")
    VSMEappPokeOutcomeDevLimitMin = DLLFunction("VSMEappPokeOutcomeDevLimitMin")
    VSMEappPokeOutcomeNames = DLLFunction("VSMEappPokeOutcomeNames")
    VSMEappPokeOutcomeName = DLLFunction("VSMEappPokeOutcomeName")
    VSMEappPokeOutcomeValues = DLLFunction("VSMEappPokeOutcomeValues")
    VSMEappPokeVariableDevLimitMax = DLLFunction("VSMEappPokeVariableDevLimitMax") 
    VSMEappPokeVariableDevLimitMin = DLLFunction("VSMEappPokeVariableDevLimitMin")
    VSMEappPokeVariableName = DLLFunction("VSMEappPokeVariableName")
    VSMEappPokeVariableNames = DLLFunction("VSMEappPokeVariableNames")
    VSMEappPokeVariableValues = DLLFunction("VSMEappPokeVariableValues")
    VSMEappUnloadVSME = DLLFunction("VSMEappUnloadVSME")
    
    # Genetic and bias algorithms
    VSMEBiasInit = DLLFunction("VSMEBiasInit")
    VSMEBiasPokeBiasValues = DLLFunction("VSMEBiasPokeBiasValues")
    VSMEGenInit = DLLFunction("VSMEGenInit")

    def __init__(
        self,
        vsme_windll: Any,
        vsme_input_filename: str,
        var_mins: List[float],
        var_maxs: List[float],
        num_output_vars: int,
        model_function: Callable,
        save_file_dir: str = ".",
        logspace: Optional[List[int]] = None,
        var_types: Optional[List[int]] = None,
        categories_list: Optional[List[List[str]]] = None
    ):
        """
        Initialize the GMOOAPI object with problem parameters.

        Args:
            vsme_windll: Ctypes object for the VSME shared library.
            vsme_input_filename: Basename for the VSME input/output files.
            var_mins: Minimum values for each model variable.
            var_maxs: Maximum values for each model variable.
            num_output_vars: Number of model objectives.
            model_function: Function to compute model's outcomes.
            save_file_dir: Directory for saving VSME files. Defaults to current directory.
            logspace: Flags indicating if variables should be sampled in log space.
                      1 for log sampling, 0 for linear sampling.
                      If None, all variables use linear sampling.
            var_types: Variable type identifiers:
                      1 = float (default)
                      2 = integer
                      3 = logical/boolean
                      4 = categorical
            categories_list: Names of categories for categorical variables.
                             Must be provided if any variable has type 4.
        """
        self.vsme_windll = vsme_windll
        self.save_file_dir = save_file_dir
        self.last_status = 0

        # Store the base filename without path manipulation or extension
        #self.vsme_input_filename = vsme_input_filename
        path_sep = "\\" if platform.system() == "Windows" else "/"
        self.vsme_input_filename = f"{self.save_file_dir}{path_sep}{vsme_input_filename}"

        # Initialize core parameters
        self.iVSME = ctypes.c_int(0)
        self.nVars = ctypes.c_int(len(var_mins))
        self.nObjs = ctypes.c_int(num_output_vars)
        self.devCaseCount = ctypes.c_int(0)

        # Store variable limits
        self.aVarLimMin = var_mins
        self.aVarLimMax = var_maxs
        self.dVarLimMin = CtypesHelper.create_double_array(var_mins)
        self.dVarLimMax = CtypesHelper.create_double_array(var_maxs)

        # Set logspace flags
        self.logspace = [ctypes.c_int(x) for x in (logspace or [0] * self.nVars.value)]
            
        # Set variable types
        self.var_types = var_types or [1] * self.nVars.value  # Default to float data type
        self.iVarType = CtypesHelper.create_int_array(self.var_types)
            
        self.categories_list = categories_list
        self.model_function = model_function

        # Initialize common parameters that will be reused
        self.cRootName = fortran_hollerith_string(py_string="ROOT", pad_len=4)

        # Create a default set of unbias values (1.0) for each variable
        self.unbias_vector = [1.0] * self.nVars.value

        # Create the component objects
        self.development = DevelopmentOperations(self)
        self.application = ApplicationOperations(self)

    def _call_function(self, func: Callable, *args: Any) -> None:
        """
        Call a ctypes function with error handling and retry logic.

        Args:
            func: The ctypes function to call.
            *args: Variable length argument list to pass to the ctypes function.
        """
        try:
            # Handle VSMEdevMain function differently as it doesn't return a status code
            if func.__name__ != "VSMEdevMain":
                # Initialize status code and add it to args
                self.iStatus = ctypes.c_int(-999)
                args = args + (ctypes.byref(self.iStatus),)
                
                # Call the function and retry if status is still -999
                while True:
                    func(*args)
                    if self.iStatus.value != -999:
                        break
                    time.sleep(0.1)  # Short pause before retry
                
                # Check for error status
                if self.iStatus.value != 0 and func.__name__ != "VSMEapp":
                    error_msg = f"Function {func.__name__} failed. Status: {self.iStatus.value}: {self.get_error_description(int(self.iStatus.value))}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            else:
                # VSMEdevMain doesn't follow the same pattern - just call it
                func(*args)
        except Exception as e:
            logger.error(f"An error occurred while calling function {func.__name__}", exc_info=True)
            raise e

    def vsme_dev_main(self, item: int) -> int:
        """
        Check the status of the VSME development process.
        
        This method is kept here as it's a lower-level DLL interaction function.
        
        Args:
            item: The status item to check (specific values depend on the DLL).
            
        Returns:
            int: Status value for the requested item (0 = false, 1 = true).
        """
        self.iRequest = ctypes.c_int(item)
        self.iVSME = ctypes.c_int(1)
        self.iHave = ctypes.c_int(0)  # 0 = false, 1 = true
        
        self._call_function(self.vsme_windll.VSMEdevMain,
                          ctypes.byref(self.iRequest),
                          ctypes.byref(self.iVSME),
                          ctypes.byref(self.iHave))

        return self.iHave.value
            
    def get_error_description(self,error_code):
            print("errorcode:",error_code)
            error_dict = {
                10001: "VSMEdevLoadVSMEName: cVSMEname cannot be a blank string",
                10002: "VSMEdevInitBackupFile: VSME does not exist to initialize the project backup file",
                10003: "VSMEdevInitBackupFile: Directory path and file name not found for the project backup file",
                10004: "VSMEdevInitBackupFile: Directory path is empty string for the project backup file",
                10005: "VSMEdevInitBackupFile: Directory does not exist for the project backup file",
                10011: "VSMEdevInitVariables: VSME does not exist to initialize the default variables",
                10012: "VSMEdevInitVariables: nVars must be a positive integer",
                10013: "VSMEdevLoadVariableNames: VSME does not exist to initialize the variable names",
                10014: "VSMEdevLoadVariableNames: nVars must be a positive integer",
                10015: "VSMEdevLoadVariableNames: cVarName cannot be a blank string",
                10016: "VSMEdevLoadVariableLimits: VSME does not exist to initialize the variable limits",
                10017: "VSMEdevLoadVariableLimits: nVars must equal to number of variables",
                10018: "VSMEdevLoadVariableLimits: dVarLimMin must be less than dVarLimMax",
                10019: "VSMEdevLoadVariableLimits: Internal error. Contact development team.",
                10021: "VSMEdevDesignAgents: VSME does not exist",
                10022: "VSMEdevDesignAgents: Variables have not been loaded",
                10023: "VSMEdevDesignAgents: Internal error. Contact development team.",
                10024: "VSMEdevDesignAgents: Internal error. Contact development team.",
                10025: "VSMEdevDesignAgents: Internal error. Contact development team.",
                10026: "VSMEdevDesignAgents: Internal error. Contact development team.",
                10027: "VSMEdevDesignAgents: Internal error. Contact development team.",
                10028: "VSMEdevDesignAgents: Internal error. Contact development team.",
                10031: "VSMEdevDesignCases: VSME does not exist",
                10032: "VSMEdevDesignCases: Variables were not loaded",
                10033: "VSMEdevDesignCases: Agents were not designed",
                10034: "VSMEdevLoadVariableName: VSME does not exist to initialize the variable names",
                10035: "VSMEdevLoadVariableName: iVar cannot be less than or equal to zero",
                10036: "VSMEdevLoadVariableName: cVarName cannot be blank",
                10037: "VSMEdevLoadVariableTypes: VSME does not exist to initialize the variable names",
                10038: "VSMEdevLoadVariableTypes: nVars must equal to number of variables",
                10039: "VSMEdevLoadVariableTypes: Internal error. Contact development team.",
                10041: "VSMEdevExportCasesBIN: VSME does not exist",
                10042: "VSMEdevExportCasesBIN: Cases were not designed",
                10043: "VSMEdevExportCasesBIN: Directory path and file name could not be found for the binary export file",
                10044: "VSMEdevExportCasesBIN: Directory path is empty string for the binary export file",
                10045: "VSMEdevExportCasesBIN: Directory does not exist for the binary export file",
                10046: "VSMEdevExportCasesBIN: Error in writing binary export file. Items written must equal number of cases.",
                10051: "VSMEdevImportResultsBIN: VSME does not exist",
                10052: "VSMEdevImportResultsBIN: Cases were not designed",
                10053: "VSMEdevImportResultsBIN: Directory path and file name could not be found for the binary import file",
                10054: "VSMEdevImportResultsBIN: Directory path is empty string for the binary import file",
                10055: "VSMEdevImportResultsBIN: Directory does not exist for the binary import file",
                10056: "VSMEdevImportResultsBIN: Error in reading binary import file",
                10057: "VSMEdevImportResultsBIN: Number of case results must match number of cases to be read",
                10061: "VSMEdevDevelopVSME: VSME does not exist",
                10062: "VSMEdevDevelopVSME: Case results were not imported",
                10063: "VSMEdevDevelopVSME: There are no valid outcomes (outcomes are same for all cases)",
                10071: "VSMEdevExportVSME: VSME does not exist",
                10072: "VSMEdevExportVSME: VSME was not developed",
                10073: "VSMEdevExportVSME: Directory path and file name could not be found for the VSME export file",
                10074: "VSMEdevExportVSME: Directory path is empty string for the VSME export file",
                10075: "VSMEdevExportVSME: Directory does not exist for the VSME export file",
                10076: "VSMEdevExportVSME: Error in writing VSME export file",
                10081: "VSMEdevExportCasesCSV: VSME does not exist when calling",
                10082: "VSMEdevExportCasesCSV: Cases were not designed",
                10083: "VSMEdevExportCasesCSV: Directory path and file name could not be found for the CSV export file",
                10084: "VSMEdevExportCasesCSV: Directory path is empty string for the CSV export file",
                10085: "VSMEdevExportCasesCSV: Directory does not exist for the CSV export file",
                10086: "VSMEdevExportCasesCSV: Error in writing CSV export file. Items written must equal number of cases.",
                10091: "VSMEdevImportResultsCSV: VSME does not exist",
                10092: "VSMEdevImportResultsCSV: Cases were not designed",
                10093: "VSMEdevImportResultsCSV: Directory path and file name could not be found for the CSV import file",
                10094: "VSMEdevImportResultsCSV: Directory path is empty string for the CSV import file",
                10095: "VSMEdevImportResultsCSV: Directory does not exist for the CSV import file",
                10096: "VSMEdevImportResultsCSV: CSV import file does not exist",
                10097: "VSMEdevImportResultsCSV: Error in reading CSV import file",
                10098: "VSMEdevImportResultsCSV: Number of case results must match number of cases to be read",
                10101: "VSMEdevInitOutcomes: VSME does not exist",
                10102: "VSMEdevInitOutcomes: nOuts must be a positive integer",
                10103: "VSMEdevLoadOutcomeNames: VSME does not exist",
                10104: "VSMEdevLoadOutcomeNames: nOuts must be a positive integer",
                10105: "VSMEdevLoadOutcomeNames: cOutName cannot be a blank string",
                10106: "VSMEdevLoadOutcomeName: VSME does not exist",
                10107: "VSMEdevLoadOutcomeName: iOut cannot be less than or equal to zero",
                10108: "VSMEdevLoadOutcomeName: cOutName cannot be blank",
                10111: "VSMEdevLoadCaseResults: VSME does not exist",
                10112: "VSMEdevLoadCaseResults: Outcomes were not defined",
                10113: "VSMEdevLoadCaseResults: Experimental cases were not designed",
                10114: "VSMEdevLoadCaseResults: Specified case number cannot be greater than the number of cases designed",
                10115: "VSMEdevLoadCaseResults: Specified case number cannot be negative or zero",
                10131: "VSMEdevLoadUserCases: VSME does not exist",
                10132: "VSMEdevLoadUserCases: nVars does not match number of variables",
                10133: "VSMEdevLoadUserCases: nOuts does not match number of outcomes",
                10141: "VSMEdevLoadUserCase: VSME does not exist",
                10142: "VSMEdevLoadUserCase: nVars does not match number of variables",
                10143: "VSMEdevLoadUserCase: nOuts does not match number of outcomes",
                10144: "VSMEdevLoadUserCase: User case was not loaded due to violation of a variable min/max limit",
                11001: "VSMEdevPokeVSMEName: VSME does not exist",
                11002: "VSMEdevPokeVariableCount: VSME does not exist",
                11003: "VSMEdevPokeVariables: VSME does not exist",
                11004: "VSMEdevPokeVariables: There are no defined variables in the project",
                11008: "VSMEdevPokeCaseCount: VSME does not exist",
                11009: "VSMEdevPokeCaseVariables: VSME does not exist",
                11010: "VSMEdevPokeCaseVariables: There are no defined variables in the project",
                11011: "VSMEdevPokeCaseVariables: There are no defined cases in the project",
                11012: "VSMEdevPokeCaseVariables: Requested case number is larger than number of cases in the project",
                11101: "VSMEdevReadBackupFile: Directory path and file name could not be found for the binary backup file",
                11102: "VSMEdevReadBackupFile: Directory path is empty string for the binary backup file",
                11103: "VSMEdevReadBackupFile: Directory does not exist for the binary backup file",
                11104: "VSMEdevReadBackupFile: Binary backup file does not exist",
                11105: "VSMEdevReadBackupFile: Error in reading binary backup file",
                19999: "VSMEdevCopyVSME: VSME was not generated to copy over",
                20001: "VSMEappLoadVSME: Directory path and file name could not be found for the binary VSME file",
                20002: "VSMEappLoadVSME: Directory path is empty string for the binary VSME file",
                20003: "VSMEappLoadVSME: Directory does not exist for the binary VSME file",
                20004: "VSMEappLoadVSME: Binary VSME file does not exist",
                20005: "VSMEappLoadVSME: Error during reading binary VSME file",
                20006: "VSMEappLoadVSME: This is not a VSME file",
                20007: "VSMEappLoadVSME: VSME file version requires an updated Program DLL",
                20008: "VSMEappLoadVSME: VSME file date requires an updated Program DLL",
                20011: "VSMEappLoadObjectiveValues: Cannot locate VSME",
                20012: "VSMEappLoadObjectiveValues: Specified nObjs must match outcomes in the VSME file",
                20021: "VSMEappLoadVariableValues: Cannot locate VSME",
                20022: "VSMEappLoadVariableValues: Specified nVars must match variables in the VSME file",
                20023: "VSMEappLoadVariableValues: A valid iBias identifier must be specified when the Bias option is active",
                20024: "VSMEappLoadVariableValues: A valid iGen identifier must be specified when the Genetic option is active",
                20031: "VSMEappLoadOutcomeValues: Cannot locate VSME",
                20032: "VSMEappLoadOutcomeValues: Specified iOuts must match outcomes in the VSME file",
                20033: "VSMEappLoadOutcomeValues: A valid iBias identifier must be specified when the Bias option is active",
                20034: "VSMEappLoadOutcomeValues: A valid iGen identifier must be specified when the Genetic option is active",
                20041: "VSMEapp: Cannot locate VSME",
                20042: "VSMEapp: Objectives vector has not been loaded",
                20043: "VSMEapp: Variables vector has not been loaded",
                20044: "VSMEapp: Outcomes vector has not been loaded",
                20051: "VSMEappPokeVariableValues: Cannot locate VSME",
                20052: "VSMEappPokeVariableValues: Specified nVars must match number of variables",
                20053: "VSMEappPokeVariableValues: A valid iBias identifier must be specified when the Bias option is active",
                20054: "VSMEappPokeVariableValues: A valid iGen identifier must be specified when the Genetic option is active",
                20055: "VSMEappPokeVariableNames: Cannot locate VSME",
                20056: "VSMEappPokeVariableNames: Specified nVars must match number of variables",
                20061: "VSMEappPokeOutcomeValues: Cannot locate VSME",
                20062: "VSMEappPokeOutcomeValues: Specified nOuts must match number of outcomes",
                20063: "VSMEappPokeOutcomeValues: A valid iGen identifier must be specified when the Genetic option is active",
                20064: "VSMEappPokeOutcomeNames: Cannot locate VSME",
                20065: "VSMEappPokeOutcomeNames: Specified nOuts must match number of outcomes",
                20071: "VSMEappPokeNorm: Cannot locate VSME",
                20081: "VSMEappExportCasesBIN: Cannot locate VSME",
                20082: "VSMEappExportCasesBIN: Directory path and file name could not be found for the binary export file",
                20083: "VSMEappExportCasesBIN: Directory path is empty string for the binary export file",
                20084: "VSMEappExportCasesBIN: Directory does not exist for the binary export file",
                20085: "VSMEappExportCasesBIN: Error in writing binary export file",
                20091: "VSMEappExportCasesCSV: Cannot locate VSME",
                20092: "VSMEappExportCasesCSV: Directory path and file name could not be found for the CSV export file",
                20093: "VSMEappExportCasesCSV: Directory path is empty string for the CSV export file",
                20094: "VSMEappExportCasesCSV: Directory does not exist for the CSV export file",
                20095: "VSMEappExportCasesCSV: Error in writing CSV export file",
                20101: "VSMEappImportResultsBIN: Cannot locate VSME",
                20102: "VSMEappImportResultsBIN: Directory path and file name could not be found for the binary import file",
                20103: "VSMEappImportResultsBIN: Directory path is empty string for the binary import file",
                20104: "VSMEappImportResultsBIN: Directory does not exist for the binary import file",
                20105: "VSMEappImportResultsBIN: Error in reading binary import file",
                20111: "VSMEappImportResultsCSV: Cannot locate VSME",
                20112: "VSMEappImportResultsCSV: Directory path and file name could not be found for the CSV import file",
                20113: "VSMEappImportResultsCSV: Directory path is empty string for the CSV import file",
                20114: "VSMEappImportResultsCSV: Directory does not exist for the CSV import file",
                20115: "VSMEappImportResultsCSV: Error in reading CSV import file",
                21011: "VSMEappLoadObjectiveUncertainty: Cannot locate VSME",
                21012: "VSMEappLoadObjectiveUncertainty: Specified nObjs must match number of outcomes",
                21013: "VSMEappLoadObjectiveValues: Cannot locate VSME",
                21014: "VSMEappLoadObjectiveValues: Specified nObjs must match number of outcomes",
                21021: "VSMEappLoadVariableLimitMin: Cannot locate VSME",
                21022: "VSMEappLoadVariableLimitMin: Specified nVars must match number of variables",
                21023: "VSMEappLoadVariableLimitMin: Specified dVarLimMin must be greater than the minimum limit in VSME",
                21024: "VSMEappLoadVariableLimitMin: Specified dVarLimMin must be smaller than the maximum limit in VSME",
                21025: "VSMEappLoadVariableLimitMin: Specified dVarLimMin must be smaller than or equal to the dVarLimMax",
                21031: "VSMEappLoadVariableLimitMax: Cannot locate VSME",
                21032: "VSMEappLoadVariableLimitMax: Specified nVars must match number of variables",
                21033: "VSMEappLoadVariableLimitMax: Specified dVarLimMax must be smaller than the maximum limit in VSME",
                21034: "VSMEappLoadVariableLimitMax: Specified dVarLimMax must be greater than the minimum limit in VSME",
                21035: "VSMEappLoadVariableLimitMax: Specified dVarLimMax must be greater than or equal to the dVarLimMin",
                21041: "VSMEappPokeDimensions: Cannot locate VSME",
                21051: "VSMEappPokeVariableName: Cannot locate VSME",
                21052: "VSMEappPokeVariableName: Specified iVar is greater than number of variables",
                21061: "VSMEappPokeOutcomeName: Cannot locate VSME",
                21062: "VSMEappPokeOutcomeName: Specified iOut is greater than number of outcomes",
                21091: "VSMEappPokeVariableDevLimitMin: Cannot locate VSME",
                21092: "VSMEappPokeVariableDevLimitMin: Specified nVars must match number of variables",
                21093: "VSMEappPokeOutcomeDevLimitMin: Cannot locate VSME",
                21094: "VSMEappPokeOutcomeDevLimitMin: Specified nOuts must match number of outcomes",
                21101: "VSMEappPokeVariableDevLimitMax: Cannot locate VSME",
                21102: "VSMEappPokeVariableDevLimitMax: Specified nVars must match number variables",
                21103: "VSMEappPokeOutcomeDevLimitMax: Cannot locate VSME",
                21104: "VSMEappPokeOutcomeDevLimitMax: Specified nOuts must match number of outcomes",
                21121: "VSMEappPokeObjectiveStatus: Cannot locate VSME",
                21122: "VSMEappPokeObjectiveStatus: Specified nObjs is greater than number of objectives"
            }
            
            return error_dict.get(error_code, "Unknown error code")