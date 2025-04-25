# -*- coding: utf-8 -*-
"""
GMOO SDK - Global Multi-Objective Optimization Software Development Kit

The GMOO SDK provides a Python interface to the GMOO (Global Multi-Objective 
Optimization) DLL, facilitating workflows for surrogate model training and
inverse design optimization.

The package simplifies common optimization tasks by abstracting away the details
of DLL management and memory handling, providing a clean, object-oriented API
for model development and application.

Core Classes:
    GMOOAPI: Main interface class for interacting with the GMOO optimization library
    DevelopmentOperations: Methods for training surrogate models
    ApplicationOperations: Methods for using models for inference and optimization
    VisualizationTools: Tools for visualizing optimization results
    
Compatibility Layer:
    GMOOAPILegacy: Legacy wrapper for the GMOOAPI class
    pyVSMEDevelopmentSetup, pyVSMEDevelopmentLoad, pyVSMEInverseSingleIter: Legacy workflow functions

Authors: Matt Freeman, Jason Hopkins
Version: 2.0.0 (Refactored)
"""

# Import core classes
from gmoo_sdk.dll_interface import GMOOAPI, GMOOException
from gmoo_sdk.development import DevelopmentOperations
from gmoo_sdk.application import ApplicationOperations
from gmoo_sdk.visualization import VisualizationTools

# Import compatibility layer
from gmoo_sdk.compatibility import (
    GMOOAPILegacy,
    pyVSMEDevelopmentSetup,
    pyVSMEDevelopmentLoad,
    pyVSMEInverseSingleIter
)

# Import helper functions
from gmoo_sdk.helpers import (
    # New style helpers
    CtypesHelper,
    fortran_hollerith_string,
    c_string_compatibility,
    validate_nan,
    write_data,
    normalize_path,
    
    # Deprecated helpers (for backward compatibility)
    fortranHollerithStringHelper,
    cStringCompatibilityHelper,
    error_nan
)

# Import high-level convenience functions from gmoo_encapsulation
from gmoo_sdk.gmoo_encapsulation import (
    get_development_cases_encapsulation,
    load_development_cases_encapsulation,
    load_user_cases_encapsulation,
    inverse_encapsulation,
    retrieve_nonlinearity_encapsulation,
    get_dimensions_encapsulation,
    get_observed_min_maxes_encapsulation,
    minmax_encapsulation,
    rescope_search_space,
    reset_mins_maxs_encapsulation
)

# Version information
__version__ = '2.0.0'

# Define what gets imported with "from gmoo_sdk import *"
__all__ = [
    # Core classes
    'GMOOAPI',
    'GMOOException',
    'DevelopmentOperations',
    'ApplicationOperations',
    'VisualizationTools',
    
    # Compatibility layer
    'GMOOAPILegacy',
    'pyVSMEDevelopmentSetup',
    'pyVSMEDevelopmentLoad',
    'pyVSMEInverseSingleIter',
    
    # New style helpers
    'CtypesHelper',
    'fortran_hollerith_string',
    'c_string_compatibility',
    'validate_nan',
    'write_data',
    'normalize_path',
    
    # Deprecated helpers (for backward compatibility)
    'fortranHollerithStringHelper',
    'cStringCompatibilityHelper',
    'error_nan',
    
    # High-level encapsulation functions
    'get_development_cases_encapsulation',
    'load_development_cases_encapsulation',
    'load_user_cases_encapsulation',
    'inverse_encapsulation',
    'retrieve_nonlinearity_encapsulation',
    'get_dimensions_encapsulation',
    'get_observed_min_maxes_encapsulation',
    'minmax_encapsulation',
    'rescope_search_space',
    'reset_mins_maxs_encapsulation'
]