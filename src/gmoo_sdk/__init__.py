# -*- coding: utf-8 -*-
"""
GMOO SDK - Global Multi-Objective Optimization Software Development Kit

The GMOO SDK provides a Python interface to the GMOO (Global Multi-Objective 
Optimization) DLL, facilitating workflows for inverse model training and
inverse design optimization.

The package simplifies common optimization tasks by abstracting away the details
of DLL management and memory handling, providing a clean, object-oriented API
for model development and application.

Core Classes:
    GMOOAPI: Main interface class for interacting with the GMOO optimization library
    DevelopmentOperations: Methods for training inverse models
    ApplicationOperations: Methods for using models for inference and optimization
    
Compatibility Layer:
    GMOOAPILegacy: Legacy wrapper for the GMOOAPI class
    pyVSMEDevelopmentSetup, pyVSMEDevelopmentLoad, pyVSMEInverseSingleIter: Legacy workflow functions

Authors: Matt Freeman, Jason Hopkins
Version: 2.0.0 (Refactored)
"""

# Import core classes
from .dll_interface import GMOOAPI, GMOOException
from .development import DevelopmentOperations
from .application import ApplicationOperations

# Import compatibility layer
from .compatibility import (
    GMOOAPILegacy,
    pyVSMEDevelopmentSetup,
    pyVSMEDevelopmentLoad,
    pyVSMEInverseSingleIter
)

# Import helper functions
from .helpers import (
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

# Import the main stateless wrapper class
from .stateless_wrapper import GmooStatelessWrapper

# Import satisfaction checking
from .satisfaction import check_satisfaction

# Version information
__version__ = '2.0.0'

# Define what gets imported with "from gmoo_sdk import *"
__all__ = [
    # Core classes
    'GMOOAPI',
    'GMOOException',
    'DevelopmentOperations',
    'ApplicationOperations',
    
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
    
    # Stateless wrapper class
    'GmooStatelessWrapper',
    
    # Satisfaction checking
    'check_satisfaction'
]