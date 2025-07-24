# -*- coding: utf-8 -*-
"""
GMOO Helpers and Utilities Module

This module provides utility functions and helper classes for the GMOO SDK,
facilitating common operations like string handling, data validation, and
type conversions needed for communication between Python and the GMOO DLL.

Classes:
    CtypesHelper: Utility class for common ctypes operations

Functions:
    create_double_array: Create a ctypes double array from a list of floats
    create_int_array: Create a ctypes int array from a list of integers
    to_numpy_array: Convert a ctypes array to a numpy array
    create_char_buffer: Create a ctypes character buffer from a string
    fortran_hollerith_string: Convert a Python string to a Fortran-compatible format
    c_string_compatibility: Convert a Python string with maximum compatibility
    validate_nan: Check if a value is NaN and raise an error if it is
    write_data: Write data to a CSV file with a .done file signal
    normalize_path: Create a normalized absolute path based on the operating system

Authors: Matt Freeman, Jason Hopkins
Version: 2.0.0 (Refactored)
"""

import csv
import ctypes
import os
import platform
import warnings
from typing import Any, List, Optional, Union

import numpy as np


class CtypesHelper:
    """Helper class for common ctypes operations."""
    
    @staticmethod
    def create_double_array(values: List[float]) -> ctypes.Array:
        """Create a ctypes double array from a list of floats."""
        return (ctypes.c_double * len(values))(*values)
    
    @staticmethod
    def create_int_array(values: List[int]) -> ctypes.Array:
        """Create a ctypes int array from a list of integers."""
        return (ctypes.c_int * len(values))(*values)
    
    @staticmethod
    def to_numpy_array(ctypes_array: ctypes.Array) -> np.ndarray:
        """Convert a ctypes array to a numpy array."""
        return np.array(ctypes_array)

    @staticmethod
    def create_char_buffer(text: str, size: int) -> ctypes.Array:
        """
        Create a ctypes character buffer from a string with specific size.
        
        Args:
            text: String to convert to a character buffer
            size: Size of the buffer in bytes
            
        Returns:
            ctypes.Array: C-compatible character buffer
        """
        return ctypes.create_string_buffer(text.encode('ascii'), size)

@staticmethod
def fortran_hollerith_string(py_string: str, pad_len: int = 32) -> ctypes.Array:
    """
    Convert a Python string to a padded byte string for Fortran compatibility.
    
    This function creates a C-compatible character array suitable for passing to
    Fortran routines that expect fixed-length character strings.
    
    Args:
        py_string: Python string to convert.
        pad_len: Length to pad the string to (default 32).
        
    Returns:
        ctypes.Array: C-compatible character array (c_char_Array).
    """
    # Encode the string in ASCII as per Fortran's compatibility
    encoded_string = py_string.encode('ascii') 

    # Pad or truncate the string to the specified length
    padded_string = encoded_string.ljust(pad_len, b'\0')[:pad_len]

    # Create a buffer compatible with C (and by extension, Fortran)
    return ctypes.create_string_buffer(padded_string, pad_len)

@staticmethod
def c_string_compatibility(py_string: str, pad_len: int = 32) -> ctypes.c_wchar_p:
    """
    Convert a Python string to a padded C wchar_p string with maximum compatibility.
    
    This approach maximizes compatibility across tested Windows environments
    and should be used when other string conversion methods fail.
    
    Args:
        py_string: Python string to convert.
        pad_len: Length to pad the string to (default 32).
        
    Returns:
        ctypes.c_wchar_p: C-compatible wide character string.
    """
    py_string = py_string.ljust(pad_len - 1)
    py_string.encode('utf-8')
    c_vsme_name = ctypes.c_wchar_p(py_string)
    return c_vsme_name

@staticmethod
def validate_nan(input_val: Union[float, np.ndarray, List[float]], name: str = "data") -> None:
    """
    Check if a value is NaN and raise an error if it is.
    
    This utility function is used to validate inputs and outputs to ensure
    they don't contain NaN values, which can cause numerical problems.
    
    Args:
        input_val: Value or array to check for NaN.
        name: Name of the variable being checked (for error messages).
        
    Raises:
        ValueError: If the input contains NaN.
    """
    if hasattr(input_val, '__iter__'):
        if np.isnan(np.array(input_val)).any():
            raise ValueError(f"{name} contains NaN values.")
    else:
        if np.isnan(input_val):
            raise ValueError(f"{name} is NaN.")

@staticmethod
def write_data(file_path: str, data: List[List[float]]) -> None:
    """
    Write data to a CSV file and create a .done file to signal completion.
    
    Args:
        file_path: Path where the CSV file will be saved.
        data: List of lists containing the data to write.
    """
    # Get the number of rows and columns from the data
    nrows = len(data)
    ncols = len(data[0]) if data else 0

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the metadata
        writer.writerow(["NROWS", nrows, "NCOLS", ncols])

        # Write the column headers
        headers = ["CaseName"] + [f"Var{i}" for i in range(ncols-1)]
        writer.writerow(headers)

        # Write the data
        for i, row in enumerate(data):
            writer.writerow([f"Case{i}"] + row)

    # Write a 'done' file to signal completion
    with open(f"{file_path}.done", 'w') as file:
        pass

@staticmethod
def normalize_path(base_dir: str, filename: str) -> str:
    """
    Create a normalized absolute path based on the operating system.
    
    Args:
        base_dir: Base directory path
        filename: Filename to append
        
    Returns:
        str: Complete normalized path with correct separators
    """
    # Determine separator based on OS
    sep = '\\' if platform.system() == "Windows" else '/'
    
    # Handle case where base_dir already has trailing separator
    if base_dir and base_dir[-1] in ['\\', '/']:
        base_dir = base_dir[:-1]
    
    return f"{base_dir}{sep}{filename}"


# ---------------------------------------------------------------------------- #
# DEPRECATED FUNCTIONS (maintained for backward compatibility)
# ---------------------------------------------------------------------------- #

def fortranHollerithStringHelper(pyString: str, padLen: int = 32) -> ctypes.Array:
    """
    [DEPRECATED] Use fortran_hollerith_string instead.
    
    Convert a Python string to a padded byte string for Fortran compatibility.
    """
    warnings.warn(
        "fortranHollerithStringHelper is deprecated, use fortran_hollerith_string instead",
        DeprecationWarning,
        stacklevel=2
    )
    return fortran_hollerith_string(pyString, padLen)


def cStringCompatibilityHelper(pyString: str, padLen: int = 32) -> ctypes.c_wchar_p:
    """
    [DEPRECATED] Use c_string_compatibility instead.
    
    Convert a Python string to a padded C wchar_p string with maximum compatibility.
    """
    warnings.warn(
        "cStringCompatibilityHelper is deprecated, use c_string_compatibility instead",
        DeprecationWarning,
        stacklevel=2
    )
    return c_string_compatibility(pyString, padLen)


def cStringHelper(pyString: str, padLen: int = 32) -> ctypes.c_wchar_p:
    """
    [DEPRECATED] Use c_string_compatibility instead.
    
    Convert a Python string to a padded C wchar_p string.
    """
    warnings.warn(
        "cStringHelper is deprecated, use c_string_compatibility instead",
        DeprecationWarning,
        stacklevel=2
    )
    pyString = pyString.ljust(padLen - 1)
    pyString.encode('utf-8')
    return ctypes.c_wchar_p(pyString)


def cStringHelper2(pyString: str, padLen: int = 32) -> ctypes.Array:
    """
    [DEPRECATED] Use create_char_buffer instead.
    
    Convert a Python string to a padded C string.
    """
    warnings.warn(
        "cStringHelper2 is deprecated, use CtypesHelper.create_char_buffer instead",
        DeprecationWarning,
        stacklevel=2
    )
    pyString = pyString.ljust(padLen)
    return ctypes.create_string_buffer(pyString.encode('utf-8'), padLen)


def byteStringHelper(pyString: str, padLen: int = 32) -> ctypes.c_char_p:
    """
    [DEPRECATED] Use fortran_hollerith_string with ctypes.cast instead.
    
    Convert a Python string to a padded C char_p byte string.
    """
    warnings.warn(
        "byteStringHelper is deprecated, use fortran_hollerith_string with ctypes.cast instead",
        DeprecationWarning,
        stacklevel=2
    )
    s = bytes(pyString, "utf-8")
    s.ljust(padLen)
    return ctypes.c_char_p(s)


def error_nan(input_val: Union[float, np.ndarray], name: str = "data") -> None:
    """
    [DEPRECATED] Use validate_nan instead.
    
    Check if a value is NaN and raise an error if it is.
    """
    warnings.warn(
        "error_nan is deprecated, use validate_nan instead",
        DeprecationWarning,
        stacklevel=2
    )
    validate_nan(input_val, name)