"""
Extended tests for helper functions in the GMOO SDK.

This module tests critical utility functions in the helpers module to improve coverage.
"""

import pytest
import numpy as np
import ctypes
import tempfile
import os
from gmoo_sdk.helpers import (
    CtypesHelper,
    fortran_hollerith_string,
    c_string_compatibility,
    validate_nan,
    write_data,
    normalize_path
)


class TestHelperCriticalFunctions:
    """Test the most-used helper functions."""
    
    def test_ctypes_helper_double_array(self):
        """Test creating ctypes double arrays."""
        # Test with list input
        data = [1.0, 2.5, 3.7, 4.2]
        arr = CtypesHelper.create_double_array(data)
        
        assert len(arr) == len(data)
        for i, val in enumerate(data):
            assert arr[i] == val
        
        # Test with empty array
        empty_arr = CtypesHelper.create_double_array([])
        assert len(empty_arr) == 0
        
        # Test with single value
        single_arr = CtypesHelper.create_double_array([42.0])
        assert len(single_arr) == 1
        assert single_arr[0] == 42.0
    
    def test_ctypes_helper_int_array(self):
        """Test creating ctypes integer arrays."""
        # Test with list input
        data = [1, 2, 3, 4, 5]
        arr = CtypesHelper.create_int_array(data)
        
        assert len(arr) == len(data)
        for i, val in enumerate(data):
            assert arr[i] == val
        
        # Test with integers that came from floats
        mixed_data = [int(1.9), int(2.1), int(3.5)]  # Explicitly convert to integers
        arr = CtypesHelper.create_int_array(mixed_data)
        
        assert arr[0] == 1
        assert arr[1] == 2
        assert arr[2] == 3
    
    def test_ctypes_helper_to_numpy(self):
        """Test converting ctypes arrays to numpy."""
        # Create ctypes array
        data = [1.5, 2.5, 3.5]
        ctypes_arr = CtypesHelper.create_double_array(data)
        
        # Convert to numpy
        np_arr = CtypesHelper.to_numpy_array(ctypes_arr)
        
        assert isinstance(np_arr, np.ndarray)
        assert len(np_arr) == len(data)
        assert np.allclose(np_arr, data)
    
    def test_ctypes_helper_char_buffer(self):
        """Test creating character buffers."""
        text = "Hello World"
        size = 32
        
        buffer = CtypesHelper.create_char_buffer(text, size)
        
        assert len(buffer) == size
        assert buffer.value == text.encode('ascii')
        
        # Test with empty string
        empty_buffer = CtypesHelper.create_char_buffer("", 10)
        assert len(empty_buffer) == 10
        assert empty_buffer.value == b""
    
    def test_fortran_hollerith_string(self):
        """Test Fortran string conversion."""
        # Test basic conversion
        py_str = "TestString"
        fortran_str = fortran_hollerith_string(py_str)
        
        assert len(fortran_str) == 32  # Default padding
        assert fortran_str.value.startswith(b"TestString")
        
        # Test with custom padding
        fortran_str_custom = fortran_hollerith_string(py_str, pad_len=16)
        assert len(fortran_str_custom) == 16
        
        # Test truncation
        long_str = "This is a very long string that should be truncated"
        fortran_str_long = fortran_hollerith_string(long_str, pad_len=10)
        assert len(fortran_str_long) == 10
        assert fortran_str_long.value == b"This is a "
    
    def test_c_string_compatibility(self):
        """Test C string compatibility conversion."""
        # Test basic conversion
        py_str = "TestString"
        c_str = c_string_compatibility(py_str)
        
        assert isinstance(c_str, ctypes.c_wchar_p)
        assert c_str.value.startswith("TestString")
        
        # Test padding
        short_str = "Hi"
        c_str_padded = c_string_compatibility(short_str, pad_len=10)
        assert len(c_str_padded.value) == 9  # pad_len - 1
    
    def test_validate_nan(self):
        """Test NaN validation."""
        # Test valid values - should not raise
        validate_nan(1.0)
        validate_nan([1.0, 2.0, 3.0])
        validate_nan(np.array([1.0, 2.0, 3.0]))
        
        # Test NaN values - should raise
        with pytest.raises(ValueError) as exc_info:
            validate_nan(np.nan)
        assert "NaN" in str(exc_info.value)
        
        # Test array with NaN
        with pytest.raises(ValueError) as exc_info:
            validate_nan([1.0, np.nan, 3.0])
        assert "NaN" in str(exc_info.value)
        
        # Test numpy array with NaN
        with pytest.raises(ValueError) as exc_info:
            validate_nan(np.array([1.0, 2.0, np.nan]))
        assert "NaN" in str(exc_info.value)
    
    def test_write_data(self):
        """Test CSV data writing with done file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_data.csv")
            
            # Test data
            data = [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0]
            ]
            
            write_data(file_path, data)
            
            # Check CSV file exists
            assert os.path.exists(file_path)
            
            # Check done file exists
            done_file = f"{file_path}.done"
            assert os.path.exists(done_file)
            
            # Check CSV content
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                # Check metadata
                assert "NROWS" in lines[0]
                assert "3" in lines[0]  # 3 rows
                assert "NCOLS" in lines[0]
                assert "3" in lines[0]  # 3 columns
                
                # Check headers
                assert "CaseName" in lines[1]
                assert "Var0" in lines[1]
                
                # Check data
                assert "Case0" in lines[2]
                assert "1.0" in lines[2]
    
    def test_normalize_path(self):
        """Test path normalization."""
        # Test Windows-style paths
        import platform
        
        if platform.system() == "Windows":
            # Windows tests
            path = normalize_path("C:\\Users\\test", "file.txt")
            assert path == "C:\\Users\\test\\file.txt"
            
            # Test with trailing separator
            path2 = normalize_path("C:\\Users\\test\\", "file.txt")
            assert path2 == "C:\\Users\\test\\file.txt"
        else:
            # Unix tests
            path = normalize_path("/home/user", "file.txt")
            assert path == "/home/user/file.txt"
            
            # Test with trailing separator
            path2 = normalize_path("/home/user/", "file.txt")
            assert path2 == "/home/user/file.txt"
        
        # Test empty base directory
        path3 = normalize_path("", "file.txt")
        sep = '\\' if platform.system() == "Windows" else '/'
        assert path3 == f"{sep}file.txt"
    
    def test_array_edge_cases(self):
        """Test edge cases in array creation."""
        # Test with very large arrays
        large_data = list(range(1000))
        arr = CtypesHelper.create_int_array(large_data)
        
        assert len(arr) == 1000
        assert arr[0] == 0
        assert arr[999] == 999
        
        # Test with negative values
        negative_data = [-1.5, -2.7, -3.9]
        arr = CtypesHelper.create_double_array(negative_data)
        
        assert arr[0] == -1.5
        assert arr[1] == -2.7
        assert arr[2] == -3.9
        
        # Test with very small/large values
        extreme_data = [1e-10, 1e10, -1e10]
        arr = CtypesHelper.create_double_array(extreme_data)
        
        assert abs(arr[0] - 1e-10) < 1e-15
        assert arr[1] == 1e10
        assert arr[2] == -1e10