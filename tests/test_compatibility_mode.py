"""
Tests for compatibility mode functions.
"""

import pytest
import numpy as np
import os
from gmoo_sdk.dll_interface import GMOOAPI
from gmoo_sdk.compatibility import GMOOAPILegacy


class TestCompatibilityMode:
    """Test compatibility mode functionality."""
    
    def test_compatibility_wrapper_basic(self, loaded_dll):
        """Test basic compatibility wrapper functionality."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        # Create a standard GMOOAPI instance first
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_compat_basic",
            var_mins=[0.0],
            var_maxs=[1.0],
            num_output_vars=1,
            model_function=lambda x: np.array([x[0]**2]),
            save_file_dir='.'
        )
        
        # Create legacy wrapper around it
        legacy = GMOOAPILegacy(model)
        
        # Test that legacy methods exist
        assert hasattr(legacy, 'dev_load_vsme_name')
        assert hasattr(legacy, 'dev_initialize_variables')
        assert hasattr(legacy, 'dev_load_variable_types')
    
    
    def test_string_handling_compatibility(self, loaded_dll):
        """Test string handling in compatibility mode."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        from gmoo_sdk.helpers import c_string_compatibility, fortran_hollerith_string
        
        # Test c_string_compatibility
        test_string = "test_model_name"
        c_str = c_string_compatibility(test_string)
        assert c_str.value == test_string.ljust(31)  # Pads to pad_len - 1
        
        # Test with unicode
        unicode_string = "模型_test"
        c_str_unicode = c_string_compatibility(unicode_string)
        assert unicode_string in c_str_unicode.value
        
        # Test fortran_hollerith_string
        fortran_str = fortran_hollerith_string(test_string, pad_len=64)
        assert len(fortran_str) == 64
        
        # Test with empty string
        empty_c_str = c_string_compatibility("")
        assert empty_c_str.value.strip() == ""
    
    def test_compatibility_error_handling(self, loaded_dll):
        """Test error handling in compatibility mode."""
        if not loaded_dll:
            pytest.skip("DLL not properly loaded")
        
        # Create a standard model first
        model = GMOOAPI(
            vsme_windll=loaded_dll,
            vsme_input_filename="test_compat_error",
            var_mins=[0.0],
            var_maxs=[1.0],
            num_output_vars=1,
            model_function=lambda x: np.array([x[0]]),
            save_file_dir='.'
        )
        
        # Create legacy wrapper
        legacy = GMOOAPILegacy(model)
        
        # Test calling non-existent method
        with pytest.raises(AttributeError):
            legacy.non_existent_method()
        
        # Clean up
        try:
            model.development.unload_vsme()
        except:
            pass
    
    
    def test_deprecated_helper_functions(self):
        """Test that deprecated helper functions still work but issue warnings."""
        from gmoo_sdk.helpers import (
            fortranHollerithStringHelper,
            cStringCompatibilityHelper,
            cStringHelper,
            error_nan
        )
        
        # Test deprecated functions issue warnings
        with pytest.warns(DeprecationWarning):
            result = fortranHollerithStringHelper("test", 32)
            assert len(result) == 32
        
        with pytest.warns(DeprecationWarning):
            result = cStringCompatibilityHelper("test")
            assert result.value.strip() == "test"
        
        with pytest.warns(DeprecationWarning):
            result = cStringHelper("test")
            assert "test" in result.value
        
        # Test error_nan
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError):
                error_nan(np.nan, "test_value")