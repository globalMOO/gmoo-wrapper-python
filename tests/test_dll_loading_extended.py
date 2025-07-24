"""
Extended tests for DLL loading to improve coverage.

This module tests various DLL loading scenarios including error cases,
platform-specific behavior, and edge cases.
"""

import pytest
import os
import sys
import tempfile
from unittest.mock import patch, Mock
import ctypes

from gmoo_sdk.load_dll import load_dll


class TestDLLLoadingErrorCases:
    """Test DLL loading failure scenarios."""
    
    def test_missing_dll_file(self):
        """Test loading a non-existent DLL file."""
        fake_path = "/path/to/nonexistent.dll"
        
        # The load_dll function should raise an error for non-existent files
        with pytest.raises((FileNotFoundError, OSError)) as exc_info:
            load_dll(fake_path)
        
        # The error message should indicate the file was not found
        error_msg = str(exc_info.value)
        # Just verify we got an exception - the exact message may vary by system
    
    
    def test_missing_dependencies(self):
        """Test DLL with missing dependencies (simulated)."""
        # This test uses mocking since we can't easily create a DLL with missing deps
        with patch('ctypes.CDLL') as mock_cdll:
            mock_cdll.side_effect = OSError("The specified module could not be found")
            
            with pytest.raises(Exception) as exc_info:
                load_dll("some_dll.dll")
            
            assert "module could not be found" in str(exc_info.value).lower()
    
    @patch('gmoo_sdk.load_dll.load_dotenv')
    def test_no_path_no_env(self, mock_load_dotenv):
        """Test loading with no path and no MOOLIB environment variable."""
        # Mock load_dotenv to not load anything from .env file
        mock_load_dotenv.return_value = None
        
        # Temporarily remove MOOLIB if it exists
        old_moolib = os.environ.get('MOOLIB')
        if 'MOOLIB' in os.environ:
            del os.environ['MOOLIB']
        
        try:
            with pytest.raises(Exception) as exc_info:
                load_dll()  # No path provided
            
            assert "MOOLIB" in str(exc_info.value) or "environment" in str(exc_info.value).lower()
        finally:
            # Restore MOOLIB if it existed
            if old_moolib:
                os.environ['MOOLIB'] = old_moolib
    
    def test_path_with_spaces(self):
        """Test loading DLL from path with spaces."""
        # Test that paths with spaces are handled correctly
        space_path = "C:\\Program Files\\My Application\\my dll.dll"
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            with patch('ctypes.CDLL') as mock_cdll:
                mock_dll = Mock()
                mock_cdll.return_value = mock_dll
                
                result = load_dll(space_path)
                
                # Should handle the path correctly
                mock_cdll.assert_called_once()
                assert result == mock_dll
    
    def test_auto_extension_handling(self):
        """Test that load_dll does NOT automatically add .dll extension."""
        base_path = "C:\\test\\mydll"
        
        # load_dll expects the full path including extension
        with pytest.raises((FileNotFoundError, OSError)):
            load_dll(base_path)  # Should fail because no .dll extension
    
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    def test_windows_dll_directories(self):
        """Test Windows-specific DLL directory handling."""
        dll_path = "test.dll"
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            with patch('os.add_dll_directory') as mock_add_dir:
                with patch('ctypes.CDLL') as mock_cdll:
                    mock_cdll.return_value = Mock()
                    
                    # Set Intel MPI environment variable
                    with patch.dict(os.environ, {'I_MPI_ROOT': 'C:\\Intel\\MPI'}):
                        load_dll(dll_path)
                        
                        # Should attempt to add Intel MPI directory
                        mock_add_dir.assert_called()
    
    def test_relative_path_handling(self):
        """Test loading DLL with relative path."""
        relative_path = "./mydll.dll"
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            with patch('ctypes.CDLL') as mock_cdll:
                mock_dll = Mock()
                mock_cdll.return_value = mock_dll
                
                result = load_dll(relative_path)
                
                # Should handle relative path
                assert result == mock_dll
    
    def test_unicode_path(self):
        """Test loading DLL with Unicode characters in path."""
        unicode_path = "C:\\Users\\用户\\dll文件.dll"
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            with patch('ctypes.CDLL') as mock_cdll:
                mock_dll = Mock()
                mock_cdll.return_value = mock_dll
                
                result = load_dll(unicode_path)
                
                # Should handle Unicode path
                assert result == mock_dll
    
    def test_network_path(self):
        """Test loading DLL from network path (UNC)."""
        unc_path = "\\\\server\\share\\folder\\mydll.dll"
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            with patch('ctypes.CDLL') as mock_cdll:
                mock_dll = Mock()
                mock_cdll.return_value = mock_dll
                
                result = load_dll(unc_path)
                
                # Should handle UNC path
                assert result == mock_dll