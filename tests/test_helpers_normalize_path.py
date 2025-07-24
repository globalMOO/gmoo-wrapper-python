"""
Tests for normalize_path edge cases in helpers.py.
"""

import pytest
import platform
from unittest.mock import patch
from gmoo_sdk.helpers import normalize_path


class TestNormalizePathEdgeCases:
    """Test edge cases for the normalize_path function."""
    
    def test_normalize_path_basic(self):
        """Test basic path normalization."""
        # Mock Windows platform
        with patch('platform.system', return_value='Windows'):
            result = normalize_path("C:\\Users\\test", "file.txt")
            assert result == "C:\\Users\\test\\file.txt"
        
        # Mock Unix platform
        with patch('platform.system', return_value='Linux'):
            result = normalize_path("/home/user", "file.txt")
            assert result == "/home/user/file.txt"
    
    def test_normalize_path_trailing_separator(self):
        """Test paths with trailing separators."""
        # Windows with trailing backslash
        with patch('platform.system', return_value='Windows'):
            result = normalize_path("C:\\Users\\test\\", "file.txt")
            assert result == "C:\\Users\\test\\file.txt"
        
        # Unix with trailing slash
        with patch('platform.system', return_value='Linux'):
            result = normalize_path("/home/user/", "file.txt")
            assert result == "/home/user/file.txt"
    
    def test_normalize_path_mixed_separators(self):
        """Test paths with mixed separators."""
        # Windows should still use backslash
        with patch('platform.system', return_value='Windows'):
            result = normalize_path("C:/Users/test", "file.txt")
            assert result == "C:/Users/test\\file.txt"
        
        # Unix with Windows-style path
        with patch('platform.system', return_value='Linux'):
            result = normalize_path("\\home\\user", "file.txt")
            assert result == "\\home\\user/file.txt"
    
    def test_normalize_path_empty_base_dir(self):
        """Test with empty base directory."""
        with patch('platform.system', return_value='Windows'):
            result = normalize_path("", "file.txt")
            assert result == "\\file.txt"
        
        with patch('platform.system', return_value='Linux'):
            result = normalize_path("", "file.txt")
            assert result == "/file.txt"
    
    def test_normalize_path_empty_filename(self):
        """Test with empty filename."""
        with patch('platform.system', return_value='Windows'):
            result = normalize_path("C:\\Users\\test", "")
            assert result == "C:\\Users\\test\\"
        
        with patch('platform.system', return_value='Linux'):
            result = normalize_path("/home/user", "")
            assert result == "/home/user/"
    
    def test_normalize_path_both_empty(self):
        """Test with both base_dir and filename empty."""
        with patch('platform.system', return_value='Windows'):
            result = normalize_path("", "")
            assert result == "\\"
        
        with patch('platform.system', return_value='Linux'):
            result = normalize_path("", "")
            assert result == "/"
    
    def test_normalize_path_relative_paths(self):
        """Test with relative path components."""
        with patch('platform.system', return_value='Windows'):
            result = normalize_path(".", "file.txt")
            assert result == ".\\file.txt"
            
            result = normalize_path("..", "file.txt")
            assert result == "..\\file.txt"
        
        with patch('platform.system', return_value='Linux'):
            result = normalize_path(".", "file.txt")
            assert result == "./file.txt"
            
            result = normalize_path("..", "file.txt")
            assert result == "../file.txt"
    
    def test_normalize_path_unicode(self):
        """Test with unicode characters in paths."""
        with patch('platform.system', return_value='Windows'):
            result = normalize_path("C:\\Users\\测试", "文件.txt")
            assert result == "C:\\Users\\测试\\文件.txt"
        
        with patch('platform.system', return_value='Linux'):
            result = normalize_path("/home/用户", "ファイル.txt")
            assert result == "/home/用户/ファイル.txt"
    
    def test_normalize_path_special_characters(self):
        """Test with special characters in paths."""
        with patch('platform.system', return_value='Windows'):
            result = normalize_path("C:\\My Documents", "file (1).txt")
            assert result == "C:\\My Documents\\file (1).txt"
            
            result = normalize_path("C:\\Test@2024", "data#1.csv")
            assert result == "C:\\Test@2024\\data#1.csv"
        
        with patch('platform.system', return_value='Linux'):
            result = normalize_path("/my documents", "file (1).txt")
            assert result == "/my documents/file (1).txt"
    
    def test_normalize_path_multiple_trailing_separators(self):
        """Test with multiple trailing separators."""
        with patch('platform.system', return_value='Windows'):
            # Only removes one trailing separator
            result = normalize_path("C:\\Users\\test\\\\", "file.txt")
            assert result == "C:\\Users\\test\\\\file.txt"
        
        with patch('platform.system', return_value='Linux'):
            result = normalize_path("/home/user//", "file.txt")
            assert result == "/home/user//file.txt"
    
    def test_normalize_path_mac_platform(self):
        """Test on macOS platform."""
        with patch('platform.system', return_value='Darwin'):
            result = normalize_path("/Users/test", "file.txt")
            assert result == "/Users/test/file.txt"