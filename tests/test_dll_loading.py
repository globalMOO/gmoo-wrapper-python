"""
Quick Test Script for DLL Loading

This script tests whether the DLL can be loaded correctly with the updated path handling.
"""

import os
import sys

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gmoo_sdk.load_dll import load_dll
from test_config import get_dll_path

def test_dll_loading():
    """Test DLL loading with different scenarios."""
    
    print("Testing DLL loading...")
    print("-" * 50)
    
    dll_loaded = False
    
    # Test 1: Try loading with explicit path
    # Use the centralized configuration
    test_path = get_dll_path()
    if not test_path:
        print("❌ No DLL path found!")
        print("   Please either:")
        print("   1. Set MOOLIB or TEST_DLL_PATH environment variable")
        print("   2. Update DEFAULT_DLL_PATH in test_config.py")
    
    print(f"Test 1: Loading DLL from explicit path")
    print(f"Path: {test_path}")
    if test_path:
        print(f"File exists: {os.path.exists(test_path)}")
    else:
        print("File exists: N/A (no path provided)")
    
    try:
        dll = load_dll(test_path)
        print("✅ SUCCESS: DLL loaded successfully with explicit path")
        dll_loaded = True
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    # Test 2: Try loading without extension (should add .dll automatically)
    if not dll_loaded:
        test_path_no_ext = r"C:\Users\mfreeman\Documents\GitHub\gmoo_sdk\examples\20250513_VSME"
        
        print(f"\nTest 2: Loading DLL without extension (should auto-add .dll)")
        print(f"Path: {test_path_no_ext}")
        
        try:
            dll = load_dll(test_path_no_ext)
            print("✅ SUCCESS: DLL loaded successfully with auto-added extension")
            dll_loaded = True
        except Exception as e:
            print(f"❌ FAILED: {e}")
    
    # Test 3: Try loading from environment variable
    if not dll_loaded:
        print(f"\nTest 3: Loading DLL from MOOLIB environment variable")
        moolib_path = os.environ.get('MOOLIB')
        print(f"MOOLIB: {moolib_path}")
        
        if moolib_path:
            print(f"File exists: {os.path.exists(moolib_path)}")
            try:
                dll = load_dll()  # No path provided, should use MOOLIB
                print("✅ SUCCESS: DLL loaded successfully from MOOLIB")
                dll_loaded = True
            except Exception as e:
                print(f"❌ FAILED: {e}")
        else:
            print("MOOLIB environment variable not set")
    
    if not dll_loaded:
        print("\n❌ All tests failed - please check your DLL path and environment setup")
    
    # Use assertion instead of return
    assert dll_loaded, "Failed to load DLL from any source"

if __name__ == "__main__":
    try:
        test_dll_loading()
        print("\n🎉 DLL loading is working correctly!")
        print("You can now run the example suites:")
        print("  python web_examples.py")
        print("  python example_suite_stateful.py")
    except AssertionError:
        print("\n💡 To fix this:")
        print("1. Make sure the DLL file exists at the specified path")
        print("2. Include the .dll extension in your path")
        print("3. Set the MOOLIB environment variable:")
        print("   set MOOLIB=C:\\path\\to\\your\\VSME.dll")
