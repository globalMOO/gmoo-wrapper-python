"""
Simple Test for Stateless Wrapper

This script tests the basic functionality of the stateless wrapper.
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import test configuration
from test_config import get_dll_path

def test_stateless_wrapper():
    """Test basic stateless wrapper functionality."""
    
    print("Testing Stateless Wrapper...")
    print("-" * 50)
    
    try:
        from gmoo_sdk.stateless_wrapper import GmooStatelessWrapper
        print("✅ Successfully imported GmooStatelessWrapper")
    except Exception as e:
        print(f"❌ Failed to import: {e}")
        assert False, f"Failed to import GmooStatelessWrapper: {e}"
    
    # Test creating wrapper instance
    try:
        # Use the centralized configuration
        dll_path = get_dll_path()
        
        if not dll_path:
            print("⚠️  No DLL path found!")
            print("   Please either:")
            print("   1. Set MOOLIB or TEST_DLL_PATH environment variable")
            print("   2. Update DEFAULT_DLL_PATH in test_config.py")
            print("   Skipping wrapper instance test")
            return
        
        print(f"Using DLL path: {dll_path}")
            
        wrapper = GmooStatelessWrapper(
            minimum_list=[0.0, 0.0],
            maximum_list=[10.0, 10.0],
            input_type_list=[1, 1],
            category_list=[[], []],
            filename_prefix="test",
            output_directory=".",
            dll_path=dll_path,
            num_outcomes=2
        )
        print("✅ Successfully created GmooStatelessWrapper instance")
        
        # Test that bridge methods exist
        bridge_methods = [
            'dev_unload_vsme', 'app_unload_vsme', 'load_model',
            'poke_variable_dev_limit_min', 'poke_variable_dev_limit_max',
            'app_load_variable_limit_min', 'app_load_variable_limit_max',
            'app_init_variables', 'load_learned_case', 'poke_dimensions10',
            'poke_learned_case', 'inverse_single_iteration'
        ]
        
        missing_methods = []
        for method in bridge_methods:
            if not hasattr(wrapper, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Missing bridge methods: {missing_methods}")
            assert False, f"Missing bridge methods: {missing_methods}"
        else:
            print("✅ All bridge methods are present")
        
    except Exception as e:
        print(f"❌ Failed to create wrapper instance: {e}")
        assert False, f"Failed to create wrapper instance: {e}"

if __name__ == "__main__":
    try:
        test_stateless_wrapper()
        print("\n🎉 Stateless wrapper is working correctly!")
        print("You can now run the example suites:")
        print("  python web_examples.py")
    except AssertionError as e:
        print("\n💡 There are issues with the stateless wrapper that need to be fixed.")
        print(f"Error: {e}")
