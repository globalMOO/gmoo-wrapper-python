import os
import ctypes
import platform
from contextlib import contextmanager
from dotenv import load_dotenv

# Track which paths have been added to avoid duplicates
added_paths = set()

def load_dll(dll_path=""):
    """
    Load the GMOO DLL with proper configuration.
    
    The function uses the following precedence for finding the DLL:
    1. Explicit dll_path parameter (if provided)
    2. MOOLIB environment variable (from system or .env file)
    3. Raises error with helpful message
    
    Parameters:
    -----------
    dll_path : str, optional
        Explicit path to the DLL file. If not provided, will check MOOLIB 
        environment variable. The path should include filename and extension.
    
    Returns:
    --------
    ctypes.CDLL
        The loaded DLL object.
        
    Raises:
    -------
    FileNotFoundError
        If neither dll_path parameter nor MOOLIB environment variable is set.
    ValueError
        If MOOLIC (license path) environment variable is not set.
    OSError
        If the DLL fails to load due to missing dependencies or other issues.
    """
    # Load environment variables from .env file if it exists
    load_dotenv()
    try:
        # Handle Intel MPI paths first
        intelRedist = os.environ.get('I_MPI_ROOT')
        
        if intelRedist:
            add_to_dll_path(intelRedist)
        else:
            default_path = r'C:\Program Files (x86)\Common Files\Intel\Shared Libraries\intel64'
            add_to_dll_path(default_path)
        
        # Handle MOOLIB path for the DLL location
        if not dll_path:
            dll_path = os.environ.get('MOOLIB')
        
        if not dll_path:
            raise FileNotFoundError("Environment variable MOOLIB and hard coded 'dll_path' are both not set. Please use either to provide the full path to the VSME.dll file (including filename and extension).")
        # Check if MOOLIC (license path) is set
        if not os.environ.get('MOOLIC'):
            raise ValueError(
                "MOOLIC environment variable not set. Please set it to the path of your license file.\n"
                "You can either:\n"
                "1. Set MOOLIC as an environment variable\n"
                "2. Create a .env file with MOOLIC=/path/to/license.lic\n"
                "3. Copy .env.example to .env and update the paths"
            )
        
        # Load the DLL
        dll = ctypes.CDLL(dll_path)
        return dll

    except OSError as e:
        raise OSError(
            f"Failed to load the DLL: {e}\n"
            f"DLL path: {dll_path}\n"
            f"Make sure the DLL exists and all dependencies are available."
        )

def add_to_dll_path(path):
    global added_paths
    if path and path not in added_paths and os.path.exists(path):
        # Check if the operating system is Windows
        if platform.system() == "Windows":
            try:
                os.add_dll_directory(path)
                added_paths.add(path)
            except Exception as e:
                # Silently ignore if we can't add the directory
                pass
        # Check if the operating system is Linux
        elif platform.system() == "Linux":
            # Modify LD_LIBRARY_PATH for Linux
            current_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
            os.environ['LD_LIBRARY_PATH'] = path + ":" + current_ld_library_path
            added_paths.add(path)

class DLLEnvironmentWrapper:
    def __init__(self, dll_path, license_path):
        self.dll_path = dll_path
        self.license_path = license_path
        self._original_env = None
        self._dll = None
    
    def __enter__(self):
        # Method 1: Temporarily set environment variable
        self._original_env = os.environ.get('MOOLIC')
        os.environ['MOOLIC'] = self.license_path
        
        # Load the DLL
        self._dll = ctypes.CDLL(self.dll_path)
        return self._dll
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original environment
        if self._original_env is None:
            del os.environ['MOOLIC']
        else:
            os.environ['MOOLIC'] = self._original_env

@contextmanager
def temporary_env_dll(dll_path, license_path):
    """Alternative context manager if you prefer function syntax"""
    original_env = os.environ.get('MOOLIC')
    os.environ['MOOLIC'] = license_path
    dll = ctypes.CDLL(dll_path)
    try:
        yield dll
    finally:
        if original_env is None:
            del os.environ['MOOLIC']
        else:
            os.environ['MOOLIC'] = original_env