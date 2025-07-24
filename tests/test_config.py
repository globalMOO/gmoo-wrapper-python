"""
Test Configuration File

Update the paths in this file to match your local environment.
This provides a central place to configure test paths without modifying
individual test files.
"""

import os

# Default DLL path - UPDATE THIS to your VSME.dll location
# You should set this in your .env file instead: MOOLIB=/path/to/VSME.dll
DEFAULT_DLL_PATH = None  # Set via environment variable

# Alternative paths to try if the default doesn't exist
ALTERNATIVE_DLL_PATHS = [
    r"C:\Program Files\GMOO\VSME.dll",
    r"C:\Program Files (x86)\GMOO\VSME.dll",
    r"D:\GMOO\VSME.dll",
    # Windows Docker container path
    r"C:\app\dll\VSME.dll",
    # Add more paths as needed
]

# License file path (optional)
# You should set this in your .env file instead: MOOLIC=/path/to/license.lic
DEFAULT_LICENSE_PATH = None  # Set via environment variable

# Intel MPI paths
INTEL_MPI_PATHS = [
    r"C:\Program Files (x86)\Intel\oneAPI\mpi\latest",
    r"C:\Program Files\Intel\oneAPI\mpi\latest",
    r"C:\Program Files (x86)\Common Files\Intel\Shared Libraries\intel64",
]


def get_dll_path():
    """
    Get the DLL path from environment or fallback to configured defaults.
    
    Returns:
        str or None: Path to the DLL if found, None otherwise
    """
    # First try environment variables
    dll_path = os.environ.get('MOOLIB') or os.environ.get('TEST_DLL_PATH')
    if dll_path and os.path.exists(dll_path):
        return dll_path
    
    # Try the default path
    if DEFAULT_DLL_PATH and os.path.exists(DEFAULT_DLL_PATH):
        return DEFAULT_DLL_PATH
    
    # Try alternative paths
    for path in ALTERNATIVE_DLL_PATHS:
        if os.path.exists(path):
            return path
    
    return None


def get_license_path():
    """
    Get the license path from environment or fallback to default.
    
    Returns:
        str or None: Path to the license if found, None otherwise
    """
    license_path = os.environ.get('MOOLIC')
    if license_path and os.path.exists(license_path):
        return license_path
    
    if DEFAULT_LICENSE_PATH and os.path.exists(DEFAULT_LICENSE_PATH):
        return DEFAULT_LICENSE_PATH
    
    return None


def get_intel_mpi_path():
    """
    Get Intel MPI path from environment or search common locations.
    
    Returns:
        str or None: Path to Intel MPI if found, None otherwise
    """
    mpi_path = os.environ.get('I_MPI_ROOT')
    if mpi_path and os.path.exists(mpi_path):
        return mpi_path
    
    for path in INTEL_MPI_PATHS:
        if os.path.exists(path):
            return path
    
    return None


def setup_test_environment():
    """
    Set up the test environment with discovered paths.
    This can be called at the beginning of test runs.
    """
    dll_path = get_dll_path()
    if dll_path and not os.environ.get('MOOLIB'):
        os.environ['MOOLIB'] = dll_path
        print(f"✅ Set MOOLIB to: {dll_path}")
    
    license_path = get_license_path()
    if license_path and not os.environ.get('MOOLIC'):
        os.environ['MOOLIC'] = license_path
        print(f"✅ Set MOOLIC to: {license_path}")
    
    mpi_path = get_intel_mpi_path()
    if mpi_path and not os.environ.get('I_MPI_ROOT'):
        os.environ['I_MPI_ROOT'] = mpi_path
        print(f"✅ Set I_MPI_ROOT to: {mpi_path}")


# Automatically set up environment when this module is imported
if __name__ != "__main__":
    setup_test_environment()