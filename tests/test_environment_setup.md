# Test Environment Setup Guide

## Configuration Options

You have two options for configuring the test environment:

### Option 1: Update test_config.py (Recommended)
Edit `tests/test_config.py` and update the `DEFAULT_DLL_PATH` variable to point to your VSME.dll file:

```python
# Default DLL path - UPDATE THIS to your VSME.dll location
DEFAULT_DLL_PATH = r"C:\path\to\your\VSME.dll"
```

This is the easiest approach and works well when you can't modify environment variables.

### Option 2: Set Environment Variables

To run the GMOO SDK tests successfully, you need to set up the following environment variables:

### 1. MOOLIB (Required)
Points to the VSME DLL file location.
```bash
# Windows
set MOOLIB=C:\path\to\your\VSME.dll

# Linux/Mac
export MOOLIB=/path/to/your/VSME.dll
```

### 2. MOOLIC (Optional but Recommended)
Points to the GMOO license file. Some functionality may be limited without it.
```bash
# Windows
set MOOLIC=C:\path\to\your\license.lic

# Linux/Mac
export MOOLIC=/path/to/your/license.lic
```

### 3. I_MPI_ROOT (Optional)
Points to Intel MPI installation directory. Required if using Intel MPI features.
```bash
# Windows
set I_MPI_ROOT=C:\Program Files (x86)\Intel\oneAPI\mpi\latest

# Linux/Mac
export I_MPI_ROOT=/opt/intel/oneapi/mpi/latest
```

### 4. TEST_DLL_PATH (For Testing)
Alternative DLL path for test scenarios.
```bash
# Windows
set TEST_DLL_PATH=C:\path\to\test\VSME.dll

# Linux/Mac
export TEST_DLL_PATH=/path/to/test/VSME.dll
```

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run With Coverage
```bash
pytest tests/ --cov=gmoo_sdk --cov-report=html
```

### Run Specific Test Categories
```bash
# Quick tests only
pytest tests/ -m quick

# Skip slow tests
pytest tests/ -m "not slow"

# Run only unit tests (no DLL required)
pytest tests/ -m "not requires_dll"
```

## Common Issues

### 1. DLL Not Found
- Ensure MOOLIB environment variable is set correctly
- Verify the DLL file exists at the specified path
- Check file permissions

### 2. Intel MPI Dependencies
- Install Intel MPI runtime or Intel OneAPI
- Set I_MPI_ROOT to the installation directory
- On Windows, may need to add Intel libraries to PATH

### 3. File Cleanup Issues
- Some tests create .gmoo files that may not be cleaned up properly
- Run `pytest tests/ --clean` to force cleanup (if implemented)
- Manually delete test_*.gmoo files if needed

### 4. License Issues
- Warning messages about MOOLIC are expected if no license is set
- Some advanced features may be disabled without a license

## CI/CD Configuration

For GitHub Actions or other CI/CD systems:

```yaml
# Example GitHub Actions setup
env:
  MOOLIB: ${{ secrets.GMOO_DLL_PATH }}
  MOOLIC: ${{ secrets.GMOO_LICENSE_PATH }}
  I_MPI_ROOT: /opt/intel/oneapi/mpi/latest
```

## Troubleshooting

1. **Enable verbose logging:**
   ```bash
   pytest tests/ -v -s --log-cli-level=DEBUG
   ```

2. **Check environment setup:**
   ```bash
   python -c "import os; print('MOOLIB:', os.environ.get('MOOLIB', 'NOT SET'))"
   ```

3. **Test DLL loading directly:**
   ```bash
   python tests/test_dll_loading.py
   ```