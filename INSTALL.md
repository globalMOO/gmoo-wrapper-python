# Installation Guide for GMOO SDK

This guide explains how to install and set up the GMOO SDK with the new project structure.

## Prerequisites

- Python 3.8 or higher
- Access to the VSME.dll file and a valid license
- Windows or Linux operating system

## Installation Options

### 1. Development Installation (Recommended for contributors)

For development work, install the package in editable mode:

```bash
# Clone the repository
git clone https://github.com/yourusername/python-wrapper-sidefork.git
cd python-wrapper-sidefork

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode without development dependencies for straightforward application
pip install -e .

# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Or install everything (dev + examples)
pip install -e ".[dev,examples]"
```

### 2. User Installation

For regular users who just want to use the SDK:

```bash
# Install from the repository
pip install .

# Or with example dependencies
pip install ".[examples]"
```

### 3. Installing from a Distribution

If you have a wheel or tarball:

```bash
pip install gmoo-sdk-2.0.0-py3-none-any.whl
```

## Configuration

### 1. Environment Variables

Create a `.env` file in your project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` to set your paths:

```ini
# Path to the VSME.dll file (including filename)
MOOLIB=/path/to/your/VSME.dll

# Path to the license file
MOOLIC=/path/to/your/license.lic
```

### 2. Alternative: System Environment Variables

You can also set these as system environment variables:

**Windows:**
```cmd
set MOOLIB=C:\path\to\VSME.dll
set MOOLIC=C:\path\to\license.lic
```

**Linux/Mac:**
```bash
export MOOLIB=/path/to/VSME.dll
export MOOLIC=/path/to/license.lic
```

## Running Examples

After installation, you can run the examples:

```bash
# Change to examples directory
cd examples

# Run a simple example
python simple_example.py

# Run the full test suite
python example_suite_stateful.py
```

## Running Tests

To run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gmoo_sdk

# Run specific test file
pytest tests/test_dll_loading.py

# Run tests with specific marker
pytest -m "not requires_dll"  # Skip tests that need the DLL
```

## Project Structure

The new structure follows Python packaging best practices:

```
python-wrapper-sidefork/
├── src/
│   └── gmoo_sdk/          # Core package code
├── examples/              # Example scripts and configurations
│   ├── functions/         # Example model functions
│   └── configs/           # Test configurations
├── tests/                 # Test suite
├── docs/                  # Documentation
├── requirements/          # Dependency specifications
│   ├── base.txt          # Core dependencies
│   ├── dev.txt           # Development dependencies
│   └── examples.txt      # Example dependencies
├── pyproject.toml        # Modern Python packaging configuration
├── setup.py              # Backward compatibility
└── MANIFEST.in           # File inclusion rules
```

## Troubleshooting

### Import Errors

If you encounter import errors when running examples directly:

1. Make sure you've installed the package: `pip install -e .`
2. Or run examples as modules: `python -m examples.simple_example`

### DLL Loading Issues

If the DLL fails to load:

1. Verify the MOOLIB path is correct and includes the filename
2. Check that all DLL dependencies are available (Intel MPI runtime)
3. Ensure the license file path (MOOLIC) is set correctly

### Missing Dependencies

If you get errors about missing packages:

```bash
# Install base requirements
pip install -r requirements.txt

# Or install with extras
pip install -e ".[dev,examples]"
```

## Building a Distribution

To create a distributable package:

```bash
# Install build tools
pip install build

# Build the package
python -m build

# This creates:
# - dist/gmoo_sdk-2.0.0-py3-none-any.whl
# - dist/gmoo-sdk-2.0.0.tar.gz
```

## Development Workflow

1. Create a virtual environment
2. Install in editable mode with dev dependencies
3. Make your changes
4. Run tests to ensure nothing is broken
5. Format code with black: `black src/gmoo_sdk`
6. Check code with flake8: `flake8 src/gmoo_sdk`
7. Commit your changes

## Getting Help

If you encounter issues:

1. Check the examples in the `examples/` directory
2. Read the API documentation in `docs/`
3. Check existing issues on GitHub
4. Create a new issue with a minimal reproducible example