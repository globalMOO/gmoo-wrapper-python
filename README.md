# GMOO SDK

A Python SDK for Global Multi-Objective Optimization

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The GMOO SDK provides a Python interface to the GMOO (Global Multi-Objective Optimization) engine, enabling inverse model training and optimization workflows. The SDK wraps a high-performance native library (VSME.dll) to solve complex inverse design and multi-objective optimization problems.

### Key Features

- **Inverse Model Training**: Train models that learn the inverse mapping G(y)=x to find inputs for desired outputs
- **Multi-Objective Optimization**: Handle multiple objectives with various constraint types
- **Parallel Optimization**: Run multiple optimization pipes simultaneously for robust convergence
- **Mixed Variable Types**: Support for continuous, integer, logical, and categorical variables
- **Multiple Objective Types**: Exact matching, percentage/absolute error bounds, inequalities, min/max

## Installation

### Prerequisites

- Python 3.8 or higher
- Windows or Linux operating system
- Access to VSME.dll and a valid license file
- Intel MPI runtime (for Windows)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/python-wrapper-sidefork.git
cd python-wrapper-sidefork

# Install in development mode
pip install -e .

# Or install with all extras
pip install -e ".[dev,examples]"
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## Configuration

### Environment Setup

Create a `.env` file from the template:

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

## Quick Start

### Simple Example

```python
import numpy as np
from gmoo_sdk.load_dll import load_dll
from gmoo_sdk.dll_interface import GMOOAPI

# Define a simple model function
def quadratic_model(inputs):
    x1, x2 = inputs[0], inputs[1]
    return np.array([x1**2 + x2, x1 + x2**2])

# Load DLL and create GMOO instance
dll = load_dll()
model = GMOOAPI(
    vsme_windll=dll,
    vsme_input_filename="my_model",
    var_mins=[0.0, 0.0],
    var_maxs=[5.0, 5.0],
    num_output_vars=2,
    model_function=quadratic_model
)

# Train inverse model
model.development.load_vsme_name()
model.development.initialize_variables()
model.development.load_variable_types()
model.development.load_variable_limits()
model.development.design_agents()
model.development.design_cases()

# Get training cases
case_count = model.development.get_case_count()
training_inputs = []
training_outputs = []

for i in range(1, case_count + 1):
    inputs = model.development.poke_case_variables(i)
    outputs = quadratic_model(inputs)
    training_inputs.append(inputs)
    training_outputs.append(outputs)

# Load results and develop model
model.development.initialize_outcomes()
for i in range(1, case_count + 1):
    model.development.load_case_results(i, training_outputs[i-1])

model.development.develop_vsme()
model.development.export_vsme()
model.development.unload_vsme()

# Use for inverse optimization
model.application.load_model()
target_outputs = [4.0, 6.0]
# ... continue with optimization loop
```

## Examples

The `examples/` directory contains several demonstration scripts:

- **`minimal_example.py`**: Simplest possible example with a linear model
- **`simple_example.py`**: Complete workflow with a quadratic model
- **`multi_pipe_example.py`**: Demonstrates parallel optimization with multiple pipes
- **`example_suite_stateful.py`**: Comprehensive test suite using the stateful API

Run examples:

```bash
cd examples
python simple_example.py
```

## Project Structure

```
python-wrapper-sidefork/
├── src/
│   └── gmoo_sdk/          # Core SDK package
│       ├── dll_interface.py    # Low-level DLL wrapper
│       ├── development.py      # Model training operations
│       ├── application.py      # Optimization operations
│       ├── satisfaction.py     # Objective checking logic
│       └── helpers.py          # Utility functions
├── examples/              # Example scripts
│   ├── functions/         # Example model functions
│   └── configs/           # Test configurations
├── tests/                 # Test suite
├── docs/                  # Documentation
└── requirements/          # Dependency specifications
```

## Core Concepts

### Inverse Models vs Surrogate Models

GMOO learns **inverse models** G(y)=x that map from desired outputs to required inputs, not traditional forward surrogate models f(x)=y. This is ideal for design problems where you know what outputs you want and need to find the inputs that produce them.

### Objective Types

The SDK supports various objective types for different optimization goals:

| Code | Type | Description |
|------|------|-------------|
| 0 | Exact Match | Output must equal target within tolerance |
| 1 | Percentage Error | Output within ±X% of target |
| 2 | Absolute Error | Output within target ± bounds |
| 11-14 | Inequalities | Less than, greater than constraints |
| 21-22 | Min/Max | Minimize or maximize (incomplete) |

### Multi-Pipe Optimization

Run multiple optimization searches in parallel from different starting points to:
- Avoid local minima
- Increase robustness
- Find multiple solutions
- Compare convergence paths

## API Reference

### Main Classes

- **`GMOOAPI`**: Core interface to the DLL
- **`DevelopmentOperations`**: Methods for training inverse models
- **`ApplicationOperations`**: Methods for optimization
- **`GmooStatelessWrapper`**: Simplified stateless interface (limited functionality)

### Key Methods

Development (Training):
- `design_cases()`: Generate training cases
- `load_case_results()`: Load evaluated outputs
- `develop_vsme()`: Train the inverse model
- `export_vsme()`: Save model to file

Application (Optimization):
- `load_model()`: Load trained model
- `perform_inverse_iteration()`: Single optimization step
- `check_satisfaction()`: Check if objectives are met

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gmoo_sdk

# Run specific test
pytest tests/test_dll_loading.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure nothing is broken
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run code formatting
black src/gmoo_sdk

# Run linting
flake8 src/gmoo_sdk
```

## Troubleshooting

### Common Issues

1. **DLL Loading Errors**
   - Verify MOOLIB path includes filename and extension
   - Check Intel MPI runtime is installed (Windows)
   - Ensure license file path (MOOLIC) is set

2. **Import Errors**
   - Install package with `pip install -e .`
   - Check Python version is 3.8+

3. **Convergence Issues**
   - Try multiple pipes for complex problems
   - Adjust uncertainty bounds
   - Check if target outputs are achievable

For more help, see the [documentation](docs/) or [open an issue](https://github.com/yourusername/python-wrapper-sidefork/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on the GMOO/VSME optimization engine
- Examples use various test functions from optimization literature