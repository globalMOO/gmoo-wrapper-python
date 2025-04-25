# GMOO SDK

Global Multi-Objective Optimization Software Development Kit

## Overview

The GMOO SDK provides a Python interface to the GMOO (Global Multi-Objective Optimization) DLL, facilitating workflows for surrogate model training and inverse design optimization.

The package simplifies common optimization tasks by abstracting away the details of DLL management and memory handling, providing a clean, object-oriented API for model development and application.

Repository: [https://github.com/globalMOO/gmoo-wrapper-python](https://github.com/globalMOO/gmoo-wrapper-python)

## Installation

### Prerequisites

- Python 3.10 or newer
- The GMOO DLL/shared library (platform-specific)
- Valid license file for the GMOO library

### Basic Installation

```bash
# Install the package with pip
pip install gmoo_sdk

# Or install from source
git clone https://github.com/globalMOO/gmoo-wrapper-python.git
cd gmoo-wrapper-python
pip install -e .
```

### DLL Configuration

The GMOO SDK requires access to the platform-specific DLL/shared library. Configure your environment as follows:

#### Windows

1. Set environment variables:
   ```
   set MOOLIB=C:\path\to\VSME.dll
   set MOOLIC=C:\path\to\license\file
   ```

2. Add Intel MPI paths (if needed):
   ```
   set I_MPI_ROOT=C:\Program Files (x86)\Common Files\Intel\Shared Libraries\intel64
   ```

#### Linux/macOS

1. Set environment variables:
   ```
   export MOOLIB=/path/to/libvsme.so
   export MOOLIC=/path/to/license/file
   ```

2. Add to library path (Linux):
   ```
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/lib/directory
   ```

## Quick Start

```python
import numpy as np
from gmoo_sdk.gmoo_encapsulation import get_development_cases_encapsulation, load_development_cases_encapsulation, inverse_encapsulation

# Define your model function
def my_model(input_arr):
    # Your model implementation
    x = np.array(input_arr)
    return np.array([x[0]**2, x[1]**2])

# Configure model parameters
var_mins = [0.0, 0.0]
var_maxs = [10.0, 10.0]
num_outputs = 2
filename = "my_model"

# Train surrogate model
input_cases = get_development_cases_encapsulation(var_mins, var_maxs, filename)
output_cases = [my_model(case) for case in input_cases]
load_development_cases_encapsulation(output_cases, var_mins, var_maxs, num_outputs, filename)

# Inverse design to find inputs for target outputs
target_outputs = [4.0, 9.0]
current_inputs = [1.0, 1.0]
current_outputs = my_model(current_inputs)

for i in range(10):
    next_inputs, l1norm, _, _ = inverse_encapsulation(
        target_outputs, var_mins, var_maxs, num_outputs,
        current_inputs, current_outputs, filename
    )
    current_inputs = next_inputs
    current_outputs = my_model(current_inputs)
    print(f"Iteration {i+1}: Inputs = {current_inputs}, Outputs = {current_outputs}")
```

## License

This software is licensed under the MIT License. See the LICENSE file for details.