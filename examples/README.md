# GMOO SDK Examples

This folder contains example scripts demonstrating how to use the GMOO SDK for various optimization tasks.

## Quick Start Examples

### `minimal_example.py`
The simplest possible example showing the core workflow:
- Defines a basic linear model (2 inputs → 2 outputs)
- Trains an inverse model
- Finds inputs that produce target outputs [8.0, 11.0]
- Uses exact matching for fast satisfaction

**Run it:**
```bash
python minimal_example.py
```

### `simple_example.py`
A more detailed example with comprehensive error handling:
- Uses a quadratic model function
- Demonstrates percentage-based satisfaction criteria
- Shows detailed progress during optimization
- Includes proper cleanup and error handling

**Run it:**
```bash
python simple_example.py
```

## Comprehensive Test Suites

### `example_suite_stateful.py`
Comprehensive test suite using the stateful API directly:
- Tests multiple optimization problems (linear, quadratic, nonlinear, constrained)
- Supports multiple optimization pipes for parallel starting points
- Benchmarks different problem types
- Can test multiple DLL versions

**Run it:**
```bash
python example_suite_stateful.py
```

### `example_suite_stateless.py`
Similar test suite using the stateless wrapper API:
- Same test problems as stateful version
- Uses simplified stateless interface
- Good for comparing API approaches

**Run it:**
```bash
python example_suite_stateless.py
```

## Helper Modules

### `example_functions.py`
Collection of test functions including:
- Linear models
- Quadratic models
- Nonlinear models (exponential, trigonometric)
- Constrained optimization problems
- Mixed variable type problems

### `check_satisfaction.py`
Satisfaction checking utilities:
- Supports various objective types (exact, percentage, absolute error, constraints)
- Will be moved to main SDK in future versions

### `example_configs.py`
Configuration definitions for all test problems:
- Problem parameters (bounds, variable types)
- Target values and satisfaction criteria
- Used by the test suites

## Prerequisites

Before running any example:

1. **Set up your environment** - Copy `.env.example` to `.env` and update paths:
   ```
   MOOLIB=/path/to/VSME.dll
   MOOLIC=/path/to/license.lic
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy python-dotenv
   ```

3. **Ensure DLL is accessible** - The VSME.dll and license file must be valid

## Understanding the Examples

### Basic Workflow
All examples follow the same pattern:
1. Define a model function (inputs → outputs)
2. Train an inverse model using the Development API
3. Use the Application API to find inputs for desired outputs

### Key Concepts
- **Inverse Model**: A fast approximation of your model function
- **Development Phase**: Training the inverse model with sample points
- **Application Phase**: Using the inverse model for optimization
- **Objective Types**: How to measure success (exact match, percentage error, etc.)

## Troubleshooting

If examples fail to run:
1. Check that MOOLIB and MOOLIC are set correctly in your `.env` file
2. Ensure the DLL file exists and is the correct architecture (32/64-bit)
3. Verify the license file is valid
4. Check that Intel MPI libraries are installed (Windows)