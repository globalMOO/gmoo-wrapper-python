# GMOO Wrapper Python Tests

This directory contains the test suite for the GMOO wrapper for Python. These tests verify the functionality of the wrapper, ensuring that it correctly interfaces with the GMOO DLL and provides reliable optimization capabilities.

## Test Suite Organization

### Core Tests

These tests focus on the fundamental functionality of the wrapper:

- **test_basic_functionality.py**: Tests core functionality like DLL loading, model creation, and basic operations.
- **test_memory_safe.py**: Tests proper memory management to avoid leaks and crashes.
- **test_final.py**: Comprehensive end-to-end test with robust error handling.

### Focused Tests

These tests target specific functionality areas:

- **test_application.py**: Tests application mode (using trained models).
- **test_design_agents_cases.py**: Tests the design of agents and cases.
- **test_development.py**: Tests the model development process, including categorical variables.
- **test_inverse_optimization.py**: Tests inverse optimization capabilities.
- **test_case_results_import.py**: Specifically addresses case results import issues.
- **test_integrated.py**: Tests integrated workflows combining development and application.
- **test_backup_diagnosis.py**: Diagnostic test for memory management and resource cleanup.

## Running Tests

### Controlling Log Output

By default, tests run with WARNING level logging to reduce verbose output. To see more detailed logs:

```bash
# Show INFO level logs
pytest tests/ -v --log-cli-level=INFO

# Show DEBUG level logs for troubleshooting
pytest tests/ -v --log-cli-level=DEBUG

# Disable all logs during test runs
pytest tests/ -v --log-cli-level=ERROR
```

When tests fail, INFO level logs are automatically captured and displayed.

### Running Individual Tests

Run a specific test file:
```bash
pytest tests/test_basic_functionality.py -v
```

Run a specific test function within a file:
```bash
pytest tests/test_development.py::test_categorical_variables -v
```

### Running Multiple Tests

Run all tests:
```bash
pytest tests/ -v
```

Run tests with a specific keyword pattern:
```bash
pytest tests/ -k "application or inverse" -v
```

### Test Coverage

The test suite is configured with comprehensive coverage reporting using pytest-cov. Coverage is automatically measured when running tests.

#### Basic Coverage Commands

Run tests with coverage (default configuration):
```bash
pytest tests/
```

This will generate:
- Terminal report with missing lines
- HTML report in `htmlcov/` directory
- XML report as `coverage.xml`

#### Using the Coverage Runner Script

A convenient `run_coverage.py` script is provided for advanced coverage scenarios:

```bash
# Run all tests with coverage and open HTML report
python run_coverage.py --html

# Run only quick tests
python run_coverage.py -q

# Run tests matching a pattern
python run_coverage.py -k "inverse"

# Run tests with a specific marker
python run_coverage.py -m nonlinearity

# Clean old coverage data and run
python run_coverage.py --clean

# Run tests in parallel (requires pytest-xdist)
python run_coverage.py --parallel

# Run without coverage
python run_coverage.py --no-cov

# Set custom coverage threshold
python run_coverage.py --cov-fail-under=80
```

#### Coverage Configuration

Coverage is configured in two places:
- `pytest.ini`: Integrated with pytest for automatic coverage during test runs
- `.coveragerc`: Standalone configuration for the coverage tool

Key coverage settings:
- **Source**: Only `gmoo_sdk` package is measured
- **Branch Coverage**: Enabled to track conditional paths
- **Minimum Coverage**: 70% (configurable)
- **Excluded Patterns**: Abstract methods, debug code, type checking blocks

#### Viewing Coverage Reports

1. **Terminal Report**: Automatically shown after test runs
2. **HTML Report**: Open `htmlcov/index.html` in a browser
3. **XML Report**: `coverage.xml` for CI/CD integration
4. **JSON Report**: `coverage.json` for custom tooling

#### Coverage Best Practices

1. **Aim for High Coverage**: Target 80%+ coverage for production code
2. **Focus on Branch Coverage**: Ensure all conditional paths are tested
3. **Review HTML Reports**: Identify untested edge cases visually
4. **Exclude Appropriately**: Use `# pragma: no cover` for truly untestable code
5. **Monitor Trends**: Track coverage over time in CI/CD

## Common Test Fixtures

The common test fixtures and utility functions are defined in `conftest.py`:

- **dll_path**: Fixture that returns the path to the GMOO DLL.
- **loaded_dll**: Fixture that loads the GMOO DLL.
- **simple_model**: Fixture that creates a basic GMOOAPI model for testing.
- **complex_model**: Fixture that creates a more complex GMOOAPI model with Rosenbrock function.
- **develop_model()**: Helper function to develop a model with robust error handling.
- **perform_inverse_optimization()**: Helper function for inverse optimization.

## Testing Best Practices

1. **Proper Resource Management**: Always use the `unload_vsme()` method to clean up resources, especially between development and application modes.

2. **Explicit Cleanup**: Use `finally` blocks to ensure cleanup happens even if tests fail:
   ```python
   try:
       # Test code here
   finally:
       model.development.unload_vsme()
       model.application.unload_vsme()
   ```

3. **Isolation**: Each test should be independent and not rely on side effects from other tests.

4. **Use Fixtures**: Leverage the predefined fixtures in `conftest.py` for consistent setup and teardown.

5. **Memory Management**: Be vigilant about releasing resources, especially when dealing with the DLL interface.

6. **Categorical Variables**: For categorical variables, remember that the minimum value must be 1.0 and the maximum value should match the number of categories.

7. **Error Handling**: Provide clear error messages and use pytest's assertion mechanisms for better error reporting.

## Common Issues and Solutions

1. **Memory Errors**: If you encounter memory errors when running multiple tests, it's usually due to improper cleanup. Make sure to call `unload_vsme()` in a `finally` block.

2. **Missing Files**: Some tests might create temporary files that others depend on. Run tests individually if you suspect file dependencies.

3. **Categorical Variables**: Categorical variables should have a minimum value of 1.0 (not 0.0) and a maximum value equal to the number of categories.

4. **Test Order Dependencies**: If tests need to run in a specific order, consider using pytest-ordering or refactoring to remove dependencies.

5. **DLL Not Found**: Ensure the `MOOLIB` environment variable is correctly set to the path of the GMOO DLL.

6. **Coverage Issues**:
   - **Import errors in coverage**: Make sure `gmoo_sdk` is installed in development mode: `pip install -e .`
   - **Missing coverage data**: Run `python run_coverage.py --clean` to clear old coverage data
   - **Parallel coverage**: If using `--parallel`, ensure pytest-xdist is installed
   - **Low coverage**: Check the HTML report to identify untested code paths

## Continuous Integration

When setting up CI/CD, use the XML coverage report:

```bash
# GitHub Actions example
pytest --cov=gmoo_sdk --cov-report=xml --cov-report=term

# Upload to Codecov
bash <(curl -s https://codecov.io/bash)
```

Or use the coverage runner:

```bash
python run_coverage.py --xml --cov-fail-under=70
```
