# Symphony Trading System Tests

This directory contains tests for the Symphony Trading System.

## Test Structure

- `test_logging.py`: Tests for the logging functionality
- Add other test files as they are created

## Running Tests

Run individual tests from the project root directory:

```bash
python -m tests.test_logging  # Test the logging functionality
```

## Guidelines for Writing Tests

- Create separate test files for each module or functionality
- Name test files with the prefix `test_`
- Place tests in the `tests` directory
- Import the module being tested using relative imports

## Test Coverage

Current test coverage includes:

1. Logging functionality:
   - Basic logging
   - Improved logging configuration
   - Symphony backtester logging integration
