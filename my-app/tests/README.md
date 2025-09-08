# Testing Documentation for my-app

This directory contains comprehensive unit tests for the my-app project. The tests are designed to serve dual purposes:

1. **Unit Testing**: Verify that all functions work correctly
2. **Documentation**: Provide clear examples of how each function should be used

## Directory Structure

```
tests/
├── __init__.py           # Package initialization and documentation
├── conftest.py           # pytest configuration and shared fixtures  
├── test_classification.py # Tests for classification.py module
├── test_helpers.py       # Common test utilities and helpers
└── README.md            # This file - testing documentation
```

## Running Tests

### Prerequisites

Install testing dependencies:
```bash
pip install pytest pytest-mock numpy
```

### Basic Test Commands

```bash
# Run all tests
pytest tests/

# Run tests with verbose output
pytest -v tests/

# Run specific test file
pytest tests/test_classification.py

# Run specific test class
pytest tests/test_classification.py::TestCategoryClass

# Run specific test function
pytest tests/test_classification.py::TestCategoryClass::test_category_creation_with_valid_data

# Run tests with coverage
pytest --cov=classification tests/

# Run tests and stop on first failure
pytest -x tests/
```

### Advanced Test Commands

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto tests/

# Run only failed tests from last run
pytest --lf tests/

# Run tests matching a pattern
pytest -k "embed" tests/

# Generate HTML coverage report
pytest --cov=classification --cov-report=html tests/
```

## Test Organization

### test_classification.py

This is the main test file for the `classification.py` module. It's organized into test classes that mirror the structure of the code:

- **TestCategoryClass**: Tests the Category data class
- **TestLoadCategories**: Tests the category loading function
- **TestEmbed**: Tests the embedding generation function
- **TestNarrowDownCategories**: Tests the similarity-based filtering
- **TestPickBestCategory**: Tests the LLM-based final selection
- **TestPickCategory**: Tests the complete classification pipeline
- **TestErrorHandling**: Tests error conditions and edge cases

Each test method includes:
- A descriptive docstring explaining the scenario
- Clear input/output examples
- Explanation of what the function does
- Verification of expected behavior

### Example Test Structure

```python
def test_function_name_describes_scenario(self):
    """
    EXAMPLE: Clear description of what this test demonstrates
    
    Input: Specific example of input data
    Output: Expected output and why
    """
    # Arrange - set up test data
    input_data = "example input"
    
    # Act - call the function
    result = function_to_test(input_data)
    
    # Assert - verify the results
    assert result == expected_output
    assert isinstance(result, expected_type)
```

### test_helpers.py

Common utilities that can be used across multiple test files:

- **create_test_category()**: Generate Category objects for testing
- **create_test_categories()**: Generate lists of test categories
- **assert_embedding_format()**: Validate embedding vector format
- **MockGenAIClient**: Mock the Google GenAI API for testing
- **Test data generators**: Create edge case test data

## Testing Philosophy

### Human-Readable Tests

Our tests are written to be easily understood by developers who are new to the codebase. Each test:

1. **Has a descriptive name** that explains the scenario being tested
2. **Includes clear documentation** showing input/output examples
3. **Explains the business logic** being tested
4. **Uses realistic data** that represents actual use cases

### Test Categories

We write tests for different scenarios:

1. **Happy Path**: Normal usage with expected inputs
2. **Edge Cases**: Boundary conditions and unusual inputs
3. **Error Conditions**: How the system handles failures
4. **Integration**: How components work together
5. **Performance**: Basic performance characteristics

### Mocking Strategy

We mock external dependencies to:

1. **Avoid API costs**: Don't make real calls to Google GenAI
2. **Ensure reproducibility**: Tests give consistent results
3. **Test error conditions**: Simulate API failures
4. **Speed up tests**: No network delays

External dependencies we mock:
- Google GenAI embedding API
- BAML classification service
- Environment variables

## Adding Tests for New Modules

When you add a new Python file to the project, create a corresponding test file:

### 1. Create the test file

```bash
# For a new module called my_new_module.py
touch tests/test_my_new_module.py
```

### 2. Use the standard template

```python
"""
Unit tests for my_new_module.py

This test file provides comprehensive coverage of all functions in my_new_module.
Each test serves as both verification and documentation.
"""

import pytest
from unittest.mock import Mock, patch

# Import functions to test
from my_new_module import function_to_test


class TestFunctionToTest:
    """
    Tests for the function_to_test function
    
    Explain what this function does and why it's important.
    """
    
    def test_function_with_valid_input(self):
        """
        EXAMPLE: Description of this test scenario
        
        Input: Example input
        Output: Expected output
        """
        # Test implementation here
        pass
```

### 3. Add to conftest.py if needed

If your new module needs special fixtures or mocks, add them to `conftest.py`:

```python
@pytest.fixture
def my_new_fixture():
    """Fixture for testing my new module."""
    return test_data
```

### 4. Update this README

Add documentation about your new test file and any special testing considerations.

## Best Practices

### Writing Good Tests

1. **One concept per test**: Each test should verify one specific behavior
2. **Descriptive names**: Test names should clearly describe the scenario
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Use fixtures**: Share common test setup via pytest fixtures
5. **Mock external dependencies**: Don't rely on external services
6. **Test edge cases**: Include boundary conditions and error cases

### Test Data

1. **Use realistic data**: Test with data that resembles production usage
2. **Include edge cases**: Empty strings, very long text, special characters
3. **Create reusable factories**: Use helper functions to generate test data
4. **Document data choices**: Explain why specific test data was chosen

### Documentation

1. **Every test has a docstring**: Explain what the test does
2. **Include examples**: Show input/output examples in docstrings
3. **Explain business logic**: Help other developers understand the purpose
4. **Keep it current**: Update tests when functionality changes

## Continuous Integration

These tests are designed to run in CI/CD pipelines. They:

1. **Are fast**: Use mocks to avoid slow external calls
2. **Are reliable**: Don't depend on external services or random data
3. **Are comprehensive**: Cover both happy paths and error conditions
4. **Provide clear output**: Failures include helpful error messages

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running tests from the correct directory
2. **Missing dependencies**: Install all required packages with `pip install -r requirements.txt`
3. **Mock issues**: Check that mocks are set up correctly for external dependencies
4. **Environment variables**: Tests should work without real API keys

### Getting Help

1. Look at existing tests for examples
2. Check the docstrings for usage patterns
3. Run tests with `-v` flag for more detailed output
4. Use `pytest --pdb` to debug failing tests

## Future Enhancements

Ideas for improving the test suite:

1. **Integration tests**: Test with real (but limited) API calls
2. **Performance tests**: Measure execution time and memory usage
3. **Property-based testing**: Use hypothesis for generating test cases
4. **Visual test reports**: Generate HTML reports showing test coverage
5. **Automated test generation**: Tools to generate basic tests from code