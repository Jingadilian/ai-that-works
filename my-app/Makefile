# Makefile for my-app development and testing
# 
# Common commands:
#   make test          - Run all tests
#   make test-fast     - Run only fast unit tests
#   make test-coverage - Run tests with coverage report
#   make lint          - Run code quality checks
#   make clean         - Clean up generated files

.PHONY: help test test-fast test-coverage test-html lint format clean install-deps

# Default target
help:
	@echo "Available commands:"
	@echo "  test          - Run all tests"
	@echo "  test-fast     - Run only unit tests (no integration tests)"
	@echo "  test-coverage - Run tests with coverage report"
	@echo "  test-html     - Run tests and generate HTML report"
	@echo "  lint          - Run code quality checks"
	@echo "  format        - Format code with black"
	@echo "  clean         - Clean up generated files"
	@echo "  install-deps  - Install all dependencies"

# Install dependencies
install-deps:
	pip install -r requirements-test.txt

# Run all tests
test:
	pytest tests/

# Run only fast unit tests (exclude slow/integration tests)
test-fast:
	pytest tests/ -m "not slow and not integration"

# Run tests with coverage
test-coverage:
	pytest --cov=. --cov-report=term-missing tests/

# Run tests and generate HTML coverage report
test-html:
	pytest --cov=. --cov-report=html tests/
	@echo "Coverage report generated in htmlcov/index.html"

# Run tests in parallel (faster)
test-parallel:
	pytest -n auto tests/

# Run specific test file
test-classification:
	pytest tests/test_classification.py -v

# Run code quality checks
lint:
	@echo "Running flake8..."
	-flake8 *.py tests/ --max-line-length=100
	@echo "Running mypy..."
	-mypy *.py --ignore-missing-imports

# Format code
format:
	black *.py tests/ --line-length=100

# Clean up generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

# Development setup
dev-setup: install-deps
	@echo "Development environment ready!"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make help' to see all available commands"