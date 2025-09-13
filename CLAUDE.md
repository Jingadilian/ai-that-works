# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

This is a monorepo containing Python and TypeScript projects:
- `my-app/` - Python application using BAML for LLM interactions with e-commerce classification
- `ai-that-works/` - Collection of various AI/TypeScript projects and tools

## Common Development Commands

### Python Project (my-app)

Run tests:
```bash
make test                    # Run all tests
make test-fast              # Run only unit tests (exclude slow/integration)
make test-coverage          # Run tests with coverage report
make test-html              # Generate HTML coverage report
pytest tests/test_classification.py -v  # Run specific test file
```

Code quality:
```bash
make lint                   # Run flake8 and mypy
make format                 # Format with black
```

### TypeScript Projects

For projects in `ai-that-works/tools/` and similar:
```bash
bun install                 # Install dependencies
bun run validate            # Validate metadata
bun run lint               # Check code
bun run build              # Build project
```

## Architecture Overview

### my-app - BAML

The application uses BAML for generating prompts functional:

- **BAML Configuration** (`my-app/baml_src/`): Contains BAML definitions for clients, functions, and generators
  - `clients.baml`: Defines LLM client configurations (CustomGeminiFlash)
  - `functions.baml`: BAML functions like `PickBestCategory` with dynamic enums
  - Generated client code in `baml_client/`

- **Classification Pipeline** (`classification2.py`):
  - Uses embeddings for initial category narrowing (Gemini embedding model)
  - Final selection via LLM with BAML TypeBuilder for dynamic category generation
  - E-commerce focused categories (search, buy, checkout, support, etc.)

### Testing Infrastructure

- **pytest Configuration** (`pytest.ini`): Tests organized with markers (unit, integration, slow, api)
- **Test Reports**: HTML reports generated in `reports/` directory
- **Coverage**: HTML coverage reports in `htmlcov/`

## Key Technologies

- **Python**: BAML, Google Genai API, pytest, pydantic
- **TypeScript/Bun**: Modern runtime for TypeScript projects in ai-that-works/
- **GitHub Actions**: Automated Claude Code reviews on PRs and issues

## Environment Variables

Required in `my-app/.env`:
- `GOOGLE_API_KEY`: For Google Genai/Gemini API access

## BAML-Specific Notes

When modifying BAML functions:
1. Edit files in `baml_src/`
2. Generated client code updates automatically in `baml_client/`
3. Use TypeBuilder for dynamic enum generation at runtime
4. Test BAML functions using the test blocks defined in `.baml` files