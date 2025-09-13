# AI That Works

A monorepo containing AI-powered applications and tools for classification and automation tasks.

## Overview

This repository contains Python and TypeScript projects focused on AI/LLM implementations:

- **`my-app/`** - Python application using BAML (Boundary Agent Markup Language) for LLM-powered e-commerce classification
- **AI Tools** - Collection of various AI and TypeScript utilities

## Key Features

### E-commerce Classification System
- Dynamic category classification using Google's Gemini models
- Embedding-based initial category narrowing for efficiency
- BAML-powered prompt generation and type-safe LLM interactions
- Comprehensive test suite with pytest and coverage reporting

### Technology Stack
- **Python**: BAML, Google Genai API, pytest, pydantic
- **LLM Framework**: BAML for structured prompt engineering
- **Testing**: pytest with markers for unit/integration/API tests
- **Code Quality**: flake8, mypy, black formatting

## Quick Start

### Prerequisites
- Python 3.8+
- Google API key for Gemini models

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Jingadilian/ai-that-works.git
cd ai-that-works
```

2. Set up the Python application:
```bash
cd my-app
pip install -r requirements-test.txt
```

3. Configure environment variables:
```bash
# Create .env file in my-app/
GOOGLE_API_KEY=your_api_key_here
```

### Running Tests

```bash
cd my-app
make test           # Run all tests
make test-fast      # Run unit tests only
make test-coverage  # Generate coverage report
```

### Code Quality

```bash
make lint    # Run flake8 and mypy
make format  # Format with black
```

## Project Structure

```
.
├── my-app/                 # Main Python application
│   ├── baml_src/          # BAML source definitions
│   ├── baml_client/       # Generated BAML client code
│   ├── tests/             # Test suite
│   └── classification2.py # Core classification logic
└── CLAUDE.md              # Repository guidelines for AI assistants
```

## Development

For detailed development instructions and repository-specific guidelines, see [CLAUDE.md](./CLAUDE.md).

## License

[License information to be added]
