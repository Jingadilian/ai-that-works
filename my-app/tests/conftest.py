"""
pytest configuration and shared fixtures for my-app tests

This file contains:
- Global pytest configuration
- Shared fixtures used across multiple test files
- Mock setups for external dependencies
"""

import pytest
from unittest.mock import Mock, patch
import os
import sys

# Add the parent directory to the Python path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def mock_genai_client():
    """
    Mock Google GenAI client for testing without making actual API calls.
    
    Returns a mock client that simulates embedding responses.
    """
    mock_client = Mock()
    
    # Mock embedding response - simulates a 3072-dimensional embedding
    mock_embedding = Mock()
    mock_embedding.values = [0.1] * 3072  # Simplified embedding vector
    
    mock_result = Mock()
    mock_result.embeddings = [mock_embedding]
    
    mock_client.models.embed_content.return_value = mock_result
    
    return mock_client

@pytest.fixture
def sample_categories():
    """
    Sample categories for testing classification functions.
    
    Returns a small set of test categories that cover different scenarios.
    """
    from classification import Category
    
    return [
        Category(
            name="Search Products", 
            embedding_text="Find products", 
            llm_description="User is looking to search for products"
        ),
        Category(
            name="Buy Product", 
            embedding_text="Buy products", 
            llm_description="User is looking to buy a product"
        ),
        Category(
            name="Contact Support", 
            embedding_text="Customer support", 
            llm_description="User needs assistance from customer support"
        ),
    ]

@pytest.fixture(autouse=True)
def mock_env_vars():
    """
    Automatically mock environment variables for all tests.
    
    This prevents tests from trying to use real API keys.
    """
    with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
        yield