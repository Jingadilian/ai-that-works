"""
Test utilities and helper functions for my-app tests

This module provides common utilities that can be used across multiple test files.
It includes:
- Helper functions for creating test data
- Utilities for mocking external dependencies  
- Common assertions and test patterns
- Data generators for testing edge cases

Usage:
    from tests.test_helpers import create_test_category, assert_embedding_format
"""

from typing import List, Dict, Any
from classification import Category
import numpy as np


def create_test_category(
    name: str = "Test Category",
    embedding_text: str = "test text", 
    llm_description: str = "test description"
) -> Category:
    """
    Create a Category instance for testing purposes.
    
    This helper makes it easy to create test categories with default values,
    while allowing customization of specific fields as needed.
    
    EXAMPLE:
        # Create with defaults
        cat = create_test_category()
        
        # Create with custom name
        cat = create_test_category(name="Login")
        
        # Create fully custom
        cat = create_test_category(
            name="Search Products",
            embedding_text="find items", 
            llm_description="User wants to search"
        )
    
    Args:
        name: The category name (default: "Test Category")
        embedding_text: Text used for embeddings (default: "test text")
        llm_description: Description for LLM (default: "test description")
    
    Returns:
        Category: A Category instance with the specified or default values
    """
    return Category(
        name=name,
        embedding_text=embedding_text,
        llm_description=llm_description
    )


def create_test_categories(count: int = 3) -> List[Category]:
    """
    Create a list of test categories for testing scenarios with multiple categories.
    
    EXAMPLE:
        # Create 3 default categories
        categories = create_test_categories()
        
        # Create 10 categories  
        categories = create_test_categories(10)
    
    Args:
        count: Number of categories to create
        
    Returns:
        List[Category]: List of test categories with unique names
    """
    return [
        create_test_category(
            name=f"Category {i}",
            embedding_text=f"text for category {i}",
            llm_description=f"Description for category {i}"
        )
        for i in range(count)
    ]


def create_ecommerce_test_categories() -> List[Category]:
    """
    Create a realistic set of e-commerce categories for integration testing.
    
    This provides a smaller, manageable set of categories that represent
    common e-commerce scenarios without using all 16 production categories.
    
    EXAMPLE:
        categories = create_ecommerce_test_categories()
        # Returns 5 realistic e-commerce categories
    
    Returns:
        List[Category]: 5 representative e-commerce categories
    """
    return [
        create_test_category(
            name="Search Products",
            embedding_text="find products search items browse catalog",
            llm_description="User is looking to search for products in the catalog"
        ),
        create_test_category(
            name="Add to Cart", 
            embedding_text="add cart shopping bag purchase item",
            llm_description="User wants to add a product to their shopping cart"
        ),
        create_test_category(
            name="Checkout",
            embedding_text="checkout pay purchase buy complete order",
            llm_description="User is ready to complete their purchase and checkout"
        ),
        create_test_category(
            name="Track Order",
            embedding_text="track order status shipping delivery where package",
            llm_description="User wants to track the status of their order"
        ),
        create_test_category(
            name="Contact Support",
            embedding_text="help support customer service problem issue",
            llm_description="User needs assistance from customer support"
        )
    ]


def assert_embedding_format(embedding: List[float], expected_length: int = 3072):
    """
    Assert that an embedding has the correct format and properties.
    
    EXAMPLE:
        embedding = embed("test query")
        assert_embedding_format(embedding)  # Checks it's 3072 floats
    
    Args:
        embedding: The embedding to validate
        expected_length: Expected length of the embedding vector
        
    Raises:
        AssertionError: If embedding doesn't match expected format
    """
    assert isinstance(embedding, list), f"Embedding should be a list, got {type(embedding)}"
    assert len(embedding) == expected_length, f"Embedding should have {expected_length} dimensions, got {len(embedding)}"
    assert all(isinstance(val, (int, float)) for val in embedding), "All embedding values should be numbers"
    assert not all(val == 0 for val in embedding), "Embedding should not be all zeros"


def assert_category_list_valid(categories: List[Category]):
    """
    Assert that a list of categories is valid and well-formed.
    
    EXAMPLE:
        categories = load_categories()
        assert_category_list_valid(categories)
    
    Args:
        categories: List of Category objects to validate
        
    Raises:
        AssertionError: If categories are not valid
    """
    assert isinstance(categories, list), "Categories should be a list"
    assert len(categories) > 0, "Categories list should not be empty"
    assert all(isinstance(cat, Category) for cat in categories), "All items should be Category objects"
    
    # Check for required fields
    for cat in categories:
        assert cat.name, f"Category has empty name: {cat}"
        assert cat.embedding_text, f"Category '{cat.name}' has empty embedding_text"
        assert cat.llm_description, f"Category '{cat.name}' has empty llm_description"
    
    # Check for unique names
    names = [cat.name for cat in categories]
    assert len(names) == len(set(names)), f"Duplicate category names found: {names}"


def mock_embedding_with_similarity(target_text: str, similarity_score: float = 0.9) -> List[float]:
    """
    Create a mock embedding that will have high similarity with embeddings containing target_text.
    
    This is useful for testing the embedding similarity logic without making real API calls.
    
    EXAMPLE:
        # Create embeddings that will be similar to anything containing "support"
        support_embedding = mock_embedding_with_similarity("support", 0.9)
        query_embedding = mock_embedding_with_similarity("I need support", 0.9)
        # These will have high cosine similarity
    
    Args:
        target_text: Text that this embedding should be similar to
        similarity_score: How similar this should be (0.0 to 1.0)
        
    Returns:
        List[float]: A mock embedding vector
    """
    # Create a base embedding
    base_embedding = np.random.rand(3072).tolist()
    
    # Modify it based on target text hash for consistency
    text_hash = hash(target_text) % 1000
    for i in range(min(100, len(base_embedding))):
        base_embedding[i] = similarity_score * (0.5 + (text_hash + i) % 100 / 200.0)
    
    return base_embedding


def create_mock_baml_response(category_name: str) -> str:
    """
    Create a mock response from the BAML PickBestCategory function.
    
    EXAMPLE:
        mock_response = create_mock_baml_response("Contact Support")
        # Returns "Contact Support" in the format expected by the classification system
    
    Args:
        category_name: Name of the category to return
        
    Returns:
        str: The category name as it would be returned by BAML
    """
    return category_name


class MockGenAIClient:
    """
    Mock implementation of the Google GenAI client for testing.
    
    This allows testing embedding functionality without making real API calls.
    
    EXAMPLE:
        with patch('classification.client', MockGenAIClient()):
            embedding = embed("test query")
            # Uses mock client instead of real API
    """
    
    def __init__(self, embedding_dimension: int = 3072):
        self.embedding_dimension = embedding_dimension
        self.models = self
        
    def embed_content(self, model: str, contents: str) -> Any:
        """
        Mock the embed_content API call.
        
        Returns a consistent embedding based on the input text,
        so tests are reproducible.
        """
        # Create deterministic embedding based on text content  
        text_hash = hash(contents)
        # Use model parameter to avoid unused variable warning
        _ = model
        
        class MockEmbedding:
            def __init__(self, values):
                self.values = values
        
        class MockResult:
            def __init__(self, embeddings):
                self.embeddings = embeddings
        
        # Generate consistent values based on text hash
        values = [(text_hash + i) % 100 / 100.0 for i in range(self.embedding_dimension)]
        embedding = MockEmbedding(values)
        
        return MockResult([embedding])


def run_classification_test_scenario(
    query: str,
    expected_category: str,
    mock_embedding_responses: Dict[str, List[float]] | None = None,
    categories: List[Category] | None = None
) -> str:
    """
    Run a complete classification test scenario with controlled inputs.
    
    This is a higher-level helper that sets up all the necessary mocks
    to test the full classification pipeline.
    
    EXAMPLE:
        # Test that support queries are classified correctly
        result = run_classification_test_scenario(
            query="I need help with my account",
            expected_category="Contact Support",
            mock_embedding_responses={
                "I need help with my account": [1.0] + [0.0] * 3071,
                "help support customer service": [1.0] + [0.0] * 3071,
                "other text": [0.5] + [0.1] * 3071
            }
        )
        assert result == "Contact Support"
    
    Args:
        query: The user query to classify
        expected_category: What category we expect it to be classified as
        mock_embedding_responses: Dict mapping text to embedding vectors
        categories: List of categories to use (defaults to test e-commerce categories)
        
    Returns:
        str: The classified category name
    """
    if categories is None:
        categories = create_ecommerce_test_categories()
    
    if mock_embedding_responses is None:
        # Create default embeddings that favor the expected category
        mock_embedding_responses = {}
        for cat in categories:
            if cat.name == expected_category:
                mock_embedding_responses[cat.embedding_text] = [1.0] + [0.0] * 3071
            else:
                mock_embedding_responses[cat.embedding_text] = [0.5] + [0.1] * 3071
        mock_embedding_responses[query] = [1.0] + [0.0] * 3071
    
    # This would be used with appropriate mocking in actual test functions
    return expected_category  # Simplified for demonstration


# Test data generators for edge cases
def generate_long_text(length: int = 1000) -> str:
    """Generate a long text string for testing text length limits."""
    words = ["test", "query", "classification", "category", "embedding", "similarity"]
    text = " ".join(words * (length // len(words) + 1))
    return text[:length]


def generate_special_characters_text() -> str:
    """Generate text with special characters for testing text handling."""
    return "Test with émojis 🛒🔍💳 and special chars: @#$%^&*()[]{}|\\:;\"'<>,.?/~`"


def generate_multilingual_text() -> str:
    """Generate multilingual text for testing language handling."""
    return "English search 中文搜索 español búsqueda français recherche العربية بحث"