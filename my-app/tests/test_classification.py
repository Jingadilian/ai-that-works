"""
Unit tests for classification.py

This test file provides comprehensive coverage of all functions in the classification module.
Each test is designed to be both a unit test and documentation that shows:
1. What inputs the function expects
2. What outputs it produces
3. How the function behaves in different scenarios

The tests use descriptive names and include detailed docstrings to serve as examples
for other developers working with this codebase.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import the functions we want to test
from classification import (
    Category,
    load_categories,
    embed,
    _narrow_down_categories,
    _pick_best_category,
    pick_category
)


class TestCategoryClass:
    """
    Tests for the Category class
    
    The Category class is a simple data container that holds information about
    each classification category including the name, embedding text, and LLM description.
    """
    
    def test_category_creation_with_valid_data(self):
        """
        EXAMPLE: Creating a Category with typical e-commerce data
        
        Input: name="Search Products", embedding_text="Find products", llm_description="User is looking to search for products"
        Output: Category object with all attributes set correctly
        """
        category = Category(
            name="Search Products",
            embedding_text="Find products", 
            llm_description="User is looking to search for products"
        )
        
        assert category.name == "Search Products"
        assert category.embedding_text == "Find products"
        assert category.llm_description == "User is looking to search for products"
    
    def test_category_creation_with_empty_strings(self):
        """
        EXAMPLE: Category can be created with empty strings (edge case)
        
        Input: All empty strings
        Output: Category object with empty string attributes
        """
        category = Category(name="", embedding_text="", llm_description="")
        
        assert category.name == ""
        assert category.embedding_text == ""
        assert category.llm_description == ""


class TestLoadCategories:
    """
    Tests for the load_categories function
    
    This function returns the predefined list of e-commerce categories that the
    classification system can choose from. It's essentially our "vocabulary" of
    possible user intents.
    """
    
    def test_load_categories_returns_expected_count(self):
        """
        EXAMPLE: Loading all predefined categories
        
        Input: None (no parameters)
        Output: List of 16 Category objects representing e-commerce actions
        """
        categories = load_categories()
        
        # We expect 16 predefined e-commerce categories
        assert len(categories) == 16
        assert all(isinstance(cat, Category) for cat in categories)
    
    def test_load_categories_contains_expected_categories(self):
        """
        EXAMPLE: Verifying specific important categories are present
        
        Input: None
        Output: List containing key e-commerce categories like "Search Products", "Buy Product", etc.
        """
        categories = load_categories()
        category_names = [cat.name for cat in categories]
        
        # Check for some key e-commerce categories
        expected_categories = [
            "Search Products",
            "Buy Product", 
            "Add to Cart",
            "Checkout",
            "Return Item",
            "Contact Support"
        ]
        
        for expected in expected_categories:
            assert expected in category_names, f"Missing expected category: {expected}"
    
    def test_load_categories_all_have_required_fields(self):
        """
        EXAMPLE: Ensuring all categories have complete data
        
        Input: None
        Output: All categories have non-empty name, embedding_text, and llm_description
        """
        categories = load_categories()
        
        for category in categories:
            assert category.name, f"Category has empty name: {category}"
            assert category.embedding_text, f"Category has empty embedding_text: {category.name}"
            assert category.llm_description, f"Category has empty llm_description: {category.name}"


class TestEmbed:
    """
    Tests for the embed function
    
    This function takes a text query and returns a vector embedding using Google's GenAI.
    The embedding is used for semantic similarity matching to find relevant categories.
    """
    
    @patch('classification.client')
    def test_embed_returns_embedding_vector(self, mock_client):
        """
        EXAMPLE: Getting an embedding for a search query
        
        Input: "I want to buy shoes"
        Output: List of 3072 float values representing the semantic embedding
        """
        # Mock the API response structure
        mock_embedding = Mock()
        mock_embedding.values = [0.1, -0.2, 0.3] * 1024  # 3072 values
        
        mock_result = Mock()
        mock_result.embeddings = [mock_embedding]
        
        mock_client.models.embed_content.return_value = mock_result
        
        # Test the function
        result = embed("I want to buy shoes")
        
        # Verify the API was called correctly
        mock_client.models.embed_content.assert_called_once_with(
            model="gemini-embedding-001",
            contents="I want to buy shoes"
        )
        
        # Verify the output format
        assert isinstance(result, list)
        assert len(result) == 3072
        assert all(isinstance(val, (int, float)) for val in result)
    
    @patch('classification.client')
    def test_embed_handles_empty_string(self, mock_client):
        """
        EXAMPLE: Handling edge case of empty input
        
        Input: ""
        Output: Still returns an embedding vector (API handles empty strings)
        """
        mock_embedding = Mock()
        mock_embedding.values = [0.0] * 3072
        
        mock_result = Mock()
        mock_result.embeddings = [mock_embedding]
        
        mock_client.models.embed_content.return_value = mock_result
        
        result = embed("")
        
        mock_client.models.embed_content.assert_called_once_with(
            model="gemini-embedding-001",
            contents=""
        )
        
        assert len(result) == 3072


class TestNarrowDownCategories:
    """
    Tests for the _narrow_down_categories function
    
    This function uses embedding similarity to find the top 5 most relevant categories
    for a given query. It's the first stage of the classification pipeline.
    """
    
    @patch('classification.embed')
    def test_narrow_down_categories_returns_top_matches(self, mock_embed, sample_categories):
        """
        EXAMPLE: Finding top categories for "I need help with my order"
        
        Input: query="I need help with my order", categories=[Search Products, Buy Product, Contact Support]
        Output: List of up to 5 categories ranked by similarity (Contact Support should rank high)
        """
        # Mock embeddings - make "Contact Support" most similar to the query
        def mock_embed_side_effect(text):
            if "help" in text.lower() or "support" in text.lower():
                return [1.0, 0.0, 0.0]  # High similarity vector
            elif "search" in text.lower() or "find" in text.lower():
                return [0.5, 0.5, 0.0]  # Medium similarity
            else:
                return [0.0, 0.0, 1.0]  # Low similarity
        
        mock_embed.side_effect = mock_embed_side_effect
        
        result = _narrow_down_categories("I need help with my order", sample_categories)
        
        # Should return all 3 categories (less than max of 5)
        assert len(result) == 3
        assert all(isinstance(cat, Category) for cat in result)
        
        # "Contact Support" should be first (highest similarity)
        assert result[0].name == "Contact Support"
    
    @patch('classification.embed')
    def test_narrow_down_categories_limits_to_five(self, mock_embed):
        """
        EXAMPLE: Handling more than 5 categories
        
        Input: query="test", categories=[7 different categories]
        Output: Exactly 5 categories (the top matches)
        """
        # Create 7 test categories
        many_categories = [
            Category(f"Category {i}", f"text {i}", f"desc {i}") 
            for i in range(7)
        ]
        
        # Mock embeddings with decreasing similarity
        def mock_embed_side_effect(text):
            if "test" in text:
                return [1.0, 0.0, 0.0]
            # Return different similarities for different categories
            for i in range(7):
                if f"text {i}" in text:
                    return [1.0 - i*0.1, 0.0, 0.0]  # Decreasing similarity
            return [0.0, 0.0, 0.0]
        
        mock_embed.side_effect = mock_embed_side_effect
        
        result = _narrow_down_categories("test query", many_categories)
        
        # Should return exactly 5 categories (the limit)
        assert len(result) == 5
    
    @patch('classification.embed')
    def test_narrow_down_categories_with_single_category(self, mock_embed, sample_categories):
        """
        EXAMPLE: Working with just one category
        
        Input: query="test", categories=[single category]
        Output: List with that single category
        """
        single_category = [sample_categories[0]]  # Just "Search Products"
        
        mock_embed.return_value = [0.5, 0.5, 0.0]
        
        result = _narrow_down_categories("test query", single_category)
        
        assert len(result) == 1
        assert result[0].name == "Search Products"


class TestPickBestCategory:
    """
    Tests for the _pick_best_category function
    
    This function uses the BAML system to make the final classification decision
    among the narrowed-down categories. It's the second stage of the pipeline.
    """
    
    @patch('classification.b.PickBestCategory')
    def test_pick_best_category_returns_matching_category(self, mock_baml, sample_categories):
        """
        EXAMPLE: LLM picks "Contact Support" for a support query
        
        Input: query="I need help", categories=[Search Products, Buy Product, Contact Support]
        Output: Contact Support Category object
        """
        # Mock BAML to return "Contact Support"
        mock_baml.return_value = "Contact Support"
        
        result = _pick_best_category("I need help with my order", sample_categories)
        
        # Verify BAML was called
        mock_baml.assert_called_once()
        call_args = mock_baml.call_args
        assert call_args[0][0] == "I need help with my order"  # Query
        assert "tb" in call_args[0][1]  # TypeBuilder passed
        
        # Verify correct category returned
        assert isinstance(result, Category)
        assert result.name == "Contact Support"
    
    @patch('classification.b.PickBestCategory')
    def test_pick_best_category_builds_type_correctly(self, mock_baml, sample_categories):
        """
        EXAMPLE: Verifying that the TypeBuilder is configured with category options
        
        Input: query="test", categories=[3 sample categories]
        Output: TypeBuilder should be configured with all 3 categories as enum values
        """
        mock_baml.return_value = "Search Products"
        
        with patch('classification.TypeBuilder') as mock_tb_class:
            mock_tb = Mock()
            mock_tb_class.return_value = mock_tb
            
            # Mock the category enum builder
            mock_category_builder = Mock()
            mock_tb.Category = mock_category_builder
            
            mock_value_builder = Mock()
            mock_category_builder.add_value.return_value = mock_value_builder
            
            result = _pick_best_category("test query", sample_categories)
            
            # Verify TypeBuilder was created
            mock_tb_class.assert_called_once()
            
            # Verify all categories were added as enum values
            expected_calls = [
                (("Search Products",),),
                (("Buy Product",),),  
                (("Contact Support",),)
            ]
            
            actual_calls = mock_category_builder.add_value.call_args_list
            assert len(actual_calls) == 3
            
            # Check that aliases and descriptions were set
            assert mock_value_builder.alias.call_count == 3
            assert mock_value_builder.description.call_count == 3


class TestPickCategory:
    """
    Tests for the main pick_category function
    
    This is the public API function that orchestrates the entire classification pipeline:
    1. Load categories
    2. Narrow down using embeddings  
    3. Pick best using LLM
    4. Return category name
    """
    
    @patch('classification._pick_best_category')
    @patch('classification._narrow_down_categories') 
    @patch('classification.load_categories')
    def test_pick_category_full_pipeline(self, mock_load, mock_narrow, mock_pick):
        """
        EXAMPLE: Complete classification of a return request
        
        Input: "I want to return my shoes"
        Output: "Return Item" (the function orchestrates the full pipeline)
        """
        # Set up the pipeline mocks
        sample_cats = [
            Category("Return Item", "return product", "User wants to return item"),
            Category("Contact Support", "support", "User needs help")
        ]
        
        mock_load.return_value = sample_cats
        mock_narrow.return_value = sample_cats[:1]  # Narrowed to just "Return Item"
        mock_pick.return_value = sample_cats[0]      # Picked "Return Item"
        
        result = pick_category("I want to return my shoes")
        
        # Verify the pipeline was executed correctly
        mock_load.assert_called_once()
        mock_narrow.assert_called_once_with("I want to return my shoes", sample_cats)
        mock_pick.assert_called_once_with("I want to return my shoes", sample_cats[:1])
        
        # Verify the final result
        assert result == "Return Item"
    
    @patch('classification._pick_best_category')
    @patch('classification._narrow_down_categories')
    @patch('classification.load_categories')
    def test_pick_category_with_different_queries(self, mock_load, mock_narrow, mock_pick):
        """
        EXAMPLE: Testing various types of user queries
        
        This test shows how different user intents should be classified
        """
        sample_cats = [
            Category("Search Products", "find products", "User searching"),
            Category("Buy Product", "purchase", "User buying"),
            Category("Contact Support", "help", "User needs support")
        ]
        
        mock_load.return_value = sample_cats
        
        test_cases = [
            ("I'm looking for running shoes", "Search Products"),
            ("I want to buy this laptop", "Buy Product"), 
            ("I need help with my account", "Contact Support")
        ]
        
        for query, expected_category in test_cases:
            # Set up mocks for this test case
            expected_cat = next(cat for cat in sample_cats if cat.name == expected_category)
            mock_narrow.return_value = [expected_cat]
            mock_pick.return_value = expected_cat
            
            result = pick_category(query)
            
            assert result == expected_category, f"Query '{query}' should classify as '{expected_category}', got '{result}'"
    
    def test_pick_category_integration_with_real_categories(self):
        """
        EXAMPLE: Integration test using real category data (but mocked external calls)
        
        Input: "How do I track my package?"
        Output: Should work with actual load_categories() data structure
        """
        with patch('classification.embed') as mock_embed, \
             patch('classification.b.PickBestCategory') as mock_baml:
            
            # Mock embeddings to favor "Track Order"
            def embed_side_effect(text):
                if "track" in text.lower() or "order" in text.lower():
                    return [1.0] + [0.0] * 3071
                else:
                    return [0.5] + [0.1] * 3071
            
            mock_embed.side_effect = embed_side_effect
            mock_baml.return_value = "Track Order"
            
            result = pick_category("How do I track my package?")
            
            assert result == "Track Order"
            
            # Verify the function called the external services
            assert mock_embed.call_count > 0  # Should call embed multiple times
            mock_baml.assert_called_once()


class TestErrorHandling:
    """
    Tests for error conditions and edge cases
    
    These tests ensure the system behaves gracefully when things go wrong.
    """
    
    @patch('classification.embed')
    def test_narrow_down_categories_with_embed_failure(self, mock_embed, sample_categories):
        """
        EXAMPLE: Handling embedding API failures
        
        Input: Valid query and categories, but embedding API fails
        Output: Should raise an appropriate exception
        """
        mock_embed.side_effect = Exception("API Error")
        
        with pytest.raises(Exception):
            _narrow_down_categories("test query", sample_categories)
    
    @patch('classification.b.PickBestCategory')
    def test_pick_best_category_with_unknown_result(self, mock_baml, sample_categories):
        """
        EXAMPLE: Handling when LLM returns unexpected category name
        
        Input: Valid query and categories, but LLM returns unknown category
        Output: Should handle gracefully (returns None)
        """
        mock_baml.return_value = "Unknown Category"
        
        result = _pick_best_category("test query", sample_categories)
        
        # Function should return None when no matching category is found
        assert result is None
    
    def test_narrow_down_categories_with_empty_categories(self):
        """
        EXAMPLE: Handling empty category list
        
        Input: query="test", categories=[]
        Output: Should return empty list without errors
        """
        with patch('classification.embed') as mock_embed:
            mock_embed.return_value = [0.5] * 3072
            
            result = _narrow_down_categories("test query", [])
            
            assert result == []


if __name__ == "__main__":
    # This allows running the tests directly with: python test_classification.py
    pytest.main([__file__, "-v"])