import numpy as np
from baml_client import b
from baml_client.type_builder import TypeBuilder
from baml_client.tracing import trace
from pydantic import BaseModel
import dotenv 
from google import genai

dotenv.load_dotenv()

client = genai.Client()

class Category:
    name: str
    embedding_text: str
    llm_description: str

    def __init__(self, name: str, embedding_text: str, llm_description: str):
        self.name = name
        self.embedding_text = embedding_text
        self.llm_description = llm_description

def load_categories() -> list[Category]:
    return [
        Category(name="Category1", embedding_text="for placeholder", llm_description="for placeholder"),
        Category(name="Category2", embedding_text="for debugging", llm_description="for debugging"),
        Category(name="Category3", embedding_text="for general use", llm_description="for general use"),
    ]

def embed(query: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query)
    return result.embeddings[0].values

def _narrow_down_categories(query: str, categories: list[Category]) -> list[Category]:
    embeddings: list[tuple[Category, list[float]]] = []
    for category in categories:
        embeddings.append((category, embed(category.embedding_text)))
    text_embedding = embed(query)
    best_matches: list[tuple[Category, float]] = []
    for category, category_embedding in embeddings:
        cosine_similarity = np.dot(text_embedding, category_embedding) / (np.linalg.norm(text_embedding) * np.linalg.norm(category_embedding))
        best_matches.append((category, cosine_similarity))
    max_matches = 5
    matches = sorted(best_matches, key=lambda x: x[1], reverse=True)[:max_matches]
    return [match[0] for match in matches]

def _pick_best_category(query: str, categories: list[Category]) -> Category:
    tb = TypeBuilder()
    for i, category in enumerate(categories):
        val = tb.Category.add_value(category.name)
        val.alias(f"k{i}")
        val.description(category.llm_description)
    
    selected_category = b.PickBestCategory(query, { "tb": tb })
    for category in categories:
        if category.name == selected_category:
            return category

def pick_category(query: str) -> str:
    categories = load_categories()
    narrowed_categories = _narrow_down_categories(query, categories)
    best_category = _pick_best_category(query, narrowed_categories)
    return best_category.name

if __name__ == "__main__":
    print(pick_category("I need to debug something"))

