from typing import List, Optional
from sentence_transformers import SentenceTransformer
from app.config import settings

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text using HuggingFace model"""
        return self.model.encode(text).tolist()
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return self.model.encode(texts).tolist() 