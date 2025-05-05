from typing import List, Dict
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from ..config import EMBEDDING_MODEL, USE_OPENAI
import os

class PaperStore:
    def __init__(self):
        self.db_path = "papers.db"
        self._ensure_db_exists()
        
        if USE_OPENAI:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.embedding_model = "text-embedding-ada-002"
        else:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    
    def _ensure_db_exists(self):
        """Ensure the database and tables exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                authors TEXT,
                abstract TEXT,
                content TEXT,
                url TEXT,
                published_date TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _get_embedding(self, text: str) -> bytes:
        """Get embedding for text using either OpenAI or SentenceTransformer."""
        if USE_OPENAI:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = np.array(response.data[0].embedding)
        else:
            embedding = self.embedding_model.encode(text)
        return embedding.tobytes()
    
    def store_paper(self, paper_data: Dict):
        """Store a new paper in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Generate embedding for the paper content
        embedding = self._get_embedding(paper_data.get("content", ""))
        
        cursor.execute("""
            INSERT INTO papers (title, authors, abstract, content, url, published_date, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            paper_data.get("title"),
            paper_data.get("authors"),
            paper_data.get("abstract"),
            paper_data.get("content"),
            paper_data.get("url"),
            paper_data.get("published_date"),
            embedding
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_papers(self, limit: int = 10) -> List[Dict]:
        """Get the most recent papers from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT title, authors, abstract, content, url, published_date
            FROM papers
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        papers = []
        for row in cursor.fetchall():
            papers.append({
                "title": row[0],
                "authors": row[1],
                "abstract": row[2],
                "content": row[3],
                "url": row[4],
                "published_date": row[5]
            })
        
        conn.close()
        return papers
    
    def search_similar_papers(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for papers similar to the query using vector similarity."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get embedding for the query
        query_embedding = self._get_embedding(query)
        
        # Get all papers with their embeddings
        cursor.execute("""
            SELECT title, authors, abstract, content, url, published_date, embedding
            FROM papers
        """)
        
        # Calculate similarity scores
        papers_with_scores = []
        for row in cursor.fetchall():
            paper = {
                "title": row[0],
                "authors": row[1],
                "abstract": row[2],
                "content": row[3],
                "url": row[4],
                "published_date": row[5]
            }
            similarity = self._cosine_similarity(query_embedding, row[6])
            papers_with_scores.append((paper, similarity))
        
        # Sort by similarity and get top k
        papers_with_scores.sort(key=lambda x: x[1], reverse=True)
        top_papers = [paper for paper, _ in papers_with_scores[:limit]]
        
        conn.close()
        return top_papers
    
    def _cosine_similarity(self, vec1: bytes, vec2: bytes) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.frombuffer(vec1, dtype=np.float32)
        vec2 = np.frombuffer(vec2, dtype=np.float32)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) 