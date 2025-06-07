# import sqlalchemy and pgvector dependencies (commented out)
# from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
# from sqlalchemy.orm import declarative_base, sessionmaker
# from pgvector.sqlalchemy import Vector
# from app.config import settings

import chromadb
from datetime import datetime
import os
from app.config import settings
import numpy as np
from typing import List, Dict, Any
import logging

# --- PostgreSQL/pgvector/SQLAlchemy setup (commented out) ---
# DATABASE_URL = settings.DATABASE_URL
# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()
#
# class ResearchPaper(Base):
#     __tablename__ = "research_papers"
#     id = Column(Integer, primary_key=True, index=True)
#     title = Column(String)
#     authors = Column(String)
#     abstract = Column(Text)
#     arxiv_id = Column(String, unique=True, index=True)
#     published_date = Column(DateTime)
#     url = Column(String)
#     embedding = Column(Vector(384))
#
# class ResearchSummary(Base):
#     __tablename__ = "research_summaries"
#     id = Column(Integer, primary_key=True, index=True)
#     paper_id = Column(String, index=True)
#     summary = Column(Text)
#     key_findings = Column(Text)
#     methodology = Column(Text)
#     embedding = Column(Vector(384))
#
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()
#
# def init_db():
#     Base.metadata.create_all(bind=engine)

# --- ChromaDB setup (active) ---
# Ensure the directory exists
os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_vector(vector: List[float]) -> List[float]:
    """Normalize vector to unit length"""
    vector = np.array(vector)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.tolist()
    return (vector / norm).tolist()

def chunk_text(text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Semantic chunking of text with overlap"""
    # Split into sentences first
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence = sentence.strip() + '. '
        sentence_size = len(sentence)
        
        if current_size + sentence_size > max_chunk_size and current_chunk:
            # Join current chunk and add to chunks
            chunks.append(' '.join(current_chunk))
            # Keep last few sentences for overlap
            overlap_sentences = []
            overlap_size = 0
            for s in reversed(current_chunk):
                if overlap_size + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_size += len(s)
                else:
                    break
            current_chunk = overlap_sentences
            current_size = overlap_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

try:
    # Initialize ChromaDB client with proper configuration
    chroma_client = chromadb.PersistentClient(
        path=settings.CHROMA_PERSIST_DIRECTORY,
        settings=chromadb.Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

    # Create collections for papers with metadata and proper configuration
    papers_collection = chroma_client.get_or_create_collection(
        name="research_papers",
        metadata={
            "hnsw:space": "cosine",  # Use cosine similarity
            "hnsw:construction_ef": 100,  # Higher value for better accuracy
            "hnsw:search_ef": 100,  # Higher value for better accuracy
            "hnsw:M": 16,  # Number of connections per element
        }
    )

    summaries_collection = chroma_client.get_or_create_collection(
        name="research_summaries",
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 100,
            "hnsw:search_ef": 100,
            "hnsw:M": 16,
        }
    )
    
    logger.info("ChromaDB initialized successfully")
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {str(e)}")
    raise

def get_db():
    """Get database session - not needed for ChromaDB but kept for API compatibility"""
    return None

def init_db():
    """Initialize database - ChromaDB collections are created on first use"""
    pass 