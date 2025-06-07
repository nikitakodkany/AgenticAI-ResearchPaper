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
import spacy
from transformers import AutoTokenizer

# Load spaCy model for sentence splitting
try:
    nlp = spacy.load("en_core_web_sm")
    use_spacy = True
except Exception:
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize
    use_spacy = False

# Load HuggingFace tokenizer for the embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

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

def chunk_text(text: str, max_tokens: int = 256, overlap_tokens: int = 30) -> list:
    """Chunk text into overlapping segments using tokenizer, with robust sentence splitting."""
    # Sentence splitting
    if use_spacy:
        sentences = [sent.text.strip() for sent in nlp(text).sents]
    else:
        sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_tokens = 0
    sentence_tokens = [len(tokenizer.encode(sent, add_special_tokens=False)) for sent in sentences]

    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_len = sentence_tokens[i]
        if current_tokens + sent_len > max_tokens and current_chunk:
            # Finalize current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            # Overlap: last N tokens (by sentences)
            overlap_count = 0
            overlap_chunk = []
            j = len(current_chunk) - 1
            while j >= 0 and overlap_count < overlap_tokens:
                overlap_chunk.insert(0, current_chunk[j])
                overlap_count += len(tokenizer.encode(current_chunk[j], add_special_tokens=False))
                j -= 1
            current_chunk = overlap_chunk
            current_tokens = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in current_chunk)
        else:
            current_chunk.append(sent)
            current_tokens += sent_len
            i += 1
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(chunk_text)
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