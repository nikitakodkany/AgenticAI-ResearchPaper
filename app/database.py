# import sqlalchemy and pgvector dependencies (commented out)
# from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
# from sqlalchemy.orm import declarative_base, sessionmaker
# from pgvector.sqlalchemy import Vector
# from app.config import settings

import chromadb
from datetime import datetime
import os
from app.config import settings

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

try:
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)

    # Create collections for papers and summaries with minimal configuration
    papers_collection = chroma_client.get_or_create_collection(
        name="research_papers"
    )

    summaries_collection = chroma_client.get_or_create_collection(
        name="research_summaries"
    )
except Exception as e:
    print(f"Error initializing ChromaDB: {str(e)}")
    raise

def get_db():
    """Get database session - not needed for ChromaDB but kept for API compatibility"""
    return None

def init_db():
    """Initialize database - ChromaDB collections are created on first use"""
    pass 