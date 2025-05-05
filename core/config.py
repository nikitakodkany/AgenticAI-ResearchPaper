"""Configuration settings for the research paper assistant."""
import os
from pathlib import Path

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Free embedding model
CHAT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Free chat model
USE_OPENAI = False  # Set to False to use free models

# Database Configuration
DB_PATH = Path(__file__).parent.parent / "data" / "papers.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ArXiv API Configuration
ARXIV_BASE_URL = "http://export.arxiv.org/api/query"
MAX_RESULTS_PER_QUERY = 10

# API Configuration
API_HOST = "localhost"  # Changed from 0.0.0.0 to localhost
API_PORT = 8000

# Frontend Configuration
STREAMLIT_PORT = 8501 