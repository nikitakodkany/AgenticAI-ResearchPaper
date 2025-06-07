class Settings:
    # Database
    DATABASE_URL = "postgresql://user:password@localhost:5432/research_db"  # This will be ignored when using ChromaDB
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"  # Directory to store ChromaDB data
    # Model Settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
    # Vector Search Settings
    VECTOR_DIMENSION = 384  # all-MiniLM-L6-v2 dimension
    SIMILARITY_THRESHOLD = 0.7
    VECTOR_BACKEND = "chroma"  # Using ChromaDB as vector backend
    # API Settings
    API_V1_STR = "/api/v1"
    PROJECT_NAME = "RAG Research Assistant"
    # ArXiv Settings
    MAX_PAPERS_PER_QUERY = 10
    API_URL = "http://localhost:8000"

settings = Settings() 