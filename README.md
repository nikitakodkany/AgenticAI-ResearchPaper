# RAG-Powered Research Assistant

A sophisticated research assistant that leverages RAG (Retrieval-Augmented Generation) to help researchers find, analyze, and summarize academic papers.

## Features

- Automated research paper fetching and processing
- Vector embeddings generation using HuggingFace models (e.g., all-MiniLM-L6-v2)
- Efficient similarity search using PostgreSQL with pgvector
- Intelligent research summarization and compilation
- Context-aware responses using RAG pipeline
- User-friendly Streamlit interface
- FastAPI backend for robust API endpoints

## Tech Stack

- LangChain & LangGraph for agent orchestration
- HuggingFace Transformers for embeddings and text generation
- PostgreSQL with pgvector for vector storage
- FastAPI for backend API
- Streamlit for frontend interface

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up PostgreSQL with pgvector extension
5. Configuration is hardcoded in `app/config.py` (no .env file needed)
6. Initialize the database:
   ```bash
   python scripts/init_db.py
   ```
7. Start the FastAPI backend:
   ```bash
   uvicorn app.main:app --reload
   ```
8. Start the Streamlit frontend:
   ```bash
   streamlit run app/frontend/app.py
   ```

## Project Structure

```
.
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── database.py          # Database connection and models
│   ├── agents/              # LangChain agents
│   ├── services/            # Business logic
│   └── frontend/            # Streamlit UI
├── scripts/
│   └── init_db.py          # Database initialization
├── tests/                   # Test files
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Usage

1. Access the Streamlit interface at `http://localhost:8501`
2. Enter your research query or topic of interest
3. The system will:
   - Search for relevant papers
   - Generate embeddings
   - Retrieve similar content
   - Provide context-aware responses

## License

MIT License 