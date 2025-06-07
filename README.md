# Research Paper Search and Analysis Tool

A powerful tool for searching, analyzing, and understanding research papers using AI. This application combines arXiv paper search with advanced AI capabilities to help researchers find and comprehend relevant papers more effectively.

## Features

### Paper Search
- **Semantic Search**: Search papers using natural language queries
- **Category Filtering**: Filter papers by specific research categories (e.g., Machine Learning, Computer Vision)
- **Year Range Filtering**: Find papers published within specific years
- **Result Limit Control**: Control the number of papers returned per search

### AI-Powered Analysis
- **Paper Summarization**: Get concise summaries of research papers
- **Key Findings Extraction**: Identify main contributions and findings
- **Methodology Analysis**: Understand the research approach and methods used
- **Similar Paper Discovery**: Find related papers based on content similarity

### Technical Stack
- **Vector Store**: Chroma for efficient paper storage and retrieval
- **Embedding Provider**: HuggingFace (all-MiniLM-L6-v2) for semantic embeddings
- **LLM Provider**: HuggingFace (mistralai/Mistral-7B-Instruct-v0.2) for text analysis

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AgenticAI-ResearchPaper.git
cd AgenticAI-ResearchPaper
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the backend server:
```bash
uvicorn app.main:app --reload
```

5. In a new terminal, start the frontend:
```bash
streamlit run app/frontend/app.py
```

## Usage

1. Open your browser and navigate to `http://localhost:8501`
2. Enter your research query in the search box
3. (Optional) Select a category to filter results
4. (Optional) Set a year range for papers
5. (Optional) Adjust the number of results to return
6. Click "Search" to find relevant papers
7. Click on any paper to view its details and get an AI-generated analysis

## API Endpoints

- `GET /api/v1/health`: Check API health status
- `POST /api/v1/research`: Search for papers with the following parameters:
  - `query`: Search query string
  - `vector_backend`: Vector store backend (default: Chroma)
  - `embedding_provider`: Embedding model provider
  - `llm_provider`: LLM provider for analysis
  - `max_results`: Maximum number of papers to return
  - `category`: Paper category filter
  - `year_range`: [start_year, end_year] filter

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 