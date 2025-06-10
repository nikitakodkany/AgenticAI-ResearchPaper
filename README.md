# CiteMind

Research Paper Search and Analysis Tool (v1.2.0) - A modern, AI-powered app to search, filter, and analyze research papers with advanced vector search, metadata filters, and evaluation metrics.

## Features

- **Provider Selection**: Choose your Vector Database, Embedding Model, and LLM Provider directly from the sidebar UI.
- **Metadata Filters**: Filter papers by category, publication year range, similarity threshold, and number of results.
- **Evaluation Metrics**: View overall and per-query metrics (precision, recall, F1 score, relevance, response time, chunk count).
- **Search History**: See your previous queries and their results.
- **Modern Sidebar UI**: All controls, filters, and metrics are organized in a clean sidebar, with a concise About section.
- **Query Evaluation**: Automatic evaluation of search results using exact match metrics.

## About

This app lets you search and analyze research papers using AI.
- **Vector DB**: Chroma (selectable)
- **Embedding**: all-MiniLM-L6-v2 (selectable)
- **LLM**: Mistral-7B (selectable)

## Tech Stack
- Python, FastAPI (backend)
- Streamlit (frontend)
- ChromaDB (default vector store, others selectable)
- HuggingFace Transformers (default embedding/LLM, others selectable)
- arXiv API (paper data)

## Setup

1. **Clone the repository**
2. **Create a virtual environment**
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Start the backend**
   ```bash
   uvicorn app.main:app --reload
   ```
5. **Start the frontend**
   ```bash
   streamlit run app/frontend/app.py
   ```

## Usage
- Open the Streamlit app in your browser (default: http://localhost:8501)
- Use the sidebar to select providers, set filters, and view metrics
- Enter your research query and search
- Expand results for details and metrics

## API Endpoints
- `GET /api/v1/health` — Health check
- `POST /api/v1/research` — Search for papers (accepts query, filters, provider options)

## Contributing
Pull requests and issues are welcome!

## License
MIT 
