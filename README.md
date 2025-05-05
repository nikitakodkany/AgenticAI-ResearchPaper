# Research Paper Assistant

A RAG-powered research paper assistant that helps you find and analyze academic papers using advanced language models and vector search capabilities.

## Features

- **Natural Language Search**: Search through research papers using conversational queries
- **Vector Similarity Search**: Find relevant papers using semantic search powered by sentence transformers or OpenAI's text-embedding-ada-002
- **Paper Analysis**: Automated summarization and analysis of research papers
- **Local Database**: Store and retrieve papers using SQLite with vector search capabilities
- **Modern Web Interface**: Streamlit-based frontend for intuitive interaction
- **RESTful API**: FastAPI backend for serving research results and analysis
- **Research Workflow**: LangGraph-powered workflow for systematic paper analysis
- **Agent State Machine**: Multi-agent system for document processing:
  - Summarizer: Condenses each retrieved document
  - Critic: Flags limitations or missing context
  - Report Generator: Composes structured, final synthesis

## Technical Architecture

### RAG Implementation
- **Document Retrieval**: Uses LangChain's vector stores with SQLiteVSS
- **Embedding Models**: Supports both OpenAI's text-embedding-ada-002 and sentence-transformers
- **LLM Integration**: Configurable between OpenAI GPT-4 and local models
- **Strict Document-Based Answering**: Ensures responses are grounded in retrieved content

### Agent State Machine (LangGraph)
The system uses LangGraph to define a state machine of specialized agents:

1. **Summarizer Agent**
   - Condenses research papers into concise summaries
   - Uses structured prompts for consistent formatting
   - Maintains document context

2. **Critic Agent**
   - Analyzes summaries for limitations
   - Identifies missing context
   - Provides constructive feedback

3. **Report Generator Agent**
   - Synthesizes information from summaries and critiques
   - Creates structured research reports
   - Ensures comprehensive coverage

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd research-paper-assistant
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/MacOS
python -m venv venv
source venv/bin/activate
```

3. Install the package and its dependencies:
```bash
pip install -e .
```

4. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=your_preferred_model_name  # e.g., gpt-4-turbo-preview
USE_OPENAI=True  # Set to False to use free models
```

5. Initialize the database:
```bash
python scripts/fetch_papers.py
```

## Running the Application

1. Start the FastAPI backend:
```bash
cd app
uvicorn main:app --reload
```

2. In a new terminal, start the Streamlit frontend:
```bash
cd frontend
streamlit run app.py
```

3. Open your browser and navigate to:
- Frontend: `http://localhost:8501`
- API Documentation: `http://localhost:8000/docs`

## Project Structure

```
research-paper-assistant/
├── app/                    # FastAPI backend application
│   ├── api/               # API endpoints
│   ├── models/            # Pydantic models
│   └── main.py            # FastAPI application entry point
├── core/                  # Core functionality
│   ├── rag/              # RAG implementation
│   │   ├── assistant.py  # RAG assistant with critic
│   │   └── retrieval.py  # Document retrieval
│   ├── workflow/         # Research workflow
│   │   └── research_workflow.py  # LangGraph state machine
│   └── database/         # Database operations
├── data/                  # Data storage
│   └── papers.db         # SQLite database with vector search
├── frontend/             # Streamlit frontend
│   ├── components/       # UI components
│   └── app.py           # Streamlit application
├── scripts/              # Utility scripts
├── requirements.txt      # Project dependencies
├── setup.py             # Package configuration
└── README.md            # Project documentation
```

## Dependencies

The project uses several key dependencies:
- FastAPI and Uvicorn for the backend API
- Streamlit for the frontend interface
- LangChain for RAG implementation and LLM chaining
- LangGraph for agent state machine
- Sentence Transformers or OpenAI for embeddings
- SQLite with vector search capabilities
- OpenAI API for language model integration

## Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests. When contributing:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a clear description of your changes

## License

This project is licensed under the MIT License - see the LICENSE file for details. 