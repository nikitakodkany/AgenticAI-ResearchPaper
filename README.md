# RAG-Powered Research Assistant

## Autonomous Multi-Agent Literature Review with LangChain, LangGraph & pgvector

A fully automated multi-agent research assistant that retrieves academic papers, extracts insights, critiques findings, and generates structured reports using LangChain, LangGraph, and pgvector.

## Features
- **Retrieval-Augmented Generation (RAG)** for knowledge-grounded responses
- **Multi-Agent Workflow** using LangGraph
- **Research Paper Retrieval** via ArXiv API
- **Embedding Storage & Similarity Search** using pgvector
- **Structured Research Report Generation** with GPT-4
- **FastAPI** for API Deployment
- **Streamlit UI** for User Interaction

## Tech Stack
| Component            | Technology Used |
|----------------------|----------------|
| LLM                 | OpenAI GPT-4    |
| Retrieval           | ArXiv API       |
| Embedding DB        | PostgreSQL + pgvector |
| Multi-Agent         | LangChain + LangGraph |
| API                 | FastAPI         |
| Frontend            | Streamlit       |
| Deployment          | Docker          |

## 📽️ [Watch the demo video](recording.mp4)

## How It Works
### Step 1: Fetch Research Papers
- The system queries ArXiv for academic papers related to the given topic.

### Step 2: Store Embeddings
- The paper summaries are embedded using OpenAI Embeddings (`text-embedding-ada-002`) and stored in `pgvector`.

### Step 3: Retrieve Most Relevant Papers (RAG)
- When a query is made, `pgvector` retrieves the top N most similar papers based on cosine similarity.

### Step 4: LLM Generates Context-Aware Answers
- The system constructs a context-aware response using LangChain, grounding the answer in retrieved research.

### Step 5: Multi-Agent Literature Review
Using LangGraph, multiple agents collaborate to:
- Summarize Findings
- Critique Biases & Limitations
- Compare & Contrast
- Generate a Structured Report

### Step 6: User Interaction
Users can query the system via:
- **FastAPI** (REST API)
- **Streamlit UI** (Web App)
