import streamlit as st
import requests
import json
from typing import Dict, List
import time
from datetime import datetime

# Version
VERSION = "1.0.0"

# Configuration
API_URL = "http://localhost:8000/api/v1"

# Backend options
BACKEND_OPTIONS = {
    "Chroma": "Chroma",
    "Pinecone": "Pinecone",
    "Weaviate": "Weaviate",
    "Milvus": "Milvus",
    "Qdrant": "Qdrant",
    "FAISS": "FAISS",
    "Elasticsearch": "Elasticsearch",
    "Redis": "Redis",
    "PostgreSQL": "PostgreSQL",
    "MongoDB": "MongoDB"
}

# Embedding options
EMBEDDING_OPTIONS = {
    "HuggingFace (all-MiniLM-L6-v2)": "sentence-transformers/all-MiniLM-L6-v2",
    "HuggingFace (BGE-Small)": "BAAI/bge-small-en",
    "HuggingFace (MPNet)": "sentence-transformers/all-mpnet-base-v2",
    "HuggingFace (E5)": "intfloat/e5-large-v2",
    "HuggingFace (GTE)": "thenlper/gte-large",
    "OpenAI (text-embedding-3-small)": "text-embedding-3-small",
    "OpenAI (text-embedding-3-large)": "text-embedding-3-large",
    "OpenAI (text-embedding-ada-002)": "text-embedding-ada-002",
    "Cohere (embed-english-v3.0)": "embed-english-v3.0",
    "Cohere (embed-multilingual-v3.0)": "embed-multilingual-v3.0",
    "Jina (jina-embedding-v2)": "jina-embedding-v2",
    "Mistral (mistral-embed)": "mistral-embed"
}

# LLM options
LLM_OPTIONS = {
    "HuggingFace (Mistral-7B)": "mistralai/Mistral-7B-Instruct-v0.2",
    "HuggingFace (Llama-2-7B)": "meta-llama/Llama-2-7b-chat-hf",
    "HuggingFace (Llama-2-13B)": "meta-llama/Llama-2-13b-chat-hf",
    "HuggingFace (Llama-2-70B)": "meta-llama/Llama-2-70b-chat-hf",
    "HuggingFace (Mixtral-8x7B)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "HuggingFace (Falcon-7B)": "tiiuae/falcon-7b-instruct",
    "HuggingFace (Falcon-40B)": "tiiuae/falcon-40b-instruct",
    "OpenAI (GPT-4)": "gpt-4",
    "OpenAI (GPT-4-Turbo)": "gpt-4-1106-preview",
    "OpenAI (GPT-3.5-Turbo)": "gpt-3.5-turbo",
    "OpenAI (GPT-3.5-Turbo-16K)": "gpt-3.5-turbo-16k",
    "Anthropic (Claude-3-Opus)": "claude-3-opus-20240229",
    "Anthropic (Claude-3-Sonnet)": "claude-3-sonnet-20240229",
    "Anthropic (Claude-3-Haiku)": "claude-3-haiku-20240307",
    "Anthropic (Claude-2)": "claude-2",
    "Anthropic (Claude-Instant)": "claude-instant",
    "Cohere (Command)": "command",
    "Cohere (Command-Light)": "command-light",
    "Cohere (Command-R)": "command-r",
    "Cohere (Command-R-Plus)": "command-r-plus"
}

# Paper categories
PAPER_CATEGORIES = [
    "All",
    "Artificial Intelligence",
    "Computation and Language",
    "Computer Vision",
    "Machine Learning",
    "Neural and Evolutionary Computing",
    "Information Retrieval",
    "Software Engineering",
    "Distributed Computing",
    "Systems and Control",
    "Programming Languages",
    "Cryptography and Security",
    "Data Structures and Algorithms",
    "Databases",
    "Hardware Architecture",
    "Operating Systems",
    "Networking and Internet Architecture",
    "Computational Geometry",
    "Computer Science and Game Theory",
    "Robotics",
    "Sound",
    "Multimedia",
    "Human-Computer Interaction",
    "Computers and Society",
    "Emerging Technologies",
    "Formal Languages and Automata Theory",
    "Logic in Computer Science",
    "Multiagent Systems",
    "Mathematical Software",
    "Performance",
    "Symbolic Computation",
    "Social and Information Networks",
    "Computer Science Theory",
    "Probability",
    "Statistics Theory",
    "Machine Learning (Statistics)",
    "Quantitative Methods",
    "Computational Physics",
    "Data Analysis and Statistics",
    "Physics and Society",
    "Econometrics",
    "Theoretical Economics",
    "Computational Finance",
    "Portfolio Management",
    "Pricing of Securities",
    "Risk Management",
    "Statistical Finance",
    "Trading and Market Microstructure"
]

def main():
    st.set_page_config(
        page_title="RAG Research Assistant",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 RAG Research Assistant")
    st.caption(f"Version {VERSION}")
    st.markdown("""
    This assistant helps you find and analyze research papers using RAG (Retrieval-Augmented Generation).
    Enter your research query below to get started.
    """)

    # Sidebar
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown("""
        This research assistant uses:
        - LangChain & LangGraph for agent orchestration
        - HuggingFace or OpenAI for embeddings and text generation
        - Chroma or PostgreSQL with pgvector for vector storage
        - RAG pipeline for context-aware responses
        """)
        # Backend selection
        backend_label = st.selectbox("Vector Backend", list(BACKEND_OPTIONS.keys()), index=0)
        backend = BACKEND_OPTIONS[backend_label]
        # Embedding provider/model selection
        embedding_label = st.selectbox("Embedding Provider", list(EMBEDDING_OPTIONS.keys()), index=0)
        embedding_provider = EMBEDDING_OPTIONS[embedding_label]
        # LLM provider/model selection
        llm_label = st.selectbox("LLM Provider", list(LLM_OPTIONS.keys()), index=0)
        llm_provider = LLM_OPTIONS[llm_label]
        
        # Filters
        st.markdown("## 🔍 Filters")
        selected_category = st.selectbox("Category", PAPER_CATEGORIES, index=0)
        year_range = st.slider(
            "Publication Year Range",
            min_value=2015,
            max_value=2025,
            value=(2015, 2024)
        )
        max_papers = st.slider(
            "Number of Papers",
            min_value=1,
            max_value=50,
            value=10,
            step=1
        )
        
        # Health check
        try:
            health = requests.get(f"{API_URL}/health")
            if health.status_code == 200:
                st.success("API Status: Healthy")
            else:
                st.error("API Status: Unhealthy")
        except:
            st.error("API Status: Unreachable")

    # Research query input
    query = st.text_input(
        "Enter your research query",
        placeholder="e.g., Recent advances in large language models"
    )

    if st.button("Search", type="primary"):
        if query:
            with st.spinner("Searching for relevant papers..."):
                try:
                    response = requests.post(
                        f"{API_URL}/research",
                        json={
                            "query": query,
                            "vector_backend": backend,
                            "embedding_provider": embedding_provider,
                            "llm_provider": llm_provider,
                            "category": selected_category if selected_category != "All" else None,
                            "year_range": year_range,
                            "max_results": max_papers
                        }
                    )
                    response.raise_for_status()
                    result = response.json()

                    # Clear previous results and display new ones
                    with st.empty():
                        st.markdown("## 📚 Relevant Papers")
                        for paper in result["papers"]:
                            with st.expander(f"📄 {paper['title']} ({paper['published_date'][:4]})"):
                                st.markdown(f"**Abstract:** {paper['abstract']}")
                                st.markdown(f"**URL:** [{paper['url']}]({paper['url']})")

                except requests.exceptions.RequestException as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a research query.")

if __name__ == "__main__":
    main() 