import streamlit as st
import requests
import json
from typing import Dict, List
import time

# Configuration
API_URL = "http://localhost:8000/api/v1"

BACKEND_OPTIONS = {
    "Chroma (local, open-source)": "chroma",
    "PostgreSQL + pgvector": "pgvector"
}
EMBEDDING_OPTIONS = {
    "HuggingFace (all-MiniLM-L6-v2)": "hf-all-MiniLM-L6-v2",
    "OpenAI (text-embedding-3-small)": "openai-text-embedding-3-small"
}
LLM_OPTIONS = {
    "HuggingFace (Mistral-7B)": "hf-mistral-7b",
    "OpenAI (GPT-4 Turbo)": "openai-gpt-4-turbo"
}

def main():
    st.set_page_config(
        page_title="RAG Research Assistant",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 RAG Research Assistant")
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
        st.info("""
        **Note:** Only Chroma (local, open-source) and HuggingFace models are currently supported in this deployment. OpenAI and pgvector options are for future use.
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
            with st.spinner("Searching and analyzing papers..."):
                try:
                    # Make API request
                    response = requests.post(
                        f"{API_URL}/research",
                        json={
                            "query": query,
                            "vector_backend": backend,
                            "embedding_provider": embedding_provider,
                            "llm_provider": llm_provider
                        }
                    )
                    response.raise_for_status()
                    result = response.json()

                    # Display results
                    st.markdown("## 📚 Relevant Papers")
                    for paper in result["papers"]:
                        with st.expander(f"📄 {paper['title']}"):
                            st.markdown(f"**Authors:** {paper['authors']}")
                            st.markdown(f"**URL:** [{paper['url']}]({paper['url']})")

                    st.markdown("## 📝 Analysis")
                    st.markdown(result["analysis"])

                except requests.exceptions.RequestException as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a research query.")

if __name__ == "__main__":
    main() 