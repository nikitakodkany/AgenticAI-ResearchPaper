import streamlit as st
import requests
import json
from typing import Dict, List
import time
from datetime import datetime

# Version
VERSION = "1.2.0"

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

# Initialize session state for storing search history and metrics
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = {}

def main():
    st.set_page_config(
        page_title="RAG Research Assistant",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Research Paper Search")
    st.caption(f"v{VERSION}")
    
    # Sidebar for filters and metrics
    with st.sidebar:
        st.header("Filters & Metrics")
        # About section
        st.markdown("""
        **About**  
        This app lets you search and analyze research papers using AI.  
        - **Vector DB:** Chroma (selectable)
        - **Embedding:** all-MiniLM-L6-v2 (selectable)
        - **LLM:** Mistral-7B (selectable)
        Filters, evaluation metrics, and advanced controls are available below.
        """)
        # Provider dropdowns
        backend_label = st.selectbox("Vector Database", ["Chroma", "Pinecone", "Weaviate", "Milvus", "Qdrant", "FAISS", "Elasticsearch", "Redis", "PostgreSQL", "MongoDB"], index=0)
        embedding_label = st.selectbox("Embedding Provider", ["all-MiniLM-L6-v2", "BAAI/bge-small-en", "all-mpnet-base-v2", "text-embedding-ada-002", "embed-english-v3.0"], index=0)
        llm_label = st.selectbox("LLM Provider", ["Mistral-7B", "Llama-2-7b-chat-hf", "gpt-3.5-turbo", "gpt-4", "claude-2", "command"], index=0)
        # Metadata Filters
        st.subheader("Metadata Filters")
        year_range = st.slider(
            "Publication Year Range",
            min_value=2010,
            max_value=datetime.now().year,
            value=(2020, datetime.now().year)
        )
        categories = [
            "All",
            "Machine Learning",
            "Computer Vision",
            "Natural Language Processing",
            "Robotics",
            "Artificial Intelligence",
            "Neural Networks",
            "Deep Learning"
        ]
        category = st.selectbox("Category", categories)
        # Number of papers and similarity threshold
        max_results = st.slider("Number of papers", 1, 10, 3)
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.05)
        # Evaluation Metrics Display
        st.subheader("Evaluation Metrics")
        if st.session_state.evaluation_metrics:
            metrics = st.session_state.evaluation_metrics
            st.metric("Precision", f"{metrics.get('precision', 0):.2f}")
            st.metric("Recall", f"{metrics.get('recall', 0):.2f}")
            st.metric("F1 Score", f"{metrics.get('f1_score', 0):.2f}")
            st.subheader("Per-Query Metrics")
            for query, metrics in metrics.get('per_query', {}).items():
                with st.expander(f"Query: {query}"):
                    st.write(f"Relevance Score: {metrics.get('relevance', 0):.2f}")
                    st.write(f"Response Time: {metrics.get('response_time', 0):.2f}s")
                    st.write(f"Chunks Retrieved: {metrics.get('chunks_retrieved', 0)}")
        # Health Check
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                st.success("API Status: Healthy")
            else:
                st.error("API Status: Unhealthy")
        except:
            st.error("API Status: Unreachable")
    
    # Main content area
    # Remove duplicate description here
    # st.markdown("""
    # Search and analyze research papers using AI. Enter your query below to find relevant papers.
    # """)
    
    # Search input
    query = st.text_input("Enter your research query", key="search_query")
    
    # Search button
    if st.button("Search", type="primary"):
        if query:
            with st.spinner("Searching papers..."):
                try:
                    # Make API request
                    response = requests.post(
                        f"{API_URL}/research",
                        json={
                            "query": query,
                            "max_results": max_results,
                            "category": category,
                            "year_range": year_range,
                            "similarity_threshold": similarity_threshold,
                            "vector_backend": backend_label,
                            "embedding_provider": embedding_label,
                            "llm_provider": llm_label
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Store search in history
                        st.session_state.search_history.append({
                            "query": query,
                            "timestamp": datetime.now().isoformat(),
                            "results_count": len(data.get("papers", [])),
                            "metrics": data.get("metrics", {})
                        })
                        
                        # Update evaluation metrics
                        st.session_state.evaluation_metrics = data.get("metrics", {})
                        
                        # Display results
                        st.subheader("Search Results")
                        for paper in data.get("papers", []):
                            with st.expander(f"{paper['title']} ({paper['published_date'][:4]})"):
                                st.write("**Authors:**", ", ".join(paper["authors"]))
                                st.write("**Abstract:**", paper["abstract"])
                                st.write("**URL:**", paper["url"])
                                
                                # Display paper-specific metrics
                                if "metrics" in paper:
                                    st.write("**Relevance Score:**", f"{paper['metrics'].get('relevance', 0):.2f}")
                                    st.write("**Chunk Count:**", paper['metrics'].get('chunk_count', 0))
                                    st.write("**Embedding Quality:**", f"{paper['metrics'].get('embedding_quality', 0):.2f}")
                    else:
                        st.error("Error searching papers. Please try again.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a search query")
    
    # Search History
    if st.session_state.search_history:
        st.subheader("Search History")
        for search in reversed(st.session_state.search_history):
            with st.expander(f"Query: {search['query']} ({search['timestamp'][:10]})"):
                st.write(f"Results found: {search['results_count']}")
                if search.get('metrics'):
                    st.subheader("Search Metrics")
                    metrics = search['metrics']
                    
                    # Helper function to format metric value
                    def format_metric(value):
                        if isinstance(value, (int, float)):
                            return f"{value:.2f}"
                        return str(value)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Precision", format_metric(metrics.get('precision', 'N/A')))
                    with col2:
                        st.metric("Recall", format_metric(metrics.get('recall', 'N/A')))
                    with col3:
                        st.metric("F1 Score", format_metric(metrics.get('f1_score', 'N/A')))

if __name__ == "__main__":
    main() 