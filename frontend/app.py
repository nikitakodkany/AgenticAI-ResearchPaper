import streamlit as st
import requests
from core.config import API_HOST, API_PORT

st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG-Powered Research Assistant")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("API URL", f"http://{API_HOST}:{API_PORT}")
    top_k = st.slider("Number of papers to retrieve", 1, 10, 5)

# Main content
query = st.text_input("Enter your research query:", placeholder="e.g., What are the latest developments in quantum computing?")

if st.button("Search"):
    if not query:
        st.warning("Please enter a query first!")
    else:
        with st.spinner("Searching through research papers..."):
            try:
                response = requests.get(f"{api_url}/research/?query={query}")
                if response.status_code == 200:
                    result = response.json()
                    st.subheader("Research Results")
                    if not result["response"].strip():
                        st.info("No relevant papers found. Please try a different query.")
                    else:
                        st.text_area("AI Response", result["response"], height=400)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Make sure the API server is running.") 