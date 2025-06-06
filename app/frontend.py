import streamlit as st
import requests
from typing import List, Dict
import json
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if "papers" not in st.session_state:
    st.session_state.papers = []
if "selected_paper" not in st.session_state:
    st.session_state.selected_paper = None
if "summary" not in st.session_state:
    st.session_state.summary = None

# Title and description
st.title("🔬 Research Assistant")
st.markdown("""
This tool helps you search and analyze research papers using AI. 
Enter a research topic to get started!
""")

# Search section
st.header("Search Papers")
search_query = st.text_input("Enter your research topic:", key="search_query")
if st.button("Search", key="search_button"):
    if search_query:
        with st.spinner("Searching for papers..."):
            response = requests.post(
                "http://localhost:8000/search",
                json={"query": search_query}
            )
            if response.status_code == 200:
                st.session_state.papers = response.json()
            else:
                st.error(f"Error: {response.text}")

# Display search results
if st.session_state.papers:
    st.subheader("Search Results")
    for paper in st.session_state.papers:
        with st.expander(f"{paper['title']} ({paper['arxiv_id']})"):
            st.write(f"**Authors:** {', '.join(paper['authors'])}")
            st.write(f"**Published:** {paper['published_date']}")
            st.write("**Abstract:**")
            st.write(paper['abstract'])
            st.write(f"[View Paper]({paper['url']})")
            
            if st.button("Generate Summary", key=f"summary_{paper['arxiv_id']}"):
                with st.spinner("Generating summary..."):
                    response = requests.post(
                        f"http://localhost:8000/summarize/{paper['arxiv_id']}"
                    )
                    if response.status_code == 200:
                        st.session_state.selected_paper = paper
                        st.session_state.summary = response.json()
                    else:
                        st.error(f"Error: {response.text}")

# Display summary if available
if st.session_state.summary:
    st.header("Paper Summary")
    st.write(f"**Title:** {st.session_state.selected_paper['title']}")
    
    st.subheader("Summary")
    st.write(st.session_state.summary['summary'])
    
    st.subheader("Key Findings")
    st.write(st.session_state.summary['key_findings'])
    
    st.subheader("Methodology")
    st.write(st.session_state.summary['methodology'])
    
    # Question answering
    st.subheader("Ask Questions")
    question = st.text_input("Ask a question about the paper:")
    if st.button("Get Answer"):
        if question:
            with st.spinner("Generating answer..."):
                response = requests.post(
                    f"http://localhost:8000/ask",
                    params={
                        "arxiv_id": st.session_state.selected_paper['arxiv_id'],
                        "question": question
                    }
                )
                if response.status_code == 200:
                    st.write("**Answer:**")
                    st.write(response.json()['response'])
                else:
                    st.error(f"Error: {response.text}")

# Similar papers section
if st.session_state.selected_paper:
    st.header("Similar Papers")
    if st.button("Find Similar Papers"):
        with st.spinner("Searching for similar papers..."):
            response = requests.post(
                "http://localhost:8000/similar",
                params={
                    "query": st.session_state.selected_paper['abstract'],
                    "limit": 5
                }
            )
            if response.status_code == 200:
                similar_papers = response.json()
                for paper in similar_papers:
                    with st.expander(f"{paper['title']} ({paper['arxiv_id']})"):
                        st.write(f"**Authors:** {paper['authors']}")
                        st.write(f"**Published:** {paper['published_date']}")
                        st.write("**Abstract:**")
                        st.write(paper['abstract'])
                        st.write(f"[View Paper]({paper['url']})")
            else:
                st.error(f"Error: {response.text}") 