import streamlit as st
import requests

st.title("📚 RAG-Powered Research Assistant")

query = st.text_input("Enter research query:")
if st.button("Get Answer"):
    with st.spinner("Fetching relevant research..."):
        response = requests.get(f"http://localhost:8000/research/?query={query}")
        if response.status_code == 200:
            st.text_area("AI Response", response.json()["response"], height=400)

