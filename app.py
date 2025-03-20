import streamlit as st
import requests
import time

st.set_page_config(page_title="Research Assistant", layout="centered")

# Sidebar Instructions
st.sidebar.markdown(
    """
    ## Instructions
    1. **Enter your research query:**  
       Type a research question related to quantum cryptography or any topic.
    2. **Generate Answer:**  
       Click the **Get Answer** button to fetch relevant papers and generate a response.
    """
)

st.title("Research Assistant")
st.write("Enter your research query and get AI-generated responses with real citations.")

query = st.text_input("Enter research query:", value="How does quantum cryptography enhance security?")

if st.button("Get Answer"):
    with st.spinner("Step 1: Fetching relevant research papers..."):
        response = requests.get(f"http://localhost:8000/research/?query={query}")
        time.sleep(1.5)
    
    if response.status_code == 200:
        data = response.json()
        research_papers = data.get("papers", [])
        ai_response = data.get("response", "No response generated.")
        review_summary = data.get("review_summary", [])
    
        with st.spinner("Step 2: Processing research papers..."):
            time.sleep(1.5)
    
        with st.spinner("Step 3: Generating a context-aware AI response..."):
            time.sleep(1.5)
    
        with st.spinner("Step 4: Conducting multi-agent literature review..."):
            time.sleep(1.5)
    
        # Display Retrieved Research Papers
        st.markdown("### Retrieved Research Papers (Cited):")
        for paper in research_papers:
            st.markdown(f"- **{paper['title']}** [[Read More]]({paper['link']})")
        
        # Display AI-generated Response
        st.markdown("### AI Generated Response:")
        st.text_area("", ai_response, height=400)
        
        # Display Multi-Agent Review Summary
        st.markdown("### Multi-Agent Review Insights:")
        for insight in review_summary:
            st.write(insight)
        
        st.success("Response generated successfully!")
    else:
        st.error("Failed to retrieve research papers. Please try again later.")
