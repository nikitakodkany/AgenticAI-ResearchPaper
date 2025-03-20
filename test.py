import streamlit as st
import time

# Set page configuration
st.set_page_config(page_title="Quantum Cryptography Research Assistant", layout="centered")

# Sidebar with instructions
st.sidebar.markdown(
    """
    ## Instructions
    1. **Enter your question:**  
       Type your research question about quantum cryptography. A default question is provided.
    2. **Generate Answer:**  
       Click the **Generate Answer** button to get an AI-generated response.
    """
)

# Custom CSS for better UI
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
    }
    .main-container {
        background: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
        max-width: 700px;
        margin: auto;
        margin-top: 2rem;
        font-family: 'Arial', sans-serif;
    }
    .title {
        text-align: center;
        color: #2C3E50;
        font-size: 2.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subheader {
        text-align: center;
        color: #7B8A8B;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .input-label {
        font-size: 1rem;
        font-weight: bold;
        color: #34495E;
    }
    .output-card {
        background: #FAFAFA;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        border: 1px solid #E0E0E0;
        box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.05);
    }
    .button-style > button {
        background-color: #3498DB !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.7rem 1.2rem !important;
        border-radius: 6px !important;
        font-size: 1rem !important;
        font-weight: bold !important;
        transition: background 0.3s;
    }
    .button-style > button:hover {
        background-color: #217DBB !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main container
with st.container():
    # st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Title and Subheader
    st.markdown('<h1 class="title">Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Enter your research question and get an AI-generated response with a citation.</p>', unsafe_allow_html=True)
    
    # User input
    st.markdown('<p class="input-label">Research Question:</p>', unsafe_allow_html=True)
    query = st.text_area("", value="How does quantum cryptography enhance security?", 
                         placeholder="Type a research question, e.g., 'What are the applications of quantum key distribution?'")
    
    # Generate Answer Button
    if st.button("Generate Answer", key="generate", help="Click to generate an AI response."):
        with st.spinner("Step 1: Fetching relevant research papers..."):
            time.sleep(1.5)
        
        with st.spinner("Step 2: Storing and analyzing paper embeddings..."):
            time.sleep(1.5)
        
        with st.spinner("Step 3: Retrieving the most relevant sources..."):
            time.sleep(1.5)
        
        with st.spinner("Step 4: Generating a context-aware AI response..."):
            time.sleep(1.5)
        
        with st.spinner("Step 5: Conducting a multi-agent literature review..."):
            time.sleep(1.5)
        
        # Simulated Retrieved Research Papers
        st.markdown("### Retrieved Research Papers (Cited):")
        st.write("1. **Quantum Key Distribution Protocols** - ArXiv (2023) [[Paper Link]](https://arxiv.org/abs/2301.00001)")
        st.write("2. **Post-Quantum Cryptography: A Survey** - Springer (2022) [[Paper Link]](https://link.springer.com/article/10.1007/s00145-022-01000-0)")
        st.write("3. **Entanglement-Based Secure Communications** - IEEE (2021) [[Paper Link]](https://ieeexplore.ieee.org/document/9456789)")
        
        # Hardcoded answer and citation
        hardcoded_answer = (
            "Quantum cryptography enhances security by leveraging the principles of quantum mechanics. "
            "It employs quantum entanglement and the no-cloning theorem to detect any eavesdropping attempt. "
            "If an unauthorized party tries to intercept the communication, the quantum state is disturbed, "
            "which alerts the communicating parties and helps ensure data integrity."
        )
                
        # Display output in a styled card
        # st.markdown('<div class="output-card">', unsafe_allow_html=True)
        st.markdown("### AI Generated Response:")
        st.write(hardcoded_answer)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Multi-Agent Review Summary
        st.markdown("### Multi-Agent Review Insights:")
        st.write("**Findings:** Quantum cryptography ensures secure communication through quantum key distribution.")
        st.write("**Limitations:** Current implementations require specialized hardware.")
        st.write("**Comparison:** Compared to classical cryptography, quantum methods provide better security but are not widely adopted yet.")
        
        st.success("Response generated successfully!")
    else:
        st.info("Enter a question and click **Generate Answer** to see the response.")
    
    st.markdown('</div>', unsafe_allow_html=True)
