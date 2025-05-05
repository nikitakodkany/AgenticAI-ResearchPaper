from core.rag.assistant import RAGResearchAssistant
from core.workflow.research_workflow import ResearchWorkflow
from core.database.store import PaperStore
import os
from dotenv import load_dotenv

load_dotenv()

def test_rag_system():
    print("Testing RAG Research Assistant...")
    
    # Initialize components
    assistant = RAGResearchAssistant()
    workflow = ResearchWorkflow()
    store = PaperStore()
    
    # Test query
    query = "What are the latest developments in quantum computing?"
    
    # Test RAG retrieval
    print("\n1. Testing RAG retrieval...")
    relevant_papers = assistant.retriever.retrieve(query)
    print(f"Found {len(relevant_papers)} relevant papers:")
    for paper in relevant_papers:
        print(f"- {paper['title']}")
    
    # Test research workflow
    print("\n2. Testing research workflow...")
    if relevant_papers:
        report = workflow.run(relevant_papers)
        print("\nGenerated Report:")
        print(report)
    else:
        print("No papers found to analyze.")

if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set up your .env file with the required API key.")
    else:
        test_rag_system() 