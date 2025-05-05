from typing import List, Dict
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import SQLiteVSS
from langchain.schema import Document
from ..config import EMBEDDING_MODEL, USE_OPENAI, DB_PATH
import os

class RAGRetriever:
    def __init__(self):
        # Initialize embeddings
        if USE_OPENAI:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL
            )
        
        # Initialize vector store
        self.vectorstore = SQLiteVSS(
            table_name="papers",
            embedding=self.embeddings,
            connection_string=f"sqlite:///{DB_PATH}"
        )
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve the most relevant documents for a given query using LangChain's vector store.
        
        Args:
            query (str): The research query
            top_k (int): Number of documents to retrieve
            
        Returns:
            List[Dict]: List of relevant documents with their metadata
        """
        # Use LangChain's similarity search
        docs = self.vectorstore.similarity_search(query, k=top_k)
        
        # Convert LangChain Documents to our format
        results = []
        for doc in docs:
            results.append({
                "title": doc.metadata.get("title", ""),
                "authors": doc.metadata.get("authors", ""),
                "abstract": doc.metadata.get("abstract", ""),
                "content": doc.page_content,
                "url": doc.metadata.get("url", ""),
                "published_date": doc.metadata.get("published_date", "")
            })
        
        return results 