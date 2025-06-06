from typing import List, Dict, Any
from transformers import pipeline
from app.config import settings

class LLMService:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model = pipeline("text2text-generation", model=model_name)
    
    def generate_summary(self, text: str) -> Dict[str, str]:
        """Generate a summary of the text"""
        prompt = f"Summarize the following research paper abstract:\n\n{text}"
        summary = self.model(prompt, max_length=150, min_length=50)[0]["generated_text"]
        
        prompt = f"Extract key findings from this research paper summary:\n\n{summary}"
        key_findings = self.model(prompt, max_length=100)[0]["generated_text"]
        
        prompt = f"Describe the methodology used in this research paper:\n\n{text}"
        methodology = self.model(prompt, max_length=100)[0]["generated_text"]
        
        return {
            "summary": summary,
            "key_findings": key_findings,
            "methodology": methodology
        }
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate a response based on the query and context"""
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        response = self.model(prompt, max_length=200)[0]["generated_text"]
        return response 