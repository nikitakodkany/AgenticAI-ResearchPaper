import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.services.paper_service import PaperService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.config import settings
from app.services.evaluation_service import EvaluationService

app = FastAPI(title="Research Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
paper_service = PaperService()
embedding_service = EmbeddingService()
llm_service = LLMService()
evaluation_service = EvaluationService()

class SearchQuery(BaseModel):
    query: str
    max_results: Optional[int] = settings.MAX_PAPERS_PER_QUERY

class PaperResponse(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: str
    published_date: str
    url: str

class SummaryResponse(BaseModel):
    summary: str
    key_findings: str
    methodology: str

class ResearchQuery(BaseModel):
    query: str
    vector_backend: str = settings.VECTOR_BACKEND
    embedding_provider: str = "hf-all-MiniLM-L6-v2"
    llm_provider: str = "hf-mistral-7b"
    max_results: Optional[int] = settings.MAX_PAPERS_PER_QUERY
    category: Optional[str] = None
    year_range: Optional[List[int]] = None

class ResearchResponse(BaseModel):
    query: str
    papers: List[Dict[str, Any]]
    analysis: Optional[str] = "Analysis not available"
    metrics: Optional[Dict[str, Any]] = {
        "precision": "N/A",
        "recall": "N/A",
        "f1_score": "N/A",
        "per_query": {}
    }

class AskRequest(BaseModel):
    query: str
    top_k: int = 3
    vector_backend: str = "chroma"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "hf-mistral-7b"
    retrieval_source: str = "ArXiv"
    year_range: List[int] = [2019, 2024]
    topic_filter: List[str] = []

class QueryRequest(BaseModel):
    query: str
    num_papers: int = 5
    category: Optional[str] = None
    year_range: Optional[List[int]] = None

@app.get("/")
async def root():
    return {"message": "Research Assistant API"}

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/search", response_model=List[PaperResponse])
async def search_papers(query: SearchQuery):
    try:
        papers = paper_service.search_papers(query.query, query.max_results)
        return papers
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/{arxiv_id}")
async def summarize_paper(arxiv_id: str):
    try:
        # Get paper from ChromaDB
        results = paper_service.papers_collection.get(ids=[arxiv_id])
        if not results["ids"]:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        paper = results["metadatas"][0]
        
        # Generate summary
        summary = llm_service.generate_summary(paper["abstract"])
        
        # Store summary
        embedding = embedding_service.get_embedding(summary["summary"])
        paper_service.store_summary(arxiv_id, summary, embedding)
        
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary/{arxiv_id}", response_model=SummaryResponse)
async def get_summary(arxiv_id: str):
    try:
        summary = paper_service.get_paper_summary(arxiv_id)
        if not summary:
            raise HTTPException(status_code=404, detail="Summary not found")
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similar")
async def find_similar_papers(query: str, limit: int = 5):
    try:
        # Generate embedding for query
        query_embedding = embedding_service.get_embedding(query)
        
        # Search for similar papers
        similar_papers = paper_service.search_similar_papers(query_embedding, limit)
        return similar_papers
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: AskRequest = Body(...)):
    try:
        papers = paper_service.search_papers(
            request.query,
            request.top_k
        )
        context = "\n\n".join([
            f"Title: {p['title']}\nAuthors: {', '.join(p['authors'])}\nAbstract: {p['abstract']}\nURL: {p['url']}" for p in papers
        ])
        answer = llm_service.generate_response(request.query, context)
        citations = [
            {
                "title": p["title"],
                "abstract": p["abstract"],
                "url": p["url"]
            } for p in papers
        ]
        return {"answer": answer, "citations": citations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/research", response_model=ResearchResponse)
async def research_query(query: ResearchQuery):
    """
    Accepts vector_backend, embedding_provider, and llm_provider, but only Chroma/HuggingFace are supported.
    Other options are commented out for future use.
    """
    # Only open-source path is active
    paper_service = PaperService()  # Only Chroma/HuggingFace
    embedding_service = EmbeddingService()
    llm_service = LLMService()

    # Search for relevant papers with filters
    papers = paper_service.search_papers(
        query.query, 
        query.max_results,
        category=query.category,
        year_range=query.year_range
    )
    
    response = {
        "query": query.query,
        "papers": papers,
        "analysis": "Analysis not available",
        "metrics": {
            "precision": "N/A",
            "recall": "N/A",
            "f1_score": "N/A",
            "per_query": {}
        }
    }
    return response

@app.post("/query")
async def query(request: QueryRequest):
    # ... existing code ...
    response = await paper_service.search_papers(request.query, request.num_papers, request.category, request.year_range)
    metrics = evaluation_service.evaluate_response(request.query, response)
    return {"response": response, "metrics": metrics}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 