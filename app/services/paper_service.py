from typing import List, Optional
import arxiv
from datetime import datetime
from app.database import papers_collection, summaries_collection
from app.config import settings

class PaperService:
    def __init__(self):
        self.papers_collection = papers_collection
        self.summaries_collection = summaries_collection

    def search_papers(self, query: str, max_results: int = settings.MAX_PAPERS_PER_QUERY) -> List[dict]:
        """Search papers on arXiv"""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        for result in search.results():
            paper = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "arxiv_id": result.entry_id.split('/')[-1],
                "published_date": result.published,
                "url": result.entry_id
            }
            papers.append(paper)
        return papers

    def store_paper(self, paper: dict, embedding: List[float]) -> str:
        """Store paper in ChromaDB"""
        paper_id = paper["arxiv_id"]
        self.papers_collection.add(
            ids=[paper_id],
            embeddings=[embedding],
            metadatas=[{
                "title": paper["title"],
                "authors": ", ".join(paper["authors"]),
                "abstract": paper["abstract"],
                "published_date": paper["published_date"].isoformat(),
                "url": paper["url"]
            }]
        )
        return paper_id

    def store_summary(self, paper_id: str, summary: dict, embedding: List[float]) -> str:
        """Store paper summary in ChromaDB"""
        summary_id = f"{paper_id}_summary"
        self.summaries_collection.add(
            ids=[summary_id],
            embeddings=[embedding],
            metadatas=[{
                "paper_id": paper_id,
                "summary": summary["summary"],
                "key_findings": summary["key_findings"],
                "methodology": summary["methodology"]
            }]
        )
        return summary_id

    def search_similar_papers(self, query_embedding: List[float], limit: int = 5) -> List[dict]:
        """Search for similar papers using vector similarity"""
        results = self.papers_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )
        
        papers = []
        for i in range(len(results["ids"][0])):
            paper = {
                "arxiv_id": results["ids"][0][i],
                **results["metadatas"][0][i]
            }
            papers.append(paper)
        return papers

    def get_paper_summary(self, paper_id: str) -> Optional[dict]:
        """Get paper summary from ChromaDB"""
        results = self.summaries_collection.get(
            ids=[f"{paper_id}_summary"]
        )
        
        if not results["ids"]:
            return None
            
        return {
            "paper_id": paper_id,
            **results["metadatas"][0]
        } 