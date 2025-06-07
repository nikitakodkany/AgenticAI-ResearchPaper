from typing import List, Optional
import arxiv
from datetime import datetime
from app.database import papers_collection, summaries_collection
from app.config import settings

class PaperService:
    def __init__(self):
        self.papers_collection = papers_collection
        self.summaries_collection = summaries_collection

    def search_papers(
        self, 
        query: str, 
        max_results: int = settings.MAX_PAPERS_PER_QUERY,
        category: Optional[str] = None,
        year_range: Optional[List[int]] = None
    ) -> List[dict]:
        """Search papers on arXiv"""
        print(f"Searching papers with query: {query}")
        print(f"Max results: {max_results}")
        print(f"Category: {category}")
        print(f"Year range: {year_range}")
        
        # Map full category names to arXiv codes
        category_mapping = {
            "Artificial Intelligence": "cs.AI",
            "Computation and Language": "cs.CL",
            "Computer Vision": "cs.CV",
            "Machine Learning": "cs.LG",
            "Neural and Evolutionary Computing": "cs.NE",
            "Information Retrieval": "cs.IR",
            "Software Engineering": "cs.SE",
            "Distributed Computing": "cs.DC",
            "Systems and Control": "cs.SY",
            "Programming Languages": "cs.PL",
            "Cryptography and Security": "cs.CR",
            "Data Structures and Algorithms": "cs.DS",
            "Databases": "cs.DB",
            "Hardware Architecture": "cs.AR",
            "Operating Systems": "cs.OS",
            "Networking and Internet Architecture": "cs.NI",
            "Computational Geometry": "cs.CG",
            "Computer Science and Game Theory": "cs.GT",
            "Robotics": "cs.RO",
            "Sound": "cs.SD",
            "Multimedia": "cs.MM",
            "Human-Computer Interaction": "cs.HC",
            "Computers and Society": "cs.CY",
            "Emerging Technologies": "cs.ET",
            "Formal Languages and Automata Theory": "cs.FL",
            "Logic in Computer Science": "cs.LO",
            "Multiagent Systems": "cs.MA",
            "Mathematical Software": "cs.MS",
            "Performance": "cs.PF",
            "Symbolic Computation": "cs.SC",
            "Social and Information Networks": "cs.SI",
            "Computer Science Theory": "cs.TH",
            "Probability": "math.PR",
            "Statistics Theory": "math.ST",
            "Machine Learning (Statistics)": "stat.ML",
            "Quantitative Methods": "q-bio.QM",
            "Computational Physics": "physics.comp-ph",
            "Data Analysis and Statistics": "physics.data-an",
            "Physics and Society": "physics.soc-ph",
            "Econometrics": "econ.EM",
            "Theoretical Economics": "econ.TH",
            "Computational Finance": "q-fin.CP",
            "Portfolio Management": "q-fin.PM",
            "Pricing of Securities": "q-fin.PR",
            "Risk Management": "q-fin.RM",
            "Statistical Finance": "q-fin.ST",
            "Trading and Market Microstructure": "q-fin.TR"
        }
        
        # Add category to query if specified
        if category and category != "All":
            arxiv_code = category_mapping.get(category)
            if arxiv_code:
                query = f"cat:{arxiv_code} AND {query}"
                print(f"Modified query with category: {query}")
            
        # If year range is specified, fetch more papers to account for filtering
        search_max_results = max_results * 10 if year_range else max_results
        print(f"Fetching {search_max_results} papers initially")
            
        search = arxiv.Search(
            query=query,
            max_results=search_max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        total_fetched = 0
        filtered_out = 0
        years_seen = set()
        
        for result in search.results():
            total_fetched += 1
            paper_year = result.published.year
            years_seen.add(paper_year)
            
            # Filter by year range if specified
            if year_range:
                if not (year_range[0] <= paper_year <= year_range[1]):
                    filtered_out += 1
                    print(f"Filtered out paper from year {paper_year}")
                    continue
                    
            paper = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "arxiv_id": result.entry_id.split('/')[-1],
                "published_date": result.published,
                "url": result.entry_id
            }
            papers.append(paper)
            print(f"Added paper: {paper['title']} ({paper['published_date'].year})")
            
            # Stop if we have enough papers after filtering
            if len(papers) >= max_results:
                break
                
        print(f"Total papers fetched: {total_fetched}")
        print(f"Papers filtered out by year: {filtered_out}")
        print(f"Years seen in results: {sorted(years_seen)}")
        print(f"Final number of papers: {len(papers)}")
        
        # If we couldn't find enough papers in the year range, try without year filtering
        if year_range and len(papers) < max_results:
            print("Not enough papers found in year range, falling back to all years")
            papers = []
            total_fetched = 0
            for result in search.results():
                total_fetched += 1
                paper = {
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary,
                    "arxiv_id": result.entry_id.split('/')[-1],
                    "published_date": result.published,
                    "url": result.entry_id
                }
                papers.append(paper)
                print(f"Added paper (fallback): {paper['title']} ({paper['published_date'].year})")
                
                if len(papers) >= max_results:
                    break
                    
            print(f"Fallback - Total papers fetched: {total_fetched}")
            print(f"Fallback - Final number of papers: {len(papers)}")
            
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