from typing import List, Optional, Dict, Any
import arxiv
from datetime import datetime
from app.database import papers_collection, summaries_collection, normalize_vector, chunk_text
from app.config import settings
from sentence_transformers import SentenceTransformer
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the embedding model with fixed version
EMBEDDING_MODEL_VERSION = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_VERSION)

class PaperService:
    def __init__(self):
        self.papers_collection = papers_collection
        self.summaries_collection = summaries_collection
        self.top_k = 5  # Default number of results to retrieve

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using the configured model"""
        try:
            # Ensure consistent tokenization by using the same model
            embedding = embedding_model.encode(text, normalize_embeddings=True)
            return normalize_vector(embedding.tolist())
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * settings.VECTOR_DIMENSION
            
    def process_paper(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process paper into chunks with metadata"""
        # Combine title and abstract for better context
        full_text = f"{paper['title']} {paper['abstract']}"
        
        # Create chunks with semantic boundaries
        chunks = chunk_text(full_text)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Generate embedding for chunk
            embedding = self.generate_embedding(chunk)
            
            # Create metadata for context-aware ranking
            metadata = {
                "paper_id": paper["arxiv_id"],
                "title": paper["title"],
                "authors": json.dumps(paper["authors"]),
                "published_date": paper["published_date"].isoformat(),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "year": paper["published_date"].year,
                "category": paper.get("category", "unknown"),
                "url": paper["url"]
            }
            
            processed_chunks.append({
                "id": f"{paper['arxiv_id']}_chunk_{i}",
                "text": chunk,
                "embedding": embedding,
                "metadata": metadata
            })
            
        return processed_chunks
        
    def store_paper(self, paper: Dict[str, Any], embedding: List[float]) -> None:
        """Store paper in ChromaDB with proper chunking and metadata"""
        try:
            # Process paper into chunks
            chunks = self.process_paper(paper)
            
            # Store each chunk
            for chunk in chunks:
                self.papers_collection.add(
                    ids=[chunk["id"]],
                    embeddings=[chunk["embedding"]],
                    documents=[chunk["text"]],
                    metadatas=[chunk["metadata"]]
                )
            logger.info(f"Stored paper {paper['arxiv_id']} in {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error storing paper {paper['arxiv_id']}: {str(e)}")
            
    def search_papers(
        self, 
        query: str, 
        max_results: int = settings.MAX_PAPERS_PER_QUERY,
        category: Optional[str] = None,
        year_range: Optional[List[int]] = None
    ) -> List[dict]:
        """Search papers on arXiv with improved filtering and retrieval"""
        logger.info(f"Searching papers with query: {query}")
        logger.info(f"Max results: {max_results}")
        logger.info(f"Category: {category}")
        logger.info(f"Year range: {year_range}")
        
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
                logger.info(f"Modified query with category: {query}")
            
        # If year range is specified, fetch more papers to account for filtering
        search_max_results = max_results * 10 if year_range else max_results
        logger.info(f"Fetching {search_max_results} papers initially")
            
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
                    logger.info(f"Filtered out paper from year {paper_year}")
                    continue
                    
            paper = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "arxiv_id": result.entry_id.split('/')[-1],
                "published_date": result.published,
                "url": result.entry_id,
                "category": category if category else "unknown"
            }
            
            # Store paper in ChromaDB with proper chunking and metadata
            try:
                self.store_paper(paper, None)  # Embedding will be generated in process_paper
                logger.info(f"Stored paper in ChromaDB: {paper['title']}")
            except Exception as e:
                logger.error(f"Error storing paper in ChromaDB: {str(e)}")
            
            papers.append(paper)
            logger.info(f"Added paper: {paper['title']} ({paper['published_date'].year})")
            
            # Stop if we have enough papers after filtering
            if len(papers) >= max_results:
                break
                
        logger.info(f"Total papers fetched: {total_fetched}")
        logger.info(f"Papers filtered out by year: {filtered_out}")
        logger.info(f"Years seen in results: {sorted(years_seen)}")
        logger.info(f"Final number of papers: {len(papers)}")
        
        # If we couldn't find enough papers in the year range, try without year filtering
        if year_range and len(papers) < max_results:
            logger.info("Not enough papers found in year range, falling back to all years")
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
                    "url": result.entry_id,
                    "category": category if category else "unknown"
                }
                
                # Store paper in ChromaDB
                try:
                    self.store_paper(paper, None)
                    logger.info(f"Stored paper in ChromaDB (fallback): {paper['title']}")
                except Exception as e:
                    logger.error(f"Error storing paper in ChromaDB: {str(e)}")
                
                papers.append(paper)
                logger.info(f"Added paper (fallback): {paper['title']} ({paper['published_date'].year})")
                
                if len(papers) >= max_results:
                    break
                    
            logger.info(f"Fallback - Total papers fetched: {total_fetched}")
            logger.info(f"Fallback - Final number of papers: {len(papers)}")
            
        return papers

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

    def search_similar_papers(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for similar papers using semantic search"""
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Search with metadata filters
            results = self.papers_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k or self.top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            similar_papers = []
            for i in range(len(results["ids"][0])):
                paper = {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
                }
                similar_papers.append(paper)
                
            return similar_papers
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []

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