import requests
import xml.etree.ElementTree as ET
from typing import List, Dict
from core.database.store import PaperStore
from core.config import ARXIV_BASE_URL, MAX_RESULTS_PER_QUERY

def parse_arxiv_response(xml_content: str) -> List[Dict]:
    """Parse arXiv API XML response."""
    root = ET.fromstring(xml_content)
    papers = []
    
    # Namespace for arXiv XML
    ns = {'arxiv': 'http://www.w3.org/2005/Atom'}
    
    for entry in root.findall('arxiv:entry', ns):
        paper = {
            'title': entry.find('arxiv:title', ns).text,
            'authors': ', '.join([author.find('arxiv:name', ns).text for author in entry.findall('arxiv:author', ns)]),
            'abstract': entry.find('arxiv:summary', ns).text,
            'url': entry.find('arxiv:id', ns).text,
            'published_date': entry.find('arxiv:published', ns).text,
            'content': f"{entry.find('arxiv:title', ns).text}\n\n{entry.find('arxiv:summary', ns).text}"
        }
        papers.append(paper)
    
    return papers

def fetch_papers_from_arxiv(query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> List[Dict]:
    """Fetch papers from arXiv API."""
    params = {
        "search_query": query,
        "max_results": max_results,
        "sortBy": "lastUpdatedDate",
        "sortOrder": "descending"
    }
    
    response = requests.get(ARXIV_BASE_URL, params=params)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch papers: {response.status_code}")
    
    return parse_arxiv_response(response.content)

def main():
    # Example queries
    queries = [
        "quantum computing",
        "machine learning",
        "artificial intelligence"
    ]
    
    # Initialize paper store
    store = PaperStore()
    
    # Fetch and store papers for each query
    for query in queries:
        print(f"Fetching papers for query: {query}")
        try:
            papers = fetch_papers_from_arxiv(query)
            for paper in papers:
                store.store_paper(paper)
                print(f"Stored paper: {paper['title']}")
        except Exception as e:
            print(f"Error fetching papers for query '{query}': {str(e)}")

if __name__ == "__main__":
    main() 