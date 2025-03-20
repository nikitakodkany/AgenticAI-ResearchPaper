import arxiv

def fetch_papers(topic, max_results=5):
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    
    for result in search.results():
        papers.append({
            "title": result.title,
            "summary": result.summary,
            "url": result.entry_id
        })
    return papers

if __name__ == "__main__":
    topic = "Quantum Computing in Cryptography"
    papers = fetch_papers(topic)
    for p in papers:
        print(f"📖 {p['title']} ({p['url']})")
