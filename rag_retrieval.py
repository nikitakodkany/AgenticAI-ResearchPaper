import openai
import psycopg2
import numpy as np

DB_NAME = "research_db"
DB_USER = "postgres"
DB_PASS = "your_password"
DB_HOST = "localhost"
DB_PORT = "5432"

openai.api_key = "your_openai_api_key"

def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

def retrieve_relevant_papers(query, top_n=3):
    query_embedding = get_embedding(query)

    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
    cur = conn.cursor()

    cur.execute("SELECT id, title, summary, url, embedding FROM research_papers;")
    results = cur.fetchall()

    similarities = []
    for result in results:
        paper_id, title, summary, url, embedding = result
        embedding_array = np.array(embedding, dtype=np.float32)
        similarity = np.dot(query_embedding, embedding_array) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding_array))
        similarities.append((similarity, title, summary, url))

    cur.close()
    conn.close()

    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:top_n]

if __name__ == "__main__":
    query = "Quantum encryption methods"
    results = retrieve_relevant_papers(query)
    for sim, title, summary, url in results:
        print(f"\n🔍 {title} ({url})\nSimilarity: {sim:.3f}\nSummary: {summary[:200]}...")
