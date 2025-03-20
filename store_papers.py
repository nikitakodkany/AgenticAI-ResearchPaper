import openai
import psycopg2
from fetch_papers import fetch_papers

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

def store_papers(topic):
    papers = fetch_papers(topic)

    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
    cur = conn.cursor()

    for paper in papers:
        embedding = get_embedding(paper["summary"])
        cur.execute(
            "INSERT INTO research_papers (title, summary, url, embedding) VALUES (%s, %s, %s, %s)",
            (paper["title"], paper["summary"], paper["url"], embedding)
        )

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Papers stored in the database.")

if __name__ == "__main__":
    store_papers("Quantum Computing in Cryptography")
