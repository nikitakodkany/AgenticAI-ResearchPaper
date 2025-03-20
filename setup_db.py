import psycopg2

DB_NAME = "research_db"
DB_USER = "postgres"
DB_PASS = "your_password"
DB_HOST = "localhost"
DB_PORT = "5432"

def create_table():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS research_papers (
        id SERIAL PRIMARY KEY,
        title TEXT,
        summary TEXT,
        url TEXT,
        embedding vector(1536) -- OpenAI embedding size
    );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Database setup complete!")

if __name__ == "__main__":
    create_table()
