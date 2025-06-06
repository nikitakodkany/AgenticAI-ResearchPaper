import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from app.config import settings
from app.database import init_db

def create_database():
    # Connect to PostgreSQL server
    conn = psycopg2.connect(
        dbname='postgres',
        user=settings.DATABASE_URL.split('://')[1].split(':')[0],
        password=settings.DATABASE_URL.split(':')[2].split('@')[0],
        host=settings.DATABASE_URL.split('@')[1].split(':')[0],
        port=settings.DATABASE_URL.split(':')[3].split('/')[0]
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    # Create database if it doesn't exist
    db_name = settings.DATABASE_URL.split('/')[-1]
    cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{db_name}'")
    exists = cur.fetchone()
    if not exists:
        cur.execute(f'CREATE DATABASE {db_name}')
    
    cur.close()
    conn.close()

def setup_vector_extension():
    # Connect to the new database
    conn = psycopg2.connect(settings.DATABASE_URL)
    cur = conn.cursor()
    
    # Create the vector extension
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    print("Creating database...")
    create_database()
    
    print("Setting up vector extension...")
    setup_vector_extension()
    
    print("Initializing database tables...")
    init_db()
    
    print("Database setup complete!") 