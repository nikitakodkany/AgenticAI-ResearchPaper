from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.rag.assistant import RAGResearchAssistant
from core.config import API_HOST, API_PORT

app = FastAPI(title="Research Paper Assistant API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the research assistant
assistant = RAGResearchAssistant()

@app.get("/research/")
async def research(query: str):
    """Get research results for a given query."""
    print("DEBUG: /research/ endpoint called with query:", query)
    response = assistant.generate_answer(query)
    print("DEBUG: LLM response:", response)  # Debug print
    return {"query": query, "response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT) 