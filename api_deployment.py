from fastapi import FastAPI
from rag_research_assistant import RAGResearchAssistant

app = FastAPI()
assistant = RAGResearchAssistant()

@app.get("/research/")
def research(query: str):
    response = assistant.generate_answer(query)
    return {"query": query, "response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
