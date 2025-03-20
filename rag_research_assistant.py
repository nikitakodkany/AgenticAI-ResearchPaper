from rag_retrieval import retrieve_relevant_papers
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

class RAGResearchAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4")

    def generate_answer(self, query):
        retrieved_papers = retrieve_relevant_papers(query)

        context = "\n".join([f"{title}: {summary}" for _, title, summary, _ in retrieved_papers])
        prompt = PromptTemplate.from_template(
            "Using the following research papers, answer the question:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        return self.llm.invoke(prompt.format(context=context, query=query))

if __name__ == "__main__":
    assistant = RAGResearchAssistant()
    response = assistant.generate_answer("How does quantum cryptography enhance security?")
    print("\n🤖 AI Response:\n", response)
