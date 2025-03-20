import langgraph
from langgraph.graph import Graph
from langchain.chat_models import ChatOpenAI
from store_papers import store_papers
import psycopg2

DB_NAME = "research_db"
DB_USER = "postgres"
DB_PASS = "your_password"
DB_HOST = "localhost"
DB_PORT = "5432"

class SummarizationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4")

    def summarize_paper(self, paper):
        return self.llm.invoke(f"Summarize this research paper:\n{paper}")

class CritiqueAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4")

    def critique_paper(self, summary):
        return self.llm.invoke(f"Analyze the limitations of this paper:\n{summary}")

class ReportGeneratorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4")

    def generate_report(self, summaries, critiques):
        return self.llm.invoke(f"Generate a report based on:\nSummaries:\n{summaries}\nCritiques:\n{critiques}")

graph = Graph()

graph.add_node("summarize_papers", SummarizationAgent().summarize_paper)
graph.add_node("critique_papers", CritiqueAgent().critique_paper)
graph.add_node("generate_report", ReportGeneratorAgent().generate_report)

graph.add_edge("summarize_papers", "critique_papers")
graph.add_edge("critique_papers", "generate_report")

research_assistant = graph.compile()

def run_research(topic):
    store_papers(topic)
    summaries = research_assistant.invoke({"summarize_papers": topic})
    critiques = research_assistant.invoke({"critique_papers": summaries})
    report = research_assistant.invoke({"generate_report": {"summaries": summaries, "critiques": critiques}})
    return report
