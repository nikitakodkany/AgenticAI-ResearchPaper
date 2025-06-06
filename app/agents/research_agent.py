from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from transformers import pipeline
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from app.config import settings
from app.services.paper_service import PaperService
from sqlalchemy.orm import Session

class ResearchAgent:
    def __init__(self, db: Session, vector_backend: Optional[str] = None, embedding_provider: Optional[str] = None, llm_provider: Optional[str] = None):
        self.db = db
        self.vector_backend = vector_backend or settings.VECTOR_BACKEND
        self.embedding_provider = embedding_provider or "hf-all-MiniLM-L6-v2"
        self.llm_provider = llm_provider or "hf-mistral-7b"
        self.paper_service = PaperService(vector_backend=self.vector_backend, embedding_provider=self.embedding_provider)
        if self.llm_provider.startswith("openai"):
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model="gpt-4-turbo-preview")
        else:
            from transformers import pipeline
            self.llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")
        self._setup_agent()

    def _setup_agent(self):
        # Define tools
        self.tools = [
            Tool(
                name="search_papers",
                func=self.paper_service.search_similar_papers,
                description="Search for similar research papers based on a query"
            ),
            Tool(
                name="fetch_papers",
                func=self.paper_service.fetch_papers,
                description="Fetch new research papers from ArXiv based on a query"
            )
        ]

        # Create the agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research assistant that helps analyze and summarize research papers.\nUse the available tools to search for and fetch relevant papers.\nWhen analyzing papers, focus on:\n1. Main findings and contributions\n2. Methodology and approach\n3. Key insights and implications\n4. Potential limitations or areas for future work"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Replace agent with a simple LLM call for now
        self.agent_executor = self

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a research query and return relevant information."""
        # First, search for similar papers
        similar_papers = await self.paper_service.search_similar_papers(query, self.db)
        
        if not similar_papers:
            # If no similar papers found, fetch new ones
            new_papers = await self.paper_service.fetch_papers(query)
            for paper in new_papers:
                await self.paper_service.process_paper(paper, self.db)
            similar_papers = await self.paper_service.search_similar_papers(query, self.db)

        # Prepare context from similar papers
        context = "\n\n".join([
            f"Title: {paper.title}\n"
            f"Authors: {paper.authors}\n"
            f"Abstract: {paper.abstract}\n"
            f"URL: {paper.url}"
            for paper in similar_papers
        ])

        # Generate response using the agent
        prompt = f"Based on the following research papers, please analyze and summarize the key findings related to: {query}\n\nContext:\n{context}"
        response = self.llm(prompt, max_new_tokens=512)[0]['generated_text']

        return {
            "query": query,
            "papers": [
                {
                    "title": paper.title,
                    "authors": paper.authors,
                    "url": paper.url
                }
                for paper in similar_papers
            ],
            "analysis": response
        }

    def create_research_graph(self):
        """Create a LangGraph for more complex research workflows."""
        workflow = StateGraph(StateType=Dict)

        # Define nodes
        workflow.add_node("search", self._search_node)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("summarize", self._summarize_node)

        # Define edges
        workflow.add_edge("search", "analyze")
        workflow.add_edge("analyze", "summarize")
        workflow.add_edge("summarize", END)

        # Set entry point
        workflow.set_entry_point("search")

        return workflow.compile()

    async def _search_node(self, state: Dict) -> Dict:
        """Node for searching papers."""
        query = state["query"]
        papers = await self.paper_service.search_similar_papers(query, self.db)
        return {"papers": papers, **state}

    async def _analyze_node(self, state: Dict) -> Dict:
        """Node for analyzing papers."""
        papers = state["papers"]
        analysis = await self.agent_executor.ainvoke({
            "input": f"Analyze these papers: {papers}"
        })
        return {"analysis": analysis["output"], **state}

    async def _summarize_node(self, state: Dict) -> Dict:
        """Node for summarizing findings."""
        summary = await self.agent_executor.ainvoke({
            "input": f"Summarize the analysis: {state['analysis']}"
        })
        return {"summary": summary["output"], **state} 