from typing import Dict, List, TypedDict, Annotated
from langgraph.graph import Graph, StateGraph
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from ..config import CHAT_MODEL, USE_OPENAI
import os

class WorkflowState(TypedDict):
    """State for the research workflow."""
    documents: List[Dict]
    summaries: List[str]
    critiques: List[str]
    final_report: str

class SummarizerAgent:
    def __init__(self):
        if USE_OPENAI:
            self.llm = ChatOpenAI(
                model_name="gpt-4-turbo-preview",
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            from langchain_community.llms import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL)
            model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=1000,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
    
    def summarize(self, state: WorkflowState) -> WorkflowState:
        """Summarize each document in the state."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a research paper summarizer. Create a concise summary of the following paper."),
            ("human", "{document}")
        ])
        
        chain = prompt | self.llm
        
        summaries = []
        for doc in state["documents"]:
            doc_text = f"Title: {doc['title']}\nAuthors: {doc['authors']}\nContent: {doc['content']}"
            summary = chain.invoke({"document": doc_text})
            summaries.append(summary)
        
        state["summaries"] = summaries
        return state

class CriticAgent:
    def __init__(self):
        if USE_OPENAI:
            self.llm = ChatOpenAI(
                model_name="gpt-4-turbo-preview",
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            from langchain_community.llms import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL)
            model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=1000,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
    
    def critique(self, state: WorkflowState) -> WorkflowState:
        """Critique each summary in the state."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a research paper critic. Analyze the following summary and identify any limitations or missing context."),
            ("human", "{summary}")
        ])
        
        chain = prompt | self.llm
        
        critiques = []
        for summary in state["summaries"]:
            critique = chain.invoke({"summary": summary})
            critiques.append(critique)
        
        state["critiques"] = critiques
        return state

class ReportGeneratorAgent:
    def __init__(self):
        if USE_OPENAI:
            self.llm = ChatOpenAI(
                model_name="gpt-4-turbo-preview",
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            from langchain_community.llms import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL)
            model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=1000,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
    
    def generate_report(self, state: WorkflowState) -> WorkflowState:
        """Generate a final report from summaries and critiques."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a research report generator. Create a structured synthesis of the following summaries and critiques."),
            ("human", "Summaries:\n{summaries}\n\nCritiques:\n{critiques}")
        ])
        
        chain = prompt | self.llm
        
        summaries_text = "\n\n".join(state["summaries"])
        critiques_text = "\n\n".join(state["critiques"])
        
        report = chain.invoke({
            "summaries": summaries_text,
            "critiques": critiques_text
        })
        
        state["final_report"] = report
        return state

class ResearchWorkflow:
    def __init__(self):
        # Initialize agents
        self.summarizer = SummarizerAgent()
        self.critic = CriticAgent()
        self.report_generator = ReportGeneratorAgent()
        
        # Create the workflow graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("summarize", self.summarizer.summarize)
        workflow.add_node("critique", self.critic.critique)
        workflow.add_node("generate_report", self.report_generator.generate_report)
        
        # Add edges
        workflow.add_edge("summarize", "critique")
        workflow.add_edge("critique", "generate_report")
        
        # Set entry point
        workflow.set_entry_point("summarize")
        
        # Compile the workflow
        self.workflow = workflow.compile()
    
    def run(self, documents: List[Dict]) -> str:
        """Run the research workflow on a list of documents."""
        # Initialize state
        state = WorkflowState(
            documents=documents,
            summaries=[],
            critiques=[],
            final_report=""
        )
        
        # Execute the workflow
        final_state = self.workflow.invoke(state)
        
        return final_state["final_report"] 