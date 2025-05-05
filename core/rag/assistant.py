from typing import List, Tuple
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from .retrieval import RAGRetriever
import os
from ..config import CHAT_MODEL, USE_OPENAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

class CriticAgent:
    def __init__(self, use_openai: bool):
        if use_openai:
            from langchain_community.chat_models import ChatOpenAI
            self.llm = ChatOpenAI(
                model_name="gpt-4-turbo-preview",
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
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

    def review_answer(self, answer: str, documents: List[dict]) -> Tuple[bool, str]:
        """Review the answer to ensure it only contains information from the documents."""
        documents_text = "\n\n".join([
            f"Document {i+1}:\nTitle: {doc['title']}\nAuthors: {doc['authors']}\nContent: {doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        prompt = PromptTemplate(
            input_variables=["documents", "answer"],
            template="""Review the following answer to ensure it only contains information from the provided documents. 
If the answer contains information not present in the documents, return 'False' and explain why. 
If the answer only uses information from the documents, return 'True' and confirm.

Documents:
{documents}

Answer to review:
{answer}

Review:"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(documents=documents_text, answer=answer)
        is_valid = "true" in response.lower()
        return is_valid, response

class RAGResearchAssistant:
    def __init__(self):
        self.retriever = RAGRetriever()
        self.critic = CriticAgent(USE_OPENAI)
        
        if USE_OPENAI:
            from langchain_community.chat_models import ChatOpenAI
            self.llm = ChatOpenAI(
                model_name="gpt-4-turbo-preview",
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
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
        
        # Initialize memory for conversation
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Create tools for the agent
        self.tools = [
            Tool(
                name="search_papers",
                func=self.retriever.retrieve,
                description="Search for relevant research papers"
            )
        ]
        
        # Create the agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self._get_agent_prompt()
        )
        
        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )
    
    def _get_agent_prompt(self) -> PromptTemplate:
        """Get the prompt template for the agent."""
        return PromptTemplate(
            input_variables=["input", "chat_history", "agent_scratchpad"],
            template="""You are a research paper assistant. Your task is to help users find and understand research papers.
You have access to a tool that can search for relevant papers.

Previous conversation:
{chat_history}

Current query: {input}

Think step by step:
1. Search for relevant papers using the search_papers tool
2. Analyze the papers to find the answer
3. Ensure your answer is based only on the retrieved papers
4. If the answer is not in the papers, say 'Not found.'

{agent_scratchpad}"""
        )
    
    def _format_documents(self, documents: List[dict]) -> str:
        """Format documents with a strict token limit."""
        MAX_TOKENS_PER_DOC = 1000  # Adjust based on your needs
        formatted_docs = []
        
        for i, doc in enumerate(documents):
            # Truncate content if needed
            content = doc['content']
            if len(content.split()) > MAX_TOKENS_PER_DOC:
                content = ' '.join(content.split()[:MAX_TOKENS_PER_DOC]) + "... [truncated]"
            
            formatted_docs.append(
                f"Document {i+1}:\nTitle: {doc['title']}\nAuthors: {doc['authors']}\nContent: {content}"
            )
        
        return "\n\n".join(formatted_docs)
    
    def generate_answer(self, query: str, max_retries: int = 3) -> str:
        """Generate an answer using the agent executor."""
        try:
            # Use the agent executor to handle the query
            response = self.agent_executor.invoke({"input": query})
            return response["output"]
        except Exception as e:
            print(f"Error in agent execution: {e}")
            return "I apologize, but I encountered an error while processing your query." 