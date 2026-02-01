# agent_core.py

import os
from dotenv import load_dotenv

from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.llms import HuggingFaceEndpoint

from rag_core import rag_answer

load_dotenv()

# -------- RAG TOOL --------
def document_rag_tool(query: str) -> str:
    """Use this tool when the question is about internal PDFs or documents."""
    return rag_answer(query)

tools = [
    Tool(
        name="Document_RAG",
        func=document_rag_tool,
        description="Use this for questions related to uploaded PDFs or internal documents.",
    ),
    Tool(
        name="DuckDuckGo_Search",
        func=DuckDuckGoSearchRun().run,
        description="Use this for current information or general web knowledge.",
    ),
]

# -------- HF AGENT LLM --------
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    temperature=0.0,
    max_new_tokens=512,
)

# -------- AGENT --------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
