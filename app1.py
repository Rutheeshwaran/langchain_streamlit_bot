# app1.py

import streamlit as st
from rag_core import rag_answer
from router import route_query
from serp_search import serp_search
from langchain_community.llms import Ollama

# Local Ollama model (used for summarizing web results)
llm = Ollama(model="llama3.1:latest", temperature=0)

st.title("🤖 RAG + Web Router (Ollama)")

query = st.text_input("Ask your question")

if query:
    with st.spinner("Thinking..."):

        route = route_query(query)

        if route == "rag":
            st.info("Tool used: Document RAG")
            response = rag_answer(query)

        else:
            st.info("Tool used: Google Search (SerpAPI)")

            # Get Google results
            search_results = serp_search(query)

            # Summarize using Ollama
            summary_prompt = f"""
            Using the Google search results below, answer clearly and concisely.

            Question:
            {query}

            Search Results:
            {search_results}

            Answer:
            """

            response = llm.invoke(summary_prompt)

        st.write(response)
