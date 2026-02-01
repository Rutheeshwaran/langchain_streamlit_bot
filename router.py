# router.py

from langchain_community.llms import Ollama

# Local Ollama model (already pulled)
llm = Ollama(
    model="llama3.1",
    temperature=0
)

def route_query(query: str) -> str:
    """
    Uses Ollama LLM to decide routing.
    Returns: 'rag' or 'web'
    """

    prompt = f"""
You are a routing classifier.

Decide whether the user's question should be answered using:
- "rag" → internal documents / PDFs / notes
- "web" → general knowledge or current information

Return ONLY one word: rag or web.

Question:
{query}

Answer:
"""

    decision = llm(prompt).strip().lower()

    if "rag" in decision:
        return "rag"
    else:
        return "web"
