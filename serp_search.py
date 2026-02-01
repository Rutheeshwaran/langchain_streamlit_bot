# serp_search.py

from serpapi import GoogleSearch
import os
from dotenv import load_dotenv

load_dotenv()

def serp_search(query, num_results=5):
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_KEY"),
        "num": num_results
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    output = ""

    for r in results.get("organic_results", []):
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        output += f"{title}\n{snippet}\n{link}\n\n"

    if not output:
        return "No good search result found."

    return output
