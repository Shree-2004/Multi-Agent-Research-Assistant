# tools/search.py
# ─────────────────────────────────────────────────────────────
# This file wraps the Tavily API into a reusable search tool.
# The Researcher agent calls this to find web + academic sources.
# ─────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv
from tavily import TavilyClient

# Load API keys from .env file
load_dotenv()


# ── Initialize Tavily Client ───────────────────────────────────
# TavilyClient reads TAVILY_API_KEY automatically from environment
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def search_web(query: str, max_results: int = None) -> list[dict]:
    """
    Searches the web using Tavily API.
    
    Args:
        query: The search query string
        max_results: How many results to return (default from .env)
    
    Returns:
        List of dicts with keys: title, url, content, score
    """
    # Use .env value if max_results not specified
    if max_results is None:
        max_results = int(os.getenv("TAVILY_MAX_RESULTS", 10))

    try:
        # search_depth="advanced" gives better quality results
        # include_answer=True gives a summary of the search
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
            include_raw_content=False  # Raw content is too large
        )

        # Extract just the results list from the response
        results = response.get("results", [])

        # Format each result to match our Source schema in state.py
        formatted = []
        for r in results:
            formatted.append({
                "title":    r.get("title", "No title"),
                "url":      r.get("url", ""),
                "abstract": r.get("content", ""),  # Tavily calls it 'content'
                "date":     r.get("published_date", "Unknown date")
            })

        return formatted

    except Exception as e:
        # Return empty list if search fails — agent will handle this
        print(f"[search.py] Tavily search failed: {e}")
        return []


def search_academic(query: str, max_results: int = 5) -> list[dict]:
    """
    Searches specifically for academic/research content using Tavily.
    Adds 'research paper' to query for better academic results.
    
    Args:
        query: The research topic
        max_results: How many academic results to return
    
    Returns:
        List of dicts with keys: title, url, abstract, date
    """
    # Append academic keywords to improve result quality
    academic_query = f"{query} research paper study findings 2024"

    return search_web(academic_query, max_results)


def test_search():
    """
    Quick test function — run this file directly to verify
    Tavily is connected and returning results correctly.
    """
    print("Testing Tavily search...")
    results = search_web("protein folding AI advances", max_results=3)

    if results:
        print(f"✓ Found {len(results)} results")
        for i, r in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"  Title: {r['title']}")
            print(f"  URL:   {r['url']}")
            print(f"  Date:  {r['date']}")
    else:
        print("✗ No results returned — check your TAVILY_API_KEY in .env")


# ── Run test if file is executed directly ─────────────────────
if __name__ == "__main__":
    test_search()