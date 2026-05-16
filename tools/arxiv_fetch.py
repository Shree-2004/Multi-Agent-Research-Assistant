# tools/arxiv_fetch.py
# ─────────────────────────────────────────────────────────────
# This file fetches academic papers directly from ArXiv.
# It is used as a BACKUP when Tavily doesn't find enough
# academic/research specific results.
# ─────────────────────────────────────────────────────────────

import arxiv
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def fetch_arxiv_papers(query: str, max_results: int = 5) -> list[dict]:
    """
    Fetches research papers from ArXiv database.

    Args:
        query: The research topic to search for
        max_results: How many papers to return (default 5)

    Returns:
        List of dicts with keys: title, url, abstract, date
    """
    try:
        # Create ArXiv search client
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,        # Most relevant first
            sort_order=arxiv.SortOrder.Descending
        )

        # Fetch and format results
        papers = []
        for result in search.results():
            papers.append({
                "title":    result.title,
                "url":      result.entry_id,              # ArXiv paper URL
                "abstract": result.summary[:500] + "...", # Trim long abstracts
                "date":     str(result.published.date())  # Publication date
            })

        return papers

    except Exception as e:
        print(f"[arxiv_fetch.py] ArXiv fetch failed: {e}")
        return []


def fetch_recent_papers(query: str, max_results: int = 5) -> list[dict]:
    """
    Fetches the MOST RECENT papers from ArXiv on a topic.
    Used when we want latest research, not just most relevant.

    Args:
        query: The research topic
        max_results: How many papers to return

    Returns:
        List of dicts with keys: title, url, abstract, date
    """
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,    # Most recent first
            sort_order=arxiv.SortOrder.Descending
        )

        papers = []
        for result in search.results():
            papers.append({
                "title":    result.title,
                "url":      result.entry_id,
                "abstract": result.summary[:500] + "...",
                "date":     str(result.published.date())
            })

        return papers

    except Exception as e:
        print(f"[arxiv_fetch.py] ArXiv recent fetch failed: {e}")
        return []


def combine_sources(
    tavily_results: list[dict],
    arxiv_results: list[dict]
) -> list[dict]:
    """
    Merges Tavily web results + ArXiv academic results.
    Removes duplicates based on URL.

    Args:
        tavily_results: Results from tools/search.py
        arxiv_results: Results from fetch_arxiv_papers()

    Returns:
        Combined deduplicated list of sources
    """
    seen_urls = set()
    combined = []

    for source in tavily_results + arxiv_results:
        url = source.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            combined.append(source)

    return combined


def test_arxiv():
    """
    Quick test — run this file directly to verify
    ArXiv is connected and returning papers correctly.
    """
    print("Testing ArXiv fetch...")
    papers = fetch_arxiv_papers("protein folding transformer", max_results=3)

    if papers:
        print(f"✓ Found {len(papers)} papers")
        for i, p in enumerate(papers, 1):
            print(f"\n  Paper {i}:")
            print(f"  Title: {p['title']}")
            print(f"  URL:   {p['url']}")
            print(f"  Date:  {p['date']}")
    else:
        print("✗ No papers returned — check your internet connection")


# ── Run test if file is executed directly ─────────────────────
if __name__ == "__main__":
    test_arxiv()