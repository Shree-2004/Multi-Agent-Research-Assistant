# agents/researcher.py
# ─────────────────────────────────────────────────────────────
# Agent 1 — Researcher
# This agent receives the user's topic, searches the web using
# Tavily + ArXiv, and returns 10-15 structured sources.
# It is the FIRST node in the LangGraph pipeline.
# ─────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

from tools.search import search_web, search_academic
from tools.arxiv_fetch import fetch_arxiv_papers, combine_sources
from graph.state import ResearchState

# Load environment variables
load_dotenv()


# ── Initialize Gemini LLM ──────────────────────────────────────
# This is the AI brain powering the Researcher agent
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,   # Low temperature = more focused, less creative
    max_tokens=int(os.getenv("MAX_TOKENS", 8192))
)


# ── System Prompt ──────────────────────────────────────────────
# This tells Gemini exactly what role it plays and how to behave.
# We keep it focused — no analysis, just finding good sources.
RESEARCHER_SYSTEM_PROMPT = """
You are a specialized Research Agent. Your ONLY job is to identify
the best search queries to find high-quality sources on a given topic.

Your responsibilities:
1. Analyze the research topic given to you
2. Generate 3 focused search queries that will find the best sources
3. Return ONLY the queries, one per line, nothing else

Rules:
- Queries must be specific and academic in nature
- Include year ranges like "2023 2024" to get recent results
- Do NOT write any explanation or commentary
- Do NOT number the queries
- Just return the raw query strings, one per line
"""


def researcher_node(state: ResearchState) -> ResearchState:
    """
    The main Researcher agent function.
    This is called by LangGraph as the first node in the pipeline.

    What it does:
    1. Takes the user's topic from state
    2. Asks Gemini to generate smart search queries
    3. Searches Tavily + ArXiv with those queries
    4. Returns combined sources back into state

    Args:
        state: The shared ResearchState object from graph/state.py

    Returns:
        Updated state with raw_sources filled in
    """
    topic = state["topic"]
    print(f"\n[Researcher] Starting research on: {topic}")

    # ── Step 1: Ask Gemini to generate smart search queries ────
    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(content=f"Generate search queries for this topic: {topic}")
    ]

    response = llm.invoke(messages)
    queries_text = response.content.strip()

    # Split the response into individual queries
    queries = [q.strip() for q in queries_text.split("\n") if q.strip()]
    print(f"[Researcher] Generated {len(queries)} search queries:")
    for q in queries:
        print(f"  → {q}")

    # ── Step 2: Search Tavily with each query ──────────────────
    all_tavily_results = []
    for query in queries:
        results = search_web(query, max_results=4)
        all_tavily_results.extend(results)
        print(f"[Researcher] Tavily found {len(results)} results for: {query}")

    # ── Step 3: Search ArXiv for academic papers ───────────────
    arxiv_results = fetch_arxiv_papers(topic, max_results=5)
    print(f"[Researcher] ArXiv found {len(arxiv_results)} papers")

    # ── Step 4: Combine and deduplicate all sources ────────────
    combined = combine_sources(all_tavily_results, arxiv_results)
    print(f"[Researcher] Total unique sources: {len(combined)}")

    # ── Step 5: Update state with sources ─────────────────────
    # We return only the fields we are updating
    return {
        "raw_sources": combined
    }


def test_researcher():
    """
    Test the Researcher agent in isolation.
    Run this file directly to verify it works before
    connecting it to the full pipeline.
    """
    print("Testing Researcher Agent...")

    # Create a minimal test state
    test_state = {
        "topic": "latest advances in protein folding AI",
        "raw_sources": [],
        "analysis_notes": "",
        "draft_report": "",
        "critic_feedback": None,
        "final_report": None,
        "iteration_count": 0
    }

    # Run the agent
    result = researcher_node(test_state)

    # Check output
    sources = result.get("raw_sources", [])
    print(f"\n✓ Researcher returned {len(sources)} sources")
    if sources:
        print("\nFirst source:")
        print(f"  Title: {sources[0]['title']}")
        print(f"  URL:   {sources[0]['url']}")
        print(f"  Date:  {sources[0]['date']}")


# ── Run test if file is executed directly ─────────────────────
if __name__ == "__main__":
    test_researcher()