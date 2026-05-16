# agents/analyst.py
# ─────────────────────────────────────────────────────────────
# Agent 2 — Analyst
# This agent receives raw sources from the Researcher and
# produces structured analysis notes. It identifies key findings,
# contradictions between sources, and emerging trends.
# It does NOT write the report — only analysis notes.
# It is the SECOND node in the LangGraph pipeline.
# ─────────────────────────────────────────────────────────────

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

from graph.state import ResearchState

# Load environment variables
load_dotenv()


# ── Initialize Gemini LLM ──────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
    max_tokens=int(os.getenv("MAX_TOKENS", 8192))
)


# ── System Prompt ──────────────────────────────────────────────
# We tell the Analyst to be structured and analytical.
# We explicitly tell it NOT to write a report — only notes.
# This separation of concerns makes each agent focused and better.
ANALYST_SYSTEM_PROMPT = """
You are a specialized Research Analyst Agent. Your job is to read
a collection of research sources and produce structured analysis notes.

Your output must follow this EXACT structure:

## KEY FINDINGS
- List the 5-7 most important findings across all sources
- Each finding must reference which source it came from
- Be specific and factual, no vague statements

## CONTRADICTIONS & DEBATES
- List any points where sources disagree with each other
- Explain what each side argues
- If no contradictions exist, write "No major contradictions found"

## EMERGING TRENDS
- List 3-5 trends you see across the sources
- Focus on what is NEW or CHANGING in this field

## RESEARCH GAPS
- List 2-3 areas that the sources suggest need more research
- These become valuable insights for the final report

## SOURCE QUALITY ASSESSMENT
- Rate the overall quality of sources: High / Medium / Low
- Note any sources that seem outdated or unreliable

Rules:
- Be objective and analytical
- Do NOT write prose paragraphs — use bullet points
- Do NOT write a report — only analysis notes
- Always cite sources by their title
"""


def analyst_node(state: ResearchState) -> ResearchState:
    """
    The main Analyst agent function.
    Called by LangGraph as the second node in the pipeline.

    What it does:
    1. Takes raw_sources from state (filled by Researcher)
    2. Also checks for critic_feedback (if this is a retry)
    3. Asks Gemini to analyze all sources
    4. Returns structured analysis notes back into state

    Args:
        state: The shared ResearchState object

    Returns:
        Updated state with analysis_notes filled in
    """
    raw_sources = state["raw_sources"]
    critic_feedback = state.get("critic_feedback", None)
    iteration = state.get("iteration_count", 0)

    print(f"\n[Analyst] Analyzing {len(raw_sources)} sources...")
    if critic_feedback:
        print(f"[Analyst] Incorporating critic feedback (iteration {iteration})")

    # ── Step 1: Format sources into readable text ──────────────
    # We convert the list of source dicts into a readable string
    # that we can pass to Gemini in the prompt
    sources_text = ""
    for i, source in enumerate(raw_sources, 1):
        sources_text += f"""
Source {i}:
Title: {source.get('title', 'Unknown')}
URL: {source.get('url', 'Unknown')}
Date: {source.get('date', 'Unknown')}
Abstract: {source.get('abstract', 'No abstract available')}
---
"""

    # ── Step 2: Build the prompt ───────────────────────────────
    # If there's critic feedback, we add it to the prompt
    # so the Analyst knows what to improve
    user_message = f"Analyze these research sources:\n\n{sources_text}"

    if critic_feedback:
        user_message += f"""
        
IMPORTANT: The Critic agent reviewed the previous analysis and found these issues:
{critic_feedback}

Please address these issues in your revised analysis.
"""

    # ── Step 3: Call Gemini ────────────────────────────────────
    messages = [
        SystemMessage(content=ANALYST_SYSTEM_PROMPT),
        HumanMessage(content=user_message)
    ]

    response = llm.invoke(messages)
    analysis_notes = response.content.strip()

    print(f"[Analyst] Analysis complete. Generated {len(analysis_notes)} characters.")

    # ── Step 4: Update state ───────────────────────────────────
    return {
        "analysis_notes": analysis_notes,
        "critic_feedback": None  # Clear feedback after addressing it
    }


def test_analyst():
    """
    Test the Analyst agent in isolation with fake sources.
    Run this file directly to verify it works.
    """
    print("Testing Analyst Agent...")

    # Create fake sources to test with (no need to run Researcher first)
    fake_sources = [
        {
            "title": "AlphaFold2: Accurate Protein Structure Prediction",
            "url": "https://arxiv.org/abs/2106.00565",
            "abstract": "We present AlphaFold2, a novel AI system that can predict protein structures with atomic accuracy. The system uses attention-based neural networks trained on the PDB database.",
            "date": "2024-01-15"
        },
        {
            "title": "Limitations of AI in Protein Folding",
            "url": "https://example.com/limitations",
            "abstract": "While AlphaFold has shown impressive results, there remain significant challenges in predicting intrinsically disordered proteins and membrane proteins.",
            "date": "2024-02-20"
        },
        {
            "title": "RoseTTAFold: Competing Approach to Structure Prediction",
            "url": "https://arxiv.org/abs/2108.01824",
            "abstract": "RoseTTAFold offers an alternative approach to protein folding using three-track neural networks, showing competitive results with faster inference times.",
            "date": "2024-03-01"
        }
    ]

    # Create test state
    test_state = {
        "topic": "latest advances in protein folding AI",
        "raw_sources": fake_sources,
        "analysis_notes": "",
        "draft_report": "",
        "critic_feedback": None,
        "final_report": None,
        "iteration_count": 0
    }

    # Run the agent
    result = analyst_node(test_state)

    # Check output
    notes = result.get("analysis_notes", "")
    print(f"\n✓ Analyst returned {len(notes)} characters of analysis")
    print("\n--- ANALYSIS PREVIEW (first 500 chars) ---")
    print(notes[:500])
    print("---")


# ── Run test if file is executed directly ─────────────────────
if __name__ == "__main__":
    test_analyst()