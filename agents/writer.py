# agents/writer.py
# ─────────────────────────────────────────────────────────────
# Agent 3 — Writer
# This agent takes the Analyst's structured notes and writes
# a clean, professional research report with proper sections
# and citations. It is the THIRD node in the LangGraph pipeline.
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
    temperature=0.4,   # Slightly higher = more natural writing
    max_tokens=int(os.getenv("MAX_TOKENS", 8192))
)


# ── System Prompt ──────────────────────────────────────────────
# We tell the Writer exactly what sections to produce and
# how to format citations. Clear structure = consistent output.
WRITER_SYSTEM_PROMPT = """
You are a specialized Research Writer Agent. Your job is to take
structured analysis notes and write a clean, professional research report.

Your report must follow this EXACT structure:

# [Topic Title]

## Executive Summary
- 3-4 sentences summarizing the entire report
- What is this topic about and why does it matter?
- What are the most important conclusions?

## Key Findings
- Write 5-7 findings as clear paragraphs (not bullet points)
- Each finding must end with a citation like [Source Title, Year]
- Findings should flow logically from one to the next

## Contradictions & Debates
- Write 2-3 paragraphs about disagreements in the field
- Present both sides fairly and objectively
- Cite sources for each position

## Emerging Trends
- Write 3-5 trends as short paragraphs
- Focus on what is changing or new in this field
- Cite relevant sources

## Research Gaps & Future Directions
- Write 2-3 paragraphs about what still needs to be studied
- This section adds unique value to the report

## Conclusion
- 2-3 sentences wrapping up the report
- What should the reader take away from this?

## References
- List ALL sources used in this format:
  [1] Title — URL — Date
  [2] Title — URL — Date

Rules:
- Write in clear, professional academic English
- Every claim MUST have a citation
- Do NOT use bullet points in Key Findings section — use paragraphs
- The report should be 800-1200 words (excluding references)
- Be objective — do not express personal opinions
"""


def writer_node(state: ResearchState) -> ResearchState:
    """
    The main Writer agent function.
    Called by LangGraph as the third node in the pipeline.

    What it does:
    1. Takes analysis_notes from state (filled by Analyst)
    2. Takes raw_sources for citation information
    3. Asks Gemini to write a full structured report
    4. Returns the draft report back into state

    Args:
        state: The shared ResearchState object

    Returns:
        Updated state with draft_report filled in
    """
    topic = state["topic"]
    analysis_notes = state["analysis_notes"]
    raw_sources = state["raw_sources"]

    print(f"\n[Writer] Writing report on: {topic}")

    # ── Step 1: Format sources for citation reference ──────────
    sources_text = ""
    for i, source in enumerate(raw_sources, 1):
        sources_text += f"[{i}] {source.get('title', 'Unknown')} — {source.get('url', '')} — {source.get('date', 'Unknown')}\n"

    # ── Step 2: Build the prompt ───────────────────────────────
    user_message = f"""
Write a research report on this topic: {topic}

Use these analysis notes as your content:
{analysis_notes}

Use these sources for citations:
{sources_text}

Remember to cite sources throughout the report using [Source Title, Year] format.
"""

    # ── Step 3: Call Gemini ────────────────────────────────────
    messages = [
        SystemMessage(content=WRITER_SYSTEM_PROMPT),
        HumanMessage(content=user_message)
    ]

    response = llm.invoke(messages)
    draft_report = response.content.strip()

    print(f"[Writer] Report written. Generated {len(draft_report)} characters.")

    # ── Step 4: Update state ───────────────────────────────────
    return {
        "draft_report": draft_report
    }


def test_writer():
    """
    Test the Writer agent in isolation with fake analysis notes.
    Run this file directly to verify it works.
    """
    print("Testing Writer Agent...")

    # Fake analysis notes (normally produced by Analyst)
    fake_analysis = """
## KEY FINDINGS
- AlphaFold2 achieves atomic accuracy in protein structure prediction
- RoseTTAFold offers faster inference as an alternative approach
- Both systems struggle with intrinsically disordered proteins

## CONTRADICTIONS & DEBATES
- AlphaFold claims superior accuracy while RoseTTAFold claims faster speed
- Some researchers argue AI cannot replace experimental validation

## EMERGING TRENDS
- Integration of AI with cryo-EM data
- Multi-chain protein complex prediction
- Drug discovery applications growing rapidly

## RESEARCH GAPS
- Membrane protein prediction remains challenging
- Dynamic protein conformations not well handled
"""

    # Fake sources
    fake_sources = [
        {
            "title": "AlphaFold2: Accurate Protein Structure Prediction",
            "url": "https://arxiv.org/abs/2106.00565",
            "abstract": "AlphaFold2 system description",
            "date": "2024-01-15"
        },
        {
            "title": "RoseTTAFold: Competing Approach",
            "url": "https://arxiv.org/abs/2108.01824",
            "abstract": "RoseTTAFold system description",
            "date": "2024-03-01"
        }
    ]

    # Create test state
    test_state = {
        "topic": "latest advances in protein folding AI",
        "raw_sources": fake_sources,
        "analysis_notes": fake_analysis,
        "draft_report": "",
        "critic_feedback": None,
        "final_report": None,
        "iteration_count": 0
    }

    # Run the agent
    result = test_writer_node(test_state)

    # Check output
    report = result.get("draft_report", "")
    print(f"\n✓ Writer returned {len(report)} characters")
    print("\n--- REPORT PREVIEW (first 500 chars) ---")
    print(report[:500])
    print("---")


def test_writer_node(state):
    return writer_node(state)


# ── Run test if file is executed directly ─────────────────────
if __name__ == "__main__":
    test_writer()