# graph/state.py
# ─────────────────────────────────────────────────────────────
# This file defines the SHARED MEMORY (state) that all 4 agents
# read from and write to. Think of it as the "baton" passed
# between agents in a relay race.
# ─────────────────────────────────────────────────────────────

from typing import TypedDict, Annotated, List, Optional
import operator


# ── Individual Source Schema ───────────────────────────────────
# Each source the Researcher finds will follow this structure
class Source(TypedDict):
    title: str        # Paper or article title
    url: str          # Link to the source
    abstract: str     # Summary or excerpt
    date: str         # Publication date


# ── Main State Schema ──────────────────────────────────────────
# This is the single object passed between ALL agents in the graph
class ResearchState(TypedDict):

    # ── Input ──────────────────────────────────────────────────
    topic: str
    # The research topic entered by the user
    # Example: "latest advances in protein folding"

    # ── Agent 1 Output ─────────────────────────────────────────
    raw_sources: Annotated[List[Source], operator.add]
    # List of sources found by the Researcher agent
    # Annotated with operator.add so new sources are APPENDED
    # not overwritten if the agent runs multiple times

    # ── Agent 2 Output ─────────────────────────────────────────
    analysis_notes: str
    # Structured analysis written by the Analyst agent
    # Contains: key findings, contradictions, trends

    # ── Agent 3 Output ─────────────────────────────────────────
    draft_report: str
    # Full report written by the Writer agent
    # Contains all sections: summary, findings, references etc.

    # ── Agent 4 Output ─────────────────────────────────────────
    critic_feedback: Optional[str]
    # Feedback from the Critic agent
    # None = report passed quality check
    # String = specific issues found, sent back to Analyst

    # ── Final Output ───────────────────────────────────────────
    final_report: Optional[str]
    # The approved report after Critic passes it
    # This is what gets shown to the user and exported

    # ── Loop Control ───────────────────────────────────────────
    iteration_count: int
    # Tracks how many times the Critic has sent feedback
    # MAX = 2 (defined in .env) to prevent infinite loops