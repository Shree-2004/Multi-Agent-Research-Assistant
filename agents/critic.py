# agents/critic.py
# ─────────────────────────────────────────────────────────────
# Agent 4 — Critic
# This agent reads the Writer's draft report and checks its
# quality. If issues are found, it sends feedback back to the
# Analyst to fix (reflection loop). If the report passes,
# it approves it as the final report.
# It is the FOURTH node in the LangGraph pipeline.
# ─────────────────────────────────────────────────────────────

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal

from graph.state import ResearchState
from agents.verifier import format_failures

# Windows consoles default to cp1252, which can't encode the
# checkmarks/arrows in the log output below — force UTF-8.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load environment variables
load_dotenv()


# ── Initialize Gemini LLM ──────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2,   # Very low = strict and consistent evaluation
    max_tokens=int(os.getenv("MAX_TOKENS", 8192))
)


# ── System Prompt ──────────────────────────────────────────────
# The Critic must be strict but fair. We give it a clear
# checklist so its feedback is always actionable, not vague.
CRITIC_SYSTEM_PROMPT = """
You are a specialized Research Critic Agent. Your job is to review
a research report and decide if it meets quality standards.

You must check ALL of the following:

CHECKLIST:
1. CITATIONS — Does every factual claim have a citation?
2. STRUCTURE — Does the report have all required sections?
   (Executive Summary, Key Findings, Contradictions, Trends, Gaps, Conclusion, References)
3. CONSISTENCY — Are there any contradictions WITHIN the report itself?
4. COMPLETENESS — Are there obvious important points missing?
5. REFERENCES — Does the References section list actual sources with URLs?

Return your review using the CriticReview tool. Feedback must be
specific and actionable for the Analyst — say WHAT is wrong and WHERE.
"""


# ── Structured Review Schema ───────────────────────────────────
# The verdict comes back as a typed field instead of free text.
# Substring-matching "NEEDS_REVISION" in prose misfired whenever
# the model echoed the format template or mentioned the phrase.
# Issues/feedback are newline-separated strings, not List[str]:
# langchain-google-genai 1.0.10 drops the array item type and
# Gemini rejects the schema.
class CriticReview(BaseModel):
    verdict: Literal["APPROVED", "NEEDS_REVISION"] = Field(
        description="APPROVED if the report passes the checklist, otherwise NEEDS_REVISION"
    )
    issues_found: str = Field(
        description="Checklist failures found, one per line; empty string if none"
    )
    specific_feedback: str = Field(
        description="Actionable fixes for the Analyst, one per line; empty string if APPROVED"
    )
    quality_score: int = Field(description="Overall quality from 1 to 10")


structured_llm = llm.with_structured_output(CriticReview)


def critic_node(state: ResearchState) -> ResearchState:
    """
    The main Critic agent function.
    Called by LangGraph as the fourth node in the pipeline.

    What it does:
    1. Reads the draft_report from state
    2. Evaluates it against quality checklist
    3. If APPROVED → copies draft to final_report
    4. If NEEDS_REVISION → sets critic_feedback for Analyst
    5. Checks iteration_count to prevent infinite loops

    Args:
        state: The shared ResearchState object

    Returns:
        Updated state with either final_report or critic_feedback
    """
    draft_report = state["draft_report"]
    iteration_count = state.get("iteration_count", 0)
    max_iterations = int(os.getenv("MAX_REFLECTION_ITERATIONS", 2))

    print(f"\n[Critic] Reviewing report (iteration {iteration_count + 1}/{max_iterations})")

    # ── Step 1: Force approve if max iterations reached ────────
    # This prevents infinite loops — after max retries,
    # we approve whatever we have
    if iteration_count >= max_iterations:
        print(f"[Critic] Max iterations reached. Force approving report.")
        return {
            "final_report": draft_report,
            "critic_feedback": None,
            "iteration_count": iteration_count
        }

    # ── Step 2: Ask Gemini to review the report ────────────────
    # The Verifier's failures go in as evidence: the Critic can see
    # a citation exists, but not whether the source supports it.
    citation_failures = format_failures(state.get("citation_checks", []))
    review_request = f"Review this research report:\n\n{draft_report}"
    if citation_failures:
        review_request += (
            "\n\nThe Citation Verifier checked each cited claim against the "
            "source text it cites and found these failures. Treat each one "
            f"as a CITATIONS problem:\n{citation_failures}"
        )

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=review_request)
    ]

    review = structured_llm.invoke(messages)

    # ── Step 3: Handle an unparseable review ───────────────────
    # If the model returns no tool call we have no verdict at all.
    # Approving silently would hide that, so say so loudly.
    if review is None:
        print("[Critic] WARNING: no structured verdict returned — "
              "force-approving an UNREVIEWED draft.")
        return {
            "final_report": draft_report,
            "critic_feedback": None,
            "iteration_count": iteration_count
        }

    verdict = review.verdict
    print(f"[Critic] Verdict: {verdict} (score {review.quality_score}/10)")
    if review.issues_found.strip():
        print(f"[Critic] Issues:\n{review.issues_found.strip()}")

    # ── Step 4: Return based on verdict ───────────────────────
    if verdict == "APPROVED":
        # Report passed — copy to final_report
        print(f"[Critic] Report APPROVED! Setting as final report.")
        return {
            "final_report": draft_report,
            "critic_feedback": None,
            "iteration_count": iteration_count
        }
    else:
        new_iteration = iteration_count + 1

        # If this revision would exceed the allowed cycles, force approve
        # the current draft now instead of looping back for another pass
        # that would just get cut off with no final_report set.
        if new_iteration >= max_iterations:
            print(f"[Critic] Max revision cycles reached. Force approving current draft.")
            # No feedback is sent, so no revision cycle happens — keep
            # the count as-is so it reports revisions actually run
            return {
                "final_report": draft_report,
                "critic_feedback": None,
                "iteration_count": iteration_count
            }

        # Report needs work — extract feedback for Analyst
        print(f"[Critic] Report needs revision. Sending feedback to Analyst.")

        # Fall back to the issue list if the model left feedback empty,
        # so the Analyst never gets a revision request with no content
        feedback = review.specific_feedback.strip() or review.issues_found.strip()

        # Pass the verifier's failures through verbatim so the Analyst
        # sees exactly which claims lacked support, not a paraphrase
        if citation_failures:
            feedback += f"\n\nUnsupported citations to fix:\n{citation_failures}"

        return {
            "final_report": None,
            "critic_feedback": feedback,
            "iteration_count": new_iteration
        }


def should_continue(state: ResearchState) -> str:
    """
    LangGraph routing function — decides what happens after Critic runs.
    This is called by the graph to determine the next node.

    critic_node already enforces the max-iteration cap (it sets
    final_report once the cap is hit instead of leaving feedback),
    so this only needs to check which of those two it set.

    Returns:
        "analyst"    → if report needs revision (loop back)
        "end"        → if report is approved (finish pipeline)
    """
    # If final_report is set, we're done (approved, or cap reached)
    if state.get("final_report"):
        print("[Router] Report approved → ending pipeline")
        return "end"

    # If there's feedback, loop back to analyst
    if state.get("critic_feedback"):
        print(f"[Router] Needs revision → routing back to analyst")
        return "analyst"

    # Default — end the pipeline
    return "end"


def test_critic():
    """
    Test the Critic agent in isolation with a fake report.
    Run this file directly to verify it works.
    """
    print("Testing Critic Agent...")

    # Fake report — intentionally missing some citations
    fake_report = """
# Latest Advances in Protein Folding AI

## Executive Summary
Protein folding prediction has been revolutionized by AI systems.
AlphaFold2 and RoseTTAFold represent major breakthroughs in this field.

## Key Findings
AlphaFold2 achieves atomic accuracy in protein structure prediction.
RoseTTAFold offers faster inference times as a competitive alternative.

## Contradictions & Debates
Some researchers argue AI cannot replace experimental validation.
Others believe AI will completely replace wet lab experiments.

## Emerging Trends
Integration with cryo-EM data is growing rapidly.
Drug discovery applications are expanding.

## Research Gaps & Future Directions
Membrane protein prediction remains challenging.
Dynamic conformations are not well handled by current systems.

## Conclusion
AI has transformed protein folding prediction significantly.

## References
[1] AlphaFold2 Paper — https://arxiv.org/abs/2106.00565 — 2024
"""

    # Create test state
    test_state = {
        "topic": "latest advances in protein folding AI",
        "raw_sources": [],
        "analysis_notes": "",
        "draft_report": fake_report,
        "critic_feedback": None,
        "final_report": None,
        "iteration_count": 0
    }

    # Run the agent
    result = critic_node(test_state)

    # Check output
    if result.get("final_report"):
        print("\n✓ Critic APPROVED the report")
    else:
        print("\n✓ Critic requested REVISION")
        print(f"Feedback: {result.get('critic_feedback', '')[:300]}")

    # Test routing function
    route = should_continue(result)
    print(f"\n✓ Router decision: → {route}")


# ── Run test if file is executed directly ─────────────────────
if __name__ == "__main__":
    test_critic()