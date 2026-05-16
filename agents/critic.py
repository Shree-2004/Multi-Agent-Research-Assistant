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

from graph.state import ResearchState

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

Your response must follow this EXACT format:

VERDICT: APPROVED or NEEDS_REVISION

ISSUES_FOUND:
- Issue 1 (if any)
- Issue 2 (if any)
- Write "None" if no issues found

SPECIFIC_FEEDBACK:
- Specific actionable feedback for the Analyst to address
- Be precise — say WHAT is wrong and WHERE
- Write "None" if verdict is APPROVED

QUALITY_SCORE: X/10
"""


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
    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=f"Review this research report:\n\n{draft_report}")
    ]

    response = llm.invoke(messages)
    review = response.content.strip()

    print(f"[Critic] Review complete.")
    print(f"[Critic] Raw verdict:\n{review[:300]}...")

    # ── Step 3: Parse the verdict ──────────────────────────────
    # We check if the Critic said APPROVED or NEEDS_REVISION
    verdict = "APPROVED"  # Default to approved
    if "NEEDS_REVISION" in review.upper():
        verdict = "NEEDS_REVISION"

    print(f"[Critic] Verdict: {verdict}")

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
        # Report needs work — extract feedback for Analyst
        print(f"[Critic] Report needs revision. Sending feedback to Analyst.")

        # Extract just the feedback section from the review
        feedback = extract_feedback(review)

        return {
            "final_report": None,
            "critic_feedback": feedback,
            "iteration_count": iteration_count + 1  # Increment loop counter
        }


def extract_feedback(review: str) -> str:
    """
    Extracts just the actionable feedback from the Critic's review.
    We only want the SPECIFIC_FEEDBACK section, not the full review.

    Args:
        review: Full review text from the Critic

    Returns:
        Just the feedback section as a string
    """
    # Try to extract SPECIFIC_FEEDBACK section
    if "SPECIFIC_FEEDBACK:" in review:
        parts = review.split("SPECIFIC_FEEDBACK:")
        if len(parts) > 1:
            feedback = parts[1].strip()
            # Cut off at next section if exists
            if "QUALITY_SCORE:" in feedback:
                feedback = feedback.split("QUALITY_SCORE:")[0].strip()
            return feedback

    # If we can't parse it, return the full review as feedback
    return review


def should_continue(state: ResearchState) -> str:
    """
    LangGraph routing function — decides what happens after Critic runs.
    This is called by the graph to determine the next node.

    Returns:
        "analyst"    → if report needs revision (loop back)
        "end"        → if report is approved (finish pipeline)
    """
    # If final_report is set, we're done
    if state.get("final_report"):
        print("[Router] Report approved → ending pipeline")
        return "end"

    # If there's feedback, loop back to analyst
    if state.get("critic_feedback"):
        iteration = state.get("iteration_count", 0)
        max_iter = int(os.getenv("MAX_REFLECTION_ITERATIONS", 2))

        if iteration >= max_iter:
            print("[Router] Max iterations reached → ending pipeline")
            return "end"

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