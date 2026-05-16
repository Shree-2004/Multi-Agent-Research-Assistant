# graph/pipeline.py
# ─────────────────────────────────────────────────────────────
# This file connects all 4 agents into a single LangGraph pipeline.
# It defines the nodes, edges, and conditional routing logic.
# Think of this as the "manager" that controls the flow.
# ─────────────────────────────────────────────────────────────

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from graph.state import ResearchState
from agents.researcher import researcher_node
from agents.analyst import analyst_node
from agents.writer import writer_node
from agents.critic import critic_node, should_continue

# Load environment variables
load_dotenv()


def build_pipeline() -> StateGraph:
    """
    Builds and compiles the full LangGraph pipeline.
    
    The pipeline flow is:
    researcher → analyst → writer → critic → (end OR back to analyst)
    
    The reflection loop:
    If critic says NEEDS_REVISION → goes back to analyst → writer → critic
    Max 2 iterations before force approving.

    Returns:
        Compiled LangGraph app ready to run
    """

    # ── Step 1: Create the graph with our state schema ─────────
    # StateGraph takes our ResearchState TypedDict as the schema
    # Every node receives and returns a ResearchState
    workflow = StateGraph(ResearchState)

    # ── Step 2: Add all 4 agents as nodes ─────────────────────
    # Each node is a function that takes state and returns state
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst",    analyst_node)
    workflow.add_node("writer",     writer_node)
    workflow.add_node("critic",     critic_node)

    # ── Step 3: Define the edges (flow between agents) ─────────
    # These are the fixed, always-run connections
    workflow.set_entry_point("researcher")       # Always start here
    workflow.add_edge("researcher", "analyst")   # Researcher → Analyst
    workflow.add_edge("analyst",    "writer")    # Analyst → Writer
    workflow.add_edge("writer",     "critic")    # Writer → Critic

    # ── Step 4: Add conditional routing after Critic ───────────
    # This is the reflection loop logic
    # should_continue() returns either "analyst" or "end"
    workflow.add_conditional_edges(
        "critic",           # After critic runs...
        should_continue,    # ...call this function to decide...
        {
            "analyst": "analyst",  # NEEDS_REVISION → back to analyst
            "end": END             # APPROVED → end pipeline
        }
    )

    # ── Step 5: Compile the graph ──────────────────────────────
    # This validates the graph structure and returns a runnable app
    app = workflow.compile()

    return app


def run_pipeline(topic: str) -> dict:
    """
    Runs the full pipeline for a given research topic.
    This is the main function called by app.py (Streamlit UI).

    Args:
        topic: The research topic entered by the user

    Returns:
        Final state dict containing the complete report
    """
    print(f"\n{'='*60}")
    print(f"Starting Multi-Agent Research Pipeline")
    print(f"Topic: {topic}")
    print(f"{'='*60}\n")

    # ── Step 1: Build the pipeline ─────────────────────────────
    app = build_pipeline()

    # ── Step 2: Set initial state ──────────────────────────────
    # This is the starting state passed to the first node
    initial_state = {
        "topic": topic,
        "raw_sources": [],
        "analysis_notes": "",
        "draft_report": "",
        "critic_feedback": None,
        "final_report": None,
        "iteration_count": 0
    }

    # ── Step 3: Run the pipeline ───────────────────────────────
    # .invoke() runs the full pipeline synchronously
    # The state is passed between agents automatically
    final_state = app.invoke(initial_state)

    print(f"\n{'='*60}")
    print(f"Pipeline Complete!")
    print(f"Sources found: {len(final_state.get('raw_sources', []))}")
    print(f"Iterations: {final_state.get('iteration_count', 0)}")
    print(f"Report length: {len(final_state.get('final_report', ''))} chars")
    print(f"{'='*60}\n")

    return final_state


def test_pipeline():
    """
    Test the full pipeline end to end.
    This is the ultimate test — all 4 agents working together.
    Run this file directly to verify the complete flow works.
    """
    print("Testing Full Pipeline...")
    print("This will run all 4 agents — may take 30-60 seconds\n")

    result = run_pipeline("latest advances in protein folding AI")

    # Check final output
    final_report = result.get("final_report", "")
    if final_report:
        print(f"\n✓ Pipeline succeeded!")
        print(f"✓ Final report: {len(final_report)} characters")
        print("\n--- REPORT PREVIEW (first 500 chars) ---")
        print(final_report[:500])
        print("---")
    else:
        print("\n✗ Pipeline failed — no final report generated")
        print(f"Draft report exists: {bool(result.get('draft_report'))}")


# ── Run test if file is executed directly ─────────────────────
if __name__ == "__main__":
    test_pipeline()