# evaluate/benchmark.py
# ─────────────────────────────────────────────────────────────
# This file measures the performance of the multi-agent pipeline.
# It tracks: execution time per agent, total pipeline time,
# source count, report quality scores, and iteration count.
# Run this to benchmark your pipeline before interviews!
# ─────────────────────────────────────────────────────────────

import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from tabulate import tabulate

from graph.state import ResearchState
from agents.researcher import researcher_node
from agents.analyst import analyst_node
from agents.writer import writer_node
from agents.critic import critic_node

load_dotenv()


def measure_agent(agent_fn, state: dict, agent_name: str) -> tuple:
    """
    Runs a single agent and measures its execution time.

    Args:
        agent_fn: The agent function to run
        state: Current pipeline state
        agent_name: Name for display purposes

    Returns:
        Tuple of (updated_state, execution_time_seconds)
    """
    print(f"\n[Benchmark] Running {agent_name}...")
    start_time = time.time()

    # Run the agent
    result = agent_fn(state)

    end_time = time.time()
    duration = round(end_time - start_time, 2)

    # Merge result back into state
    updated_state = {**state, **result}

    print(f"[Benchmark] {agent_name} completed in {duration}s")
    return updated_state, duration


def calculate_quality_score(state: dict) -> dict:
    """
    Calculates quality metrics for the final report.
    These are simple heuristic scores — not perfect but useful.

    Metrics:
    - source_count: How many sources were found
    - report_length: Character count of final report
    - has_all_sections: Whether all required sections exist
    - citation_count: How many citations are in the report
    - iteration_count: How many revision loops happened

    Args:
        state: Final pipeline state

    Returns:
        Dict of quality metrics with scores
    """
    final_report = state.get("final_report", "") or state.get("draft_report", "")
    raw_sources = state.get("raw_sources", [])

    # Check for required sections
    required_sections = [
        "## Executive Summary",
        "## Key Findings",
        "## Contradictions",
        "## Emerging Trends",
        "## Conclusion",
        "## References"
    ]

    sections_found = sum(
        1 for section in required_sections
        if section.lower() in final_report.lower()
    )
    sections_score = round((sections_found / len(required_sections)) * 10, 1)

    # Count citations (lines starting with [ or containing [Source)
    citation_count = final_report.count("[") 

    # Report length score (target: 800-1200 words)
    word_count = len(final_report.split())
    if 800 <= word_count <= 1500:
        length_score = 10
    elif 500 <= word_count < 800:
        length_score = 7
    elif word_count > 1500:
        length_score = 8
    else:
        length_score = 4

    # Source quality score
    source_score = min(10, round(len(raw_sources) / 1.5, 1))

    # Overall score — weighted average
    overall = round(
        (sections_score * 0.4) +
        (length_score * 0.3) +
        (source_score * 0.3),
        1
    )

    return {
        "source_count":     len(raw_sources),
        "word_count":       word_count,
        "sections_found":   f"{sections_found}/{len(required_sections)}",
        "citation_count":   citation_count,
        "sections_score":   f"{sections_score}/10",
        "length_score":     f"{length_score}/10",
        "source_score":     f"{source_score}/10",
        "overall_score":    f"{overall}/10"
    }


def run_benchmark(topic: str) -> dict:
    """
    Runs the full pipeline with timing measurements for each agent.
    This gives you detailed performance data for each agent.

    Args:
        topic: Research topic to benchmark on

    Returns:
        Dict containing timing data, quality scores, and full results
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK RUN")
    print(f"Topic: {topic}")
    print(f"Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Track timing for each agent
    timings = {}
    total_start = time.time()

    # ── Initial state ──────────────────────────────────────────
    state = {
        "topic": topic,
        "raw_sources": [],
        "analysis_notes": "",
        "draft_report": "",
        "critic_feedback": None,
        "final_report": None,
        "iteration_count": 0
    }

    # ── Run each agent with timing ─────────────────────────────
    state, timings["researcher"] = measure_agent(
        researcher_node, state, "Researcher"
    )

    state, timings["analyst"] = measure_agent(
        analyst_node, state, "Analyst"
    )

    state, timings["writer"] = measure_agent(
        writer_node, state, "Writer"
    )

    state, timings["critic"] = measure_agent(
        critic_node, state, "Critic"
    )

    # Handle reflection loop if needed
    iteration = 1
    max_iter = int(os.getenv("MAX_REFLECTION_ITERATIONS", 2))

    while state.get("critic_feedback") and iteration < max_iter:
        print(f"\n[Benchmark] Reflection loop iteration {iteration}")

        state, analyst_time = measure_agent(
            analyst_node, state, f"Analyst (revision {iteration})"
        )
        timings[f"analyst_revision_{iteration}"] = analyst_time

        state, writer_time = measure_agent(
            writer_node, state, f"Writer (revision {iteration})"
        )
        timings[f"writer_revision_{iteration}"] = writer_time

        state, critic_time = measure_agent(
            critic_node, state, f"Critic (revision {iteration})"
        )
        timings[f"critic_revision_{iteration}"] = critic_time

        iteration += 1

    total_time = round(time.time() - total_start, 2)

    # ── Calculate quality scores ───────────────────────────────
    quality = calculate_quality_score(state)

    # ── Display results ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")

    # Agent timing table
    timing_data = [
        [agent.replace("_", " ").title(), f"{t}s"]
        for agent, t in timings.items()
    ]
    timing_data.append(["TOTAL", f"{total_time}s"])

    print("\n📊 Agent Execution Times:")
    print(tabulate(
        timing_data,
        headers=["Agent", "Time"],
        tablefmt="rounded_outline"
    ))

    # Quality scores table
    quality_data = [
        ["Sources Found",    quality["source_count"]],
        ["Word Count",       quality["word_count"]],
        ["Sections Found",   quality["sections_found"]],
        ["Citation Count",   quality["citation_count"]],
        ["Sections Score",   quality["sections_score"]],
        ["Length Score",     quality["length_score"]],
        ["Source Score",     quality["source_score"]],
        ["OVERALL SCORE",    quality["overall_score"]],
    ]

    print("\n📈 Quality Metrics:")
    print(tabulate(
        quality_data,
        headers=["Metric", "Value"],
        tablefmt="rounded_outline"
    ))

    print(f"\n✓ Benchmark complete!")
    print(f"✓ Total time: {total_time}s")
    print(f"✓ Overall quality: {quality['overall_score']}")

    # ── Save results to CSV ────────────────────────────────────
    save_benchmark_results(topic, timings, quality, total_time)

    return {
        "timings":    timings,
        "quality":    quality,
        "total_time": total_time,
        "state":      state
    }


def save_benchmark_results(
    topic: str,
    timings: dict,
    quality: dict,
    total_time: float
):
    """
    Saves benchmark results to a CSV file for tracking over time.
    You can show this CSV in interviews to demonstrate performance.

    Args:
        topic: Research topic
        timings: Dict of agent timing data
        quality: Dict of quality metrics
        total_time: Total pipeline execution time
    """
    os.makedirs("evaluate", exist_ok=True)

    results = {
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "topic":          topic,
        "total_time_s":   total_time,
        "researcher_s":   timings.get("researcher", 0),
        "analyst_s":      timings.get("analyst", 0),
        "writer_s":       timings.get("writer", 0),
        "critic_s":       timings.get("critic", 0),
        "source_count":   quality["source_count"],
        "word_count":     quality["word_count"],
        "overall_score":  quality["overall_score"],
        "iterations":     quality.get("iteration_count", 0)
    }

    csv_path = "evaluate/benchmark_results.csv"

    # Append to existing CSV or create new one
    df_new = pd.DataFrame([results])
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(csv_path, index=False)
    print(f"\n[Benchmark] Results saved to {csv_path}")


# ── Run benchmark if file is executed directly ─────────────────
if __name__ == "__main__":
    run_benchmark("latest advances in protein folding AI")