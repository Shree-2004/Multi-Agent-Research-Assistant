# app.py
# ─────────────────────────────────────────────────────────────
# Main Streamlit UI for the Multi-Agent Research Assistant.
# This is the entry point of the application.
# Run with: streamlit run app.py
# ─────────────────────────────────────────────────────────────

import os
import sys
import streamlit as st
from dotenv import load_dotenv

from graph.pipeline import run_pipeline
from output.report_exporter import export_report

# Load environment variables
load_dotenv()


# ── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .agent-status {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    .status-running {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .status-done {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .status-waiting {
        background-color: #f8f9fa;
        border-left: 4px solid #dee2e6;
        color: #999;
    }
    .report-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    # Model info
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    st.markdown(f"**🤖 Model:** `{model}`")
    st.markdown(f"**🔍 Search:** Tavily + ArXiv")
    st.markdown(f"**🔄 Max Iterations:** {os.getenv('MAX_REFLECTION_ITERATIONS', 2)}")

    st.markdown("---")
    st.markdown("## 📋 Pipeline")
    st.markdown("""
    1. 🔍 **Researcher** — Finds sources
    2. 🧠 **Analyst** — Extracts insights  
    3. ✍️ **Writer** — Writes report
    4. 🔎 **Critic** — Reviews quality
    """)

    st.markdown("---")
    st.markdown("## 💡 Example Topics")
    example_topics = [
        "Latest advances in protein folding AI",
        "Quantum computing breakthroughs 2024",
        "Large language model alignment research",
        "CRISPR gene editing recent developments",
        "Climate change mitigation technologies"
    ]
    for topic in example_topics:
        if st.button(topic, key=topic, use_container_width=True):
            st.session_state.topic_input = topic


# ── Main Content ───────────────────────────────────────────────
st.markdown('<div class="main-header">🔬 Multi-Agent Research Assistant</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">4 AI agents work together to research any topic and generate a professional report</div>',
            unsafe_allow_html=True)

# ── Topic Input ────────────────────────────────────────────────
topic = st.text_input(
    "Enter your research topic:",
    value=st.session_state.get("topic_input", ""),
    placeholder="e.g. Latest advances in protein folding AI",
    key="topic_input"
)

col1, col2 = st.columns([1, 4])
with col1:
    run_button = st.button(
        "🚀 Generate Report",
        type="primary",
        use_container_width=True,
        disabled=not topic.strip()
    )

# ── Pipeline Execution ─────────────────────────────────────────
if run_button and topic.strip():

    # ── Agent Status Display ───────────────────────────────────
    st.markdown("### 🔄 Pipeline Running...")

    # Create status placeholders for each agent
    status_researcher = st.empty()
    status_analyst    = st.empty()
    status_writer     = st.empty()
    status_critic     = st.empty()

    def update_status(agent: str, state: str):
        """Updates the visual status of each agent card."""
        icons = {
            "researcher": "🔍",
            "analyst":    "🧠",
            "writer":     "✍️",
            "critic":     "🔎"
        }
        labels = {
            "researcher": "Researcher — Searching web + ArXiv",
            "analyst":    "Analyst — Extracting insights",
            "writer":     "Writer — Writing report",
            "critic":     "Critic — Reviewing quality"
        }
        css_class = {
            "running": "status-running",
            "done":    "status-done",
            "waiting": "status-waiting"
        }
        prefix = {
            "running": "⏳ Running: ",
            "done":    "✅ Done: ",
            "waiting": "⏸️ Waiting: "
        }

        html = f"""
        <div class="agent-status {css_class[state]}">
            {icons[agent]} {prefix[state]}{labels[agent]}
        </div>
        """

        placeholders = {
            "researcher": status_researcher,
            "analyst":    status_analyst,
            "writer":     status_writer,
            "critic":     status_critic
        }
        placeholders[agent].markdown(html, unsafe_allow_html=True)

    # Set all to waiting initially
    for agent in ["researcher", "analyst", "writer", "critic"]:
        update_status(agent, "waiting")

    # ── Progress Bar ───────────────────────────────────────────
    progress = st.progress(0, text="Starting pipeline...")

    try:
        # ── Run each agent with status updates ─────────────────
        # We run the full pipeline but update UI at each step

        update_status("researcher", "running")
        progress.progress(10, text="🔍 Researcher searching the web...")

        # Run the full pipeline
        with st.spinner("Running multi-agent pipeline..."):
            result = run_pipeline(topic)

        # Update all statuses to done
        update_status("researcher", "done")
        progress.progress(40, text="🧠 Analyst extracting insights...")
        update_status("analyst", "done")
        progress.progress(70, text="✍️ Writer composing report...")
        update_status("writer", "done")
        progress.progress(90, text="🔎 Critic reviewing quality...")
        update_status("critic", "done")
        progress.progress(100, text="✅ Pipeline complete!")

        # ── Display Results ────────────────────────────────────
        final_report = result.get("final_report", "")
        raw_sources  = result.get("raw_sources", [])
        iterations   = result.get("iteration_count", 0)

        if final_report:
            st.success(f"✅ Report generated successfully! "
                      f"({len(raw_sources)} sources, "
                      f"{iterations} revision(s), "
                      f"{len(final_report)} characters)")

            # ── Stats Row ──────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            m1.metric("Sources Found",   len(raw_sources))
            m2.metric("Revisions Made",  iterations)
            m3.metric("Report Length",   f"{len(final_report)} chars")

            # ── Report Display ─────────────────────────────────
            st.markdown("### 📄 Final Report")
            with st.container():
                st.markdown(final_report)

            # ── Export Buttons ─────────────────────────────────
            st.markdown("### 💾 Download Report")
            export_col1, export_col2 = st.columns(2)

            # Export files
            paths = export_report(topic, final_report)

            with export_col1:
                # Markdown download
                with open(paths["markdown"], "r", encoding="utf-8") as f:
                    md_content = f.read()
                st.download_button(
                    label="📥 Download Markdown (.md)",
                    data=md_content,
                    file_name=os.path.basename(paths["markdown"]),
                    mime="text/markdown",
                    use_container_width=True
                )

            with export_col2:
                # PDF download
                with open(paths["pdf"], "rb") as f:
                    pdf_content = f.read()
                st.download_button(
                    label="📥 Download PDF (.pdf)",
                    data=pdf_content,
                    file_name=os.path.basename(paths["pdf"]),
                    mime="application/pdf",
                    use_container_width=True
                )

            # ── Sources Expander ───────────────────────────────
            with st.expander(f"📚 View All Sources ({len(raw_sources)})"):
                for i, source in enumerate(raw_sources, 1):
                    st.markdown(f"**{i}. {source.get('title', 'Unknown')}**")
                    st.markdown(f"🔗 {source.get('url', 'No URL')}")
                    st.markdown(f"📅 {source.get('date', 'Unknown date')}")
                    st.markdown(f"_{source.get('abstract', 'No abstract')[:200]}..._")
                    st.markdown("---")
        else:
            st.error("Pipeline completed but no report was generated. Please try again.")

    except Exception as e:
        st.error(f"❌ Pipeline failed: {str(e)}")
        st.exception(e)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#999; font-size:0.85rem'>"
    "Built with LangGraph • Gemini 2.0 Flash • Tavily Search • Streamlit"
    "</div>",
    unsafe_allow_html=True
)