# 🔬 Multi-Agent Research Assistant

**LangGraph · Gemini 2.0 Flash · Tavily · ArXiv API · Streamlit · fpdf2**

A production-grade AI research pipeline powered by **4 specialized LangGraph agents** that collaboratively research any topic and generate professional, citation-backed reports — with a built-in quality reflection loop.

> Architected a 4-agent LangGraph pipeline (Researcher → Analyst → Writer → Critic) with a reflection loop — Critic agent evaluates output quality and routes back to Analyst for up to 2 revision cycles. Integrated dual-source retrieval (Tavily + ArXiv), benchmarking module, and full Markdown + PDF export via Streamlit.

---

## 🧠 Architecture

```
User Input: "Latest advances in quantum computing"
        ↓
┌───────────────────┐
│   RESEARCHER      │  Generates smart search queries via Gemini
│   Agent 1         │  Searches Tavily (web) + ArXiv (academic)
│                   │  Returns 10-15 deduplicated sources
└────────┬──────────┘
         ↓
┌───────────────────┐
│   ANALYST         │  Reads all sources, extracts structured notes
│   Agent 2         │  Key findings, contradictions, trends, gaps
│                   │  Incorporates Critic feedback on revision loops
└────────┬──────────┘
         ↓
┌───────────────────┐
│   WRITER          │  Transforms analysis into a structured report
│   Agent 3         │  Executive Summary → Findings → Conclusion
│                   │  Every claim backed by source citations
└────────┬──────────┘
         ↓
┌───────────────────┐
│   CRITIC          │  Reviews against 5-point quality checklist
│   Agent 4         │  Returns QUALITY_SCORE (X/10)
│                   │  APPROVED → final report
│                   │  NEEDS_REVISION → routes back to Analyst
│                   │  (max 2 revision cycles)
└────────┬──────────┘
         ↓
   Final Report → Markdown + PDF export
```

### Reflection Loop

The Critic agent evaluates every draft against a strict checklist:

1. **Citations** — Does every factual claim have a source?
2. **Structure** — Are all required sections present?
3. **Consistency** — Any internal contradictions?
4. **Completeness** — Are important points missing?
5. **References** — Do sources include URLs?

If the report scores below threshold → feedback is routed back to the **Analyst** (not the Writer), who revises the analysis with specific feedback. This loops through Analyst → Writer → Critic for up to **2 revision cycles** before force-approving.

---

## 🚀 Quick Start

### 1. Clone & setup
```bash
git clone https://github.com/yourusername/multi-agent-research-assistant.git
cd multi-agent-research-assistant
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
# Create .env file with:
GOOGLE_API_KEY=your_gemini_key        # https://aistudio.google.com
TAVILY_API_KEY=your_tavily_key        # https://app.tavily.com
```

### 3. Run
```bash
streamlit run app.py
```
Open `http://localhost:8501` → enter any topic → one-click report generation.

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Agent Orchestration** | LangGraph `StateGraph` | Defines nodes, edges, conditional routing, shared state |
| **LLM** | Gemini 2.0 Flash | Powers all 4 agents with tuned temperatures (0.2–0.4) |
| **Web Search** | Tavily API (`search_depth=advanced`) | Real-time web retrieval, 10+ results per query |
| **Academic Search** | ArXiv API (`arxiv` library) | Peer-reviewed papers, sorted by relevance |
| **State Management** | `TypedDict` + `Annotated` reducers | Shared memory across agents with append-only source list |
| **Frontend** | Streamlit | Live agent status, progress bar, download buttons |
| **PDF Export** | fpdf2 | Custom `ReportPDF` class with headers, footers, styled sections |
| **Benchmarking** | pandas + tabulate | Per-agent timing, quality scores, CSV persistence |

---

## 📁 Project Structure

```
multi-agent-research-assistant/
│
├── app.py                        # Streamlit UI — entry point
│
├── agents/
│   ├── researcher.py             # Agent 1 — generates queries, searches Tavily + ArXiv
│   ├── analyst.py                # Agent 2 — structured analysis with feedback incorporation
│   ├── writer.py                 # Agent 3 — citation-backed report writing
│   └── critic.py                 # Agent 4 — quality checklist + reflection routing
│
├── graph/
│   ├── state.py                  # ResearchState TypedDict — shared agent memory
│   └── pipeline.py               # StateGraph definition — nodes, edges, conditional routing
│
├── tools/
│   ├── search.py                 # Tavily wrapper — web + academic search
│   └── arxiv_fetch.py            # ArXiv fetcher + source deduplication
│
├── output/
│   ├── report_exporter.py        # Markdown + PDF export pipeline
│   └── reports/                  # Generated reports saved here
│
├── evaluate/
│   └── benchmark.py              # Per-agent timing + quality scoring + CSV tracking
│
├── requirements.txt              # Pinned dependencies with comments
├── .env                          # API keys (gitignored)
└── README.md
```

---

## 📊 Benchmarking

Run the benchmark module to measure per-agent execution time and report quality:

```bash
python evaluate/benchmark.py
```

**Real run** (`evaluate/benchmark_results.csv`, topic: "latest advances in protein
folding AI") — this particular run needed one full reflection/revision cycle
(the Critic sent it back once for missing citation dates before force-approving
on the second pass), so the total includes a full extra Analyst→Writer→Critic
pass, not just the fast path:

```
📊 Agent Execution Times:
╭─────────────────────┬─────────╮
│ Agent                │ Time    │
├─────────────────────┼─────────┤
│ Researcher           │ 21.01s  │
│ Analyst              │ 22.42s  │
│ Writer               │ 22.18s  │
│ Critic                │ 23.25s  │
│ Analyst (revision 1) │ 29.72s  │
│ Writer (revision 1)  │ 23.43s  │
│ Critic (revision 1)  │ 13.95s  │
│ TOTAL                 │ 155.97s │
╰─────────────────────┴─────────╯

📈 Quality Metrics:
╭──────────────────┬────────╮
│ Metric           │ Value  │
├──────────────────┼────────┤
│ Sources Found    │ 17     │
│ Word Count       │ 1816   │
│ Sections Found   │ 6/6    │
│ Citation Count   │ 27     │
│ OVERALL SCORE    │ 9.4/10 │
╰──────────────────┴────────╯
```

A run that the Critic approves on the first pass (no revision cycle) finishes in
roughly a third of that time — the reflection loop trades latency for the Critic's
own quality checklist actually being enforced rather than rubber-stamped.

Results are automatically appended to `evaluate/benchmark_results.csv` for cross-run comparison.

---

## 🔑 API Keys Required

| Key | Source | Free Tier |
|-----|--------|-----------|
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | 1M tokens/day |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) | 1000 searches/month |

---

## ✨ Key Features

- **4-Agent LangGraph Pipeline** — Researcher → Analyst → Writer → Critic with clear separation of concerns
- **Reflection Loop** — Critic evaluates quality and routes back to Analyst for up to 2 revision cycles
- **Dual-Source Retrieval** — Tavily for real-time web search + ArXiv API for peer-reviewed academic papers
- **Quality Scoring** — Every report gets a structured quality score (sections, citations, length, sources)
- **Benchmarking** — Track per-agent execution times and quality scores across runs via CSV
- **Full Export** — Download reports as Markdown or PDF with one click
- **Production Patterns** — Built with LangGraph StateGraph, the same framework used in production AI systems

---

## 🧪 Testing Individual Agents

Each agent has a built-in test function — run any file directly:

```bash
python agents/researcher.py     # Test search + source gathering
python agents/analyst.py        # Test analysis with fake sources
python agents/writer.py         # Test report writing with fake analysis
python agents/critic.py         # Test quality evaluation with fake report
python graph/pipeline.py        # Test full end-to-end pipeline
python output/report_exporter.py  # Test Markdown + PDF export
```

---
