# 🔬 Multi-Agent Research Assistant

**LangGraph · Gemini 2.5 Flash · Tavily · ArXiv API · Streamlit · fpdf2**

A production-grade AI research pipeline powered by **5 specialized LangGraph agents** that collaboratively research any topic and generate citation-backed reports — with a Citation Verifier that checks each cited claim against its source, and a quality reflection loop.

> Architected a 5-agent LangGraph pipeline (Researcher → Analyst → Writer → Verifier → Critic) with a reflection loop — a Citation Verifier labels every cited claim against the retrieved source text, and the Critic routes failures back to the Analyst for revision. Integrated dual-source retrieval (Tavily + ArXiv), a benchmark that measures citation precision, and Markdown + PDF export via Streamlit.

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
│                   │  Cites sources by number, e.g. [3]
└────────┬──────────┘
         ↓
┌───────────────────┐
│   VERIFIER        │  Pulls out every cited sentence
│   Agent 4         │  Checks it against the text of the source it cites
│                   │  SUPPORTED / PARTIAL / UNSUPPORTED / DANGLING
└────────┬──────────┘
         ↓
┌───────────────────┐
│   CRITIC          │  Reviews against 5-point quality checklist,
│   Agent 5         │  with the Verifier's failures as evidence
│                   │  Returns a typed verdict + score (X/10)
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

If the Critic returns `NEEDS_REVISION`, feedback is routed back to the **Analyst** (not the Writer), who revises the analysis. Any claims the Verifier found unsupported are passed along verbatim. This loops through Analyst → Writer → Verifier → Critic, with at most **1 revision cycle** at the default `MAX_REFLECTION_ITERATIONS=2`; after that the current draft is force-approved.

### Citation Verifier

The Critic can see that a citation *exists*; it can't see whether the source says what the report claims. The Verifier checks that:

- Each sentence with a numeric citation (`[3]`, `[2, 5]`, `[2-4]`) is extracted deterministically.
- Gemini labels it against the cited sources' retrieved text only: **SUPPORTED**, **PARTIAL** (adds specifics the source doesn't state) or **UNSUPPORTED**.
- A citation to a source number that doesn't exist is **DANGLING**, decided without a model call.
- A claim the model returns no verdict for is **UNVERIFIED** and is excluded from the score, never counted as supported.

**Citation precision** = supported ÷ claims judged. It shows `n/a`, not 100%, when there's nothing to check.

**Limitation:** the Verifier only sees what the pipeline retrieved — Tavily excerpts and ArXiv abstracts trimmed to 500 characters — not full papers. `UNSUPPORTED` means "not supported by the text the pipeline read", not "false".

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
| **LLM** | Gemini 2.5 Flash (`GEMINI_MODEL` in `.env`) | Powers all 5 agents with tuned temperatures (0–0.4) |
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
│   ├── verifier.py               # Agent 4 — checks each cited claim against its source
│   └── critic.py                 # Agent 5 — quality checklist + reflection routing
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

**Real run** (2026-09-27, topic: "latest advances in protein folding AI"). The
Critic sent the first draft back once, then force-approved the revision at the cap,
so the total includes one full extra Analyst → Writer → Verifier → Critic pass:

```
📊 Agent Execution Times:
╭─────────────────────┬─────────╮
│ Agent               │ Time    │
├─────────────────────┼─────────┤
│ Researcher          │ 17.73s  │
│ Analyst             │ 28.43s  │
│ Writer              │ 20.01s  │
│ Verifier            │ 33.69s  │
│ Critic              │ 9.02s   │
│ Analyst Revision 1  │ 33.28s  │
│ Writer Revision 1   │ 28.65s  │
│ Verifier Revision 1 │ 55.18s  │
│ Critic Revision 1   │ 16.89s  │
│ TOTAL               │ 242.87s │
╰─────────────────────┴─────────╯

📈 Quality Metrics (final report):
╭────────────────────┬─────────╮
│ Metric             │ Value   │
├────────────────────┼─────────┤
│ Sources Found      │ 17      │
│ Word Count         │ 1451    │
│ Sections Found     │ 6/6     │
│ Cited Claims       │ 16      │
│   Supported        │ 14      │
│   Partial          │ 1       │
│   Unsupported      │ 1       │
│ Citation Precision │ 0.875   │
│ OVERALL SCORE      │ 10.0/10 │
╰────────────────────┴─────────╯
```

What this run shows:

- **The revision made citations worse, not better.** The first draft scored 16/16 supported (precision 1.0). The Critic asked for revision; the rewrite introduced one unsupported and one partial claim (0.875), and the cap then force-approved it. This is a single run, not a trend, but the heuristic `OVERALL SCORE` (10/10) could not have shown it at all.
- **Verification is the slowest stage** — 34–55s per draft for 16 claims, checked in sequential batches of 15.

Results are appended to `evaluate/benchmark_results.csv` for cross-run comparison (local only — the file is gitignored).

---

## 🔑 API Keys Required

| Key | Source | Free Tier |
|-----|--------|-----------|
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | 1M tokens/day |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) | 1000 searches/month |

---

## ✨ Key Features

- **5-Agent LangGraph Pipeline** — Researcher → Analyst → Writer → Verifier → Critic with clear separation of concerns
- **Citation Verification** — every cited claim is checked against the source text it cites; citation precision is reported per run
- **Reflection Loop** — Critic evaluates quality and routes back to Analyst for up to 2 revision cycles
- **Dual-Source Retrieval** — Tavily for real-time web search + ArXiv API for peer-reviewed academic papers
- **Quality Scoring** — A heuristic score (sections, length, source count). It does not check whether citations are correct — citation precision does
- **Benchmarking** — Track per-agent execution times, quality scores and citation precision across runs via CSV
- **Full Export** — Download reports as Markdown or PDF with one click
- **Production Patterns** — Built with LangGraph StateGraph, the same framework used in production AI systems

---

## 🧪 Testing Individual Agents

Each agent has a built-in test function — run any file directly:

```bash
python agents/researcher.py     # Test search + source gathering
python agents/analyst.py        # Test analysis with fake sources
python agents/writer.py         # Test report writing with fake analysis
python agents/verifier.py       # Test citation checks with a planted false citation
python agents/critic.py         # Test quality evaluation with fake report
python graph/pipeline.py        # Test full end-to-end pipeline
python output/report_exporter.py  # Test Markdown + PDF export
```

---
