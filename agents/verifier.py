# agents/verifier.py
# ─────────────────────────────────────────────────────────────
# Citation Verifier
# This agent checks whether each cited claim in the Writer's
# draft is actually supported by the source it cites. The Critic
# only checks that citations EXIST; this checks they are TRUE.
# It runs between the Writer and the Critic.
#
# Scope: sources are only the snippets the pipeline retrieved
# (Tavily content, ArXiv abstracts trimmed to 500 chars), so
# UNSUPPORTED means "not supported by what the pipeline read",
# not "false".
# ─────────────────────────────────────────────────────────────

import os
import re
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import google.generativeai as genai

from graph.state import ResearchState

# Windows consoles default to cp1252, which can't encode the
# checkmarks/arrows in the log output below — force UTF-8.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load environment variables
load_dotenv()

LABELS = ["SUPPORTED", "PARTIAL", "UNSUPPORTED"]

# Claims per Gemini call — keeps prompts small without one call per claim
BATCH_SIZE = 15


# ── Gemini Backend ─────────────────────────────────────────────
# Called directly through google-generativeai rather than
# LangChain: langchain-google-genai 1.0.10 drops array item types
# from tool schemas, so a batch of verdicts can't be typed there.
# JSON mode + response_schema gives a typed list in one call.
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel(
    os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    system_instruction="""
You are a Citation Verifier. For each claim you are given the exact
source text it cites. Judge ONLY against that source text — not
against your own knowledge.

- SUPPORTED: the source text states or directly implies the claim.
- PARTIAL: the source supports part of the claim, but the claim adds
  specifics (numbers, comparisons, causes) the source does not state.
- UNSUPPORTED: the source does not support the claim, or contradicts it.

Give a one-sentence reason for each verdict.
"""
)

VERDICT_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "claim_id": {"type": "INTEGER"},
            "label":    {"type": "STRING", "format": "enum", "enum": LABELS},
            "reason":   {"type": "STRING"}
        },
        "required": ["claim_id", "label", "reason"]
    }
}


# ── Claim Extraction ───────────────────────────────────────────
# Deterministic: a claim is a sentence carrying one or more
# numeric citations like [3], [2, 5] or [2-4].
CITATION_RE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def parse_citation_ids(group: str) -> list[int]:
    """Expands a citation group like '2, 5' or '2-4' into ids."""
    ids = []
    for part in re.split(r"\s*,\s*", group):
        bounds = re.split(r"\s*[–-]\s*", part)
        if len(bounds) == 2:
            ids.extend(range(int(bounds[0]), int(bounds[1]) + 1))
        else:
            ids.append(int(bounds[0]))
    return ids


def extract_claims(report: str) -> list[dict]:
    """
    Pulls every cited sentence out of the report body.
    The References section is excluded — it lists sources,
    it doesn't make claims.

    Returns:
        List of dicts with keys: id, text, source_ids
    """
    body = re.split(r"^##\s*References", report, flags=re.MULTILINE | re.IGNORECASE)[0]

    claims = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sentence in SENTENCE_SPLIT_RE.split(line):
            groups = CITATION_RE.findall(sentence)
            if not groups:
                continue
            source_ids = sorted({i for g in groups for i in parse_citation_ids(g)})
            claims.append({
                "id":         len(claims) + 1,
                "text":       CITATION_RE.sub("", sentence).strip(),
                "source_ids": source_ids
            })
    return claims


# ── Judging ────────────────────────────────────────────────────
def judge_claims(claims: list[dict], sources: list[dict]) -> dict[int, dict]:
    """
    Asks Gemini to label a batch of claims against their cited sources.
    This is the only model-dependent function in this file — swap it
    to try another backend (e.g. a System One model like Jev).

    Returns:
        Dict of claim_id -> {"label", "reason"} for the claims the
        model returned a valid verdict for. Missing ids are left out.
    """
    blocks = []
    for c in claims:
        cited = "\n".join(
            f"  [{i}] {sources[i - 1].get('title', '')}: {sources[i - 1].get('abstract', '')}"
            for i in c["source_ids"]
        )
        blocks.append(f"CLAIM {c['id']}: {c['text']}\nCITED SOURCES:\n{cited}")

    response = model.generate_content(
        "Label every claim below.\n\n" + "\n\n".join(blocks),
        generation_config={
            "response_mime_type": "application/json",
            "response_schema":    VERDICT_SCHEMA,
            "temperature":        0
        }
    )

    wanted = {c["id"] for c in claims}
    verdicts = {}
    for v in json.loads(response.text):
        if v.get("claim_id") in wanted and v.get("label") in LABELS:
            verdicts[v["claim_id"]] = {"label": v["label"], "reason": v.get("reason", "")}
    return verdicts


def verify_report(report: str, sources: list[dict]) -> list[dict]:
    """
    Extracts claims and labels each one.

    Labels beyond the model's three:
    - DANGLING:   cites a number that isn't in the source list
                  (decided here, no model call)
    - UNVERIFIED: the model returned no verdict for it — counted
                  separately so it is never mistaken for SUPPORTED
    """
    claims = extract_claims(report)
    to_judge = []

    for c in claims:
        bad_ids = [i for i in c["source_ids"] if not 1 <= i <= len(sources)]
        if bad_ids:
            c["label"] = "DANGLING"
            c["reason"] = f"cites {bad_ids}, not in the {len(sources)}-source list"
        else:
            to_judge.append(c)

    for start in range(0, len(to_judge), BATCH_SIZE):
        batch = to_judge[start:start + BATCH_SIZE]
        try:
            verdicts = judge_claims(batch, sources)
        except Exception as e:
            print(f"[Verifier] Batch failed: {e}")
            verdicts = {}
        for c in batch:
            v = verdicts.get(c["id"])
            c["label"] = v["label"] if v else "UNVERIFIED"
            c["reason"] = v["reason"] if v else "no verdict returned"

    return claims


def summarize_checks(checks: list[dict]) -> dict:
    """
    Counts labels and computes citation precision:
    SUPPORTED / (claims that got a verdict).
    UNVERIFIED claims are excluded from the denominator rather than
    silently counted either way. Precision is None when nothing was
    checkable, so a report with no numeric citations can't score 100%.
    """
    counts = {label: 0 for label in LABELS + ["DANGLING", "UNVERIFIED"]}
    for c in checks:
        counts[c["label"]] += 1

    judged = len(checks) - counts["UNVERIFIED"]
    precision = round(counts["SUPPORTED"] / judged, 3) if judged else None
    return {"claims_checked": len(checks), **counts, "citation_precision": precision}


def verifier_node(state: ResearchState) -> ResearchState:
    """
    The main Verifier agent function.
    Called by LangGraph between the Writer and the Critic.

    Args:
        state: The shared ResearchState object

    Returns:
        Updated state with citation_checks filled in
    """
    print(f"\n[Verifier] Checking citations against retrieved sources")

    checks = verify_report(state["draft_report"], state["raw_sources"])
    summary = summarize_checks(checks)

    print(f"[Verifier] {summary['claims_checked']} cited claims: "
          f"{summary['SUPPORTED']} supported, {summary['PARTIAL']} partial, "
          f"{summary['UNSUPPORTED']} unsupported, {summary['DANGLING']} dangling, "
          f"{summary['UNVERIFIED']} unverified")
    print(f"[Verifier] Citation precision: {summary['citation_precision']}")

    return {"citation_checks": checks}


def format_failures(checks: list[dict]) -> str:
    """Formats UNSUPPORTED and DANGLING claims as feedback lines."""
    return "\n".join(
        f"- Claim cites {c['source_ids']} but {c['label']}: \"{c['text'][:150]}\" ({c['reason']})"
        for c in checks
        if c["label"] in ("UNSUPPORTED", "DANGLING")
    )


def test_verifier():
    """
    Test the Verifier with a planted false citation.
    Claim 2 cites the RoseTTAFold source for something only the
    AlphaFold source says, and claim 3 cites a source that
    doesn't exist. Run this file directly to verify it works.
    """
    print("Testing Citation Verifier...")

    fake_sources = [
        {
            "title": "Highly accurate protein structure prediction with AlphaFold",
            "url": "https://www.nature.com/articles/s41586-021-03819-2",
            "abstract": "AlphaFold predicts protein structures with atomic accuracy "
                        "even where no similar structure is known, and was the top "
                        "method in CASP14.",
            "date": "2021-07-15"
        },
        {
            "title": "Accurate prediction of protein structures with RoseTTAFold",
            "url": "https://www.science.org/doi/10.1126/science.abj8754",
            "abstract": "RoseTTAFold is a three-track neural network that predicts "
                        "protein structures with accuracy approaching AlphaFold2 "
                        "at lower computational cost.",
            "date": "2021-07-15"
        }
    ]

    fake_report = """
# Protein Folding AI

## Key Findings
AlphaFold predicts protein structures with atomic accuracy and ranked first in CASP14 [1]. RoseTTAFold won CASP14 with atomic accuracy [2]. Both methods now run on mobile phones [3].

## References
[1] Highly accurate protein structure prediction with AlphaFold — https://www.nature.com/articles/s41586-021-03819-2 — 2021
[2] Accurate prediction of protein structures with RoseTTAFold — https://www.science.org/doi/10.1126/science.abj8754 — 2021
"""

    checks = verify_report(fake_report, fake_sources)
    for c in checks:
        print(f"  claim {c['id']} {c['source_ids']} → {c['label']}: {c['reason']}")

    expected = ["SUPPORTED", "UNSUPPORTED", "DANGLING"]
    got = [c["label"] for c in checks]
    print(f"\n{'✓' if got == expected else '✗'} expected {expected}, got {got}")
    print(f"Summary: {summarize_checks(checks)}")


# ── Run test if file is executed directly ─────────────────────
if __name__ == "__main__":
    test_verifier()
