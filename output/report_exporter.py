# output/report_exporter.py
# ─────────────────────────────────────────────────────────────
# This file handles exporting the final report to:
# 1. Markdown (.md) file — clean text format
# 2. PDF (.pdf) file — professional downloadable format
# Both files are saved to output/reports/ folder
# ─────────────────────────────────────────────────────────────

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from fpdf import FPDF
import markdown2
from dotenv import load_dotenv

load_dotenv()

# ── Output directory ───────────────────────────────────────────
# Read from .env, default to output/reports
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/reports")


def sanitize_filename(topic: str) -> str:
    """
    Converts a topic string into a safe filename.
    Example: "Latest AI in 2024!" → "latest_ai_in_2024"

    Args:
        topic: Raw topic string from user

    Returns:
        Safe filename string without special characters
    """
    # Lowercase, replace spaces with underscores
    filename = topic.lower().strip()
    filename = filename.replace(" ", "_")

    # Remove any character that isn't alphanumeric or underscore
    filename = "".join(c for c in filename if c.isalnum() or c == "_")

    # Limit length to 50 characters
    filename = filename[:50]

    return filename


def export_markdown(topic: str, report: str) -> str:
    """
    Saves the final report as a .md (Markdown) file.

    Args:
        topic: Research topic (used for filename)
        report: The final report text in Markdown format

    Returns:
        Full path to the saved .md file
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{sanitize_filename(topic)}_{timestamp}.md"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Write the report to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[Exporter] Markdown saved: {filepath}")
    return filepath


class ReportPDF(FPDF):
    """
    Custom PDF class extending FPDF.
    Adds header and footer to every page automatically.
    """

    def header(self):
        # Header — shown at top of every page
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)  # Gray color
        self.cell(0, 10, "Multi-Agent Research Report", align="C")
        self.ln(5)
        # Draw a line under header
        self.set_draw_color(200, 200, 200)
        self.line(10, 20, 200, 20)
        self.ln(5)

    def footer(self):
        # Footer — shown at bottom of every page
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def export_pdf(topic: str, report: str) -> str:
    """
    Saves the final report as a .pdf file.
    Converts Markdown formatting to PDF-friendly text.

    Args:
        topic: Research topic (used for filename + title)
        report: The final report text in Markdown format

    Returns:
        Full path to the saved .pdf file
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{sanitize_filename(topic)}_{timestamp}.pdf"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # ── Create PDF ─────────────────────────────────────────────
    pdf = ReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Process report line by line
    lines = report.split("\n")

    for line in lines:
        line = line.strip()

        if not line:
            # Empty line — add small spacing
            pdf.ln(3)

        elif line.startswith("# "):
            # H1 — Main title
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(30, 30, 30)
            title_text = line[2:].encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 10, title_text)
            pdf.ln(3)

        elif line.startswith("## "):
            # H2 — Section heading
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(50, 80, 140)  # Blue color for headings
            heading_text = line[3:].encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 8, heading_text)
            pdf.ln(2)

        elif line.startswith("### "):
            # H3 — Sub heading
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(70, 70, 70)
            subheading_text = line[4:].encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 7, subheading_text)

        elif line.startswith("- ") or line.startswith("* "):
            # Bullet point
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            bullet_text = "• " + line[2:]
            bullet_text = bullet_text.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 6, bullet_text)

        elif line.startswith("[") and "] " in line:
            # Reference line
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(100, 100, 100)
            ref_text = line.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 6, ref_text)

        else:
            # Regular paragraph text
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            para_text = line.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 6, para_text)

    # Save the PDF
    pdf.output(filepath)

    print(f"[Exporter] PDF saved: {filepath}")
    return filepath


def export_report(topic: str, report: str) -> dict:
    """
    Exports the report in BOTH formats — Markdown and PDF.
    This is the main function called by app.py.

    Args:
        topic: Research topic
        report: Final report text

    Returns:
        Dict with paths to both files:
        {"markdown": "path/to/file.md", "pdf": "path/to/file.pdf"}
    """
    print(f"\n[Exporter] Exporting report for: {topic}")

    md_path = export_markdown(topic, report)
    pdf_path = export_pdf(topic, report)

    return {
        "markdown": md_path,
        "pdf": pdf_path
    }


def test_exporter():
    """
    Test the exporter with a fake report.
    Run this file directly to verify both exports work.
    """
    print("Testing Report Exporter...")

    fake_report = """# Latest Advances in Protein Folding AI

## Executive Summary
Protein folding prediction has been revolutionized by AI systems like
AlphaFold2 and RoseTTAFold. These systems achieve atomic-level accuracy
and are transforming drug discovery.

## Key Findings
AlphaFold2 achieves atomic accuracy in protein structure prediction
[AlphaFold2 Paper, 2024]. This represents a major breakthrough in
computational biology.

RoseTTAFold offers faster inference as a competitive alternative
[RoseTTAFold Paper, 2024].

## Conclusion
AI has fundamentally transformed protein folding prediction.

## References
[1] AlphaFold2 Paper — https://arxiv.org/abs/2106.00565 — 2024
[2] RoseTTAFold Paper — https://arxiv.org/abs/2108.01824 — 2024
"""

    result = export_report("latest advances in protein folding AI", fake_report)

    print(f"\n✓ Markdown: {result['markdown']}")
    print(f"✓ PDF:      {result['pdf']}")
    print("\nCheck the output/reports/ folder for your files!")


# ── Run test if file is executed directly ─────────────────────
if __name__ == "__main__":
    test_exporter()