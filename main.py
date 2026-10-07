"""
main.py — Entry point for the Contract-Invoice Audit System.

Usage:
    python main.py

Prerequisites:
    1. Activate your virtual environment.
    2. pip install -r requirements.txt
    3. Copy .env.example → .env and set your API key + MODEL.
    4. Ensure both PDF files are in this directory:
         purchase_terms_conditions.pdf
         sample_invoice.pdf

Outputs:
    discrepancy_report.txt   — human-readable audit report
    discrepancy_report.json  — machine-readable structured report
    Console output           — live agent reasoning + final report
"""

import os
import json
import textwrap
from datetime import datetime

# Load .env BEFORE importing any crew components that read env vars at import time
from dotenv import load_dotenv
load_dotenv()

from contract_invoice_crew.crew import build_crew
from contract_invoice_crew.models.schemas import DiscrepancyReport


# ── Report rendering ───────────────────────────────────────────────────────────

def format_report_as_text(report: DiscrepancyReport) -> str:
    """
    Render a DiscrepancyReport Pydantic model as a human-readable plain-text
    report suitable for email attachment or console output.
    """
    W = 72  # line width
    SEP  = "=" * W
    THIN = "-" * W

    def pad(label: str, value: str, width: int = 27) -> str:
        return f"  {label:<{width}} {value}"

    lines = [
        SEP,
        "  CONTRACT vs INVOICE DISCREPANCY REPORT",
        pad("Invoice audited:", report.invoice_number),
        pad("Report generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        SEP,
        "",
        "OVERALL SUMMARY",
        THIN,
        *textwrap.wrap(report.overall_summary, width=W),
        "",
        pad("Total discrepancies found:", str(report.total_discrepancies_found)),
        pad("Contract-based subtotal:",   f"${report.contract_based_subtotal:>8.2f}"),
        pad("Invoice subtotal:",          f"${report.invoice_subtotal:>8.2f}"),
        pad("Subtotal difference:",       f"${report.subtotal_difference:>+8.2f}"
            + ("  (overcharge)" if report.subtotal_difference > 0.005
               else "  (undercharge)" if report.subtotal_difference < -0.005
               else "  (no net difference)")),
        "",
        SEP,
        "LINE-ITEM AND CLAUSE FINDINGS",
        SEP,
    ]

    for idx, f in enumerate(report.line_item_findings, start=1):
        badge = "[ DISCREPANCY ]" if f.status == "DISCREPANCY" else "[     OK      ]"
        lines += [
            f"\n{idx:>2}. {f.description}",
            f"    Status:           {badge}",
            f"    Type:             {f.discrepancy_type}",
            f"    Invoice says:     {f.invoice_says}",
            f"    Contract says:    {f.contract_says}",
            f"    Difference:       {f.difference}",
            f"    Contract clause:  {f.contract_clause}",
            "    " + THIN[4:],
        ]

    lines += [
        "",
        SEP,
        "END OF REPORT",
        SEP,
    ]

    return "\n".join(lines)


# ── API key validation ─────────────────────────────────────────────────────────

def _check_api_keys() -> None:
    """Raise a clear error if no LLM API key is configured."""
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "\n[ERROR] No LLM API key found.\n"
            "  • Copy .env.example to .env and fill in your key:\n"
            "      GEMINI_API_KEY=...             (for Google Gemini — default)\n"
            "      ANTHROPIC_API_KEY=sk-ant-...   (for Claude)\n"
            "      OPENAI_API_KEY=sk-...          (for GPT-4o)\n"
            "  • Also set MODEL= to match the provider (see .env.example).\n"
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Build the crew, run the pipeline, print and save the report."""

    print("\n" + "=" * 72)
    print("  Contract-Invoice Audit System  |  CrewAI Multi-Agent Pipeline")
    print("  Model:", os.getenv("MODEL", "gemini/gemini-2.0-flash"))
    print("=" * 72 + "\n")

    _check_api_keys()

    # ── Run the crew ───────────────────────────────────────────────────────────
    crew   = build_crew()
    result = crew.kickoff()

    # ── Retrieve the DiscrepancyReport pydantic object ─────────────────────────
    # CrewAI stores the last task's pydantic output in CrewOutput.pydantic.
    # If that's None (rare edge case), fall back to parsing result.raw.
    report: DiscrepancyReport | None = result.pydantic

    if report is None:
        print("\n[WARNING] result.pydantic was None; attempting to parse result.raw.\n")
        raw = result.raw.strip()
        # Strip markdown code fences if the LLM wrapped the JSON
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()
        report = DiscrepancyReport.model_validate_json(raw)

    # ── Render text report ─────────────────────────────────────────────────────
    text_report = format_report_as_text(report)

    # ── Print to console ───────────────────────────────────────────────────────
    print("\n" + text_report)

    # ── Write text report ──────────────────────────────────────────────────────
    txt_path = "discrepancy_report.txt"
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(text_report)
    print(f"\n[INFO] Text report saved  → {txt_path}")

    # ── Write JSON report ──────────────────────────────────────────────────────
    json_path = "discrepancy_report.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    print(f"[INFO] JSON report saved  → {json_path}\n")


if __name__ == "__main__":
    main()
