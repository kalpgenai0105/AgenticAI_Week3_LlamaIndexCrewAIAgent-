"""
Agent definitions for the contract-invoice audit crew.

Three specialist agents with non-overlapping responsibilities:
  1. ContractExtractor  — reads the contract PDF, outputs ContractTerms
  2. InvoiceExtractor   — reads the invoice PDF, outputs InvoiceData
  3. DiscrepancyAnalyst — compares both extractions, outputs DiscrepancyReport

LLM is configured from environment variables so switching providers requires
only a .env change, not a code change.

Supported LLM configurations (set in .env):
  Google Gemini:     GEMINI_API_KEY    + MODEL=gemini/gemini-2.0-flash  (default)
  Anthropic Claude:  ANTHROPIC_API_KEY + MODEL=claude-3-5-sonnet-20241022
  OpenAI:            OPENAI_API_KEY    + MODEL=gpt-4o-mini  (or gpt-4o)
"""

import os
from crewai import Agent, LLM
from contract_invoice_crew.tools.pdf_tools import (
    read_contract_pdf,
    read_invoice_pdf,
    query_contract_document,
    query_invoice_document,
)


def _build_llm() -> LLM:
    """
    Construct a CrewAI LLM instance from environment variables.

    MODEL defaults to Claude 3.5 Sonnet. For OpenAI models set
    MODEL=gpt-4o or MODEL=gpt-4o-mini in your .env.
    """
    model = os.getenv("MODEL", "gemini/gemini-2.0-flash")
    # Pass temperature for reproducibility; lower = more consistent JSON output
    return LLM(model=model, temperature=0.0)


def build_contract_extractor_agent() -> Agent:
    """
    Agent 1 – Contract Data Extractor.

    Sole responsibility: read the purchase T&C PDF and produce a structured
    ContractTerms object. It holds the 'Read Contract PDF' tool and uses it
    to obtain the raw document text before extracting.
    """
    return Agent(
        role="Contract Data Extractor",
        goal=(
            "Read the Purchase Terms and Conditions PDF in full using the provided tool, "
            "then extract every commercial term into a clean structured format: "
            "contracted services with agreed unit prices and clause references, "
            "payment terms, payment method, tax policy, and any other clause that "
            "could affect invoice validity."
        ),
        backstory=(
            "You are a meticulous legal-commercial analyst with 15 years of experience "
            "reviewing vendor contracts. You never miss a clause, always cite the exact "
            "section number, and produce structured output that downstream agents can "
            "rely on without second-guessing."
        ),
        tools=[read_contract_pdf, query_contract_document],
        llm=_build_llm(),
        verbose=True,
        # Allow the agent to call the tool once and then reason from its output
        max_iter=3,
    )


def build_invoice_extractor_agent() -> Agent:
    """
    Agent 2 – Invoice Data Extractor.

    Sole responsibility: read the invoice PDF and produce a structured
    InvoiceData object. Captures every field exactly as printed — no rounding,
    no inference.
    """
    return Agent(
        role="Invoice Data Extractor",
        goal=(
            "Read the sample invoice PDF in full using the provided tool, then extract "
            "every data point with complete fidelity: invoice header (number, dates, "
            "parties, payment terms/method), every line item (description, quantity, "
            "unit price, line amount), and all totals (subtotal, tax rate, tax amount, "
            "total due). Reproduce numbers exactly as they appear — do not round or infer."
        ),
        backstory=(
            "You are an expert accounts-payable clerk who has processed tens of thousands "
            "of invoices. You extract data with absolute precision — the quantity '2' is "
            "not '1', and $220.00 is not $210.00. Your structured output is the sole "
            "source of truth for the auditor downstream."
        ),
        tools=[read_invoice_pdf, query_invoice_document],
        llm=_build_llm(),
        verbose=True,
        max_iter=3,
    )


def build_discrepancy_analyst_agent() -> Agent:
    """
    Agent 3 – Contract-Invoice Discrepancy Analyst.

    Sole responsibility: receive the two structured extractions via task context
    and compare them exhaustively. Produces a DiscrepancyReport that covers
    every contracted item, every total, and every relevant clause — including
    explicit 'OK' findings so nothing is silently skipped.
    """
    return Agent(
        role="Contract-Invoice Discrepancy Analyst",
        goal=(
            "Compare the structured contract terms against the structured invoice data "
            "item by item. Identify every discrepancy — pricing mismatches, quantity "
            "irregularities, items not in the contract scope, payment-term conflicts, "
            "tax calculation errors, subtotal errors, and total errors. "
            "For each contracted service and each relevant clause, produce an explicit "
            "finding with status DISCREPANCY or OK, the contract clause reference, and "
            "the exact difference. Never omit an item — if it matches, say OK."
        ),
        backstory=(
            "You are a forensic auditor specialising in vendor invoice compliance reviews. "
            "Your reports are used by finance controllers to approve or dispute payments. "
            "You verify: unit prices match the contract, line totals equal qty × price, "
            "subtotal equals the sum of line totals, tax equals rate × subtotal, and "
            "the total equals subtotal + tax. You cite every contract clause precisely."
        ),
        tools=[],  # Works entirely from structured context — no PDF access needed
        llm=_build_llm(),
        verbose=True,
        max_iter=5,  # Allow extra reasoning cycles for the comparison logic
    )
