"""
Task definitions for the contract-invoice audit crew.

Execution order (Process.sequential):
  Task 1 – extract_contract_task    → ContractTerms  (Agent 1)
  Task 2 – extract_invoice_task     → InvoiceData    (Agent 2)
  Task 3 – discrepancy_report_task  → DiscrepancyReport (Agent 3)
             └─ context=[task1, task2]: Agent 3 receives both prior outputs

output_pydantic instructs CrewAI to parse each agent's JSON response into
the specified Pydantic model, giving downstream tasks strongly-typed context.
"""

from crewai import Task
from contract_invoice_crew.models.schemas import (
    ContractTerms,
    InvoiceData,
    DiscrepancyReport,
)


def build_tasks(contract_agent, invoice_agent, analyst_agent):
    """
    Construct and return the three tasks.

    Args:
        contract_agent: Agent 1 (Contract Data Extractor)
        invoice_agent:  Agent 2 (Invoice Data Extractor)
        analyst_agent:  Agent 3 (Discrepancy Analyst)

    Returns:
        Tuple(contract_task, invoice_task, discrepancy_task)
    """

    # ── Task 1: Extract Contract Terms ─────────────────────────────────────────
    # Agent 1 reads the PDF and returns a ContractTerms JSON object.
    # The task description is deliberately detailed so the LLM knows exactly
    # what fields the Pydantic schema requires.
    contract_task = Task(
        description=(
            "Step 1 of 3: Extract all commercial terms from the purchase contract.\n\n"
            "Instructions:\n"
            "1. Call the 'Read Contract PDF' tool (no arguments needed) to retrieve the "
            "   full text of purchase_terms_conditions.pdf. You may also call the "
            "   'Query Contract Document' tool with a natural-language question to "
            "   retrieve specific information from the contract (e.g. 'What is the "
            "   agreed unit price for Cloud Consulting Services?').\n"
            "2. From that text extract the following into a JSON object:\n"
            "   - vendor_name: the vendor/supplier named in the document\n"
            "   - invoice_reference: the invoice number the contract governs\n"
            "   - line_items: a list where each entry has:\n"
            "       * description: exact service name\n"
            "       * unit_price: agreed price as a float (e.g. 85.0)\n"
            "       * clause_reference: the section that states this price\n"
            "   - payment_terms: full payment terms string\n"
            "   - payment_method: accepted payment method(s)\n"
            "   - tax_policy: the tax clause, verbatim\n"
            "   - additional_notes: any other clause relevant to invoice compliance "
            "     (delivery, penalties, revision charges, etc.) — null if none\n\n"
            "Use numbers exactly as written in the contract. Do not round or infer."
        ),
        expected_output=(
            "A valid JSON object conforming to the ContractTerms schema, containing "
            "all four contracted services with their exact agreed unit prices, the "
            "contract clause that sets each price, and all payment and tax terms."
        ),
        agent=contract_agent,
        output_pydantic=ContractTerms,
    )

    # ── Task 2: Extract Invoice Data ───────────────────────────────────────────
    # Runs in parallel conceptually (no dependency on Task 1) but in sequential
    # mode it follows Task 1. Agent 2 reads its own PDF and returns InvoiceData.
    invoice_task = Task(
        description=(
            "Step 2 of 3: Extract all data from the invoice.\n\n"
            "Instructions:\n"
            "1. Call the 'Read Invoice PDF' tool (no arguments needed) to retrieve the "
            "   full text of sample_invoice.pdf. You may also call the "
            "   'Query Invoice Document' tool with a natural-language question to "
            "   retrieve specific values (e.g. 'What is the subtotal on the invoice?').\n"
            "2. From that text extract the following into a JSON object:\n"
            "   - invoice_number: e.g. 'INV-2026-104'\n"
            "   - invoice_date: date string as printed\n"
            "   - due_date: due date string as printed\n"
            "   - vendor_name: vendor issuing the invoice\n"
            "   - client_name: client being billed\n"
            "   - payment_terms: payment terms string on the invoice\n"
            "   - payment_method: payment method on the invoice\n"
            "   - line_items: a list where each entry has:\n"
            "       * description: service description as printed\n"
            "       * quantity: quantity as a float (e.g. 2.0)\n"
            "       * unit_price: unit price as a float (e.g. 85.0)\n"
            "       * amount: line total as a float (e.g. 170.0)\n"
            "   - subtotal: pre-tax subtotal float\n"
            "   - tax_rate: rate as decimal float (e.g. 0.0825 for 8.25%)\n"
            "   - tax_amount: tax amount float\n"
            "   - total_due: final total due float\n\n"
            "Reproduce every number exactly as printed — do not round, estimate, "
            "or re-calculate. If a field is not on the invoice, use null."
        ),
        expected_output=(
            "A valid JSON object conforming to the InvoiceData schema, containing "
            "all four line items with their exact quantities, unit prices, and line "
            "amounts, plus all header fields and totals exactly as printed."
        ),
        agent=invoice_agent,
        output_pydantic=InvoiceData,
    )

    # ── Task 3: Discrepancy Analysis & Report ──────────────────────────────────
    # context=[contract_task, invoice_task] makes both prior outputs available
    # to Agent 3 as part of its prompt context in sequential mode.
    discrepancy_task = Task(
        description=(
            "Step 3 of 3: Perform a full contract-vs-invoice audit and produce the "
            "discrepancy report.\n\n"
            "You have the ContractTerms (Task 1 output) and InvoiceData (Task 2 output) "
            "available in your context. Use them directly — do NOT call any PDF tools.\n\n"
            "Audit checklist — check every item and produce a LineItemDiscrepancy entry "
            "for each one:\n\n"
            "A. For EACH service in contract.line_items:\n"
            "   1. Locate the matching invoice line item by description.\n"
            "   2. Compare unit_price (contract vs invoice). Flag if different.\n"
            "      Difference = invoice_price − contract_price (positive = overcharge).\n"
            "   3. Note the quantity billed. The contract lists services 'per unit' with "
            "      no explicit maximum; flag qty > 1 only as a note if the contract does "
            "      not explicitly permit multiples for that service.\n"
            "   4. Verify line amount = quantity × invoice_unit_price (arithmetic check).\n"
            "   5. Flag if the service description on the invoice differs materially.\n\n"
            "B. For any invoice line item NOT found in contract.line_items:\n"
            "   Flag as item_not_in_contract.\n\n"
            "C. Payment terms: compare contract.payment_terms vs invoice.payment_terms.\n"
            "D. Payment method: compare contract.payment_method vs invoice.payment_method.\n"
            "E. Subtotal arithmetic: verify invoice.subtotal == sum of all line amounts.\n"
            "F. Tax arithmetic: verify invoice.tax_amount ≈ invoice.tax_rate × invoice.subtotal "
            "   (allow ±$0.02 rounding tolerance). Note the tax_policy from the contract.\n"
            "G. Total arithmetic: verify invoice.total_due == invoice.subtotal + invoice.tax_amount.\n"
            "H. Contract-based subtotal: recalculate using contract unit prices × invoice "
            "   quantities, then compute subtotal_difference = invoice_subtotal − contract_based_subtotal.\n\n"
            "Status rules:\n"
            "  - Status = 'DISCREPANCY' if the values do not match.\n"
            "  - Status = 'OK' if the values match (within $0.02 rounding for tax).\n"
            "  - Every contracted service AND every check (A–H) must have an entry.\n\n"
            "Return a DiscrepancyReport JSON with all required fields."
        ),
        expected_output=(
            "A complete DiscrepancyReport JSON conforming to the schema, with one "
            "LineItemDiscrepancy entry per contracted service plus entries for payment "
            "terms, tax, subtotal, and total checks. Every discrepancy includes the "
            "exact difference, its type, and the relevant contract clause. Every OK "
            "item is explicitly marked OK — nothing is omitted."
        ),
        agent=analyst_agent,
        # context passes Task 1 and Task 2 outputs into this task's prompt
        context=[contract_task, invoice_task],
        output_pydantic=DiscrepancyReport,
    )

    return contract_task, invoice_task, discrepancy_task
