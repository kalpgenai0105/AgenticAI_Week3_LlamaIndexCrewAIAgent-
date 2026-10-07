"""
Pydantic v2 schemas for structured outputs produced by each agent.

Using typed models (rather than free text) keeps comparisons deterministic:
the analyst agent receives clean floats and strings instead of needing to
re-parse natural-language sentences.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Contract models ────────────────────────────────────────────────────────────

class ContractLineItem(BaseModel):
    """One service or deliverable listed in the contract with its agreed price."""
    description: str = Field(description="Exact name of the service or good")
    unit_price: float = Field(description="Agreed unit price in USD, e.g. 85.00")
    clause_reference: str = Field(
        description="Contract section that defines this price, e.g. 'Section 2 – Pricing'"
    )


class ContractTerms(BaseModel):
    """All commercial terms extracted from the Purchase Terms & Conditions PDF."""
    vendor_name: str = Field(description="Vendor/supplier named in the contract")
    invoice_reference: str = Field(
        description="Invoice number this contract governs, e.g. 'INV-2026-104'"
    )
    line_items: List[ContractLineItem] = Field(
        description="Every contracted service with its agreed unit price and clause ref"
    )
    payment_terms: str = Field(description="Payment terms, e.g. 'Net 14 days from invoice date'")
    payment_method: str = Field(description="Accepted payment method(s)")
    tax_policy: str = Field(
        description="Tax clause verbatim, e.g. 'Applicable sales tax shall be added and paid by the Buyer'"
    )
    additional_notes: Optional[str] = Field(
        default=None,
        description="Any other clauses that could affect invoice validity (delivery, penalties, etc.)"
    )


# ── Invoice models ─────────────────────────────────────────────────────────────

class InvoiceLineItem(BaseModel):
    """One line item as it appears on the invoice."""
    description: str = Field(description="Service or good description as written on the invoice")
    quantity: float = Field(description="Quantity billed")
    unit_price: float = Field(description="Unit price on the invoice in USD")
    amount: float = Field(description="Line total (qty × unit_price) as printed on the invoice")


class InvoiceData(BaseModel):
    """Complete structured extraction of the invoice PDF."""
    invoice_number: str = Field(description="Invoice identifier, e.g. 'INV-2026-104'")
    invoice_date: str = Field(description="Date the invoice was issued")
    due_date: str = Field(description="Payment due date from the invoice")
    vendor_name: str = Field(description="Vendor issuing the invoice")
    client_name: str = Field(description="Client/buyer being billed")
    payment_terms: str = Field(description="Payment terms stated on the invoice")
    payment_method: str = Field(description="Payment method stated on the invoice")
    line_items: List[InvoiceLineItem] = Field(
        description="All line items with quantity, unit price, and line amount"
    )
    subtotal: float = Field(description="Pre-tax subtotal as printed on the invoice")
    tax_rate: Optional[float] = Field(
        default=None,
        description="Tax rate as a decimal, e.g. 0.0825 for 8.25%"
    )
    tax_amount: float = Field(description="Tax amount charged on the invoice")
    total_due: float = Field(description="Final total due as printed on the invoice")


# ── Discrepancy / Report models ────────────────────────────────────────────────

class LineItemDiscrepancy(BaseModel):
    """
    Audit finding for a single line item or contract clause comparison.
    Every contracted item gets exactly one entry — including items with no discrepancy.
    """
    description: str = Field(description="The service/item being audited")
    invoice_says: str = Field(
        description="What the invoice states, e.g. 'Unit price $220.00, Qty 1, Amount $220.00'"
    )
    contract_says: str = Field(
        description="What the contract stipulates, e.g. 'Unit price $210.00 per unit'"
    )
    discrepancy_type: str = Field(
        description=(
            "One of: price_mismatch | quantity_issue | line_total_error | "
            "item_not_in_contract | payment_term_conflict | tax_error | "
            "subtotal_error | total_error | no_discrepancy"
        )
    )
    difference: str = Field(
        description="Size and direction of the gap, e.g. '+$10.00 overcharge' or 'N/A'"
    )
    contract_clause: str = Field(
        description="Relevant contract section, e.g. 'Section 2 – Pricing and Payment'"
    )
    status: str = Field(description="Either 'DISCREPANCY' or 'OK'")


class DiscrepancyReport(BaseModel):
    """
    Final audit report produced by the Discrepancy Analyst agent.
    Covers every line item and every relevant contract clause.
    """
    invoice_number: str = Field(description="Invoice being audited")
    overall_summary: str = Field(
        description="One-paragraph summary of all findings, suitable for a business reader"
    )
    line_item_findings: List[LineItemDiscrepancy] = Field(
        description=(
            "One entry per contracted service plus entries for payment terms, tax, "
            "subtotal, and total — so nothing is silently omitted"
        )
    )
    total_discrepancies_found: int = Field(
        description="Count of entries whose status is 'DISCREPANCY'"
    )
    contract_based_subtotal: float = Field(
        description=(
            "Recalculated subtotal using contract unit prices × invoice quantities. "
            "This is what the subtotal should be."
        )
    )
    invoice_subtotal: float = Field(description="Actual subtotal printed on the invoice")
    subtotal_difference: float = Field(
        description="invoice_subtotal minus contract_based_subtotal (positive = overcharge)"
    )
