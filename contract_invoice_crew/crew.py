"""
Crew assembly for the contract-invoice audit pipeline.

Combines the three agents and three tasks into a single CrewAI Crew using
sequential process so tasks run in dependency order (1 → 2 → 3).
"""

from crewai import Crew, Process
from contract_invoice_crew.agents import (
    build_contract_extractor_agent,
    build_invoice_extractor_agent,
    build_discrepancy_analyst_agent,
)
from contract_invoice_crew.tasks import build_tasks


def build_crew() -> Crew:
    """
    Assemble and return the audit Crew.

    Task execution order (Process.sequential):
      1. Contract extraction  (Agent 1)
      2. Invoice extraction   (Agent 2)
      3. Discrepancy analysis (Agent 3) — receives context from tasks 1 & 2

    Returns:
        A configured Crew instance ready for kickoff().
    """
    # Instantiate one copy of each specialist agent
    contract_agent = build_contract_extractor_agent()
    invoice_agent  = build_invoice_extractor_agent()
    analyst_agent  = build_discrepancy_analyst_agent()

    # Build tasks; analyst_task holds context=[contract_task, invoice_task]
    contract_task, invoice_task, discrepancy_task = build_tasks(
        contract_agent, invoice_agent, analyst_agent
    )

    crew = Crew(
        agents=[contract_agent, invoice_agent, analyst_agent],
        tasks=[contract_task, invoice_task, discrepancy_task],
        process=Process.sequential,  # tasks run in list order
        verbose=True,                # stream agent reasoning to console
    )

    return crew
