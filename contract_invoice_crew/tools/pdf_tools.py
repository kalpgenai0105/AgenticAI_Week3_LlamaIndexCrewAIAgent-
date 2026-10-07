"""
LlamaIndex-powered PDF tools for the contract-invoice audit crew.

Two tool pairs are provided per document:
  - read_*_pdf()              : full-document text via LlamaIndex SimpleDirectoryReader
  - query_*_document(query)   : semantic query engine over a per-document VectorStoreIndex

LlamaIndex is configured automatically from the same env vars used by CrewAI:
  OPENAI_API_KEY    → OpenAI LLM + OpenAI embeddings (text-embedding-3-small)
  GEMINI_API_KEY    → Gemini LLM + Gemini embeddings
  ANTHROPIC_API_KEY → Anthropic LLM + OpenAI embeddings (set OPENAI_API_KEY too)
"""

import os
import logging
from pathlib import Path
from functools import lru_cache

from crewai.tools import tool

_BASE_DIR      = Path(__file__).resolve().parents[2]
_CONTRACT_PDF  = _BASE_DIR / "purchase_terms_conditions.pdf"
_INVOICE_PDF   = _BASE_DIR / "sample_invoice.pdf"

# Suppress noisy LlamaIndex tokenizer warnings
logging.getLogger("llama_index").setLevel(logging.WARNING)


# ── LlamaIndex setup ───────────────────────────────────────────────────────────

def _configure_settings() -> None:
    """
    Wire LlamaIndex Settings (LLM + embed model) from the project's env vars.
    Called once before any index is built.
    """
    from llama_index.core import Settings

    model_env = os.getenv("MODEL", "gemini/gemini-2.0-flash")

    if os.getenv("OPENAI_API_KEY"):
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.openai import OpenAIEmbedding
        Settings.llm         = OpenAI(model="gpt-4o-mini", temperature=0.0)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    elif os.getenv("GEMINI_API_KEY"):
        from llama_index.llms.gemini import Gemini
        from llama_index.embeddings.gemini import GeminiEmbedding
        Settings.llm         = Gemini(model_name="models/gemini-2.0-flash", temperature=0.0)
        Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")

    elif os.getenv("ANTHROPIC_API_KEY"):
        from llama_index.llms.anthropic import Anthropic
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        Settings.llm         = Anthropic(model="claude-3-5-sonnet-20241022", temperature=0.0)
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def _load_documents(pdf_path: Path):
    """Load all pages from a PDF as LlamaIndex Document objects."""
    from llama_index.core import SimpleDirectoryReader
    return SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()


@lru_cache(maxsize=2)
def _build_query_engine(pdf_path_str: str):
    """
    Build and cache a VectorStoreIndex query engine for a given PDF.
    Returns None if the PDF doesn't exist or setup fails.
    """
    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        return None
    try:
        from llama_index.core import VectorStoreIndex
        _configure_settings()
        docs   = _load_documents(pdf_path)
        index  = VectorStoreIndex.from_documents(docs, show_progress=False)
        return index.as_query_engine(similarity_top_k=8)
    except Exception as exc:
        return f"ERROR building query engine: {exc}"


# ── Full-document read tools ───────────────────────────────────────────────────

@tool("Read Contract PDF")
def read_contract_pdf() -> str:
    """
    Load purchase_terms_conditions.pdf with LlamaIndex and return the complete
    text of every page.  Use this tool once to obtain all contract clauses,
    agreed unit prices, payment terms, tax policy, and commercial conditions
    before extracting structured data.  No input required.
    """
    if not _CONTRACT_PDF.exists():
        return f"ERROR: Contract PDF not found at '{_CONTRACT_PDF}'."

    docs  = _load_documents(_CONTRACT_PDF)
    parts = []
    for i, doc in enumerate(docs, start=1):
        page_label = doc.metadata.get("page_label") or doc.metadata.get("page") or i
        parts.append(f"\n{'─' * 60}")
        parts.append(f"PAGE {page_label}")
        parts.append(f"{'─' * 60}")
        parts.append(doc.text.strip())

    return "\n".join(parts) if parts else "No content extracted from contract PDF."


@tool("Read Invoice PDF")
def read_invoice_pdf() -> str:
    """
    Load sample_invoice.pdf with LlamaIndex and return the complete text of
    every page, including all line items, header fields, and totals.
    Use this tool once before extracting structured invoice data.
    No input required.
    """
    if not _INVOICE_PDF.exists():
        return f"ERROR: Invoice PDF not found at '{_INVOICE_PDF}'."

    docs  = _load_documents(_INVOICE_PDF)
    parts = []
    for i, doc in enumerate(docs, start=1):
        page_label = doc.metadata.get("page_label") or doc.metadata.get("page") or i
        parts.append(f"\n{'─' * 60}")
        parts.append(f"PAGE {page_label}")
        parts.append(f"{'─' * 60}")
        parts.append(doc.text.strip())

    return "\n".join(parts) if parts else "No content extracted from invoice PDF."


# ── Semantic query engine tools ────────────────────────────────────────────────

@tool("Query Contract Document")
def query_contract_document(query: str) -> str:
    """
    Run a semantic query against a LlamaIndex VectorStoreIndex built from the
    contract PDF.  Use this tool to retrieve precise answers about specific
    clauses, prices, or terms — e.g. 'What is the agreed unit price for Cloud
    Consulting Services?' — without re-reading the entire document.
    Input: a natural-language question about the contract.
    """
    engine = _build_query_engine(str(_CONTRACT_PDF))
    if engine is None:
        return f"ERROR: Contract PDF not found at '{_CONTRACT_PDF}'."
    if isinstance(engine, str):      # error string from _build_query_engine
        return engine
    result = engine.query(query)
    return str(result)


@tool("Query Invoice Document")
def query_invoice_document(query: str) -> str:
    """
    Run a semantic query against a LlamaIndex VectorStoreIndex built from the
    invoice PDF.  Use this tool to retrieve precise values — e.g. 'What is the
    unit price charged for Cloud Consulting Services?' or 'What is the invoice
    subtotal?' — without re-reading the full document.
    Input: a natural-language question about the invoice.
    """
    engine = _build_query_engine(str(_INVOICE_PDF))
    if engine is None:
        return f"ERROR: Invoice PDF not found at '{_INVOICE_PDF}'."
    if isinstance(engine, str):
        return engine
    result = engine.query(query)
    return str(result)
