"""
retriever.py - CiteWise Legal Context Retriever
================================================
Orchestrates the full retrieval pipeline:
  1. Calls hybrid_search (which internally uses MilvusClient).
  2. Formats retrieved chunks into a structured context block
     with mandatory Hebrew citations (מקור / עמוד).
  3. Exposes a clean retrieve() function for the generation layer.

No direct Milvus imports here — all vector access is via ingest.index.

Author: CiteWise Senior Legal AI Architect
PEP 8 compliant.
"""

import logging
import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document

from retrieval.hybrid_search import hybrid_search

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Citation Formatter
# ---------------------------------------------------------------------------

def _format_citation(doc: Document) -> str:
    """
    Format a single Document into a citation string.
    Mandatory format: מקור: [filename], עמוד: [page_number]
    """
    source = doc.metadata.get("source", "לא ידוע")
    page = doc.metadata.get("page", "?")
    return f"מקור: {source}, עמוד: {page}"


def _format_context_block(
    general_docs: List[Document],
    definition_docs: List[Document],
) -> str:
    """
    Build the full context string injected into the LLM prompt.

    Structure:
      [DEFINITIONS SECTION] — injected first so the LLM resolves terminology
      before reading the main evidence chunks.
      [EVIDENCE SECTION] — ranked general chunks with inline citations.

    Parameters
    ----------
    general_docs : List[Document]
        Reranked general document chunks.
    definition_docs : List[Document]
        Relevant definitions from the secondary index.

    Returns
    -------
    str
        Formatted context block (Hebrew section headers).
    """
    parts: List[str] = []

    # -- Definitions --
    if definition_docs:
        parts.append("## הגדרות רלוונטיות ##")
        for doc in definition_docs:
            citation = _format_citation(doc)
            parts.append(f"{doc.page_content.strip()}\n[{citation}]")
        parts.append("")   # blank line separator

    # -- Main Evidence --
    if general_docs:
        parts.append("## קטעים רלוונטיים מהמסמכים ##")
        for i, doc in enumerate(general_docs, start=1):
            citation = _format_citation(doc)
            parts.append(
                f"[{i}] {doc.page_content.strip()}\n[{citation}]"
            )

    if not parts:
        return "לא נמצאו מסמכים רלוונטיים במאגר."

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
) -> Tuple[str, List[Document], List[Document]]:
    """
    Execute the full retrieval pipeline for a user query.

    Parameters
    ----------
    query : str
        The user's Hebrew legal question.

    Returns
    -------
    Tuple[str, List[Document], List[Document]]
        - context_block : Formatted string ready to inject into the LLM prompt.
        - general_docs  : Raw ranked Document list (for UI source display).
        - definition_docs: Raw definition Document list (for UI status display).
    """
    logger.info("Retrieving context for query: %s", query[:80])

    try:
        general_docs, definition_docs = hybrid_search(query=query)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        return "שגיאה: לא ניתן לגשת למאגר הידע.", [], []

    context_block = _format_context_block(general_docs, definition_docs)

    logger.info(
        "Retrieval complete: %d general + %d definition chunk(s).",
        len(general_docs),
        len(definition_docs),
    )

    return context_block, general_docs, definition_docs


def get_sources_summary(docs: List[Document]) -> List[Dict]:
    """
    Return a deduplicated list of source citations for UI display.

    Parameters
    ----------
    docs : List[Document]
        Retrieved documents from retrieve().

    Returns
    -------
    List[Dict]
        Each dict: {source: str, page: int}
    """
    seen = set()
    sources = []
    for doc in docs:
        key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0))
        if key not in seen:
            seen.add(key)
            sources.append({"source": key[0], "page": key[1]})
    return sources
