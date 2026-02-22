"""
chunker.py - CiteWise Legal Document Chunker
============================================
Responsible for:
  1. Splitting cleaned Document objects into legal-grade chunks using
     RecursiveCharacterTextSplitter with \n\n as primary boundary.
  2. Detecting and extracting "Definitions" sections via LLM (Ollama/llama3).
  3. Tagging every chunk with full metadata:
       {source, page, chunk_id, is_definition}

Author: CiteWise Senior Legal AI Architect
PEP 8 compliant.
"""

import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (from .env)
# ---------------------------------------------------------------------------

LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

# ---------------------------------------------------------------------------
# Core splitter
# ---------------------------------------------------------------------------

# \n\n is the primary separator (preserves legal clause boundaries),
# followed by sentence, then word-level fallbacks.
_LEGAL_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", "؟", "!", " ", ""],
    length_function=len,
    is_separator_regex=False,
)

# ---------------------------------------------------------------------------
# Definitions Detection Helpers
# ---------------------------------------------------------------------------

# Common Hebrew/English section headers that indicate a Definitions section.
_DEFINITION_HEADER_PATTERNS = [
    re.compile(r"^\s*(הגדרות|פרשנות|מונחים|הגדרה)\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*\d+[\.\)]\s*(הגדרות|פרשנות)\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Definitions\s*$", re.MULTILINE | re.IGNORECASE),
]


def _heuristic_is_definitions_chunk(text: str) -> bool:
    """
    Fast heuristic check: does this chunk look like it contains definitions?
    Uses regex patterns to avoid an LLM call when the answer is obvious.
    """
    for pattern in _DEFINITION_HEADER_PATTERNS:
        if pattern.search(text):
            return True
    # Also flag chunks that have high density of 'פירושו', 'יהיה', 'means'
    definition_signals = ["פירושו", "פירושה", "משמעותו", "means", "shall mean", "יהיה פירושו"]
    matches = sum(1 for sig in definition_signals if sig in text)
    return matches >= 2


def _extract_definitions_via_llm(text: str) -> List[Dict[str, str]]:
    """
    Use Ollama/llama3 to extract structured definitions from a definitions chunk.

    Returns a list of dicts: [{"term": "...", "definition": "..."}, ...]
    Returns an empty list on failure or if no definitions are found.
    """
    try:
        import ollama  # type: ignore
    except ImportError:
        logger.error("ollama package not installed. Cannot extract definitions via LLM.")
        return []

    prompt = (
        "אתה עוזר משפטי מומחה. הטקסט הבא לקוח ממסמך משפטי ומכיל הגדרות.\n"
        "חלץ את כל ההגדרות ממנו והחזר אותן כ-JSON בלבד, ללא טקסט נוסף.\n"
        "הפורמט הנדרש: [{\"term\": \"...\", \"definition\": \"...\"}]\n\n"
        f"טקסט:\n{text}\n\n"
        "JSON בלבד:"
    )

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response["message"]["content"].strip()

        # Strip markdown code fences if the model wraps JSON in them
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        definitions = json.loads(raw)
        if isinstance(definitions, list):
            return definitions
        return []

    except json.JSONDecodeError as exc:
        logger.warning("LLM returned invalid JSON for definitions extraction: %s", exc)
        return []
    except Exception as exc:
        logger.error("LLM call failed during definitions extraction: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Chunk ID Generation
# ---------------------------------------------------------------------------

def _generate_chunk_id(source: str, page: int, chunk_index: int, text: str) -> str:
    """
    Create a deterministic, unique chunk ID based on source, page,
    chunk index, and a short content hash. This ensures idempotent indexing.
    """
    content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{source}__p{page}__c{chunk_index}__{content_hash}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_documents(
    documents: List[Document],
) -> Tuple[List[Document], List[Document]]:
    """
    Split a list of Documents into legal-grade chunks and separate them
    into general chunks and definition chunks.

    Process:
    1. Split each Document with the legal RecursiveCharacterTextSplitter.
    2. For each chunk, run a heuristic check for definitions.
    3. If the chunk looks like a definitions section, run the LLM extractor.
    4. Tag every chunk with full metadata.

    Parameters
    ----------
    documents : List[Document]
        Cleaned Document objects from load_docs.load_document().

    Returns
    -------
    Tuple[List[Document], List[Document]]
        (general_chunks, definition_chunks)
        Both lists contain LangChain Document objects with enriched metadata.
    """
    general_chunks: List[Document] = []
    definition_chunks: List[Document] = []

    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)

        # Split the cleaned text into chunks
        raw_chunks: List[str] = _LEGAL_SPLITTER.split_text(doc.page_content)

        logger.debug("  %s (p.%d): %d chunk(s) produced.", source, page, len(raw_chunks))

        for idx, chunk_text in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue

            chunk_id = _generate_chunk_id(source, page, idx, chunk_text)
            is_def = _heuristic_is_definitions_chunk(chunk_text)

            base_metadata: Dict[str, Any] = {
                "source": source,
                "page": page,
                "chunk_id": chunk_id,
                "is_definition": is_def,
            }

            chunk_doc = Document(
                page_content=chunk_text,
                metadata=base_metadata,
            )

            if is_def:
                # Attempt LLM-based structured extraction
                extracted = _extract_definitions_via_llm(chunk_text)
                if extracted:
                    # Create one Document per extracted definition for
                    # high-precision retrieval from the definitions index
                    for def_item in extracted:
                        term = def_item.get("term", "").strip()
                        definition = def_item.get("definition", "").strip()
                        if not term or not definition:
                            continue
                        def_text = f"{term}: {definition}"
                        def_id = _generate_chunk_id(source, page, idx, def_text)
                        def_doc = Document(
                            page_content=def_text,
                            metadata={
                                "source": source,
                                "page": page,
                                "chunk_id": def_id,
                                "is_definition": True,
                                "term": term,
                            },
                        )
                        definition_chunks.append(def_doc)
                    logger.info(
                        "  Extracted %d definition(s) from %s p.%d chunk %d.",
                        len(extracted), source, page, idx,
                    )
                else:
                    # LLM extraction failed or found nothing — still store
                    # the raw chunk in the definitions index as a fallback
                    definition_chunks.append(chunk_doc)
                    logger.debug(
                        "  Definitions heuristic hit but LLM extracted 0 items. "
                        "Storing raw chunk in definitions index."
                    )
            else:
                general_chunks.append(chunk_doc)

    logger.info(
        "Chunking complete: %d general chunk(s), %d definition chunk(s).",
        len(general_chunks),
        len(definition_chunks),
    )
    return general_chunks, definition_chunks
