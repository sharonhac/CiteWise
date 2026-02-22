"""
index.py - CiteWise Legal Vector Indexer
=========================================
Manages two Milvus Lite vector collections.

ARCHITECTURE NOTE — why we do NOT import pymilvus at module level:
  pymilvus>=2.4 ships an ORM singleton (pymilvus.orm.connections.Connections)
  that is instantiated at *import time* of pymilvus/__init__.py. Its __init__
  reads the MILVUS_URI environment variable and attempts to parse it as a
  network address. Any local path — absolute or relative — raises:
    ConnectionConfigException: Illegal uri: [./citewise_db.db]

  The only safe approach is:
    1. Pop MILVUS_URI from os.environ BEFORE any pymilvus import.
    2. Import MilvusClient lazily — inside functions, never at top level.
    3. Pass the db path directly to MilvusClient(), never via env var.

  Also: if langchain-milvus is still installed in your venv, its __init__.py
  triggers pymilvus.orm unconditionally. Uninstall it:
    pip uninstall -y langchain-milvus

Author: CiteWise Senior Legal AI Architect
PEP 8 compliant.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional, Set

from dotenv import load_dotenv
from langchain_core.documents import Document

# ── Step 1: load .env values into os.environ ──────────────────────────────
load_dotenv()

# ── Step 2: grab & resolve the DB path, then REMOVE it from the environment
#    so that pymilvus ORM cannot find it when imported later.
_raw_uri: str = os.environ.pop("MILVUS_URI", "./citewise_db.db")
DB_PATH: str = str(Path(_raw_uri).resolve())

# ── Step 3: rest of config (safe — no Milvus involved) ────────────────────
DATA_PATH: str = os.getenv("DATA_PATH", "./data")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "legal_docs")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
DEFS_COLLECTION_NAME: str = f"{COLLECTION_NAME}_defs"
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "768"))

logger = logging.getLogger(__name__)
logger.info("CiteWise DB path: %s", DB_PATH)


# ---------------------------------------------------------------------------
# Lazy singletons — nothing Milvus-related at module level
# ---------------------------------------------------------------------------

_milvus_client = None
_embeddings = None


def _get_client():
    """
    Lazily create and return the MilvusClient.

    pymilvus is imported HERE (inside the function), never at module level.
    By this point MILVUS_URI has already been removed from os.environ, so
    the ORM singleton has nothing to parse.
    """
    global _milvus_client
    if _milvus_client is None:
        from pymilvus import MilvusClient  # noqa: PLC0415  ← lazy, safe
        _milvus_client = MilvusClient(DB_PATH)
        logger.info("MilvusClient opened: %s", DB_PATH)
    return _milvus_client


def _get_embeddings():
    """Lazily create and return the OllamaEmbeddings instance."""
    global _embeddings
    if _embeddings is None:
        from langchain_ollama import OllamaEmbeddings  # noqa: PLC0415
        _embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        logger.info("Embeddings ready: %s", EMBEDDING_MODEL)
    return _embeddings


# ---------------------------------------------------------------------------
# Collection Bootstrap
# ---------------------------------------------------------------------------

def _ensure_collection(collection_name: str) -> None:
    """
    Create the collection + HNSW index if it does not exist yet.

    Schema fields
    -------------
    id            VARCHAR(256)  primary key  — our deterministic chunk_id
    vector        FLOAT_VECTOR(768)          — HNSW COSINE index
    text          VARCHAR(65535)             — chunk content
    source        VARCHAR(512)               — source filename
    page          INT32                      — page number
    is_definition BOOL                       — definition flag
    chunk_id      VARCHAR(256)               — copy of id for explicit query
    """
    client = _get_client()
    if client.has_collection(collection_name):
        return

    from pymilvus import DataType  # noqa: PLC0415

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id",            DataType.VARCHAR,      max_length=256,   is_primary=True)
    schema.add_field("vector",        DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field("text",          DataType.VARCHAR,      max_length=65535)
    schema.add_field("source",        DataType.VARCHAR,      max_length=512)
    schema.add_field("page",          DataType.INT32)
    schema.add_field("is_definition", DataType.BOOL)
    schema.add_field("chunk_id",      DataType.VARCHAR,      max_length=256)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    logger.info("Created collection '%s'.", collection_name)


def _bootstrap() -> None:
    """Ensure both collections exist. Call once on startup."""
    _ensure_collection(COLLECTION_NAME)
    _ensure_collection(DEFS_COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch-embed a list of strings via Ollama."""
    return _get_embeddings().embed_documents(texts)


def embed_query(query: str) -> List[float]:
    """Embed a single query string. Called by retrieval/hybrid_search.py."""
    return _get_embeddings().embed_query(query)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def _index_chunks(chunks: List[Document], collection_name: str) -> int:
    """
    Embed and upsert Document chunks into the named Milvus collection.
    Returns the count of successfully stored chunks.
    """
    if not chunks:
        return 0

    _ensure_collection(collection_name)
    client = _get_client()

    texts = [c.page_content for c in chunks]
    try:
        vectors = _embed_texts(texts)
    except Exception as exc:
        logger.error("Embedding failed (%d chunks): %s", len(chunks), exc)
        return 0

    rows = []
    for chunk, vector in zip(chunks, vectors):
        meta = chunk.metadata
        pk = (meta.get("chunk_id") or str(uuid.uuid4()))[:256]
        rows.append({
            "id":            pk,
            "vector":        vector,
            "text":          chunk.page_content[:65000],
            "source":        meta.get("source", "")[:512],
            "page":          int(meta.get("page", 0)),
            "is_definition": bool(meta.get("is_definition", False)),
            "chunk_id":      pk,
        })

    try:
        client.upsert(collection_name=collection_name, data=rows)
        logger.info("Upserted %d row(s) → '%s'.", len(rows), collection_name)
        return len(rows)
    except Exception as exc:
        logger.error("Upsert failed for '%s': %s", collection_name, exc)
        return 0


def index_file(file_path: Path) -> dict:
    """
    Full ingestion pipeline: load → clean → chunk → embed → upsert.

    Parameters
    ----------
    file_path : Path
        Path to a PDF or Word document.

    Returns
    -------
    dict
        {source, general_chunks, definition_chunks, status}
    """
    from ingest.load_docs import load_document    # noqa: PLC0415
    from ingest.chunker import chunk_documents    # noqa: PLC0415

    logger.info("Ingesting: %s", file_path.name)
    summary = {
        "source": file_path.name,
        "general_chunks": 0,
        "definition_chunks": 0,
        "status": "error",
    }

    try:
        docs = load_document(file_path)
        if not docs:
            logger.warning("No content in %s.", file_path.name)
            summary["status"] = "empty"
            return summary

        general_chunks, definition_chunks = chunk_documents(docs)
        summary["general_chunks"] = _index_chunks(general_chunks, COLLECTION_NAME)
        summary["definition_chunks"] = _index_chunks(definition_chunks, DEFS_COLLECTION_NAME)
        summary["status"] = "ok"

    except Exception as exc:
        logger.error("Ingestion error [%s]: %s", file_path.name, exc)

    return summary


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def delete_file_vectors(source_filename: str) -> None:
    """
    Delete all vectors for the given source filename from both collections.

    Parameters
    ----------
    source_filename : str
        Basename, e.g. "contract.pdf".
    """
    client = _get_client()
    safe = source_filename.replace('"', '\\"')
    expr = f'source == "{safe}"'
    for col in (COLLECTION_NAME, DEFS_COLLECTION_NAME):
        try:
            if not client.has_collection(col):
                continue
            client.delete(collection_name=col, filter=expr)
            logger.info("Deleted '%s' from '%s'.", source_filename, col)
        except Exception as exc:
            logger.error("Delete error '%s'/'%s': %s", source_filename, col, exc)


# ---------------------------------------------------------------------------
# Source Tracker
# ---------------------------------------------------------------------------

def _get_indexed_sources() -> Set[str]:
    """
    Return the set of source filenames currently stored in the general
    collection. Used by sync_folder() to detect files removed from disk.
    """
    client = _get_client()
    try:
        if not client.has_collection(COLLECTION_NAME):
            return set()
        rows = client.query(
            collection_name=COLLECTION_NAME,
            filter='source != ""',
            output_fields=["source"],
            limit=16384,
        )
        return {r["source"] for r in rows if r.get("source")}
    except Exception as exc:
        logger.error("Could not query indexed sources: %s", exc)
        return set()


# ---------------------------------------------------------------------------
# Vector Search  (called by retrieval/hybrid_search.py)
# ---------------------------------------------------------------------------

def search_collection(
    query: str,
    collection_name: str,
    top_k: int = 10,
) -> List[Document]:
    """
    Semantic HNSW search against the named collection.

    Parameters
    ----------
    query : str
        Hebrew legal query string.
    collection_name : str
        Either COLLECTION_NAME or DEFS_COLLECTION_NAME.
    top_k : int
        Number of nearest-neighbour results to return.

    Returns
    -------
    List[Document]
        LangChain Documents with metadata, ordered by COSINE similarity.
    """
    client = _get_client()
    _ensure_collection(collection_name)

    try:
        query_vector = embed_query(query)
    except Exception as exc:
        logger.error("Query embedding failed: %s", exc)
        return []

    try:
        results = client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["text", "source", "page", "is_definition", "chunk_id"],
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        )
    except Exception as exc:
        logger.error("Milvus search error in '%s': %s", collection_name, exc)
        return []

    docs: List[Document] = []
    for hit in results[0]:
        entity = hit.get("entity", hit)
        docs.append(Document(
            page_content=entity.get("text", ""),
            metadata={
                "source":        entity.get("source", ""),
                "page":          entity.get("page", 0),
                "is_definition": entity.get("is_definition", False),
                "chunk_id":      entity.get("chunk_id", ""),
                "score":         hit.get("distance", 0.0),
            },
        ))
    return docs


# ---------------------------------------------------------------------------
# Sync Mechanism
# ---------------------------------------------------------------------------

def sync_folder(data_dir: Optional[Path] = None) -> dict:
    """
    Diff-based sync of the data/ directory vs the vector store.

    Algorithm
    ---------
    1. List supported files on disk.
    2. Query Milvus for currently indexed source names.
    3. Index files that are on disk but not yet indexed.
    4. Delete vectors for files that were removed from disk.

    Returns
    -------
    dict
        {added, deleted, errors, total_on_disk}
    """
    from ingest.load_docs import SUPPORTED_EXTENSIONS  # noqa: PLC0415

    data_dir = data_dir or Path(DATA_PATH)
    logger.info("sync_folder: %s", data_dir.resolve())

    report: dict = {"added": [], "deleted": [], "errors": [], "total_on_disk": 0}

    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        return report

    disk_files: dict = {
        p.name: p
        for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    }
    report["total_on_disk"] = len(disk_files)
    disk_names: Set[str] = set(disk_files.keys())
    indexed_names: Set[str] = _get_indexed_sources()

    # Add new files
    for name in sorted(disk_names - indexed_names):
        logger.info("Sync ADD: %s", name)
        s = index_file(disk_files[name])
        (report["added"] if s["status"] == "ok" else report["errors"]).append(name)

    # Remove deleted files
    for name in sorted(indexed_names - disk_names):
        logger.info("Sync DELETE: %s", name)
        try:
            delete_file_vectors(name)
            report["deleted"].append(name)
        except Exception as exc:
            logger.error("Sync delete error '%s': %s", name, exc)
            report["errors"].append(name)

    logger.info(
        "Sync complete — +%d -%d err:%d",
        len(report["added"]), len(report["deleted"]), len(report["errors"]),
    )
    return report


# ---------------------------------------------------------------------------
# Status Helper  (called by GET /status and Streamlit sidebar)
# ---------------------------------------------------------------------------

def get_index_status() -> dict:
    """
    Return current index statistics. Safe to call even before any documents
    have been indexed (collections may not exist yet).

    Returns
    -------
    dict
        {general_count: int, definition_count: int, sources: List[str]}
    """
    status = {"general_count": 0, "definition_count": 0, "sources": []}
    try:
        client = _get_client()

        if client.has_collection(COLLECTION_NAME):
            stats = client.get_collection_stats(COLLECTION_NAME)
            status["general_count"] = int(stats.get("row_count", 0))
            try:
                rows = client.query(
                    collection_name=COLLECTION_NAME,
                    filter='source != ""',
                    output_fields=["source"],
                    limit=16384,
                )
                status["sources"] = sorted(
                    {r["source"] for r in rows if r.get("source")}
                )
            except Exception as qe:
                logger.warning("Source query failed: %s", qe)

        if client.has_collection(DEFS_COLLECTION_NAME):
            stats = client.get_collection_stats(DEFS_COLLECTION_NAME)
            status["definition_count"] = int(stats.get("row_count", 0))

    except Exception as exc:
        logger.error("get_index_status error: %s", exc)

    return status


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _bootstrap()
    report = sync_folder()
    print("\n=== Sync Report ===")
    print(f"  Files on disk : {report['total_on_disk']}")
    print(f"  Added         : {report['added']}")
    print(f"  Deleted       : {report['deleted']}")
    print(f"  Errors        : {report['errors']}")