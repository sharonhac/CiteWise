"""
CiteWise Ingestion Package
--------------------------
Exposes the three core ingestion modules:
  - load_docs : document loading & cleaning
  - chunker   : legal-aware chunking & definition extraction
  - index     : Milvus Lite vector indexing & folder sync

NOTE: Imports are intentionally lazy (not at package level) to prevent
pymilvus from attempting to connect at import time.
"""


def load_document(file_path):
    from ingest.load_docs import load_document as _f
    return _f(file_path)


def load_all_documents(data_dir):
    from ingest.load_docs import load_all_documents as _f
    return _f(data_dir)


def chunk_documents(documents):
    from ingest.chunker import chunk_documents as _f
    return _f(documents)


def index_file(file_path):
    from ingest.index import index_file as _f
    return _f(file_path)


def sync_folder(data_dir=None):
    from ingest.index import sync_folder as _f
    return _f(data_dir)


def get_index_status():
    from ingest.index import get_index_status as _f
    return _f()


__all__ = [
    "load_document",
    "load_all_documents",
    "chunk_documents",
    "index_file",
    "sync_folder",
    "get_index_status",
]
