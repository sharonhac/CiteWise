"""
hybrid_search.py - CiteWise Hybrid Search Engine
=================================================
Combines:
  1. Semantic (vector) search via Milvus Lite HNSW (through index.search_collection)
  2. Keyword (BM25) scoring of semantic hits — fully offline, no Elasticsearch
  3. FlashRank cross-encoder re-ranking for final precision

The two-tier architecture searches BOTH the general index and the
definitions index on every query, injecting relevant definitions into
the result context for consistent Hebrew legal terminology.

NOTE: All Milvus access goes through ingest.index.search_collection which
uses MilvusClient (Milvus Lite compatible). No langchain_milvus imports here.

Author: CiteWise Senior Legal AI Architect
PEP 8 compliant.
"""

import logging
import math
import os
import re
from collections import Counter
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEMANTIC_TOP_K: int = int(os.getenv("SEMANTIC_TOP_K", "10"))
KEYWORD_TOP_K: int = int(os.getenv("KEYWORD_TOP_K", "10"))
RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))
DEFS_TOP_K: int = int(os.getenv("DEFS_TOP_K", "3"))

# ---------------------------------------------------------------------------
# Tokeniser (Hebrew + Latin aware)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[\w\u0590-\u05FF]+", re.UNICODE)


def _tokenise(text: str) -> List[str]:
    """Simple whitespace + punctuation tokeniser for Hebrew/Latin text."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# ---------------------------------------------------------------------------
# Lightweight in-memory BM25 scorer
# ---------------------------------------------------------------------------

class _BM25Scorer:
    """
    Minimal BM25 implementation for offline keyword scoring.
    Only used when a corpus is supplied (post-vector-retrieval re-scoring).
    k1 = 1.5, b = 0.75 (standard defaults).
    """

    K1 = 1.5
    B = 0.75

    def __init__(self, corpus: List[str]) -> None:
        self._tokenised = [_tokenise(doc) for doc in corpus]
        self._n = len(corpus)
        self._avgdl = (
            sum(len(t) for t in self._tokenised) / self._n if self._n else 1
        )
        # Document frequency per term
        self._df: Counter = Counter()
        for tokens in self._tokenised:
            for term in set(tokens):
                self._df[term] += 1

    def score(self, query: str, doc_index: int) -> float:
        """BM25 score for a single document."""
        query_terms = _tokenise(query)
        tokens = self._tokenised[doc_index]
        tf_map = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.K1 + 1)
            denominator = tf + self.K1 * (
                1 - self.B + self.B * dl / self._avgdl
            )
            score += idf * numerator / denominator
        return score


# ---------------------------------------------------------------------------
# FlashRank Reranker
# ---------------------------------------------------------------------------

def _flashrank_rerank(
    query: str,
    candidates: List[Document],
    top_k: int,
) -> List[Document]:
    """
    Re-rank candidates with FlashRank cross-encoder.
    Falls back to original order on import error.
    """
    try:
        from flashrank import Ranker, RerankRequest  # type: ignore
    except ImportError:
        logger.warning("flashrank not installed – skipping rerank step.")
        return candidates[:top_k]

    try:
        ranker = Ranker()
        passages = [
            {"id": i, "text": doc.page_content}
            for i, doc in enumerate(candidates)
        ]
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        # results is a list of dicts with 'id' and 'score' keys
        ranked = sorted(results, key=lambda r: r["score"], reverse=True)
        reranked_docs = [candidates[r["id"]] for r in ranked[:top_k]]
        logger.debug("FlashRank reranked %d → %d candidates.", len(candidates), len(reranked_docs))
        return reranked_docs
    except Exception as exc:
        logger.error("FlashRank error: %s – returning original order.", exc)
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# Public Search API
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
) -> Tuple[List[Document], List[Document]]:
    """
    Execute a full hybrid search across both Milvus collections.

    Steps:
      1. Semantic search on general index (top SEMANTIC_TOP_K).
      2. Semantic search on definitions index (top DEFS_TOP_K).
      3. BM25 keyword re-scoring of semantic hits.
      4. Merge and deduplicate by chunk_id.
      5. FlashRank reranking of merged general results.

    Parameters
    ----------
    query : str
        The user's Hebrew legal query.

    Returns
    -------
    Tuple[List[Document], List[Document]]
        (reranked_general_docs, definition_docs)
    """
    # Lazy import avoids circular/module-level Milvus initialisation
    from ingest.index import (                          # noqa: PLC0415
        search_collection,
        COLLECTION_NAME,
        DEFS_COLLECTION_NAME,
    )

    # ---- Step 1: Semantic search (general) ----
    try:
        semantic_hits: List[Document] = search_collection(
            query=query,
            collection_name=COLLECTION_NAME,
            top_k=SEMANTIC_TOP_K,
        )
        logger.debug("Semantic search returned %d hits.", len(semantic_hits))
    except Exception as exc:
        logger.error("Semantic search failed: %s", exc)
        semantic_hits = []

    # ---- Step 2: Definitions semantic search ----
    try:
        def_hits: List[Document] = search_collection(
            query=query,
            collection_name=DEFS_COLLECTION_NAME,
            top_k=DEFS_TOP_K,
        )
        logger.debug("Definitions search returned %d hits.", len(def_hits))
    except Exception as exc:
        logger.error("Definitions search failed: %s", exc)
        def_hits = []

    # ---- Step 3: BM25 keyword scoring of semantic hits ----
    if semantic_hits:
        corpus = [doc.page_content for doc in semantic_hits]
        bm25 = _BM25Scorer(corpus)
        keyword_scores = [bm25.score(query, i) for i in range(len(corpus))]

        # Normalise BM25 scores to [0, 1]
        max_kw = max(keyword_scores) if keyword_scores else 1.0
        norm_kw = [s / max_kw if max_kw > 0 else 0.0 for s in keyword_scores]

        # Combine: semantic rank position score + keyword score
        # semantic score derived from rank (higher rank = higher score)
        n = len(semantic_hits)
        combined: List[Tuple[float, Document]] = []
        for i, doc in enumerate(semantic_hits):
            sem_score = (n - i) / n          # 1.0 → 1/n descending
            kw_score = norm_kw[i]
            combined_score = 0.6 * sem_score + 0.4 * kw_score
            combined.append((combined_score, doc))

        # Sort by combined score, descending
        combined.sort(key=lambda x: x[0], reverse=True)
        merged_docs = [doc for _, doc in combined]
    else:
        merged_docs = []

    # ---- Step 4: Deduplicate by chunk_id ----
    seen_ids: set = set()
    deduped: List[Document] = []
    for doc in merged_docs:
        cid = doc.metadata.get("chunk_id", doc.page_content[:32])
        if cid not in seen_ids:
            seen_ids.add(cid)
            deduped.append(doc)

    # ---- Step 5: FlashRank reranking ----
    final_docs = _flashrank_rerank(query, deduped, top_k=RERANK_TOP_K)

    return final_docs, def_hits
