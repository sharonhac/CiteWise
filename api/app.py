"""
app.py - CiteWise FastAPI Application
======================================
Central API coordinator for the CiteWise legal RAG system.

Endpoints:
  POST /query          — Main RAG query endpoint (streaming)
  POST /sync           — On-demand folder sync via BackgroundTasks
  GET  /status         — Index status (document count, definitions count)
  POST /upload         — Upload a new document to data/ and index it
  GET  /health         — Health check

Scheduled Tasks:
  - sync_folder() runs every 30 minutes via APScheduler.
  - APScheduler is started on application startup (lifespan).

Author: CiteWise Senior Legal AI Architect
PEP 8 compliant.
"""

# ── CRITICAL: must run before ANY other import ─────────────────────────────
# pymilvus.orm.connections is a module-level singleton that reads MILVUS_URI
# from os.environ when first imported. Removing it here ensures no import
# downstream can trigger the network-URI parser with a local file path.
import os
from dotenv import load_dotenv
load_dotenv()
os.environ.pop("MILVUS_URI", None)   # ingest.index will re-read from .env itself
# ───────────────────────────────────────────────────────────────────────────

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH: Path = Path(os.getenv("DATA_PATH", "./data"))
SYNC_INTERVAL_MINUTES: int = int(os.getenv("SYNC_INTERVAL_MINUTES", "30"))
ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ---------------------------------------------------------------------------
# APScheduler Setup
# ---------------------------------------------------------------------------

from apscheduler.schedulers.background import BackgroundScheduler  # noqa: E402

_scheduler = BackgroundScheduler(timezone="Asia/Jerusalem")


def _get_sync_folder():
    """Lazy import of sync_folder to avoid module-level pymilvus init."""
    from ingest.index import sync_folder  # noqa: PLC0415
    return sync_folder


def _get_index_status():
    """Lazy import of get_index_status."""
    from ingest.index import get_index_status  # noqa: PLC0415
    return get_index_status


def _scheduled_sync() -> None:
    """Wrapper for scheduled sync — catches all exceptions to prevent crash."""
    logger.info("APScheduler: running scheduled sync_folder().")
    try:
        report = _get_sync_folder()(DATA_PATH)
        logger.info("Scheduled sync complete: %s", report)
    except Exception as exc:
        logger.error("Scheduled sync failed: %s", exc)


# ---------------------------------------------------------------------------
# Application Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start scheduler on startup, shut down on exit."""
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    _scheduler.add_job(
        _scheduled_sync,
        trigger="interval",
        minutes=SYNC_INTERVAL_MINUTES,
        id="sync_job",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info(
        "APScheduler started – sync every %d minutes.", SYNC_INTERVAL_MINUTES
    )
    yield
    _scheduler.shutdown(wait=False)
    logger.info("APScheduler stopped.")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CiteWise Legal RAG API",
    description="מערכת RAG משפטית מתקדמת לעורכי דין ישראליים",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Payload for the /query endpoint."""
    question: str
    history: Optional[List[dict]] = []   # [{"role": "user"|"assistant", "content": "..."}]
    stream: bool = True


class SyncResponse(BaseModel):
    """Response schema for /sync endpoint."""
    added: List[str]
    deleted: List[str]
    errors: List[str]
    total_on_disk: int


class StatusResponse(BaseModel):
    """Response schema for /status endpoint."""
    general_count: int
    definition_count: int
    sources: List[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    """Simple health check – confirms the API is running."""
    return {"status": "ok", "service": "CiteWise Legal RAG API"}


@app.get("/status", response_model=StatusResponse, tags=["System"])
async def index_status():
    """
    Return current index statistics:
      - general_count   : number of general chunk vectors
      - definition_count: number of definition vectors
      - sources         : list of indexed source filenames
    """
    try:
        status = _get_index_status()()
        return StatusResponse(**status)
    except Exception as exc:
        logger.error("Status check failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/sync", response_model=SyncResponse, tags=["Ingestion"])
async def on_demand_sync(background_tasks: BackgroundTasks):
    """
    Trigger an immediate, on-demand sync of the data/ folder.
    The sync runs as a FastAPI BackgroundTask so the response is returned
    immediately, and the UI can poll /status for updated counts.
    """
    def _run_sync():
        logger.info("On-demand sync triggered via API.")
        try:
            sync_folder(DATA_PATH)
        except Exception as exc:
            logger.error("On-demand sync failed: %s", exc)

    background_tasks.add_task(_run_sync)
    return SyncResponse(added=[], deleted=[], errors=[], total_on_disk=0)


@app.post("/sync/blocking", response_model=SyncResponse, tags=["Ingestion"])
async def on_demand_sync_blocking():
    """
    Blocking sync variant — waits for completion and returns the full report.
    Useful for testing and the Streamlit sync button with result display.
    """
    try:
        report = _get_sync_folder()(DATA_PATH)
        return SyncResponse(**report)
    except Exception as exc:
        logger.error("Blocking sync failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/upload", tags=["Ingestion"])
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    """
    Upload a PDF or Word document to the data/ directory and index it
    immediately in the background.

    Accepted types: .pdf, .docx, .doc
    """
    from ingest.load_docs import SUPPORTED_EXTENSIONS
    from ingest.index import index_file

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"סוג קובץ לא נתמך: {suffix}. קבצים מותרים: {SUPPORTED_EXTENSIONS}",
        )

    dest = DATA_PATH / file.filename
    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info("Uploaded file saved to: %s", dest)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"שמירת הקובץ נכשלה: {exc}")

    def _index():
        summary = index_file(dest)
        logger.info("Upload index summary: %s", summary)

    if background_tasks:
        background_tasks.add_task(_index)
    else:
        _index()

    return {
        "message": f"הקובץ '{file.filename}' הועלה בהצלחה ויאונדקס ברקע.",
        "filename": file.filename,
    }


@app.post("/query", tags=["Query"])
async def query(request: QueryRequest):
    """
    Main RAG query endpoint.

    Workflow:
      1. Retrieve context (hybrid search + definitions injection).
      2. Build the Hebrew legal prompt.
      3. Stream the LLM response token-by-token.

    Returns a StreamingResponse for real-time Streamlit display.
    If stream=False, collects the full response and returns JSON.
    """
    from retrieval.retriever import retrieve
    from generation.prompt import build_rag_prompt, format_history
    from generation.llm import generate

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="השאלה אינה יכולה להיות ריקה.")

    # 1. Retrieve context
    try:
        context_block, general_docs, def_docs = retrieve(question)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"שגיאת אחזור: {exc}")

    # 2. Build prompt
    history_str = format_history(request.history or [])
    prompt = build_rag_prompt(
        context=context_block,
        question=question,
        history=history_str,
    )

    # 3. Stream or return
    if request.stream:
        async def _stream_generator() -> AsyncGenerator[str, None]:
            try:
                for token in generate(prompt=prompt, stream=True):
                    yield token
            except Exception as exc:
                logger.error("Streaming error: %s", exc)
                yield f"\n\n[שגיאה: {exc}]"

        return StreamingResponse(
            _stream_generator(),
            media_type="text/plain; charset=utf-8",
        )
    else:
        from generation.llm import generate_full
        answer = generate_full(prompt=prompt)
        return {
            "answer": answer,
            "sources": [
                {"source": d.metadata.get("source"), "page": d.metadata.get("page")}
                for d in general_docs
            ],
        }