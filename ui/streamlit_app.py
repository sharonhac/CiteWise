"""
streamlit_app.py - CiteWise Legal AI Frontend
===============================================
Professional Hebrew RTL Streamlit interface for Israeli law firms.

Features:
  - Full RTL layout with professional dark-gold law firm theme
  - Real-time streaming answer display
  - Persistent chat history with context-aware follow-up questions
  - Index status panel (document count, definitions count, source list)
  - On-demand sync button
  - Document upload widget
  - Mobile-responsive CSS

Run with:
    streamlit run ui/streamlit_app.py

Author: CiteWise Senior Legal AI Architect
PEP 8 compliant.
"""

import os
import sys
import time
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on the path when running from ui/
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE: str = os.getenv("API_BASE", "http://localhost:8000")
PAGE_TITLE: str = "CiteWise | יועץ משפטי AI"
LOGO_TEXT: str = "⚖️ CiteWise"
TAGLINE: str = "מערכת ייעוץ משפטי מבוססת בינה מלאכותית"

# ---------------------------------------------------------------------------
# CSS — Law Firm Theme + Full RTL
# ---------------------------------------------------------------------------

LAW_FIRM_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Frank+Ruhl+Libre:wght@400;700;900&family=Assistant:wght@400;600&display=swap');

/* ── Root palette ── */
:root {
    --gold:       #C9A84C;
    --gold-light: #E8C87A;
    --dark:       #0D0D0D;
    --surface:    #141414;
    --card:       #1C1C1C;
    --border:     #2E2E2E;
    --text:       #E8E0D0;
    --muted:      #8A8070;
    --accent:     #4A90D9;
    --danger:     #C0392B;
    --success:    #27AE60;
}

/* ── Global RTL + Font ── */
html, body, [class*="css"] {
    direction: rtl !important;
    text-align: right !important;
    font-family: 'Assistant', 'Frank Ruhl Libre', sans-serif !important;
    background-color: var(--dark) !important;
    color: var(--text) !important;
}

/* ── Main container ── */
.main .block-container {
    padding: 1.5rem 2rem !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden !important; }

/* ── Header Banner ── */
.citewise-header {
    background: linear-gradient(135deg, #1A1408 0%, #2C2010 50%, #1A1408 100%);
    border-bottom: 2px solid var(--gold);
    padding: 1.2rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-radius: 8px;
}
.citewise-logo {
    font-family: 'Frank Ruhl Libre', serif !important;
    font-size: 2rem !important;
    font-weight: 900 !important;
    color: var(--gold) !important;
    letter-spacing: 1px;
}
.citewise-tagline {
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 0.2rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-left: 1px solid var(--border) !important;
    border-right: none !important;
    direction: rtl !important;
}
[data-testid="stSidebar"] * {
    direction: rtl !important;
    text-align: right !important;
}
.sidebar-section-title {
    color: var(--gold);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 1rem 0 0.5rem 0;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.3rem;
}

/* ── Status Cards ── */
.status-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.status-card .label { color: var(--muted); font-size: 0.82rem; }
.status-card .value {
    color: var(--gold);
    font-weight: 700;
    font-size: 1.1rem;
}

/* ── Chat messages ── */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 1.5rem;
    max-height: 60vh;
    overflow-y: auto;
    padding: 0.5rem;
}
.message-user {
    background: linear-gradient(135deg, #1E3A5F, #1A3050);
    border: 1px solid var(--accent);
    border-radius: 12px 12px 4px 12px;
    padding: 0.9rem 1.2rem;
    max-width: 82%;
    align-self: flex-start;
    direction: rtl;
    color: #D0E8FF;
}
.message-assistant {
    background: linear-gradient(135deg, #1A1408, #221A0A);
    border: 1px solid var(--gold);
    border-radius: 12px 12px 12px 4px;
    padding: 0.9rem 1.2rem;
    max-width: 92%;
    align-self: flex-end;
    direction: rtl;
    line-height: 1.8;
}
.message-role {
    font-size: 0.72rem;
    color: var(--muted);
    margin-bottom: 0.4rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.message-content { white-space: pre-wrap; word-break: break-word; }

/* ── Citation badges ── */
.citation {
    display: inline-block;
    background: rgba(201, 168, 76, 0.15);
    border: 1px solid var(--gold);
    color: var(--gold-light);
    font-size: 0.72rem;
    padding: 0.1rem 0.5rem;
    border-radius: 4px;
    margin: 0.2rem 0.1rem;
}

/* ── Input box ── */
.stTextArea textarea {
    direction: rtl !important;
    text-align: right !important;
    background-color: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Assistant', sans-serif !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.2) !important;
}

/* ── Buttons ── */
.stButton button {
    background: linear-gradient(135deg, var(--gold), #A07830) !important;
    color: #0D0D0D !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.5rem !important;
    font-family: 'Assistant', sans-serif !important;
    cursor: pointer !important;
    transition: opacity 0.2s ease !important;
}
.stButton button:hover { opacity: 0.85 !important; }
.stButton.secondary button {
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}

/* ── Spinner ── */
.thinking-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--muted);
    font-size: 0.85rem;
    padding: 0.5rem 0;
    direction: rtl;
}
.thinking-dot {
    width: 7px; height: 7px;
    background: var(--gold);
    border-radius: 50%;
    animation: pulse 1.4s infinite;
}
.thinking-dot:nth-child(2) { animation-delay: 0.2s; }
.thinking-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes pulse {
    0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1); }
}

/* ── Source list ── */
.source-item {
    background: var(--card);
    border-right: 3px solid var(--gold);
    padding: 0.4rem 0.8rem;
    margin: 0.3rem 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.82rem;
    color: var(--muted);
    direction: rtl;
}
.source-item span { color: var(--text); font-weight: 600; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    direction: rtl !important;
    background: var(--card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Mobile ── */
@media (max-width: 768px) {
    .main .block-container { padding: 0.8rem 0.8rem !important; }
    .message-user, .message-assistant { max-width: 98% !important; }
    .citewise-logo { font-size: 1.4rem !important; }
}

/* ── Select boxes & dropdowns ── */
.stSelectbox div[data-baseweb="select"] {
    direction: rtl !important;
    background: var(--card) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
"""


# ---------------------------------------------------------------------------
# API Helpers
# ---------------------------------------------------------------------------

def api_get(endpoint: str) -> dict:
    """GET request to the FastAPI backend. Returns {} on failure."""
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"שגיאת תקשורת עם השרת: {exc}")
        return {}


def api_post(endpoint: str, payload: dict, timeout: int = 10) -> dict:
    """POST request to the FastAPI backend. Returns {} on failure."""
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"שגיאת תקשורת עם השרת: {exc}")
        return {}


def api_stream_query(question: str, history: list):
    """
    Stream the answer from POST /query.
    Yields text chunks as they arrive.
    """
    try:
        with requests.post(
            f"{API_BASE}/query",
            json={"question": question, "history": history, "stream": True},
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
    except Exception as exc:
        yield f"\n\n[שגיאה בחיבור לשרת: {exc}]"


def upload_file(uploaded_file) -> dict:
    """Upload a file via POST /upload."""
    try:
        r = requests.post(
            f"{API_BASE}/upload",
            files={"file": (uploaded_file.name, uploaded_file.getvalue())},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# UI Component Renderers
# ---------------------------------------------------------------------------

def render_header():
    """Render the top law firm branding banner."""
    st.markdown(
        f"""
        <div class="citewise-header">
            <div>
                <div class="citewise-logo">{LOGO_TEXT}</div>
                <div class="citewise-tagline">{TAGLINE}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """
    Render the sidebar with:
      - Index status (document chunks, definitions count)
      - Source file list
      - Sync button
      - Upload widget
      - Settings placeholder
    """
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-section-title">📊 מצב מאגר הידע</div>',
            unsafe_allow_html=True,
        )

        # Status cards
        status = st.session_state.get("index_status", {})
        gen_count = status.get("general_count", 0)
        def_count = status.get("definition_count", 0)
        sources = status.get("sources", [])

        st.markdown(
            f"""
            <div class="status-card">
                <span class="label">📄 קטעי מסמכים</span>
                <span class="value">{gen_count:,}</span>
            </div>
            <div class="status-card">
                <span class="label">📖 הגדרות</span>
                <span class="value">{def_count:,}</span>
            </div>
            <div class="status-card">
                <span class="label">📁 קבצים</span>
                <span class="value">{len(sources)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Source file list
        if sources:
            st.markdown(
                '<div class="sidebar-section-title">📂 מסמכים מאונדקסים</div>',
                unsafe_allow_html=True,
            )
            for src in sources:
                st.markdown(
                    f'<div class="source-item">📄 <span>{src}</span></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # Sync button
        st.markdown(
            '<div class="sidebar-section-title">🔄 עדכון מאגר</div>',
            unsafe_allow_html=True,
        )
        if st.button("🔄 סנכרן עכשיו", use_container_width=True):
            with st.spinner("מסנכרן..."):
                report = api_post("/sync/blocking", {}, timeout=300)
                if report:
                    added = report.get("added", [])
                    deleted = report.get("deleted", [])
                    errors = report.get("errors", [])
                    st.success(
                        f"✅ סנכרון הושלם\n"
                        f"נוספו: {len(added)} | הוסרו: {len(deleted)} | שגיאות: {len(errors)}"
                    )
                    _refresh_status()

        st.markdown("---")

        # Upload widget
        st.markdown(
            '<div class="sidebar-section-title">📤 העלאת מסמך</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "בחר קובץ PDF או Word",
            type=["pdf", "docx", "doc"],
            label_visibility="collapsed",
        )
        if uploaded:
            if st.button("📤 העלה ואנדקס", use_container_width=True):
                with st.spinner(f"מעלה את {uploaded.name}..."):
                    result = upload_file(uploaded)
                    if "error" in result:
                        st.error(f"שגיאה: {result['error']}")
                    else:
                        st.success(result.get("message", "הקובץ הועלה בהצלחה."))
                        time.sleep(1)
                        _refresh_status()

        st.markdown("---")

        # Clear chat
        if st.button("🗑 נקה שיחה", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_chat_history():
    """Render all messages in the conversation history."""
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            st.markdown(
                f"""
                <div class="message-user">
                    <div class="message-role">👤 שאלה</div>
                    <div class="message-content">{content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="message-assistant">
                    <div class="message-role">⚖️ עורך דין AI</div>
                    <div class="message-content">{content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_thinking_indicator():
    """Show an animated 'thinking' indicator while the LLM is generating."""
    st.markdown(
        """
        <div class="thinking-indicator">
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
            <span>מעבד את שאלתך...</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session State Helpers
# ---------------------------------------------------------------------------

def _init_session():
    """Initialise session state keys on first load."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "index_status" not in st.session_state:
        _refresh_status()


def _refresh_status():
    """Fetch and cache the index status from the API."""
    status = api_get("/status")
    if status:
        st.session_state.index_status = status


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject CSS
    st.markdown(LAW_FIRM_CSS, unsafe_allow_html=True)

    # Init state
    _init_session()

    # Header
    render_header()

    # Sidebar
    render_sidebar()

    # --- Main chat area ---
    st.markdown(
        """
        <div style="color:#8A8070; font-size:0.85rem; margin-bottom:0.5rem; direction:rtl;">
            שאל שאלה משפטית מפורטת. המערכת תאחזר מידע מהמסמכים המאונדקסים ותספק תשובה
            משפטית מנומקת עם ציטוטי מקורות.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Chat history
    render_chat_history()

    # Input area
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_question = st.text_area(
            "שאלה משפטית",
            placeholder="לדוגמה: מהן חובות הגילוי של מוכר דירה לפי חוק המכר?",
            height=100,
            label_visibility="collapsed",
            key="question_input",
        )
    with col_btn:
        st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
        send_clicked = st.button("שלח ▶", use_container_width=True)

    # Handle submission
    if send_clicked and user_question.strip():
        question = user_question.strip()

        # Add to history
        st.session_state.messages.append({"role": "user", "content": question})

        # Build history list for API (last 6 turns)
        history_for_api = st.session_state.messages[-6:]

        # Show thinking indicator placeholder
        thinking_placeholder = st.empty()
        with thinking_placeholder:
            render_thinking_indicator()

        # Stream answer
        answer_placeholder = st.empty()
        full_answer = ""

        with answer_placeholder:
            answer_container = st.markdown(
                '<div class="message-assistant"><div class="message-role">⚖️ עורך דין AI</div>'
                '<div class="message-content"></div></div>',
                unsafe_allow_html=True,
            )

        for token in api_stream_query(question, history_for_api):
            full_answer += token
            # Update the displayed answer in real-time
            answer_placeholder.markdown(
                f"""
                <div class="message-assistant">
                    <div class="message-role">⚖️ עורך דין AI</div>
                    <div class="message-content">{full_answer}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Clear thinking indicator
        thinking_placeholder.empty()

        # Persist to history and rerun to render clean
        st.session_state.messages.append(
            {"role": "assistant", "content": full_answer}
        )
        st.rerun()

    elif send_clicked and not user_question.strip():
        st.warning("אנא הכנס שאלה לפני השליחה.")

    # Keyboard shortcut hint
    st.markdown(
        '<div style="color:#4A4A4A; font-size:0.72rem; margin-top:0.5rem; direction:rtl;">'
        'טיפ: לחץ על "שלח" לאחר כתיבת השאלה.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
