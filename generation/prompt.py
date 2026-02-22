"""
prompt.py - CiteWise Legal Prompt Templates
============================================
Defines all prompt templates used by the generation layer.

Design principles:
  - All prompts are in Hebrew to enforce professional legal register.
  - The system persona is "עורך דין ישראלי בכיר" (Senior Israeli Attorney).
  - Every answer must cite sources using the mandatory format.
  - Precision over brevity: the LLM is instructed to prefer detailed,
    accurate answers over short summaries.
  - Definitions are injected before the main context so terminology
    is resolved before the model reads the evidence.

Author: CiteWise Senior Legal AI Architect
PEP 8 compliant.
"""

from string import Template

# ---------------------------------------------------------------------------
# System Persona
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """אתה עורך דין ישראלי בכיר עם התמחות בניתוח חוזים ומסמכים משפטיים.
תפקידך לספק ייעוץ משפטי מקצועי, מדויק ומפורט בהתבסס אך ורק על המסמכים שסופקו לך.

כללים מחייבים:
1. השב תמיד בעברית משפטית רשמית ומקצועית.
2. כל טענה עובדתית חייבת להיות מצוטטת עם המקור המדויק:
   פורמט חובה: (מקור: [שם קובץ], עמוד: [מספר])
3. אם המידע אינו קיים במסמכים שסופקו – הצהר זאת במפורש.
   לעולם אל תמציא מידע שאינו קיים במאגר.
4. העדף פירוט ודיוק על פני תמציתיות.
5. כאשר קיימות הגדרות רלוונטיות, השתמש בהן לפרשנות מונחים.
6. בסיום כל תשובה, ספק סיכום קצר של המקורות שנסתמכת עליהם.
"""

# ---------------------------------------------------------------------------
# RAG Answer Template
# ---------------------------------------------------------------------------

# Placeholders:
#   $context   — the formatted context block from retriever.py
#   $history   — prior conversation turns (may be empty string)
#   $question  — the user's current question
RAG_TEMPLATE = Template("""
$system_prompt

---
היסטוריית שיחה:
$history

---
הקשר מתוך המסמכים:
$context

---
שאלת המשתמש:
$question

---
תשובה משפטית מפורטת (בעברית, עם ציטוטי מקורות):
""")


def build_rag_prompt(
    context: str,
    question: str,
    history: str = "",
) -> str:
    """
    Build the full RAG prompt string for a user query.

    Parameters
    ----------
    context : str
        The formatted context block from retriever.format_context_block().
    question : str
        The user's Hebrew legal question.
    history : str
        Formatted prior conversation turns. Empty string if first turn.

    Returns
    -------
    str
        The complete prompt ready to send to the LLM.
    """
    return RAG_TEMPLATE.substitute(
        system_prompt=SYSTEM_PROMPT,
        context=context,
        question=question,
        history=history or "אין היסטוריה קודמת.",
    )


# ---------------------------------------------------------------------------
# Definitions Extraction Prompt (used by chunker.py)
# ---------------------------------------------------------------------------

DEFINITIONS_EXTRACTION_SYSTEM = (
    "אתה עוזר משפטי מומחה. חלץ הגדרות ממסמכים משפטיים."
)

DEFINITIONS_EXTRACTION_TEMPLATE = Template("""
חלץ את כל ההגדרות מהטקסט הבא והחזר אותן כ-JSON בלבד.
פורמט: [{"term": "...", "definition": "..."}]
אל תוסיף טקסט נוסף, הסברים, או גרשיים.

טקסט:
$text

JSON:
""")


def build_definitions_prompt(text: str) -> str:
    """
    Build the definitions extraction prompt for a chunk of text.

    Parameters
    ----------
    text : str
        A raw text chunk suspected to contain legal definitions.

    Returns
    -------
    str
        The extraction prompt.
    """
    return DEFINITIONS_EXTRACTION_TEMPLATE.substitute(text=text)


# ---------------------------------------------------------------------------
# Conversation History Formatter
# ---------------------------------------------------------------------------

def format_history(messages: list) -> str:
    """
    Format a list of conversation messages into a prompt-safe string.

    Parameters
    ----------
    messages : list
        List of dicts: [{"role": "user"|"assistant", "content": "..."}]

    Returns
    -------
    str
        Formatted conversation history string.
    """
    if not messages:
        return ""
    lines = []
    role_map = {"user": "משתמש", "assistant": "עורך דין AI"}
    for msg in messages:
        role = role_map.get(msg.get("role", "user"), "משתמש")
        content = msg.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)
