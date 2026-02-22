"""
llm.py - CiteWise LLM Abstraction Layer
=========================================
Provides a unified LLM interface that supports:
  - Local: Ollama (llama3, mistral, etc.) — default, offline operation
  - Cloud: OpenAI-compatible API (set LLM_PROVIDER=openai in .env)
  - Cloud: Anthropic Claude (set LLM_PROVIDER=anthropic in .env)

The generate() function is the single entry point used by the API layer.
Supports streaming for real-time UI response display.

Author: CiteWise Senior Legal AI Architect
PEP 8 compliant.
"""

import logging
import os
from typing import Generator, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").lower()
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# Cloud provider keys (only needed if LLM_PROVIDER != "ollama")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")


# ---------------------------------------------------------------------------
# Ollama Provider
# ---------------------------------------------------------------------------

def _generate_ollama(
    prompt: str,
    system: Optional[str] = None,
    stream: bool = False,
) -> Generator[str, None, None]:
    """
    Generate a response via the local Ollama server.

    Yields text chunks if stream=True, else yields a single complete string.
    """
    try:
        import ollama  # type: ignore
    except ImportError:
        raise RuntimeError(
            "ollama package not installed. Run: pip install ollama"
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        if stream:
            response_stream = ollama.chat(
                model=LLM_MODEL,
                messages=messages,
                stream=True,
                options={
                    "temperature": LLM_TEMPERATURE,
                    "num_predict": LLM_MAX_TOKENS,
                },
            )
            for chunk in response_stream:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
        else:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=messages,
                stream=False,
                options={
                    "temperature": LLM_TEMPERATURE,
                    "num_predict": LLM_MAX_TOKENS,
                },
            )
            yield response["message"]["content"]

    except Exception as exc:
        logger.error("Ollama generation error: %s", exc)
        yield f"שגיאה בהפעלת המודל המקומי: {exc}"


# ---------------------------------------------------------------------------
# OpenAI Provider
# ---------------------------------------------------------------------------

def _generate_openai(
    prompt: str,
    system: Optional[str] = None,
    stream: bool = False,
) -> Generator[str, None, None]:
    """Generate via OpenAI-compatible API (also supports local vLLM etc.)."""
    try:
        import openai  # type: ignore
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        if stream:
            with client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                stream=True,
            ) as resp_stream:
                for chunk in resp_stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        yield delta
        else:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            yield response.choices[0].message.content or ""

    except Exception as exc:
        logger.error("OpenAI generation error: %s", exc)
        yield f"שגיאה בגישה ל-API: {exc}"


# ---------------------------------------------------------------------------
# Anthropic Provider
# ---------------------------------------------------------------------------

def _generate_anthropic(
    prompt: str,
    system: Optional[str] = None,
    stream: bool = False,
) -> Generator[str, None, None]:
    """Generate via Anthropic Claude API."""
    try:
        import anthropic  # type: ignore
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    kwargs = {
        "model": LLM_MODEL,
        "max_tokens": LLM_MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    try:
        if stream:
            with client.messages.stream(**kwargs) as stream_resp:
                for text in stream_resp.text_stream:
                    yield text
        else:
            response = client.messages.create(**kwargs)
            yield response.content[0].text

    except Exception as exc:
        logger.error("Anthropic generation error: %s", exc)
        yield f"שגיאה בגישה ל-Anthropic API: {exc}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "ollama": _generate_ollama,
    "openai": _generate_openai,
    "anthropic": _generate_anthropic,
}


def generate(
    prompt: str,
    system: Optional[str] = None,
    stream: bool = False,
) -> Generator[str, None, None]:
    """
    Unified LLM generation entry point.

    Dispatches to the correct provider based on LLM_PROVIDER env variable.
    Supports streaming for real-time Streamlit display.

    Parameters
    ----------
    prompt : str
        The fully assembled prompt string (from prompt.py).
    system : Optional[str]
        System instruction override. If None, uses the prompt's embedded
        system context.
    stream : bool
        If True, yields text chunks progressively. Ideal for Streamlit
        st.write_stream() usage.

    Yields
    ------
    str
        Text tokens or complete response depending on stream flag.
    """
    provider_fn = _PROVIDERS.get(LLM_PROVIDER)
    if provider_fn is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{LLM_PROVIDER}'. "
            f"Valid options: {list(_PROVIDERS.keys())}"
        )

    logger.info(
        "Generating with provider='%s' model='%s' stream=%s",
        LLM_PROVIDER, LLM_MODEL, stream,
    )
    yield from provider_fn(prompt=prompt, system=system, stream=stream)


def generate_full(prompt: str, system: Optional[str] = None) -> str:
    """
    Convenience wrapper: collect full streamed response into a single string.

    Use this when streaming is not needed (e.g. background tasks,
    definition extraction, testing).
    """
    return "".join(generate(prompt=prompt, system=system, stream=False))
