"""LLM binding — the single provider seam (Gemini via langchain-google-genai).

Imports are lazy so that offline code paths (tests using stub/heuristic planners,
the tools, the state model) load without langchain/gemini installed. Everything
that reaches the model goes through here so the model id / provider is swappable
in one place (config.py).
"""

from __future__ import annotations

from typing import Any, Optional

from .config import Settings, get_settings
from .logging_config import get_logger

log = get_logger("llm")


def get_chat_model(
    settings: Optional[Settings] = None,
    *,
    model_id: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Any:
    """Build a Gemini chat model. Raises a clear error if no key is configured."""
    cfg = settings or get_settings()
    if not cfg.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to .env (see .env.example) or pass a "
            "stub/heuristic planner for offline runs."
        )
    from langchain_google_genai import ChatGoogleGenerativeAI  # lazy

    return ChatGoogleGenerativeAI(
        model=model_id or cfg.model_id,
        temperature=cfg.temperature if temperature is None else temperature,
        google_api_key=cfg.google_api_key,
    )
