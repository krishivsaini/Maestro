"""LLM binding — the single provider seam.

Imports are lazy so that offline code paths (tests using stub/heuristic planners,
the tools, the state model) load without langchain/gemini installed. Everything
that reaches the model goes through here so the model id / provider is swappable
in one place (config.py).

A model id may name its provider inline — ``groq:llama-3.3-70b-versatile``,
``nvidia:meta/llama-3.3-70b-instruct``, ``google:gemini-3.6-flash`` — and an
unprefixed id uses ``Settings.llm_provider``. Carrying the provider *in the id*
means one process can address several providers at once without swapping global
config, which is what the benchmark harness and any cross-provider fallback need.

Providers other than Google live in the ``providers`` extra:
``uv sync --extra providers``.
"""

from __future__ import annotations

from typing import Any, Optional

from .config import Settings, get_settings
from .logging_config import get_logger

log = get_logger("llm")


PROVIDERS = ("google", "groq", "nvidia")


def split_model_id(model_id: str, default_provider: str) -> tuple[str, str]:
    """``"groq:llama-3.3"`` -> ``("groq", "llama-3.3")``; unprefixed uses the default.

    Only a known provider counts as a prefix — NVIDIA ids are themselves namespaced
    (``meta/llama-3.3-70b-instruct``) and Google's are not, but a bare colon could
    still appear in a future id, so an unrecognised prefix is left alone.
    """
    prefix, sep, rest = model_id.partition(":")
    if sep and prefix in PROVIDERS:
        return prefix, rest
    return default_provider, model_id


def get_chat_model(
    settings: Optional[Settings] = None,
    *,
    model_id: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Any:
    """Build a chat model for the requested provider. Clear error if no key is set."""
    cfg = settings or get_settings()
    provider, model = split_model_id(model_id or cfg.model_id, cfg.llm_provider)
    temp = cfg.temperature if temperature is None else temperature

    if provider == "google":
        if not cfg.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to .env (see .env.example) or pass a "
                "stub/heuristic planner for offline runs."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI  # lazy

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temp,
            google_api_key=cfg.google_api_key,
            # Both are load-bearing; see Settings.llm_timeout_seconds. Without the
            # timeout a stalled request hangs the run forever with nothing surfaced.
            timeout=cfg.llm_timeout_seconds,
            max_retries=cfg.llm_max_retries,
        )

    if provider == "groq":
        if not cfg.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to .env.")
        try:
            from langchain_groq import ChatGroq  # lazy
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("langchain-groq missing. Run: uv sync --extra providers") from exc

        return ChatGroq(
            model=model,
            temperature=temp,
            api_key=cfg.groq_api_key,
            timeout=cfg.llm_timeout_seconds,
            max_retries=cfg.llm_max_retries,
        )

    if provider == "nvidia":
        if not cfg.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set. Add it to .env.")
        try:
            # NIM is OpenAI-compatible, so the OpenAI client covers it — no extra SDK,
            # and structured output goes through the same tool-calling path.
            from langchain_openai import ChatOpenAI  # lazy
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("langchain-openai missing. Run: uv sync --extra providers") from exc

        return ChatOpenAI(
            model=model,
            temperature=temp,
            api_key=cfg.nvidia_api_key,
            base_url=cfg.nvidia_base_url,
            timeout=cfg.llm_timeout_seconds,
            max_retries=cfg.llm_max_retries,
        )

    raise RuntimeError(f"Unknown llm provider {provider!r}. Expected one of {PROVIDERS}.")
