"""Central configuration — every tunable lives here (no magic numbers elsewhere).

All settings are overridable via environment variables with the ``MAESTRO_`` prefix
(e.g. ``MAESTRO_MAX_PARALLEL=3``), except the two provider keys which use their
conventional names (``GOOGLE_API_KEY``, ``SEARCH_API_KEY``).

Rate-limit note (verified live 2026-07 against ai.google.dev): the current stable
free-tier Flash model is ``gemini-3.5-flash`` (~10 RPM / 250K TPM / 1500 RPD).
A multi-agent run multiplies calls, and RPM during parallel bursts is the binding
constraint, so ``max_parallel`` defaults to 2 and every call is backoff-wrapped
(see maestro/resilience.py). ``gemini-3.1-flash-lite`` (~30 RPM) is the
higher-concurrency fallback.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MAESTRO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),  # allow field names starting with `model_`
    )

    # --- Provider / model (verify live before demo; see module docstring) ---
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    model_id: str = "gemini-3.5-flash"
    fallback_model_id: str = "gemini-3.1-flash-lite"  # ~30 RPM, higher concurrency
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # local, free (sentence-transformers)
    temperature: float = 0.2

    # --- Search tool ---
    search_provider: str = "duckduckgo"  # keyless default
    search_api_key: str = Field(default="", alias="SEARCH_API_KEY")
    search_max_results: int = 5

    # --- Concurrency & plan bounds ---
    max_parallel: int = 2  # Gemini free tier ~10 RPM -> keep small
    max_subtasks: int = 6  # a good decomposition is tight, not sprawling

    # --- Loop / cost ceilings ---
    max_steps: int = 40  # global backstop: total node executions per run
    max_critic_iters: int = 3
    max_recovery_attempts: int = 2
    loop_detect_threshold: int = 5  # same delegation dispatched this many times -> loop

    # --- LLM request bounds ---
    # A request with no timeout can stall forever; tenacity below only retries calls
    # that *return*, so an un-timed-out hang never reaches the backoff ladder and the
    # run stops dead with no error to show. Bound the call so a stall becomes a
    # retryable exception. max_retries=0 leaves retrying to resilience.py — the
    # provider client retries internally too, and stacking them multiplies attempts.
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 0

    # --- Backoff / rate-limit resilience (tenacity) ---
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 30.0
    backoff_max_attempts: int = 6
    backoff_jitter_seconds: float = 1.0

    # --- Fault injection (makes visible recovery demoable) ---
    fault_injection: bool = False
    fault_injection_tool: str = "web_search"

    # --- Demo: force the Critic to REJECT its first N reviews ---
    # A genuine LLM critic can reject on its own; this makes a rejection reliably
    # triggerable on demand for the reject->revise->pass demo. 0 = off.
    force_critic_reject: int = 0

    # --- Storage ---
    trace_db_path: str = "maestro_runs.db"
    memory_store_dir: str = "memory_store"

    # --- Service ---
    # Comma-separated origins allowed to call the API from a browser. Empty (the
    # default) keeps CORS off entirely, which is correct when FastAPI serves the
    # viewer itself at ``/`` — same-origin needs no headers. Set this only when the
    # viewer is deployed to a separate static host, e.g.
    # ``MAESTRO_CORS_ORIGINS=https://maestro.pages.dev``.
    cors_origins: str = ""

    # Local dev binds loopback; containers set MAESTRO_HOST=0.0.0.0 (see Dockerfile).
    host: str = "127.0.0.1"
    # PaaS hosts (Render, Cloud Run, Fly) inject a bare ``PORT`` and expect the process
    # to bind exactly it, so accept that name alongside the prefixed ``MAESTRO_PORT``.
    port: int = Field(default=8000, validation_alias=AliasChoices("MAESTRO_PORT", "PORT"))


    def cors_origin_list(self) -> list[str]:
        """``cors_origins`` split into a list; empty when CORS should stay off."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cached settings — used by tests that mutate the environment."""
    get_settings.cache_clear()
