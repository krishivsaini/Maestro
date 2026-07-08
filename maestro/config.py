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

from pydantic import Field
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
    max_parallel: int = 2  # Gemini free tier ~10 RPM -> keep small (§5)
    max_subtasks: int = 6  # a good decomposition is tight, not sprawling (§10)

    # --- Loop / cost ceilings (§15) ---
    max_steps: int = 40
    max_critic_iters: int = 3
    max_recovery_attempts: int = 2

    # --- Backoff / rate-limit resilience (tenacity; §15) ---
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 30.0
    backoff_max_attempts: int = 6
    backoff_jitter_seconds: float = 1.0

    # --- Fault injection (makes visible recovery demoable; §13) ---
    fault_injection: bool = False
    fault_injection_tool: str = "web_search"

    # --- Storage ---
    trace_db_path: str = "maestro_runs.db"
    memory_store_dir: str = "memory_store"

    # --- Service ---
    host: str = "127.0.0.1"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cached settings — used by tests that mutate the environment."""
    get_settings.cache_clear()
