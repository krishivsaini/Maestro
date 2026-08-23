# Maestro — deploy image for the FastAPI service + coordination viewer.
#
# Two stages so the toolchain never ships: the builder resolves the locked
# dependency set into a venv, the runtime carries only that venv plus source.
#
# Deliberately omits the `embeddings` extra (sentence-transformers -> torch,
# ~670MB installed). LongTermMemory defaults to HashingEmbedder, so the service
# is fully functional without it — and the image fits Render's free 512MB tier.

# ---------- builder ----------
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies resolve from the lockfile in their own layer, so editing source
# does not re-download the tree. --no-install-project: source is copied below.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY maestro/ ./maestro/
COPY README.md ./
RUN uv sync --frozen --no-dev

# ---------- runtime ----------
FROM python:3.11-slim AS runtime

# faiss-cpu's wheel links against libgomp (OpenMP); python:*-slim omits it and
# the import dies with "libgomp.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 maestro

WORKDIR /app

# WORKDIR creates /app as root, and COPY --chown only covers what it copies — so the
# non-root user could not create the trace DB in the working dir. Give runtime state
# its own owned directory and point the defaults at it, so `docker run` needs no env.
RUN mkdir -p /app/data && chown -R maestro:maestro /app

COPY --from=builder --chown=maestro:maestro /app/.venv /app/.venv
COPY --chown=maestro:maestro maestro/ ./maestro/
COPY --chown=maestro:maestro viewer/ ./viewer/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Bind all interfaces — loopback is unreachable from the host's proxy.
    MAESTRO_HOST=0.0.0.0 \
    # Fallback only; the platform's injected PORT wins (see config.Settings.port).
    PORT=8000 \
    # Writable by the non-root user. Ephemeral unless a volume is mounted here.
    MAESTRO_TRACE_DB_PATH=/app/data/maestro_runs.db \
    MAESTRO_MEMORY_STORE_DIR=/app/data/memory_store

USER maestro

EXPOSE 8000

# Honours PORT/MAESTRO_HOST via get_settings(); see maestro/serve.py:main.
CMD ["maestro-serve"]
