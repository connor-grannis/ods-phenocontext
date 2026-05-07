# Multi-stage build for the PhenoContext production inference image.
#
# Stage 1 (builder): installs uv and resolves/installs runtime-only deps into
#   /app/.venv using the frozen lockfile — no dev or teacher group.
# Stage 2 (runtime): copies the venv, package source, AND the Python interpreter
#   from the builder into the CUDA runtime image.  Copying Python avoids the
#   broken-symlink problem that occurs when apt installs Python to a different
#   path than the one the uv venv was built against (/usr/local vs /usr/bin).
#
# The pytorch-cu124 index is declared in pyproject.toml so uv pulls the
# CUDA wheel on Linux automatically from the lockfile.
# Build with: docker build --platform linux/amd64 -t ods-phenocontext:dev .

# ── Stage 1: dependency builder ─────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

# Copy only the files uv needs to resolve and install deps.
# Source code comes later so this layer is cached independently.
COPY pyproject.toml uv.lock ./

# Install runtime deps into a local venv using the exact locked versions.
# --no-dev and --no-group teacher keep AWS/langchain out of the image.
# --no-install-project defers building the package itself until src/ is copied.
RUN uv sync --frozen --no-dev --no-group teacher --no-install-project

# Copy source and install the package itself into the venv
COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-group teacher

# ── Stage 2: runtime image ───────────────────────────────────────────────────
# nvidia/cuda:12.4.1-runtime matches the cu124 wheel index used for Linux.
# Falls back to CPU gracefully if no GPU is attached.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

# libgomp is required by torch/numpy for multi-threaded ops
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the Python interpreter from the builder so venv symlinks resolve correctly.
# The uv builder image installs Python under /usr/local; copying it here keeps
# the venv's shebang lines and symlinks intact without a separate apt install.
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin/python3.11 /usr/local/bin/python3.11
COPY --from=builder /usr/local/bin/python3 /usr/local/bin/python3
COPY --from=builder /usr/local/bin/python /usr/local/bin/python

# Copy the fully-populated venv and package source from the builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Prepend the venv's bin dir so `python` and all console scripts resolve first
ENV PATH="/app/.venv/bin:$PATH"

# Sanity check: print torch version and CUDA availability on startup
CMD ["python", "-c", "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"]
