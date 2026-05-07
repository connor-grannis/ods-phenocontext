# Multi-stage build for the PhenoContext production inference image.
#
# Stage 1 (builder): installs uv and resolves/installs runtime-only deps into
#   /app/.venv using the frozen lockfile — no dev or teacher group.
# Stage 2 (runtime): copies the venv + package source into a lean CUDA runtime
#   image; no build tools, no AWS creds, no langchain.
#
# The pytorch-cu124 index is declared in pyproject.toml so uv will pull the
# CUDA wheel on Linux automatically from the lockfile.

# ── Stage 1: dependency builder ─────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

# Copy only the files uv needs to resolve and install deps.
# Source code comes later so dep layer is cached independently.
COPY pyproject.toml uv.lock ./

# Install runtime deps into a local venv using the exact locked versions.
# --no-dev and --no-group teacher keep AWS/langchain out of the image.
# --no-install-project skips building the package itself (done after COPY src).
RUN uv sync --frozen --no-dev --no-group teacher --no-install-project

# Now copy source and install the package itself
COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-group teacher

# ── Stage 2: runtime image ───────────────────────────────────────────────────
# nvidia/cuda:12.4.1-runtime matches the cu124 wheel index used for Linux.
# Falls back to CPU gracefully if no GPU is present.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

# Install Python 3.11 — the CUDA image ships Ubuntu without it
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-distutils \
        libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the fully-populated venv and source from the builder stage
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Activate the venv by prepending its bin dir to PATH
ENV PATH="/app/.venv/bin:$PATH"
# Tell Python where to find the package source
ENV PYTHONPATH="/app/src"

# Sanity: print torch version and CUDA availability on startup
CMD ["python", "-c", "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"]
