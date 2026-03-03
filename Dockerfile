# Python Worker - Single comprehensive worker for 80-90% of use cases
#
# Includes: pandas, polars, numpy, scikit-learn, duckdb, pyarrow,
#           sentence-transformers, tiktoken, transformers, and core utilities
#
# Uses CPU-only PyTorch (~200 MB) instead of CUDA build (~2.5 GB)
# Multi-stage build keeps build-essential out of the final image
#
# Estimated image size (uncompressed / compressed): ~2.3 GB / ~500 MB
#
# Usage (from kruxiaflow-python/ directory):
#   docker build --target py-std -t kruxia/kruxiaflow-py-std .

# =============================================================================
# Builder stage - install all dependencies with build tools
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies for compiled packages (numpy, scikit-learn, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy package source
COPY pyproject.toml README.md ./
COPY kruxiaflow/ ./kruxiaflow/

# Install CPU-only PyTorch first, then the rest of the std dependencies
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[std]"

# =============================================================================
# Runtime stage - clean base without build tools
# =============================================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# =============================================================================
# py-std: Comprehensive Python worker
# Copies installed packages from builder into clean runtime image
# =============================================================================
FROM runtime AS py-std

# Copy installed Python packages and scripts from builder
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application source
COPY --from=builder /app /app

ENV WORKER_TYPE=py-std
CMD ["kruxiaflow-worker"]
