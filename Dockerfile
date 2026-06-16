# syntax=docker/dockerfile:1
# Multi-stage build keeps the runtime image small.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 is required by LightGBM/XGBoost (OpenMP runtime).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e '.[ml,dev]'

# Tests live outside the package; copy them for the CI/test target.
COPY tests ./tests

# Default: build the dataset, run EDA, then the baselines.
CMD ["bash", "-lc", "fraud-data && fraud-eda && fraud-baseline"]
