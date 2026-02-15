FROM python:3.12-slim-bookworm
ARG RELEASE_VERSION="main"

# Install system tools for sandboxed execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app
COPY . .
RUN uv sync --no-cache --locked --link-mode copy

ENV PRODUCTION_MODE=True \
    RELEASE_VERSION=${RELEASE_VERSION}

RUN mkdir -p /workspace && chown -R 1001:1001 /app /workspace
USER 1001

CMD ["uv", "run", "--no-sync", "server"]
