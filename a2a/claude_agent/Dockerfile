# Claude needs Node (the CLI is an npm package); the A2A server is Python.
# Pinned by multi-arch index digest for reproducible/supply-chain-safe builds.
FROM node:22-bookworm-slim@sha256:7af03b14a13c8cdd38e45058fd957bf00a72bbe17feac43b1c15a689c029c732
ARG RELEASE_VERSION="main"

# System deps: git + ripgrep are used by Claude; ca-certificates for TLS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ripgrep ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The Claude CLI (`claude`). Pinned for reproducible/supply-chain-safe builds.
RUN npm install -g @anthropic-ai/claude-code@2.1.161

# uv (Python package manager); it will fetch a managed CPython during `uv sync`.
# Pinned to a fixed version (avoid floating :latest).
COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /usr/local/bin/uv

# Install uv's managed CPython into a world-readable location so the non-root
# user can use it (the default lives under root's HOME, which is mode 700).
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python

WORKDIR /app
COPY . .
RUN uv sync --no-cache --locked --link-mode copy

# Non-root user with a writable HOME (for ~/.claude) and workspace root.
RUN useradd -u 1001 -m -d /home/agent agent \
    && mkdir -p /workspace \
    && chown -R 1001:1001 /app /workspace /home/agent /opt/uv

ENV HOME=/home/agent \
    WORKSPACE_ROOT=/workspace \
    PRODUCTION_MODE=True \
    RELEASE_VERSION=${RELEASE_VERSION}

USER 1001

CMD ["uv", "run", "--no-sync", "server"]
