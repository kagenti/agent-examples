FROM python:3.12-slim-bookworm
ARG RELEASE_VERSION="main"

# Install system tools for sandboxed execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    # Install GitHub CLI
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
       -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
       > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app
ARG CACHE_BUST
COPY . .
RUN uv sync --no-cache --locked --link-mode copy

ENV PRODUCTION_MODE=True \
    RELEASE_VERSION=${RELEASE_VERSION}

# Create workspace and set permissions.
# Use chmod g+w so OCP arbitrary UIDs (same group) can write to /app.
RUN mkdir -p /workspace && chown -R 1001:0 /app /workspace && chmod -R g+w /app /workspace
USER 1001

CMD ["uv", "run", "--no-sync", "server"]
