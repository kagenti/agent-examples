FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
ARG RELEASE_VERSION="main"

WORKDIR /app
COPY . .
RUN uv sync --no-cache --locked --link-mode copy

ENV PRODUCTION_MODE=True \
    RELEASE_VERSION=${RELEASE_VERSION}

RUN chown -R 1001:1001 /app
USER 1001

# Default port 8000; override with PORT env var.
# The kagenti operator sets PORT during proxy-sidecar port-stealing.
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "server"]