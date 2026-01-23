FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NO_SHOW_URLS=1 \
    PYTHONPATH=/app/src \
    APPWORLD_CACHE=/app/.cache/appworld

WORKDIR /app

# Create non-root user early
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser \
    && mkdir /home/appuser \
    && chown -R appuser:appuser /home/appuser \
    && chown -R appuser:appuser /app

COPY entrypoint.py /app/entrypoint.py

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev git git-lfs curl \
    && git lfs install \
    && pip install "typer<0.13.0" "click<8.2.0"

# Switch to non-root user
USER appuser

# Add .local/bin to PATH for the appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"

RUN git clone https://github.com/StonyBrookNLP/appworld.git /tmp/appworld \
    && cd /tmp/appworld \
    && git lfs pull \
    && pip install ".[mcp]" \
    && cd /app \
    && appworld install \
    && appworld download data

# Switch back to root to clean up
USER root

# Clean up only build tools, keep git and git-lfs
RUN apt-get purge -y --auto-remove build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Switch back to non-root user for runtime
USER appuser

EXPOSE 8000 8001

ENV APIS_PORT=8000 \
    APPWORLD_ROOT=/app \
    APIS_ON_DISK=1 \
    MCP_TRANSPORT=http \
    MCP_PORT=8001 \
    REMOTE_APIS_URL=http://localhost:8000

CMD ["python", "/app/entrypoint.py"]
