FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NO_SHOW_URLS=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY entrypoint.py /app/entrypoint.py

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev git git-lfs curl \
    && git lfs install \
    && pip install "typer<0.13.0" "click<8.2.0"

RUN git clone https://github.com/StonyBrookNLP/appworld.git /tmp/appworld \
    && cd /tmp/appworld \
    && git lfs pull \
    && pip install "." \
    && pip install ".[mcp]" \
    && appworld install \
    && cd /app \
    && appworld download data

# Keep git and source files for runtime - needed by appworld in Kubernetes
# RUN apt-get purge -y --auto-remove build-essential python3-dev git \
#     && rm -rf /var/lib/apt/lists/*

# Clean up only build tools, keep git and git-lfs
RUN apt-get purge -y --auto-remove build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*


EXPOSE 8000 8001

ENV APIS_PORT=8000 \
    APIS_ON_DISK=1 \
    MCP_TRANSPORT=http \
    MCP_PORT=8001 \
    REMOTE_APIS_URL=http://localhost:8000

CMD ["python", "/app/entrypoint.py"]
