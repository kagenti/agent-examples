FROM quay.io/pdettori/python:3.12-slim

RUN apt-get update && apt-get install -y git

COPY --from=ghcr.io/astral-sh/uv:0.6.8 /uv /uvx /bin/

ADD . /app

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv uv sync

ENV PATH="/app/.venv/bin:$PATH"

CMD ["uv", "run", "server", "--host", "0.0.0.0", "--port", "8000"]
