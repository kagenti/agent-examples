FROM docker.io/library/python:3.12-slim-bookworm
ARG RELEASE_VERSION="main"

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app
COPY . .
RUN uv sync --no-cache --locked --link-mode copy

ENV PRODUCTION_MODE=True \
    RELEASE_VERSION=${RELEASE_VERSION}

RUN chown -R 1001:1001 /app
USER 1001

CMD ["uv", "run", "--no-sync", "weather_tool.py"]
