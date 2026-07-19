FROM quay.io/fedora/python-314@sha256:381b2b1135f4c506cc1287223eff3ec6ff585638297ecb141667203a89116375

USER root
RUN dnf install -y git && dnf clean all
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml /app/
COPY wiki_service.py /app/
COPY run_local.py /app/
COPY mcp_server.py /app/
RUN uv sync --no-dev && chmod -R 755 /app/.venv

RUN mkdir -p /tmp/uv-cache && chmod 755 /tmp/uv-cache
USER 1001

ENV UV_CACHE_DIR=/tmp/uv-cache
ENV UV_LINK_MODE=copy
ENV WIKI_ROOT=/data/wiki
ENV ACL_FILE=/config/acl.yaml
ENV SPIFFE_TRUST_DOMAIN=rossoctl.example.com

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "wiki_service:app", "--host", "0.0.0.0", "--port", "8000"]
