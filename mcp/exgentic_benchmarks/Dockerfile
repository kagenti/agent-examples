FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Build arguments
ARG BENCHMARK_NAME
ARG RELEASE_VERSION="main"

# Validate BENCHMARK_NAME is provided
RUN if [ -z "$BENCHMARK_NAME" ]; then \
        echo "ERROR: BENCHMARK_NAME build argument is required"; \
        echo "Usage: docker build --build-arg BENCHMARK_NAME=<benchmark> -t exgentic-mcp-<benchmark> ."; \
        exit 1; \
    fi

# Install system dependencies (git is needed for cloning)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configure git to handle large repositories and avoid HTTP/2 issues
RUN git config --global http.postBuffer 524288000 && \
    git config --global http.version HTTP/1.1 && \
    git config --global core.compression 0

# Set working directory
WORKDIR /app

# Clone the Exgentic repository
RUN git clone https://github.com/Exgentic/exgentic.git /app/exgentic

# Checkout the feature/mcp-command branch
WORKDIR /app/exgentic
RUN git checkout feature/mcp-command

# Install Exgentic and its dependencies using uv
RUN uv pip install --system --no-cache -e .

# Setup the benchmark with UV_SYSTEM_PYTHON to ensure uv uses system Python
ENV UV_SYSTEM_PYTHON=1
RUN exgentic setup --benchmark ${BENCHMARK_NAME}

# Copy entrypoint script
WORKDIR /app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set environment variables
ENV BENCHMARK_NAME=${BENCHMARK_NAME} \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO \
    PRODUCTION_MODE=True \
    RELEASE_VERSION=${RELEASE_VERSION}

# Create non-root user and set permissions
RUN useradd -m -u 1001 exgentic && \
    chown -R 1001:0 /app && \
    chmod -R g+rwX /app
    
USER 1001

# Set HOME environment variable to /app
ENV HOME=/app


# Expose the default port
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]