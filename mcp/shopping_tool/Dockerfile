FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY shopping_agent.py .
COPY __init__.py .
COPY README.md .

# Install dependencies
RUN uv pip install --system -e .

# Environment variables (can be overridden at runtime)
ENV HOST=0.0.0.0
ENV PORT=8000
ENV MCP_TRANSPORT=http
ENV LOG_LEVEL=INFO

# Expose the port
EXPOSE 8000

# Run the server
CMD ["python", "shopping_agent.py"]

