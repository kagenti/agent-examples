#!/bin/bash
set -e

# Read environment variables with defaults
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
LOG_LEVEL=${LOG_LEVEL:-INFO}

# Validate BENCHMARK_NAME is set
if [ -z "$BENCHMARK_NAME" ]; then
    echo "ERROR: BENCHMARK_NAME environment variable is not set"
    exit 1
fi

echo "Starting Exgentic MCP Server"
echo "Benchmark: $BENCHMARK_NAME"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Log Level: $LOG_LEVEL"

# Change to the exgentic directory
cd /app/exgentic

# Run the exgentic MCP server
exec exgentic mcp --benchmark "$BENCHMARK_NAME" --host "$HOST" --port "$PORT"

# Made with Bob
