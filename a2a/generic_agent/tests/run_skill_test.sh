#!/bin/bash

# Script to run the Generic Agent skill loading test
# This script sets up the environment and runs the test

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "========================================"
echo "Generic Agent - Skill Loading Test"
echo "========================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Set default environment variables
export SKILL_FOLDERS="${SKILL_FOLDERS:-$PROJECT_ROOT/skills/summarizer}"
export LLM_MODEL="${LLM_MODEL:-llama3.2:3b-instruct-fp16}"
export LLM_API_BASE="${LLM_API_BASE:-http://localhost:11434/v1}"
export LLM_API_KEY="${LLM_API_KEY:-dummy}"
export MCP_URLS="${MCP_URLS:-http://localhost:8000/mcp}"
export MCP_TRANSPORT="${MCP_TRANSPORT:-streamable_http}"

echo -e "${GREEN}Configuration:${NC}"
echo "  Project Root: $PROJECT_ROOT"
echo "  Skill Folders: $SKILL_FOLDERS"
echo "  LLM Model: $LLM_MODEL"
echo "  LLM API Base: $LLM_API_BASE"
echo ""

# Check if Ollama is running
echo -e "${YELLOW}Checking prerequisites...${NC}"
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${RED}✗ ERROR: Ollama is not running or not accessible at http://localhost:11434${NC}"
    echo "  Please start Ollama and try again."
    echo "  Visit: https://ollama.ai for installation instructions"
    exit 1
fi
echo -e "${GREEN}✓ Ollama is running${NC}"

# Check if the model is available
if ! ollama list | grep -q "$LLM_MODEL"; then
    echo -e "${YELLOW}⚠ Warning: Model '$LLM_MODEL' not found locally${NC}"
    echo "  Attempting to pull the model..."
    if ! ollama pull "$LLM_MODEL"; then
        echo -e "${RED}✗ ERROR: Failed to pull model '$LLM_MODEL'${NC}"
        echo "  Available models:"
        ollama list
        exit 1
    fi
fi
echo -e "${GREEN}✓ Model '$LLM_MODEL' is available${NC}"

# Check if skill folder exists
if [ ! -d "$SKILL_FOLDERS" ]; then
    echo -e "${RED}✗ ERROR: Skill folder does not exist: $SKILL_FOLDERS${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Skill folder exists${NC}"

# Check if SKILL.md exists
if [ ! -f "$SKILL_FOLDERS/SKILL.md" ]; then
    echo -e "${RED}✗ ERROR: SKILL.md not found in: $SKILL_FOLDERS${NC}"
    exit 1
fi
echo -e "${GREEN}✓ SKILL.md found${NC}"

echo ""
echo "========================================"
echo "Running Test"
echo "========================================"
echo ""

# Change to the script directory
cd "$SCRIPT_DIR"

# Run the test
python manual_test_skill_loading.py

# Capture exit code
TEST_EXIT_CODE=$?

echo ""
echo "========================================"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Test completed successfully!${NC}"
else
    echo -e "${RED}✗ Test failed with exit code: $TEST_EXIT_CODE${NC}"
fi
echo "========================================"

exit $TEST_EXIT_CODE

# Made with Bob
