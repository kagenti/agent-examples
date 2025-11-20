#!/bin/bash

echo "🧪 Testing Shopping Agent MCP Server"
echo "===================================="
echo ""

# Test 1: Check if server is running
echo "Test 1: Server Health Check"
echo "----------------------------"
response=$(curl -s -w "\n%{http_code}" http://localhost:8000/health 2>&1)
echo "Response: $response"
echo ""

# Test 2: List MCP capabilities (using SSE endpoint)
echo "Test 2: Initialize MCP Connection"
echo "----------------------------------"
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test-client","version":"1.0.0"}}}' \
  2>&1 | head -20
echo ""
echo ""

# Test 3: Call the recommend_products tool directly (if using standard REST)
echo "Test 3: Test recommend_products tool"
echo "-------------------------------------"
echo "Note: This requires proper MCP protocol - use MCP Inspector for full testing"
echo ""

echo "✅ If you see responses above, your server is working!"
echo ""
echo "To properly test the server, use the MCP Inspector:"
echo "  1. Keep your server running on port 8000"
echo "  2. Run: npx @modelcontextprotocol/inspector"
echo "  3. Add server with URL: http://localhost:8000"
echo "  4. Connect and test the tools"

