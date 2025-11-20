# Shopping Agent MCP Server - Troubleshooting Guide

## ❌ Error: "Not Acceptable: Client must accept text/event-stream"

### What This Error Means

This error is **NORMAL and EXPECTED** when you try to access `http://localhost:8000/mcp` in a web browser!

**Why it happens:**
- The `/mcp` endpoint uses the MCP protocol (JSON-RPC over HTTP/SSE)
- Browsers send standard HTTP headers that don't include `Accept: text/event-stream`
- The server correctly rejects the request because it's not a proper MCP client

**This does NOT mean your server is broken!** Your server is running correctly - it's just protecting itself from improper requests.

---

## ✅ How to Properly Test Your MCP Server

### Method 1: Use the MCP Inspector (RECOMMENDED)

The MCP Inspector is the official tool for testing MCP servers:

1. **Start your server** (in one terminal):
   ```bash
   cd mcp/shopping_agent
   export OPENAI_API_KEY="your-key-here"
   export SERPAPI_API_KEY="your-key-here"
   python3 shopping_agent.py
   ```

2. **Launch the Inspector** (in another terminal):
   ```bash
   npx @modelcontextprotocol/inspector
   ```

3. **Connect to your server** (in the Inspector UI at http://localhost:5173):
   - Click **"Add Server"**
   - Name: `Shopping Agent`
   - Transport: **HTTP** (or "Streamable HTTP" in older versions)
   - URL: `http://localhost:8000`
   - Click **"Connect"**

4. **Test the tools**:
   - Go to the **"Tools"** tab
   - Select `recommend_products` or `search_products`
   - Fill in the parameters
   - Click "Run"
   - See the results!

### Method 2: Direct Function Testing

Test the functions directly without the MCP protocol layer:

```bash
cd mcp/shopping_agent
export OPENAI_API_KEY="your-key-here"
export SERPAPI_API_KEY="your-key-here"
python3 test_simple.py
```

This imports and calls the functions directly, bypassing the MCP protocol.

### Method 3: Using MCP JSON-RPC (Advanced)

If you want to test the MCP protocol directly with curl:

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2024-11-05",
      "capabilities": {},
      "clientInfo": {"name": "test-client", "version": "1.0.0"}
    }
  }'
```

Then to call a tool:

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "recommend_products",
      "arguments": {
        "query": "I want to buy a scarf for 40 dollars",
        "maxResults": 5
      }
    }
  }'
```

---

## Common Issues

### "API key not configured"

**Solution:** Set your environment variables before starting the server:

```bash
export OPENAI_API_KEY="sk-..."
export SERPAPI_API_KEY="..."
```

Then restart the server.

### "Server shows running but I get errors"

**This is expected!** The server IS running correctly. You're just accessing it the wrong way. See the testing methods above.

### "I want to see if the server is alive"

The server doesn't have a `/health` endpoint currently. To verify it's running:

1. Check the terminal - you should see: `Starting MCP server transport=http host=0.0.0.0 port=8000`
2. Use the MCP Inspector to connect
3. Or use the curl JSON-RPC examples above

---

## Understanding MCP Servers

MCP (Model Context Protocol) servers are NOT traditional REST APIs:

| Traditional REST API | MCP Server |
|---------------------|------------|
| GET /api/products | JSON-RPC method: tools/call |
| Direct browser access ✅ | Direct browser access ❌ |
| Simple curl works | Needs proper MCP client |
| Returns JSON | Uses JSON-RPC 2.0 protocol |

**Key Point:** MCP servers need proper clients. You can't just open them in a browser!

---

## Quick Reference

### Is my server working?
✅ Yes, if you see: `Starting MCP server transport=http host=0.0.0.0 port=8000`

### Why can't I access it in my browser?
🚫 MCP protocol requires proper clients, not browsers

### How do I test it?
✅ Use MCP Inspector: `npx @modelcontextprotocol/inspector`

### Where do I connect?
✅ URL: `http://localhost:8000` (NOT /mcp!)

---

## Summary

**The error you're seeing is CORRECT behavior!**

Your server is:
- ✅ Running properly
- ✅ Protecting itself from improper requests
- ✅ Ready to accept MCP protocol requests

To use it, you need:
- ✅ MCP Inspector (recommended)
- ✅ Proper MCP client
- ✅ Or direct function calls for testing

**Next step:** Launch the MCP Inspector and connect to `http://localhost:8000`

