#!/bin/bash
# Demo script for Restaurant Reservation MCP Server
# This script demonstrates the full workflow of the reservation system

set -e

echo "=================================="
echo "Restaurant Reservation MCP Demo"
echo "=================================="
echo

# Check if server is running
MCP_URL="${MCP_URL:-http://localhost:8000/mcp}"
echo "Using MCP endpoint: $MCP_URL"
echo

# Initialize MCP session
echo "Step 1: Initializing MCP session..."
INIT_RESPONSE=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  "$MCP_URL" \
  --data '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2025-03-26",
      "capabilities": {},
      "clientInfo": {
        "name": "DemoClient",
        "version": "1.0.0"
      }
    }
  }')

SESSION_ID=$(echo "$INIT_RESPONSE" | grep -i "mcp-session-id:" | sed 's/mcp-session-id: //I' | tr -d '\r')
echo "Session ID: $SESSION_ID"
echo

# Send initialized notification
curl -s -X POST \
  -H "mcp-session-id: $SESSION_ID" \
  -H "Content-Type: application/json" \
  "$MCP_URL" \
  --data '{
    "jsonrpc": "2.0",
    "method": "notifications/initialized"
  }' > /dev/null

echo "Step 2: Listing available tools..."
curl -s -X POST \
  -H "mcp-session-id: $SESSION_ID" \
  -H "Content-Type: application/json" \
  "$MCP_URL" \
  --data '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list"
  }' | jq '.result.tools[] | {name: .name, description: .description}'
echo

echo "Step 3: Searching for Italian restaurants in Boston..."
SEARCH_RESULT=$(curl -s -X POST \
  -H "mcp-session-id: $SESSION_ID" \
  -H "Content-Type: application/json" \
  "$MCP_URL" \
  --data '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "name": "search_restaurants",
      "arguments": {
        "city": "Boston",
        "cuisine": "Italian"
      }
    }
  }')

echo "$SEARCH_RESULT" | jq -r '.result.content[0].text' | jq '.[0] | {name, cuisine, price_tier, rating, address: .location.address}'
RESTAURANT_ID=$(echo "$SEARCH_RESULT" | jq -r '.result.content[0].text' | jq -r '.[0].id')
echo "Selected restaurant ID: $RESTAURANT_ID"
echo

echo "Step 4: Checking availability for party of 4 on 2025-12-25..."
AVAILABILITY_RESULT=$(curl -s -X POST \
  -H "mcp-session-id: $SESSION_ID" \
  -H "Content-Type: application/json" \
  "$MCP_URL" \
  --data "{
    \"jsonrpc\": \"2.0\",
    \"id\": 4,
    \"method\": \"tools/call\",
    \"params\": {
      \"name\": \"check_availability\",
      \"arguments\": {
        \"restaurant_id\": \"$RESTAURANT_ID\",
        \"date_time\": \"2025-12-25T18:00:00\",
        \"party_size\": 4
      }
    }
  }")

echo "$AVAILABILITY_RESULT" | jq -r '.result.content[0].text' | jq '[.[] | select(.available == true)] | .[0:3] | .[] | {time, max_party_size, available}'
SLOT_TIME=$(echo "$AVAILABILITY_RESULT" | jq -r '.result.content[0].text' | jq -r '[.[] | select(.available == true)][0].time')
echo "Selected time slot: $SLOT_TIME"
echo

echo "Step 5: Placing a reservation..."
RESERVATION_RESULT=$(curl -s -X POST \
  -H "mcp-session-id: $SESSION_ID" \
  -H "Content-Type: application/json" \
  "$MCP_URL" \
  --data "{
    \"jsonrpc\": \"2.0\",
    \"id\": 5,
    \"method\": \"tools/call\",
    \"params\": {
      \"name\": \"place_reservation\",
      \"arguments\": {
        \"restaurant_id\": \"$RESTAURANT_ID\",
        \"date_time\": \"$SLOT_TIME\",
        \"party_size\": 4,
        \"name\": \"Jane Smith\",
        \"phone\": \"+1-555-987-6543\",
        \"email\": \"jane@example.com\",
        \"notes\": \"Window seat preferred\"
      }
    }
  }")

echo "$RESERVATION_RESULT" | jq -r '.result.content[0].text' | jq '{id, restaurant_name, date_time, party_size, guest_name, confirmation_code, status}'
RESERVATION_ID=$(echo "$RESERVATION_RESULT" | jq -r '.result.content[0].text' | jq -r '.id')
echo "Reservation ID: $RESERVATION_ID"
echo

echo "Step 6: Listing reservations for jane@example.com..."
curl -s -X POST \
  -H "mcp-session-id: $SESSION_ID" \
  -H "Content-Type: application/json" \
  "$MCP_URL" \
  --data '{
    "jsonrpc": "2.0",
    "id": 6,
    "method": "tools/call",
    "params": {
      "name": "list_reservations",
      "arguments": {
        "user_id": "jane@example.com"
      }
    }
  }' | jq -r '.result.content[0].text' | jq '.[] | {id, restaurant_name, date_time, party_size}'
echo

echo "Step 7: Cancelling the reservation..."
curl -s -X POST \
  -H "mcp-session-id: $SESSION_ID" \
  -H "Content-Type: application/json" \
  "$MCP_URL" \
  --data "{
    \"jsonrpc\": \"2.0\",
    \"id\": 7,
    \"method\": \"tools/call\",
    \"params\": {
      \"name\": \"cancel_reservation\",
      \"arguments\": {
        \"reservation_id\": \"$RESERVATION_ID\",
        \"reason\": \"Demo completed\"
      }
    }
  }" | jq -r '.result.content[0].text' | jq '{reservation_id, restaurant_name, cancelled_at, reason}'
echo

echo "=================================="
echo "Demo completed successfully!"
echo "=================================="
