"""Simple test by directly calling the tool functions"""

import os
import sys

# Make sure we have the API keys
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY not set")
    sys.exit(1)

if not os.getenv("SERPAPI_API_KEY"):
    print("❌ Error: SERPAPI_API_KEY not set")
    sys.exit(1)

print("✅ API keys found")
print("🛍️  Testing Shopping Agent Functions Directly")
print("=" * 60)

# Import the functions from shopping_agent
from shopping_agent import recommend_products, search_products

# Test 1: Recommend products
print("\n📊 Test 1: Recommend scarves under $40")
print("-" * 60)
result1 = recommend_products(
    query="I want to buy a scarf for 40 dollars",
    maxResults=3
)
print(result1)

print("\n\n📊 Test 2: Search for winter jackets")
print("-" * 60)
result2 = search_products(
    query="winter jacket waterproof",
    maxResults=5
)
# Print first 500 chars to avoid overwhelming output
print(result2[:500] + "..." if len(result2) > 500 else result2)

print("\n\n" + "=" * 60)
print("✅ Tests completed!")
print("=" * 60)
print("\n💡 To test via MCP protocol, use the MCP Inspector:")
print("   1. Make sure server is running: python3 shopping_agent.py")
print("   2. Run: npx @modelcontextprotocol/inspector")
print("   3. Connect to: http://localhost:8000")

